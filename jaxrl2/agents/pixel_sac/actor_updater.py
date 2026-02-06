from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


# 更新 Actor 网络（策略网络）
# 使用策略梯度方法最大化期望回报，同时保持策略的熵（探索性）
# Actor 的损失函数：L = E[α * log π(a|s) - Q(s, a)]

def update_actor(key: PRNGKey,                          # 随机数生成密钥（用于动作采样）
                 actor: TrainState,                     # Actor 网络（策略网络 π(a|s)）
                 critic: TrainState,                    # Critic 网络（Q 函数 Q(s,a)）
                 temp: TrainState,                      # 温度参数 α（熵正则化系数）
                 batch: DatasetDict,                   # 批次数据（包含观测和动作）
                 cross_norm: bool = False,              # 是否使用交叉归一化
                 critic_reduction: str = 'min'          # Q 网络聚合方式：'min' 或 'mean'
) -> Tuple[TrainState, Dict[str, float]]:               # 返回：新 Actor 状态 + 监控指标

    # 将随机数密钥拆分为两个：一个用于 Actor 损失函数内部，一个用于动作采样
    # key：用于可能的后续操作，key_act：专门用于采样动作
    key, key_act = jax.random.split(key, num=2)

    # 步骤 1：定义 Actor 损失函数
    # 这是一个内部函数，会被 jax.grad 自动微分
    # 输入：actor_params（Actor 网络参数）
    # 输出：(actor_loss, (info_dict, new_model_state))
    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:

        # 步骤 2：前向传播 - 使用 Actor 网络计算动作分布
        # 如果 Actor 使用批归一化（batch normalization），需要处理 batch_stats
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            # mutable=['batch_stats'] 表示 batch_stats 会在前向传播中更新（如更新移动平均）
            dist, new_model_state = actor.apply_fn(
                {'params': actor_params, 'batch_stats': actor.batch_stats},
                batch['observations'],
                mutable=['batch_stats'])

            # 如果启用 cross_norm，也对 next_observations 计算分布（用于某些归一化策略）
            if cross_norm:
                next_dist = actor.apply_fn(
                    {'params': actor_params, 'batch_stats': actor.batch_stats},
                    batch['next_observations'],
                    mutable=['batch_stats'])
            else:
                next_dist = actor.apply_fn(
                    {'params': actor_params, 'batch_stats': actor.batch_stats},
                    batch['next_observations'])

            # 处理返回值可能是 tuple 的情况（有些网络返回状态和 batch_stats）
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            # 如果 Actor 不使用批归一化，直接计算分布
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}

        # 步骤 3：提取分布参数（仅用于日志记录和监控）
        # mean_dist: 分布的均值（确定性部分）
        # std_diag_dist: 分布的标准差（随机性部分）
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        # 步骤 4：从分布中采样动作并计算对数概率
        # actions: 实际采样的动作，用于计算 Q 值和策略梯度
        # log_probs: log π(a|s)，策略的对数概率，用于策略梯度定理和熵计算
        actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        # 步骤 5：使用 Critic 网络评估采样动作的价值
        # qs: Q(s, a)，Critic 对当前状态-动作对的价值估计
        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            # 如果 Critic 使用批归一化，需要处理 batch_stats
            qs, _ = critic.apply_fn(
                {'params': critic.params, 'batch_stats': critic.batch_stats},
                batch['observations'],
                actions,
                mutable=['batch_stats'])
        else:
            # 如果 Critic 不使用批归一化，直接计算 Q 值
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)

        # 步骤 6：Critic Reduction - 聚合多个 Q 网络的预测
        # 如果使用 Q 网络集成（ensemble），需要选择一个 Q 值
        if critic_reduction == 'min':
            q = qs.min(axis=0)      # 取最小值（保守，减少过估计）
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)     # 取平均值（更平滑）
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")

        # 步骤 7：计算 Actor 损失函数
        # 策略梯度定理：∇J = E[∇log π(a|s) * Q(s,a)]
        # SAC 变体：L = E[α * log π(a|s) - Q(s,a)]
        # 第一项是熵正则化（鼓励探索），第二项是最大化 Q 值
        # temp.apply_fn({'params': temp.params}) 返回温度参数 α
        actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()

        # 步骤 8：准备监控指标
        # 这些指标会被记录到 wandb 用于可视化训练过程
        things_to_log = {
            'actor_loss': actor_loss,                  # Actor 损失值
            'entropy': -log_probs.mean(),              # 策略熵（探索程度，越大越随机）
            'q_pi_in_actor': q.mean(),                 # Q(s,a) 在 Actor 中的平均值
            'mean_pi_norm': mean_dist_norm.mean(),     # 策略均值范数（策略幅度）
            'std_pi_norm': std_dist_norm.mean(),       # 策略标准差范数（探索幅度）
            'mean_pi_avg': mean_dist.mean(),           # 策略均值平均值
            'mean_pi_max': mean_dist.max(),            # 策略均值最大值
            'mean_pi_min': mean_dist.min(),            # 策略均值最小值
            'std_pi_avg': std_diag_dist.mean(),        # 策略标准差平均值
            'std_pi_max': std_diag_dist.max(),         # 策略标准差最大值
            'std_pi_min': std_diag_dist.min(),         # 策略标准差最小值
        }

        # 返回损失值和监控指标（包括 model_state 用于批归一化）
        return actor_loss, (things_to_log, new_model_state)

    # 步骤 9：计算梯度并更新 Actor 网络
    # jax.grad 自动微分：计算 actor_loss_fn 关于 actor_params 的梯度
    # has_aux=True 表示损失函数返回 (损失值, 辅助信息)
    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

    # 步骤 10：应用梯度下降更新参数
    # 如果有 batch_stats（批归一化统计量），需要更新它们
    if 'batch_stats' in new_model_state:
        # 更新参数和批归一化统计量
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        # 只更新参数（不使用批归一化或批归一化统计量未更新）
        new_actor = actor.apply_gradients(grads=grads)

    # 返回：更新后的 Actor 训练状态 + 监控指标字典
    # new_actor 包含更新后的网络参数
    # info 包含所有训练指标，可用于 wandb 日志记录
    return new_actor, info