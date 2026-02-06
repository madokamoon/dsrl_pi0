from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


# 更新 Critic 网络（Q 函数）
# 使用 Bellman 方程计算 TD 目标，最小化 TD 误差
# Critic 网络负责评估状态-动作对的价值 Q(s, a)


def update_critic(
        key: PRNGKey,                          # 随机数生成密钥（用于采样动作）
        actor: TrainState,                     # Actor 网络（策略网络 π(a|s)）
        critic: TrainState,                    # Critic 网络（Q 函数 Q(s,a)）
        target_critic: TrainState,             # 目标 Critic 网络（稳定的目标）
        temp: TrainState,                      # 温度参数 α（熵正则化系数）
        batch: DatasetDict,                   # 批次数据（包含 obs, next_obs, actions, rewards 等）
        discount: float,                       # 折扣因子 γ
        backup_entropy: bool = False,          # 是否在 TD 目标中备份熵（默认 False）
        critic_reduction: str = 'min'          # Q 网络聚合方式：'min' 或 'mean'
) -> Tuple[TrainState, Dict[str, float]]:      # 返回：新 Critic 状态 + 监控指标

    # 步骤 1：从 Actor 采样下一动作（用于计算 TD 目标）
    # 使用当前 Actor 策略在下一观测上采样动作 a' ~ π(·|s')
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    # 步骤 2：使用目标 Critic 计算下一状态的 Q 值
    # 输入：(s', a')，输出：Q(s', a')
    # next_qs 的形状：[num_qs, batch_size]，num_qs 是 Q 网络的数量（通常是 10）
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)

    # 步骤 3：Critic Reduction - 聚合多个 Q 网络的预测
    # 使用多个 Q 网络可以减少过估计偏差（overestimation bias）
    # 'min'：取最小值（保守，减少过估计，更稳定）
    # 'mean'：取平均值（更平滑）
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)      # 沿第一维取最小值，得到 [batch_size]
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)     # 沿第一维取平均值，得到 [batch_size]
    else:
        raise NotImplemented()

    # 步骤 4：计算 TD 目标（Target Q 值）
    # Bellman 方程：Q_target(s,a) = r + γ * (1 - done) * Q(s', a')
    # batch['masks'] = 1 - done（如果是终止状态，mask=0）
    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    # 步骤 5：熵正则化（可选）
    # 如果 backup_entropy=True，在 TD 目标中加入熵项
    # 这会鼓励策略保持随机性，公式：target_q -= γ * α * log π(a'|s')
    # 更强的探索能力，避免过早收敛到确定性策略
    if backup_entropy:
        target_q -= batch["discount"] * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs

    # 步骤 6：定义 Critic 损失函数
    # 这是一个内部函数，会被 jax.grad 自动微分
    # 输入：critic_params（Critic 网络参数）
    # 输出：(critic_loss, info_dict)
    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # 步骤 6.1：使用当前 Critic 网络计算 Q(s, a)
        # 输入：(batch['observations'], batch['actions'])
        # 输出：qs - 当前状态-动作对的 Q 值预测
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])

        # 步骤 6.2：计算 TD 误差的均方误差（MSE Loss）
        # critic_loss = E[(Q(s,a) - Q_target(s,a))^2]
        # 这个损失衡量当前 Q 预测与 TD 目标的差距
        critic_loss = ((qs - target_q)**2).mean()

        # 步骤 6.3：返回损失值和监控指标字典
        # 这些指标会被记录到 wandb 用于可视化训练过程
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
            
        }

    # 步骤 7：计算梯度并更新 Critic 网络
    # jax.grad 自动微分：计算 critic_loss_fn 关于 critic_params 的梯度
    # has_aux=True 表示损失函数返回 (损失值, 辅助信息)
    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)

    # 步骤 8：应用梯度下降更新参数
    # 使用 Critic 中保存的优化器（通常是 Adam）更新参数
    # 更新规则：params = params - lr * grads
    new_critic = critic.apply_gradients(grads=grads)

    # 返回：更新后的 Critic 训练状态 + 监控指标字典
    # new_critic 包含更新后的网络参数和优化器状态
    # info 包含所有训练指标，可用于 wandb 日志记录
    return new_critic, info
