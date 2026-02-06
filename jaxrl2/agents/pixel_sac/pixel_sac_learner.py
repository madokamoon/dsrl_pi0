"""
Pixel SAC (Soft Actor-Critic) 学习器实现
用于连续控制任务的强化学习算法，基于论文 https://arxiv.org/abs/1812.05905
这是 DSRL (Diffusion Steering via Reinforcement Learning) 项目的核心组件
"""

import matplotlib  # 导入 matplotlib 用于可视化
matplotlib.use('Agg')  # 设置非交互式后端，避免显示窗口
from flax.training import checkpoints  # Flax 检查点工具，用于模型保存和加载
import pathlib  # 路径处理
import matplotlib.pyplot as plt  # 绘图库
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # 非交互式画布

import numpy as np  # NumPy 数组操作
import copy  # 深拷贝工具
import functools  # 函数工具，用于装饰器
from typing import Dict, Optional, Sequence, Tuple, Union  # 类型提示

import jax  # JAX 框架，用于自动微分和加速计算
import jax.numpy as jnp  # JAX 的 NumPy 实现
import optax  # JAX 优化库
from flax.core.frozen_dict import FrozenDict  # Flax 冻结字典
from flax.training import train_state  # Flax 训练状态管理
from typing import Any  # 任意类型

from jaxrl2.agents.agent import Agent  # Agent 基类
from jaxrl2.data.augmentations import batched_random_crop, color_transform  # 数据增强：随机裁剪和颜色变换
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer  # 视觉编码器和像素多路复用器
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder  # IMPALA 编码器
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall  # ResNet V1 编码器
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder  # ResNet V2 编码器
from jaxrl2.agents.pixel_sac.actor_updater import update_actor  # Actor 更新函数
from jaxrl2.agents.pixel_sac.critic_updater import update_critic  # Critic 更新函数
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature  # 温度参数更新函数
from jaxrl2.agents.pixel_sac.temperature import Temperature  # 温度参数模块
from jaxrl2.data.dataset import DatasetDict  # 数据集字典
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy  # 可学习标准差的高斯策略
from jaxrl2.networks.values import StateActionEnsemble  # 状态-动作值函数集成
from jaxrl2.types import Params, PRNGKey  # 类型定义：参数和随机数生成器密钥
from jaxrl2.utils.target_update import soft_target_update  # 软更新目标网络


class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter',  'aug_next', 'num_cameras'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    """
    JIT 编译的单步 SAC 更新函数

    这是 SAC 算法的核心更新逻辑，包含以下步骤：
    1. 数据增强（随机裁剪 + 颜色抖动）
    2. 更新 Critic 网络（Q 函数）
    3. 软更新目标 Critic 网络
    4. 更新 Actor 网络（策略）
    5. 更新温度参数（α）

    参数:
        rng: 随机数生成器密钥
        actor: Actor 训练状态
        critic: Critic 训练状态
        target_critic_params: 目标 Critic 参数
        temp: 温度参数训练状态
        batch: 从回放缓冲区采样的批次数据
        discount: 折扣因子 γ
        tau: 软更新系数 τ
        target_entropy: 目标熵
        critic_reduction: Critic 聚合方式 ('min' 或 'mean')
        color_jitter: 是否启用颜色抖动增强
        aug_next: 是否对 next_observations 进行增强
        num_cameras: 相机数量（用于多相机颜色抖动）

    返回:
        Tuple 包含：
        - rng: 新的随机数生成器密钥
        - actor: 更新后的 Actor 状态
        - critic: 更新后的 Critic 状态
        - target_critic_params: 更新后的目标 Critic 参数
        - temp: 更新后的温度参数状态
        - info: 包含所有训练指标的字典
    """
    # 初始化增强后的像素数据（默认为原始数据）
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']

    # 检查是不是纯2D数据（状态向量而非图像）
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        # 如果是图像（3D/4D），才进行随机裁剪
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        # 如果启用颜色抖动，则应用颜色变换
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                # 多相机情况：分别对每个相机的 RGB 通道应用颜色变换
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                # 单相机情况：直接对整个图像应用颜色变换
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    # 用增强后的像素数据更新批次中的观测
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    # 对 next_observations 应用相同的增强（如果启用）
    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        # 对下一时刻的观测图像进行随机裁剪
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                # 多相机颜色抖动
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                # 单相机颜色抖动
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

        # 更新批次中的 next_observations
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})

    # 分割随机数密钥用于 Critic 更新
    key, rng = jax.random.split(rng)
    # 创建目标 Critic 状态
    target_critic = critic.replace(params=target_critic_params)
    # 更新 Critic 网络（Q 函数）
    new_critic, critic_info = update_critic(
        key, actor, critic, target_critic, temp, batch, discount, critic_reduction=critic_reduction
    )
    # 软更新目标 Critic 参数
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    # 分割随机数密钥用于 Actor 更新
    key, rng = jax.random.split(rng)
    # 更新 Actor 网络（策略）
    new_actor, actor_info = update_actor(
        key, actor, new_critic, temp, batch, critic_reduction=critic_reduction
    )
    # 更新温度参数（熵正则化系数）
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    # 返回更新后的状态和训练指标
    return rng, new_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,  # Critic 相关指标
        **actor_info,   # Actor 相关指标
        **alpha_info    # 温度参数相关指标
    }


class PixelSACLearner(Agent):
    """
    基于像素的 Soft Actor-Critic (SAC) 学习器

    这是 DSRL (Diffusion Steering via Reinforcement Learning) 的核心算法实现，
    用于在 π₀ 策略的潜在噪声空间中学习动作扰动。

    主要特点：
    - 基于图像观测（像素输入）
    - 软 Actor-Critic 算法，包含熵正则化
    - 支持数据增强（随机裁剪、颜色抖动）
    - 支持多个 Critic 网络（Q 函数集成）
    - 支持多种视觉编码器架构（CNN、ResNet、IMPALA）

    继承自 Agent 基类，实现了标准的 RL 接口。
    """

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter = True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1
                 ):
        """
        PixelSACLearner 初始化函数

        基于 Soft Actor-Critic 论文 (https://arxiv.org/abs/1812.05905) 的实现，
        但扩展为基于像素输入和 DSRL 的特定需求。

        参数:
            seed: 随机种子，用于确保实验可复现
            observations: 观测数据结构，用于推断观测空间
            actions: 动作数组，用于推断动作空间和维度
            actor_lr: Actor 网络学习率，默认 3e-4
            critic_lr: Critic 网络学习率，默认 3e-4
            temp_lr: 温度参数学习率，默认 3e-4
            decay_steps: 余弦退火衰减步数（可选）
            hidden_dims: Actor/Critic 网络的隐藏层维度，默认 (256, 256)
            cnn_features: CNN 编码器的特征通道数，默认 (32, 32, 32, 32)
            cnn_strides: CNN 编码器的步长，默认 (2, 1, 1, 1)
            cnn_padding: CNN 编码器的填充方式，默认 'VALID'（无填充）
            latent_dim: Bottleneck 层的潜在维度，默认 50
            discount: 折扣因子 γ，默认 0.99
            tau: 软更新目标网络的系数，默认 0.005
            critic_reduction: 多个 Critic 的聚合方式，'mean' 或 'min'，默认 'mean'
            dropout_rate: Dropout 比率（可选）
            encoder_type: 编码器类型，可选：'small', 'impala', 'impala_small', 'resnet_small', 'resnet_18_v1', 'resnet_34_v1', 'resnet_small_v2', 'resnet_18_v2', 'resnet_34_v2'
            encoder_norm: 编码器归一化方式，默认 'group'
            color_jitter: 是否启用颜色抖动增强，默认 True
            use_spatial_softmax: 是否使用空间 Softmax，默认 True
            softmax_temperature: Softmax 温度，默认 1
            aug_next: 是否对 next_observations 进行增强，默认 True
            use_bottleneck: 是否使用瓶颈层，默认 True
            init_temperature: 初始温度参数 α，默认 1.0
            num_qs: Critic 网络数量（Q 函数集成），默认 2
            target_entropy: 目标熵值，None 或 'auto' 表示自动计算
            action_magnitude: 动作幅度限制，默认 1.0
            num_cameras: 相机数量（用于多相机设置），默认 1
        """

        # 存储数据增强相关配置
        self.aug_next = aug_next  # 是否对下一个观测进行增强
        self.color_jitter = color_jitter  # 是否启用颜色抖动
        self.num_cameras = num_cameras  # 相机数量

        # 计算动作维度
        self.action_dim = np.prod(actions.shape[-2:])  # 动作总维度（处理后扁平化）
        self.action_chunk_shape = actions.shape[-2:]  # 动作块形状（例如 [action_dim, 1]）

        # 存储 SAC 超参数
        self.tau = tau  # 目标网络软更新系数
        self.discount = discount  # 折扣因子
        self.critic_reduction = critic_reduction  # Critic 聚合方式

        # 初始化随机数生成器
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)  # 为不同网络生成独立密钥

        # 根据配置选择视觉编码器
        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')  # 使用标准 IMPALA 编码器
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')  # 使用小型 IMPALA 编码器
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature
            )
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature
            )
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature
            )
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError(f'encoder type {encoder_type} not found!')  # 编码器类型无效

        # 如果指定了 decay_steps，使用余弦退火调度器
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # 如果 hidden_dims 只有一个值，则重复三次
        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        # 创建策略网络定义（Actor）
        policy_def = LearnedStdTanhNormalPolicy(
            hidden_dims,
            self.action_dim,
            dropout_rate=dropout_rate,
            low=-action_magnitude,  # 动作下界
            high=action_magnitude   # 动作上界
        )

        # 创建 Actor 网络：PixelMultiplexer 将编码器和策略网络组合
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck
        )
        print(actor_def)  # 打印 Actor 网络结构

        # 初始化 Actor 参数
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        # 提取批归一化统计量（如果存在）
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        # 创建 Actor 训练状态
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),  # Adam 优化器
            batch_stats=actor_batch_stats
        )

        # 创建 Critic 网络定义（Q 函数集成）
        critic_base_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(
            encoder=encoder_def,  # 与 Actor 共享编码器
            network=critic_base_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck
        )
        print(critic_def)  # 打印 Critic 网络结构

        # 初始化 Critic 参数
        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init['params']  # 保存初始参数（用于调试）

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None

        # 创建 Critic 训练状态
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),  # Adam 优化器
            batch_stats=critic_batch_stats
        )

        # 创建目标 Critic 参数（深拷贝 Critic 参数）
        target_critic_params = copy.deepcopy(critic_params)

        # 创建温度参数模块（可学习的熵正则化系数 α）
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),  # Adam 优化器
            batch_stats=None  # 温度参数没有批归一化
        )

        # 存储所有训练状态
        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp

        # 计算目标熵
        if target_entropy is None or target_entropy == 'auto':
            # 自动计算：-dim(A)/2，这是 SAC 论文中的推荐值
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)

        print(f'target_entropy: {self.target_entropy}')
        print(self.critic_reduction)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        """
        执行一次 SAC 更新

        这是训练期间调用的主要方法，使用 JIT 编译的 _update_jit 函数
        执行完整的 SAC 更新步骤。

        参数:
            batch: 从回放缓冲区采样的批次数据（FrozenDict 格式）
                   应包含：observations, actions, rewards, next_observations, masks

        返回:
            info: 包含所有训练指标的字典
                  包含：critic_loss, actor_loss, temperature_loss, entropy, q_values 等
        """
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params, self._temp, batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras
            )

        self._rng = new_rng  # 更新随机数生成器状态
        self._actor = new_actor  # 更新 Actor 状态
        self._critic = new_critic  # 更新 Critic 状态
        self._target_critic_params = new_target_critic  # 更新目标 Critic 参数
        self._temp = new_temp  # 更新温度参数
        return info  # 返回训练指标

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        """
        执行评估并生成可视化

        在训练过程中的评估阶段调用，生成 Q 值和奖励的可视化图表。

        参数:
            variant: 配置字典
            i: 当前训练步数
            wandb_logger: Weights & Biases 日志记录器
            eval_buffer: 评估回放缓冲区
            eval_buffer_iterator: 评估缓冲区迭代器
            eval_env: 评估环境
        """
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        """
        为给定的轨迹生成 Q 值和奖励可视化

        从轨迹中提取观测、动作和奖励，计算每个时间步的 Q 值预测，
        并生成包含图像、Q 值曲线、奖励曲线和掩码的可视化图表。

        参数:
            variant: 配置字典（未使用，保留兼容性）
            trajs: 轨迹字典，包含：
                   - observations: 观测序列
                   - next_observations: 下一观测序列
                   - actions: 动作序列
                   - rewards: 奖励序列
                   - masks: 掩码序列（1 - done）

        返回:
            numpy 数组：拼接所有轨迹的可视化图像
        """
        # 获取轨迹数量
        num_traj = len(trajs['rewards'])
        traj_images = []  # 存储每条轨迹的可视化

        # 遍历每条轨迹
        for itraj in range(num_traj):
            # 提取单条轨迹数据
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []  # 存储 Q 值预测

            # 遍历轨迹中的每个时间步
            for t in range(0, len(actions)):
                action = actions[t][None]  # 添加批次维度 [1, action_dim]
                obs_pixels = observations['pixels'][t]  # 当前观测图像
                next_obs_pixels = next_observations['pixels'][t]  # 下一观测图像

                # 构建观测字典（包含非像素观测）
                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]

                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                # 使用 Critic 网络计算 Q 值
                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            # 生成轨迹可视化
            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))

        print('finished reward value visuals.')  # 完成可视化生成
        return np.concatenate(traj_images, 0)  # 拼接所有轨迹的可视化

    @property
    def _save_dict(self):
        """
        用于保存模型的字典

        包含所有需要保存的训练状态，包括 Actor、Critic、目标 Critic 和温度参数。
        Flax 的检查点系统会自动序列化这些对象。

        返回:
            dict: 包含所有需要保存的对象
        """
        save_dict = {
            'critic': self._critic,  # Critic 训练状态
            'target_critic_params': self._target_critic_params,  # 目标 Critic 参数
            'actor': self._actor,  # Actor 训练状态
            'temp': self._temp  # 温度参数训练状态
        }
        return save_dict

    def restore_checkpoint(self, dir):
        """
        从检查点恢复模型

        从指定目录加载保存的模型检查点，恢复所有训练状态。

        参数:
            dir: 检查点目录路径

        断言:
            检查点目录必须存在
        """
        # 验证目录存在
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."

        # 从检查点加载
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        # 恢复所有训练状态
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']

        print('restored from ', dir)  # 打印恢复信息
        
    
@functools.partial(jax.jit)  # JIT 编译的函数，用于高效计算 Q 值
def get_value(action, observation, critic):
    """
    JIT 编译的函数，用于查询 Critic 网络的 Q 值

    参数:
        action: 动作 [batch_size, action_dim]
        observation: 观测字典，包含像素和其他状态信息
        critic: Critic 训练状态

    返回:
        q_pred: 预测的 Q 值
    """
    input_collections = {'params': critic.params}  # 只使用参数（无批归一化统计量）
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    """
    NumPy 辅助函数：将数组沿指定轴拆分为列表

    参数:
        array: 输入 NumPy 数组
        axis: 要拆分的轴

    返回:
        list: 拆分后的数组列表
    """
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]  # 移除单维度
    return arr


def make_visual(q_estimates, rewards, masks, images):
    """
    生成 Q 值、奖励和图像的可视化图表

    创建包含 4 个子图的图表：
    1. 轨迹关键帧图像
    2. Q 值预测曲线
    3. 奖励曲线
    4. 掩码（1 - done）曲线

    参数:
        q_estimates: Q 值预测列表，每个元素是 [batch_size, num_qs] 或 [batch_size]
        rewards: 奖励数组 [episode_length]
        masks: 掩码数组（1 表示非终止，0 表示终止）
        images: 图像数组 [episode_length, H, W, C, stack_size]

    返回:
        out_image: 生成的可视化图像数组 [H, W, 3]
    """

    # 将 Q 值列表堆叠为数组
    q_estimates_np = np.stack(q_estimates, 0).squeeze()

    # 创建 4 行 1 列的子图布局
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)  # 非交互式画布
    plt.xlim([0, len(q_estimates_np)])  # 设置 X 轴范围

    # 验证图像维度并处理图像堆栈
    assert len(images.shape) == 5
    images = images[..., -1]  # 只取堆栈中的最新图像（最后一张）
    assert images.shape[-1] == 3

    # 选择 4 个关键帧用于显示（均匀采样）
    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)  # 水平拼接

    # 子图 1：显示关键帧图像
    axs[0].imshow(sel_images)

    # 子图 2：绘制 Q 值曲线
    if len(q_estimates_np.shape) == 2:
        # 多个 Q 函数：绘制每条曲线（虚线 + 圆圈标记）
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        # 单个 Q 函数
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('q values')  # Y 轴标签

    # 子图 3：绘制奖励曲线
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])  # 设置 X 轴范围

    # 子图 4：绘制掩码曲线（菱形标记）
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('masks')  # 1 表示非终止，0 表示终止
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()  # 自动调整子图间距

    # 绘制画布并转换为 numpy 数组
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # 关闭图表释放内存
    return out_image