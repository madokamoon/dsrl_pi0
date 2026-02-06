"""
可学习标准差的高斯策略网络实现

该模块实现了三种策略网络：
1. LearnedStdNormalPolicy: 可学习标准差的高斯策略
2. TanhMultivariateNormalDiag: 带 Tanh 变换的多变量正态分布
3. LearnedStdTanhNormalPolicy: 结合前两者，带 Tanh 压缩和动作范围限制

这是 DSRL 项目中 Actor 网络的核心组件，用于在 π₀ 的潜在空间中学习策略分布。
"""

from typing import Optional, Sequence

import distrax  # 概率分布库
import flax.linen as nn  # Flax 神经网络模块
import jax.numpy as jnp  # JAX NumPy

from jaxrl2.networks import MLP  # 多层感知机
from jaxrl2.networks.constants import default_init  # 默认初始化

class LearnedStdNormalPolicy(nn.Module):
    """
    可学习标准差的高斯策略网络

    该策略网络同时学习动作的均值和标准差，输出一个多变量对角高斯分布。
    与固定标准差的策略相比，能够更好地适应不同状态的不确定性。

    网络结构：
    observations → MLP → [mean_head, log_std_head] → MultivariateNormalDiag

    参数:
        hidden_dims: 隐藏层维度序列，例如 (256, 256)
        action_dim: 动作空间维度
        dropout_rate: Dropout 比率（可选）
        log_std_min: log(标准差) 的最小值，用于防止标准差过小
        log_std_max: log(标准差) 的最大值，用于防止标准差过大
    """

    hidden_dims: Sequence[int]  # 隐藏层维度
    action_dim: int  # 动作维度
    dropout_rate: Optional[float] = None  # Dropout 比率
    log_std_min: Optional[float] = -20  # log(std) 最小值
    log_std_max: Optional[float] = 2  # log(std) 最大值

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,  # 观测 [batch_size, obs_dim]
        training: bool = False  # 是否训练模式（影响 Dropout）
    ) -> distrax.Distribution:
        """
        前向传播

        参数:
            observations: 观测数组
            training: 是否训练模式

        返回:
            distribution: 多变量对角高斯分布 distrax.MultivariateNormalDiag
        """
        # 通过 MLP 编码观测
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,  # 最后一层使用激活函数
            dropout_rate=self.dropout_rate
        )(observations, training=training)

        # 计算动作的均值（均值头）
        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(1e-2)  # 小初始化防止初始输出过大
        )(outputs)

        # 计算动作的标准差（对数标准差头）
        log_stds = nn.Dense(
            self.action_dim,
            kernel_init=default_init(1e-2)
        )(outputs)

        # 裁剪 log_stds 防止数值不稳定
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # 构建多变量对角高斯分布
        distribution = distrax.MultivariateNormalDiag(
            loc=means,  # 均值
            scale_diag=jnp.exp(log_stds)  # 标准差（从对数空间转换回来）
        )

        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):
    """
    带 Tanh 变换的双射（Bijector）分布

    该类通过链式组合多个变换，将无约束空间的高斯分布映射到约束空间：
    1. 底层分布：无约束的 MultivariateNormalDiag 分布
    2. Tanh 变换：将输出压缩到 (-1, 1) 范围
    3. （可选）缩放变换：将 (-1, 1) 映射到 [low, high] 范围

    这是实现约束动作空间的标准方法，保持了可逆性和概率密度计算。

    参数:
        loc: 底层高斯分布的均值
        scale_diag: 底层高斯分布的对角标准差
        low: 动作下界（可选）
        high: 动作上界（可选）
    """

    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None
    ):
        """
        初始化带 Tanh 变换的分布

        通过链式组合多个双射变换：
        1. 首先应用 Tanh 压缩到 (-1, 1)
        2. 然后（如果指定了 low/high）缩放到目标范围 [low, high]

        为了保持概率密度的一致性，每个变换都实现了对数雅可比行列式（log det Jacobian）。
        """
        # 创建底层的多变量对角高斯分布
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []  # 存储双射变换层（按应用顺序）

        # 如果指定了动作范围，添加缩放变换
        if not (low is None or high is None):

            def rescale_from_tanh(x):
                """
                将 Tanh 输出 (-1, 1) 缩放到目标范围 [low, high]

                变换公式：
                x' = (x + 1) / 2 * (high - low) + low
                """
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                """
                缩放变换的对数雅可比行列式

                这个值用于概率密度计算中的体积变化校正。
                """
                high_ = jnp.broadcast_to(high, x.shape)  # 广播到输入形状
                low_ = jnp.broadcast_to(low, x.shape)    # 广播到输入形状
                # 对数雅可比 = sum(log(缩放因子))
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            # 添加 Lambda 变换（自定义缩放）
            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,   # 输入事件维度
                    event_ndims_out=1   # 输出事件维度
                )
            )

        # 添加 Tanh 变换（Block 包装器用于指定事件维度）
        layers.append(distrax.Block(distrax.Tanh(), 1))

        # 链式组合所有变换（顺序：从右到左应用）
        bijector = distrax.Chain(layers)

        # 调用父类初始化
        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        """
        获取分布的众数（Mode）

        众数是高斯分布的均值，通过双射变换映射到约束空间。

        返回:
            mode: 变换后的众数（确定性动作）
        """
        # 底层分布的众数（均值）通过双射变换
        return self.bijector.forward(self.distribution.mode())

class LearnedStdTanhNormalPolicy(nn.Module):
    """
    可学习标准差的 Tanh 高斯策略网络

    这是 DSRL 项目中使用的主要策略网络，结合了以下特性：
    1. 可学习的标准差（不确定性估计）
    2. Tanh 变换（动作约束到 [-1, 1] 或 [low, high]）
    3. 支持 Dropout 正则化

    网络结构：
    observations → MLP → [mean_head, log_std_head] → TanhMultivariateNormalDiag

    参数:
        hidden_dims: 隐藏层维度序列
        action_dim: 动作空间维度
        dropout_rate: Dropout 比率（可选）
        log_std_min: log(标准差) 的最小值
        log_std_max: log(标准差) 的最大值
        low: 动作下界（可选，用于自定义范围）
        high: 动作上界（可选，用于自定义范围）
    """

    hidden_dims: Sequence[int]  # 隐藏层维度
    action_dim: int  # 动作维度
    dropout_rate: Optional[float] = None  # Dropout 比率
    log_std_min: Optional[float] = -20  # log(std) 最小值
    log_std_max: Optional[float] = 2  # log(std) 最大值
    low: Optional[float] = None  # 动作下界
    high: Optional[float] = None  # 动作上界

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,  # 观测 [batch_size, obs_dim]
        training: bool = False  # 是否训练模式
    ) -> distrax.Distribution:
        """
        前向传播

        参数:
            observations: 观测数组
            training: 是否训练模式（影响 Dropout）

        返回:
            distribution: 带 Tanh 约束的多变量高斯分布
                         TanhMultivariateNormalDiag
        """
        # 通过 MLP 编码观测
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,  # 最后一层使用激活函数
            dropout_rate=self.dropout_rate
        )(observations, training=training)

        # 计算动作均值
        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(1e-2)
        )(outputs)

        # 计算动作标准差（对数空间）
        log_stds = nn.Dense(
            self.action_dim,
            kernel_init=default_init(1e-2)
        )(outputs)

        # 裁剪 log_stds 防止数值不稳定
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # 构建带 Tanh 约束的分布
        distribution = TanhMultivariateNormalDiag(
            loc=means,  # 均值
            scale_diag=jnp.exp(log_stds),  # 标准差
            low=self.low,  # 下界
            high=self.high  # 上界
        )

        return distribution