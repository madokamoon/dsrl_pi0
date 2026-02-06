"""
状态-动作值函数集成（State-Action Value Ensemble）

该模块实现了 Q 函数集成网络，通过创建多个独立的 Q 函数来减少过估计偏差。
这是 SAC 算法中常用的技术，通过取多个 Q 值的最小值或平均值来提高训练稳定性。

在 DSRL 项目中用于 PixelSACLearner 的 Critic 网络。
"""

from typing import Callable, Sequence

import flax.linen as nn  # Flax 神经网络模块
import jax.numpy as jnp  # JAX NumPy

from jaxrl2.networks.values.state_action_value import StateActionValue  # 导入单个 Q 函数


class StateActionEnsemble(nn.Module):
    """
    状态-动作值函数集成网络

    通过 vmap（向量化映射）创建多个独立的 Q 函数（num_qs 个），
    每个 Q 函数有独立的参数，但共享相同的网络结构。

    在训练时，可以使用：
    - min: 取所有 Q 值的最小值（更保守，减少过估计）
    - mean: 取所有 Q 值的平均值（较稳定）

    网络结构：
    states + actions → [Q₁, Q₂, ..., Qₙ] → Q_values [num_qs]

    参数:
        hidden_dims: 隐藏层维度序列
        activations: 激活函数，默认 nn.relu
        num_qs: Q 函数数量（集成大小），默认 2
        use_action_sep: 是否使用动作分离的 MLP（每步都拼接动作）
    """

    hidden_dims: Sequence[int]  # 隐藏层维度
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu  # 激活函数
    num_qs: int = 2  # Q 函数数量（默认值 2）
    use_action_sep: bool = False  # 是否使用动作分离

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        """
        前向传播

        使用 nn.vmap 向量化创建多个 Q 函数，每个有独立参数。

        参数:
            states: 状态观测 [batch_size, state_dim]
            actions: 动作 [batch_size, action_dim]
            training: 是否训练模式

        返回:
            qs: Q 值数组 [num_qs, batch_size]
                包含 num_qs 个 Q 函数的预测值
        """
        # 使用 vmap 创建多个 Q 函数
        # variable_axes={'params': 0}: 每个 Q 函数有独立的参数（第 0 维）
        # split_rngs={'params': True}: 为每个 Q 函数分配独立的随机数
        # axis_size=self.num_qs: 创建 num_qs 个 Q 函数
        VmapCritic = nn.vmap(
            StateActionValue,  # 基础 Q 函数类
            variable_axes={'params': 0},  # 参数沿第 0 维分开
            split_rngs={'params': True},  # 分割随机数生成器
            in_axes=None,  # 输入不映射（所有 Q 函数共享输入）
            out_axes=0,  # 输出沿第 0 维堆叠
            axis_size=self.num_qs  # Q 函数数量
        )

        # 实例化并调用 vmap 后的 Q 函数集成
        qs = VmapCritic(
            self.hidden_dims,
            activations=self.activations,
            use_action_sep=self.use_action_sep
        )(states, actions, training)

        # 返回形状: [num_qs, batch_size]
        # 例如: [2, 256] 表示 2 个 Q 函数，每个对 256 个样本的预测
        return qs
