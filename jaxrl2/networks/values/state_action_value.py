from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

import jax

from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.mlp import MLPActionSep
from jaxrl2.networks.constants import default_init

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

# PRNGKey（伪随机数生成器密钥）
PRNGKey = Any
# Shape（张量形状）
Shape = Tuple[int, ...]
# Dtype（数据类型）
Dtype = Any
# Array（数组类型）
Array = Any
# PrecisionLike（精度类型）
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

# 默认的核初始化方法（使用lecun正态分布）
default_kernel_init = nn.initializers.lecun_normal()


# StateActionValue（状态-动作值网络，即Q网络）
# 用于计算给定状态和动作的Q值Q(s,a)

class StateActionValue(nn.Module):
    # hidden_dims（隐藏层维度序列）
    hidden_dims: Sequence[int]
    # activations（激活函数，默认为ReLU）
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # use_action_sep（是否使用动作分离的MLP）
    use_action_sep: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        # 将观测和动作组合成输入字典
        inputs = {'states': observations, 'actions': actions}

        # 根据配置选择MLP类型：动作分离版或标准版
        if self.use_action_sep:
            # 使用动作分离的MLP（在每层都拼接动作信息）
            critic = MLPActionSep(
                (*self.hidden_dims, 1),  # 输出维度为1（Q值）
                activations=self.activations,
                use_layer_norm=True)(inputs, training=training)
        else:
            # 使用标准MLP
            critic = MLP((*self.hidden_dims, 1),  # 输出维度为1（Q值）
                        activations=self.activations,
                        use_layer_norm=True)(inputs, training=training)

        # 移除最后一维（从[batch, 1]变为[batch]）
        return jnp.squeeze(critic, -1)
