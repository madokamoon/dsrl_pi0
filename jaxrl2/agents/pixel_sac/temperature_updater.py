from typing import Dict, Tuple
# 导入类型注解，Dict（字典类型）和 Tuple（元组类型）用于类型提示

import jax
# 导入 JAX 库，用于自动微分和数组计算
from flax.training.train_state import TrainState
# 从 flax 导入 TrainState，表示可训练的模型状态（包含参数、优化器等）


def update_temperature(
        temp: TrainState, entropy: float,
        target_entropy: float) -> Tuple[TrainState, Dict[str, float]]:
    # 更新温度参数（Temperature）函数
    # 在 SAC（Soft Actor-Critic）中，温度参数控制熵正则化的强度
    # 通过自动调整温度，使策略熵接近目标熵

    def temperature_loss_fn(temp_params):
        # 定义温度参数的损失函数（temperature_loss_fn）
        # 计算温度损失用于优化温度参数

        temperature = temp.apply_fn({'params': temp_params})
        # 使用当前参数计算温度值（Temperature）
        # temperature 是一个大于 0 的标量，控制熵的权重

        temp_loss = temperature * (entropy - target_entropy).mean()
        # 计算温度损失（Temperature Loss）
        # 损失函数为：温度 × (当前熵 - 目标熵)
        # 当熵高于目标时，损失为负（鼓励降低温度）
        # 当熵低于目标时，损失为正（鼓励提高温度）

        return temp_loss, {
            'temperature': temperature,
            'temperature_loss': temp_loss
        }
        # 返回损失值和辅助信息字典
        # 包含当前温度值和损失值

    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    # 使用 JAX 的自动微分计算梯度（grads）
    # has_aux=True 表示损失函数返回额外的辅助信息
    # 第一个返回值是梯度，第二个是辅助信息

    new_temp = temp.apply_gradients(grads=grads)
    # 应用梯度更新温度参数
    # new_temp 是更新后的 TrainState，包含新的参数

    return new_temp, info
    # 返回更新后的温度状态和训练信息
    # info 字典包含当前温度值和温度损失