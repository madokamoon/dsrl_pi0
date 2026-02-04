#!/usr/bin/env python
# ==============================================================================
# 模拟环境训练主程序 - DSRL (Diffusion Steering via Reinforcement Learning)
# ==============================================================================
# 这是整个 DSRL 训练流程的核心文件，负责：
# 1. 初始化环境和 π₀ 预训练策略
# 2. 创建 SAC (Soft Actor-Critic) Agent
# 3. 设置经验回放缓冲区
# 4. 启动训练循环
#
# 关键概念：
# - π₀ (pi-zero): 预训练的扩散策略，是一个通用的机器人控制策略
# - DSRL: 在 π₀ 的潜在噪声空间中学习扰动，以改进特定任务的性能
# - SAC: 一种高效的无模型强化学习算法，使用熵正则化鼓励探索
# - Pixel-based: 直接从像素/图像观测中学习，不需要手动提取特征
# ==============================================================================

import os

# ==============================================================================
# JAX 性能优化 - Triton GEMM 加速
# ==============================================================================
# Triton 是 OpenAI 开发的高性能矩阵乘法库
# 使用 Triton GEMM 可以在某些 GPU 上提升约 30% 的性能
# 参考: https://github.com/huggingface/gym-aloha/tree/main#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

# ==============================================================================
# JAX 编译缓存 - 加速重复编译
# ==============================================================================
# JAX 使用 JIT (Just-In-Time) 编译将 Python 代码编译为高效的机器码
# 编译缓存可以保存编译结果，避免重复编译，显著加快后续运行速度
home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def _get_libero_env(task, resolution, seed):
    """
    初始化并返回 LIBERO 环境及任务描述。

    参数:
        task: LIBERO 任务对象，包含任务描述和 BDDL 文件信息
        resolution: 相机图像分辨率（高度和宽度）
        seed: 随机种子，影响物体初始位置

    返回:
        env: LIBERO 环境对象
        task_description: 自然语言任务描述（如 "put the spoon on the plate"）

    注意:
        BDDL (Benchmark for Describing Dynamic Logic) 是一种用于描述机器人任务的形式化语言
        它定义了任务的初始状态、目标和约束条件
    """
    task_description = task.language  # 获取自然语言任务描述
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file  # BDDL 文件路径
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}  # 环境参数
    env = OffScreenRenderEnv(**env_args)  # 创建离屏渲染环境（不使用显示器）
    env.seed(seed)  # 设置随机种子（重要：即使使用固定初始状态，种子也会影响物体位置）
    return env, task_description


def shard_batch(batch, sharding):
    """
    将批次数据分片到多个设备（GPU）上。

    这是多 GPU 训练的关键函数，它将批次数据均匀分布到所有可用的 GPU 上，
    每个 GPU 处理一部分数据，从而实现并行加速。

    参数:
        batch: 一个 PyTree 结构的数据批次（可能包含多个数组）
        sharding: JAX 的 Sharding 对象，定义数据如何分片

    返回:
        sharded_batch: 分片后的数据批次

    示例:
        如果有 2 个 GPU，批次大小为 256，则每个 GPU 处理 128 个样本
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """
    虚拟环境 - 用于定义观测空间和动作空间。

    这个环境不与真实环境交互，它的唯一作用是提供一个标准的 Gym 接口，
    让 RL 算法知道观测和动作的形状、范围等信息，从而正确构建神经网络。

    在强化学习中，环境必须定义两个关键空间：
    1. Observation Space (观测空间): Agent 能看到什么
    2. Action Space (动作空间): Agent 能做什么
    """

    def __init__(self, variant):
        self.variant = variant  # 配置参数

        # 图像形状: (高度, 宽度, 通道数, 1)
        # 3 * variant.num_cameras: 每个相机有 RGB 3 个通道
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)

        # 构建观测空间字典
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)  # 像素观测，范围 0-255

        # 如果配置要求添加低维状态信息
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8  # Libero 的状态维度
            elif variant.env == 'aloha_cube':
                state_dim = 14  # Aloha 的状态维度
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)  # 状态观测，范围 -1 到 1

        self.observation_space = Dict(obs_dict)  # 观测空间是字典类型，包含 'pixels' 和可选的 'state'

        # 动作空间: π₀ 的噪声空间是 32 维
        # DSRL 的策略输出噪声扰动，而不是直接输出关节角度
        # 扰动的形状是 (1, 32)，值范围在 -1 到 1 之间
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)


def main(variant):
    """
    主训练函数 - 整个 DSRL 训练流程的入口。

    这个函数执行以下步骤：
    1. 设置多 GPU 并行训练（如果可用）
    2. 创建实验目录和日志记录器
    3. 初始化仿真环境
    4. 加载预训练的 π₀ 策略
    5. 创建 SAC Agent
    6. 创建经验回放缓冲区
    7. 启动训练循环

    参数:
        variant: 配置字典，包含所有超参数和设置
    """
    # ==============================================================================
    # 多 GPU 设置 - 分布式训练加速
    # ==============================================================================
    # 获取所有可用的 JAX 设备（GPU 或 TPU）
    devices = jax.local_devices()
    num_devices = len(devices)
    # 确保批次大小能被设备数量整除，这样才能均匀分配
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # ==============================================================================
    # TensorFlow 设置 - 防止 TF 占用 GPU
    # ==============================================================================
    # 虽然主框架使用 JAX，但某些库可能内部使用 TensorFlow
    # 这里禁止 TF 使用 GPU，避免与 JAX 冲突
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    # 如果使用余弦衰减学习率，设置衰减步数
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
    # 如果没有指定前缀，生成一个随机的 5 字符 UUID
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    # 创建实验名称：前缀 + 种子 + 后缀
    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    # ==============================================================================
    # 环境初始化 - 创建仿真环境
    # ==============================================================================
    # 根据不同的环境类型创建对应的仿真环境
    if variant.env == 'libero':
        # Libero 环境设置
        benchmark_dict = benchmark.get_benchmark_dict()  # 获取所有可用的任务套件
        task_suite = benchmark_dict["libero_90"]()  # 使用 libero_90 套件（90 个任务）
        task_id = 57  # 选择第 58 个任务（从 0 开始计数）
        task = task_suite.get_task(task_id)  # 获取具体任务

        # 创建环境，分辨率 256x256
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env  # Libero 的评估环境和训练环境相同

        variant.task_description = task_description  # 保存任务描述（如 "put the spoon on the plate"）
        variant.env_max_reward = 1  # 环境最大奖励（成功时为 1）
        variant.max_timesteps = 400  # 每个 episode 的最大步数

    elif variant.env == 'aloha_cube':
        # Aloha Cube Transfer 环境设置
        from gymnasium.envs.registration import register

        # 注册自定义环境
        register(
            id="gym_aloha/AlohaTransferCube-v0",  # 环境 ID
            entry_point="gym_aloha.env:AlohaEnv",  # 环境类路径
            max_episode_steps=400,  # 最大回合步数
            nondeterministic=True,  # 是否非确定性
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},  # 传递给环境构造器的参数
        )

        # 创建环境
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)  # 深拷贝创建评估环境（独立副本）

        variant.env_max_reward = 4  # Aloha 环境最大奖励为 4
        variant.max_timesteps = 400  # 每个 episode 的最大步数

    # ==============================================================================
    # Weights & Biases 日志设置 - 在线监控训练
    # ==============================================================================
    # W&B 是一个流行的实验跟踪平台，可以在线查看训练曲线、比较实验等
    group_name = variant.prefix + '_' + variant.launch_group_id  # 实验分组
    wandb_output_dir = tempfile.mkdtemp()  # W&B 临时文件目录

    # 创建 W&B 日志记录器
    wandb_logger = WandBLogger(
        variant.prefix != '',  # 是否启用 W&B（有前缀才启用）
        variant,  # 配置字典（记录所有超参数）
        variant.wandb_project,  # 项目名称
        experiment_id=expname,  # 实验 ID
        output_dir=wandb_output_dir,  # 输出目录
        group_name=group_name  # 分组名称
    )

    # ==============================================================================
    # 虚拟环境创建 - 用于定义观测和动作空间
    # ==============================================================================
    # DummyEnv 不执行实际仿真，只提供空间定义
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    
    # ==============================================================================
    # π₀ 策略加载 - 预训练的基础策略
    # ==============================================================================
    # π₀ 是一个在大量机器人数据上预训练的扩散策略，能够理解自然语言指令
    # DSRL 的目标是学习一个扰动策略来增强 π₀，而不是从头学习
    if variant.env == 'libero':
        # 加载 Libero 版本的 π₀
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        # 加载 Aloha 版本的 π₀
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    # ==============================================================================
    # SAC Agent 创建 - 学习扰动策略
    # ==============================================================================
    # PixelSACLearner 是 DSRL 的核心组件，它：
    # 1. 学习一个策略 (Actor)，输出噪声扰动
    # 2. 学习价值函数 (Critic)，评估状态-动作对的好坏
    # 3. 使用熵正则化鼓励探索
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # ==============================================================================
    # 经验回放缓冲区创建 - 存储交互数据
    # =============================================================================
    # 计算缓冲区大小: 总步数 // UTD 比率
    # 因为每个环境步会执行 multi_grad_step 次梯度更新
    online_buffer_size = variant.max_steps // variant.multi_grad_step

    # 创建重放缓冲区，指定观测空间和动作空间
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,  # 观测空间
        dummy_env.action_space,  # 动作空间
        int(online_buffer_size)  # 缓冲区大小
    )

    replay_buffer = online_replay_buffer  # DSRL 只使用在线数据
    replay_buffer.seed(variant.seed)  # 设置缓冲区采样随机种子

    # ==============================================================================
    # 启动训练循环 - 开始训练
    # ==============================================================================
    # trajwise_alternating_training_loop 是训练的核心循环，它：
    # 1. 交替进行数据收集和策略更新
    # 2. 每个循环收集一条完整轨迹 (trajectory)
    # 3. 使用收集的数据执行多次梯度更新
    # 4. 记录日志和评估策略性能

    trajwise_alternating_training_loop(
        variant,  # 配置参数
        agent,  # SAC Agent
        env,  # 训练环境
        eval_env,  # 评估环境
        online_replay_buffer,  # 在线数据缓冲区
        replay_buffer,  # 训练数据缓冲区（与 online_replay_buffer 相同）
        wandb_logger,  # W&B 日志记录器
        shard_fn=shard_fn,  # 数据分片函数（多 GPU）
        agent_dp=agent_dp  # π₀ 策略
    )
 