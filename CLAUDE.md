# DSRL for π₀ 项目结构文档

## 项目概述 (Project Overview)

本项目实现了 **DSRL (Diffusion Steering via Reinforcement Learning)**，用于通过潜在空间强化学习来引导预训练的通用策略 π₀。

- **论文**: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025)
- **网站**: https://diffusion-steering.github.io
- **基础框架**: JAX-based 实现
- **应用环境**:
  - 模拟环境: Libero, Aloha
  - 真实机器人: Franka

---

## 项目根目录结构

```
dsrl_pi0/
├── examples/              # 训练示例和脚本
├── jaxrl2/               # 核心算法实现（基于 jaxrl2）
├── LIBERO/               # Libero 环境子模块
├── openpi/               # OpenPI (π₀) 策略子模块
├── setup.py              # Python 包安装配置
├── requirements.txt      # 项目依赖
├── README.md            # 项目说明文档
└── .gitignore           # Git 忽略文件配置
```

---

## `/examples` 文件夹详细说明

这是项目的**训练入口**，包含所有训练相关的脚本和工具函数。

### 核心文件结构

```
examples/
├── scripts/                      # Shell 脚本
│   ├── run_libero.sh            # Libero 环境训练脚本
│   ├── run_aloha.sh             # Aloha 环境训练脚本
│   └── run_real.sh              # 真实机器人训练脚本
├── launch_train_sim.py          # 模拟环境训练启动器
├── launch_train_real.py         # 真实环境训练启动器
├── train_sim.py                 # 模拟环境训练主程序
├── train_real.py                # 真实环境训练主程序
├── train_utils_sim.py           # 模拟环境训练工具函数
├── train_utils_real.py          # 真实环境训练工具函数
└── __init__.py                  # 包初始化文件
```

---

### 文件功能详解

#### 1. **Shell 脚本 (scripts/)**

##### `run_libero.sh`
- **作用**: 启动 Libero 环境的 DSRL 训练
- **主要配置**:
  - 项目名称: `DSRL_pi0_Libero`
  - 环境变量: MUJOCO_GL=egl (使用 EGL 渲染)
  - 训练参数:
    - batch_size=256
    - max_steps=500,000
    - multi_grad_step=20 (UTD ratio)
    - resize_image=64
    - action_magnitude=1.0
    - query_freq=20
- **依赖**: mujoco==3.3.1
- **调用**: `python3 examples/launch_train_sim.py --env libero ...`

##### `run_aloha.sh`
- **作用**: 启动 Aloha 环境的 DSRL 训练
- **主要配置**:
  - 项目名称: `DSRL_pi0_Aloha`
  - 环境: aloha_cube (Transfer Cube 任务)
  - 训练参数:
    - batch_size=256
    - max_steps=3,000,000
    - multi_grad_step=20
    - resize_image=64
    - action_magnitude=2.0
    - query_freq=50
- **依赖**: mujoco==2.3.7
- **调用**: `python3 examples/launch_train_sim.py --env aloha_cube ...`

##### `run_real.sh`
- **作用**: 启动真实 Franka 机器人的 DSRL 训练
- **主要配置**:
  - 项目名称: `DSRL_pi0_FrankaDroid`
  - 环境: franka_droid (使用 DROID 包)
  - 训练参数:
    - batch_size=256
    - max_steps=500,000
    - multi_grad_step=30
    - resize_image=128
    - action_magnitude=2.5
    - query_freq=10
    - hidden_dims=1024 (更大的网络)
    - num_qs=2 (使用 2 个 Q 网络)
- **环境变量要求**:
  - `LEFT_CAMERA_ID`: 左侧相机 ID
  - `RIGHT_CAMERA_ID`: 右侧相机 ID
  - `WRIST_CAMERA_ID`: 手腕相机 ID
  - `remote_host`: π₀ 模型远程服务器地址
  - `remote_port`: π₀ 模型远程服务器端口
- **调用**: `python3 examples/launch_train_real.py ...`

---

#### 2. **启动器文件 (Launch Files)**

##### `launch_train_sim.py`
- **作用**: 模拟环境训练的**命令行参数解析器**
- **功能**:
  - 解析训练超参数（learning rates, network architecture, etc.）
  - 设置默认的训练配置字典
  - 调用 `train_sim.py` 的 `main()` 函数
- **关键参数**:
  - `--env`: 环境名称 (libero/aloha_cube)
  - `--seed`: 随机种子
  - `--batch_size`: 批次大小
  - `--max_steps`: 最大训练步数
  - `--multi_grad_step`: UTD (Update-to-Data) ratio
  - `--resize_image`: 图像缩放尺寸
  - `--query_freq`: π₀ 查询频率
  - `--add_states`: 是否添加低维状态信息
- **训练配置**:
  ```python
  actor_lr=1e-4
  critic_lr=3e-4
  hidden_dims=(128, 128, 128)
  cnn_features=(32, 32, 32, 32)
  latent_dim=50
  discount=0.999
  num_qs=10
  encoder_type='small'
  use_spatial_softmax=True
  ```

##### `launch_train_real.py`
- **作用**: 真实机器人训练的**命令行参数解析器**
- **功能**: 类似 `launch_train_sim.py`，但调用 `train_real.py`
- **特殊参数**:
  - `--instruction`: 机器人的语言指令（如 "put the spoon on the plate"）
  - `--num_initial_traj_collect`: 开始训练前收集的轨迹数量
- **训练配置差异**:
  ```python
  hidden_dims=(1024, 1024, 1024)  # 更大的网络
  discount=0.99  # 更小的折扣因子
  critic_reduction='min'  # 使用最小 Q 值
  num_qs=2  # 只使用 2 个 Q 网络
  action_magnitude=2.5  # 更大的动作幅度
  num_cameras=3  # 使用 3 个相机
  ```

---

#### 3. **训练主程序 (Main Training Programs)**

##### `train_sim.py`
- **作用**: 模拟环境训练的**核心逻辑**
- **主要流程**:
  1. **环境初始化**:
     - Libero: 使用 `OffScreenRenderEnv`，加载 BDDL 任务文件
     - Aloha: 使用 `gym_aloha` 的 Transfer Cube 环境
  2. **策略加载**:
     - 从 S3 下载预训练的 π₀ 模型
     - Libero: `s3://openpi-assets/checkpoints/pi0_libero`
     - Aloha: `s3://openpi-assets/checkpoints/pi0_aloha_sim`
  3. **Agent 初始化**:
     - 创建 `PixelSACLearner` (DSRL 的 Actor-Critic)
     - 创建 `DummyEnv` 用于定义观测和动作空间
  4. **数据缓冲区**:
     - 创建 `ReplayBuffer` 用于存储在线交互数据
  5. **训练循环**:
     - 调用 `trajwise_alternating_training_loop()` 进行训练
  6. **JAX 优化**:
     - 使用 Triton GEMM 加速 (~30% 性能提升)
     - 支持多设备并行训练（数据分片）
- **关键类**:
  - `DummyEnv`: 虚拟环境，定义观测空间（pixels + state）和动作空间（32 维噪声）

##### `train_real.py`
- **作用**: 真实机器人训练的**核心逻辑**
- **主要差异**:
  1. **策略通信**:
     - 使用 `WebsocketClientPolicy` 连接远程 π₀ 服务器
     - 通过 `remote_host` 和 `remote_port` 环境变量配置
  2. **环境初始化**:
     - 使用 DROID 的 `RobotEnv`
     - 配置 3 个相机（左侧、右侧、手腕）
     - 动作空间: joint_velocity，夹爪: position
  3. **状态空间**:
     - 包含 8 维本体感受状态 + 2024 维图像表示
  4. **检查点恢复**:
     - 支持从 `restore_path` 恢复训练
- **关键配置**:
  ```python
  robot_config = dict(
      camera_to_use='right',
      left_camera_id=os.environ['LEFT_CAMERA_ID'],
      right_camera_id=os.environ['RIGHT_CAMERA_ID'],
      wrist_camera_id=os.environ['WRIST_CAMERA_ID'],
      max_timesteps=200
  )
  ```

---

#### 4. **训练工具函数 (Training Utils)**

##### `train_utils_sim.py`
- **作用**: 模拟环境训练的**辅助函数库**
- **主要函数**:
  - `obs_to_img(obs, variant)`:
    - 将原始观测转换为 DSRL 使用的缩放图像
    - Libero: 使用 "agentview_image"（翻转）
    - Aloha: 使用 "pixels"]["top"
  - `obs_to_pi_zero_input(obs, variant)`:
    - 将观测转换为 π₀ 输入格式
    - 使用 224x224 的填充缩放（pad & resize）
  - `_quat2axisangle(quat)`:
    - 四元数转轴角表示（从 robosuite 复制）
  - `trajwise_alternating_training_loop()`:
    - **核心训练循环**（轨迹级交替训练）
    - 收集轨迹 → 添加到 buffer → 执行梯度更新

##### `train_utils_real.py`
- **作用**: 真实机器人训练的**辅助函数库**
- **主要函数**:
  - `trajwise_alternating_training_loop()`:
    - 真实机器人版本的训练循环
    - 包含轨迹收集、数据添加、梯度更新逻辑
  - `collect_traj()`:
    - 收集单条轨迹
    - 与 π₀ 策略交互获取动作
    - 与机器人环境交互获取观测和奖励
  - `add_online_data_to_buffer()`:
    - 将收集的轨迹添加到 replay buffer
- **关键特性**:
  - 支持视频录制（使用 moviepy）
  - 支持键盘中断（termios）

---

## `/jaxrl2` 文件夹详细结构

核心算法实现，基于 [jaxrl2](https://github.com/ikostrikov/jaxrl2) 和 [PTR](https://github.com/Asap7772/PTR) 框架。

```
jaxrl2/
├── agents/                   # 强化学习智能体
│   ├── agent.py             # Agent 基类
│   ├── common.py            # 通用组件
│   └── pixel_sac/           # Pixel-based SAC 算法
│       ├── pixel_sac_learner.py    # SAC Learner 主类
│       ├── actor_updater.py        # Actor 网络更新逻辑
│       ├── critic_updater.py       # Critic 网络更新逻辑
│       ├── temperature.py          # 温度参数模块
│       └── temperature_updater.py  # 温度参数更新逻辑
├── data/                     # 数据处理和增强
│   ├── dataset.py           # 数据集抽象类
│   ├── replay_buffer.py     # 经验回放缓冲区
│   └── augmentations.py     # 图像增强（随机裁剪、高斯模糊、颜色变换）
├── networks/                 # 神经网络模块
│   ├── constants.py         # 网络初始化常量
│   ├── mlp.py              # 多层感知机（MLP）
│   ├── normal_policy.py     # 高斯策略
│   ├── normal_tanh_policy.py  # Tanh 高斯策略
│   ├── learned_std_normal_policy.py  # 可学习标准差的高斯策略
│   ├── encoders/            # 视觉编码器
│   │   ├── networks.py         # 基础编码器和 PixelMultiplexer
│   │   ├── impala_encoder.py   # IMPALA 编码器
│   │   ├── resnet_encoderv1.py # ResNet V1 编码器（ResNet18/34/Small）
│   │   ├── resnet_encoderv2.py # ResNet V2 编码器
│   │   ├── spatial_softmax.py  # Spatial Softmax 层
│   │   └── cross_norm.py       # Cross Normalization
│   └── values/              # Value 网络
│       ├── state_value.py          # 状态值函数 V(s)
│       ├── state_action_value.py   # 状态-动作值函数 Q(s,a)
│       └── state_action_ensemble.py # Q 网络集成（多个 Q 网络）
├── utils/                    # 工具函数
│   ├── wandb_logger.py      # Weights & Biases 日志记录器
│   ├── launch_util.py       # 训练启动工具
│   ├── general_utils.py     # 通用工具函数
│   ├── target_update.py     # 目标网络软更新
│   ├── visualization_utils.py  # 可视化工具
│   └── wandb_config_example.py # W&B 配置示例
└── types.py                 # 类型定义
```

---

### `/jaxrl2/agents` - 智能体模块

#### `pixel_sac/` - Pixel-based SAC 算法

DSRL 的核心算法实现，使用 **Soft Actor-Critic (SAC)** 算法在 π₀ 的潜在噪声空间中学习动作扰动。

##### `pixel_sac_learner.py`
- **核心类**: `PixelSACLearner(Agent)`
- **功能**: SAC 算法的主控制器
- **主要方法**:
  - `__init__()`: 初始化 Actor、Critic、温度参数、编码器
  - `update()`: 执行一步梯度更新（调用 `_update_jit`）
  - `sample_actions()`: 采样动作（用于与环境交互）
  - `save_checkpoint()` / `restore_checkpoint()`: 保存/恢复模型检查点
- **关键特性**:
  - 使用 JIT 编译 (`@jax.jit`) 加速更新
  - 支持多设备并行训练（数据分片）
  - 支持数据增强（随机裁剪、颜色抖动）
- **JIT 更新函数**: `_update_jit()`
  ```python
  def _update_jit(rng, actor, critic, target_critic_params, temp, batch, ...):
      # 1. 数据增强（随机裁剪 + 颜色抖动）
      # 2. 更新 Critic
      # 3. 软更新目标 Critic
      # 4. 更新 Actor
      # 5. 更新温度参数
      return new_actor, new_critic, new_target_critic_params, new_temp, info
  ```

##### `actor_updater.py`
- **函数**: `update_actor()`
- **功能**: 更新 Actor 网络参数
- **损失函数**:
  ```python
  actor_loss = (log_probs * temperature - Q_values).mean()
  ```
  - 最大化 Q 值，同时保持熵正则化
- **支持的 Critic Reduction**:
  - `'min'`: 使用最小 Q 值（更保守）
  - `'mean'`: 使用平均 Q 值
- **记录的指标**:
  - `actor_loss`: Actor 损失
  - `entropy`: 策略熵
  - `q_pi_in_actor`: Actor 采样动作的 Q 值
  - `mean_pi_norm`: 策略均值的范数
  - `std_pi_norm`: 策略标准差的范数

##### `critic_updater.py`
- **函数**: `update_critic()`
- **功能**: 更新 Critic 网络参数
- **损失函数**:
  ```python
  target_q = reward + discount * mask * next_q
  critic_loss = ((Q_values - target_q) ** 2).mean()
  ```
- **目标 Q 值计算**:
  1. 使用 Actor 在 next_state 采样动作
  2. 使用目标 Critic 计算 Q 值
  3. 应用 Bellman 更新
- **记录的指标**:
  - `critic_loss`: Critic 损失
  - `q`: 当前 Q 值
  - `target_q`: 目标 Q 值
  - `next_q_pi`: 下一状态的 Q 值

##### `temperature_updater.py`
- **函数**: `update_temperature()`
- **功能**: 自动调整 SAC 的温度参数 α
- **目标**: 保持策略熵接近目标熵 `target_entropy`
- **更新规则**:
  ```python
  temperature_loss = -temperature * (entropy - target_entropy)
  ```

##### `temperature.py`
- **类**: `Temperature`
- **功能**: 可学习的温度参数模块
- **实现**: 使用 log(α) 来保证 α > 0

---

### `/jaxrl2/networks` - 神经网络模块

#### 编码器 (`encoders/`)

##### `networks.py`
- **类 1**: `Encoder`
  - **功能**: 基础 CNN 编码器
  - **结构**: 4 层卷积网络（默认）
  - **配置**:
    - `features=(32, 32, 32, 32)`: 卷积通道数
    - `strides=(2, 1, 1, 1)`: 卷积步长
    - `padding='VALID'`: 填充方式
  - **输出**: 展平的特征向量

- **类 2**: `PixelMultiplexer`
  - **功能**: 视觉观测的多路复用器
  - **作用**: 将像素编码器 + Bottleneck 层 + 下游网络组合在一起
  - **流程**:
    ```
    观测['pixels'] → Encoder → Bottleneck (可选) → 下游网络
                                ↓
                          Dense(latent_dim) → LayerNorm → Tanh
    ```
  - **参数**:
    - `encoder`: 图像编码器（CNN/ResNet/IMPALA）
    - `network`: 下游网络（MLP）
    - `latent_dim`: 潜在表示维度（默认 50）
    - `use_bottleneck`: 是否使用瓶颈层

##### `resnet_encoderv1.py`
- **类**:
  - `ResNet18`: 18 层 ResNet
  - `ResNet34`: 34 层 ResNet
  - `ResNetSmall`: 轻量级 ResNet
- **用途**: 更强的视觉特征提取能力

##### `impala_encoder.py`
- **类**:
  - `ImpalaEncoder`: 标准 IMPALA 编码器
  - `SmallerImpalaEncoder`: 更小的 IMPALA 编码器
- **来源**: DeepMind 的 IMPALA 架构

##### `spatial_softmax.py`
- **功能**: Spatial Softmax 层
- **作用**: 将空间特征图转换为关键点表示
- **用途**: 机器人视觉任务中的常用技术

#### 策略网络 (`policies`)

##### `mlp.py`
- **类 1**: `MLP`
  - **功能**: 标准多层感知机
  - **特性**:
    - 支持 Dropout
    - 支持 LayerNorm
    - 可配置激活函数
  - **辅助函数**: `_flatten_dict(x)` - 展平嵌套的观测字典

- **类 2**: `MLPActionSep`
  - **功能**: 动作单独处理的 MLP
  - **用途**: 在每层都拼接动作信息

##### `learned_std_normal_policy.py`
- **类 1**: `LearnedStdNormalPolicy`
  - **功能**: 可学习标准差的高斯策略
  - **输出**: `MultivariateNormalDiag` 分布
  - **参数**:
    - `log_std_min=-20`: 最小对数标准差
    - `log_std_max=2`: 最大对数标准差

- **类 2**: `TanhMultivariateNormalDiag`
  - **功能**: 带 Tanh 压缩的高斯分布
  - **作用**: 将动作限制在 [-1, 1] 范围内
  - **实现**: 使用 `distrax.Transformed`

- **类 3**: `LearnedStdTanhNormalPolicy`
  - **功能**: 结合 Tanh 的高斯策略
  - **用途**: DSRL 的 Actor 网络
  - **输出**: 动作在指定范围 [low, high] 内

#### Value 网络 (`values/`)

##### `state_action_ensemble.py`
- **类**: `StateActionEnsemble`
- **功能**: Q 网络集成
- **实现**: 使用 `nn.vmap` 创建多个 Q 网络
- **参数**:
  - `num_qs=2`: Q 网络数量（Libero 使用 10，Real 使用 2）
  - `hidden_dims`: 隐藏层维度
- **用途**: 减少 Q 值过估计（通过取最小值）

##### `state_action_value.py`
- **类**: `StateActionValue`
- **功能**: 单个 Q(s, a) 网络
- **实现**: `MLP(观测 + 动作) → Q 值`

##### `state_value.py`
- **类**: `StateValue`
- **功能**: 状态值函数 V(s)
- **实现**: `MLP(观测) → V 值`

---

### `/jaxrl2/data` - 数据模块

#### `replay_buffer.py`
- **类**: `ReplayBuffer(Dataset)`
- **功能**: 经验回放缓冲区
- **存储的数据**:
  ```python
  {
      'observations': Dict[str, np.ndarray],
      'next_observations': Dict[str, np.ndarray],
      'actions': np.ndarray,
      'next_actions': np.ndarray,
      'rewards': np.ndarray,
      'masks': np.ndarray,  # 1 - done
      'discount': np.ndarray
  }
  ```
- **主要方法**:
  - `insert()`: 添加单步经验
  - `insert_batch()`: 批量添加经验
  - `get_iterator()`: 获取批次迭代器
  - `get_random_trajs()`: 获取随机轨迹
  - `increment_traj_counter()`: 增加轨迹计数器
- **特性**:
  - 支持字典观测空间（pixels + state）
  - 支持轨迹级采样
  - 循环缓冲区实现

#### `augmentations.py`
- **函数**:
  - `random_crop()`: 单张图像随机裁剪
  - `batched_random_crop()`: 批量随机裁剪
  - `color_transform()`: 颜色变换（亮度、对比度、饱和度）
  - `_gaussian_blur_single_image()`: 高斯模糊
- **用途**: 数据增强以提高泛化能力
- **实现**: 使用 JAX 的 `jax.lax` 高效实现

#### `dataset.py`
- **类**: `Dataset`
- **功能**: 数据集抽象基类
- **类**: `DatasetDict`
- **功能**: 支持嵌套字典的数据集

---

### `/jaxrl2/utils` - 工具模块

#### `wandb_logger.py`
- **类**: `WandBLogger`
- **功能**: Weights & Biases 日志记录
- **主要方法**:
  - `log()`: 记录标量指标
  - `log_video()`: 记录视频
  - `log_image()`: 记录图像
- **函数**: `create_exp_name()` - 创建带时间戳的实验名称

#### `launch_util.py`
- **函数**: `parse_training_args()`
- **功能**: 解析命令行参数和训练配置
- **返回**: `variant` 字典（包含所有超参数）

#### `target_update.py`
- **函数**: `soft_target_update()`
- **功能**: 软更新目标网络
- **公式**:
  ```python
  target_params = (1 - tau) * target_params + tau * params
  ```
- **参数**: `tau=0.005` (默认)

#### `general_utils.py`
- **函数**:
  - `add_batch_dim()`: 添加批次维度
  - `flatten_dict()`: 展平嵌套字典
- **用途**: 通用数据处理工具

#### `visualization_utils.py`
- **功能**: 可视化工具（绘制曲线、热图等）

---

### `/jaxrl2/types.py` - 类型定义

```python
DataType = Union[np.ndarray, Dict[str, 'DataType']]
PRNGKey = Any  # JAX 随机数生成器密钥
Params = flax.core.FrozenDict[str, Any]  # Flax 模型参数
```

---

### DSRL 算法流程（基于 jaxrl2）

```
1. 初始化 PixelSACLearner:
   ├── Actor: PixelMultiplexer + LearnedStdTanhNormalPolicy
   ├── Critic: PixelMultiplexer + StateActionEnsemble
   ├── Target Critic: 目标网络副本
   └── Temperature: 可学习的温度参数

2. 数据收集:
   ├── 使用 π₀ 策略 + Actor 输出的噪声扰动
   └── 存储到 ReplayBuffer

3. 训练循环 (每步):
   ├── 从 ReplayBuffer 采样批次
   ├── 数据增强（随机裁剪 + 颜色抖动）
   ├── 更新 Critic:
   │   ├── 计算目标 Q 值（使用目标网络）
   │   └── 最小化 TD 误差
   ├── 软更新目标 Critic
   ├── 更新 Actor:
   │   ├── 最大化 Q 值
   │   └── 熵正则化
   └── 更新温度参数

4. 评估:
   ├── 使用确定性策略（均值动作）
   └── 记录成功率、奖励等指标
```

---

### 关键设计选择

1. **编码器**: 使用轻量级 CNN + Bottleneck 层
   - Libero/Aloha: `encoder_type='small'`, `latent_dim=50`
   - Real: `latent_dim=50`

2. **Critic Reduction**:
   - Libero/Aloha: `'mean'` (平均 10 个 Q 网络)
   - Real: `'min'` (取 2 个 Q 网络的最小值)

3. **数据增强**:
   - 随机裁剪: `padding=4`
   - 颜色抖动: 可选（`color_jitter=False`）
   - 仅增强当前观测和下一观测

4. **Spatial Softmax**:
   - 用于提取空间关键点特征
   - `use_spatial_softmax=True`

5. **Batch Normalization**:
   - 使用 `LayerNorm` 而非 `BatchNorm`
   - 更稳定的训练（特别是在小批次下）

---

## 子模块说明

### LIBERO
- **来源**: https://github.com/ARISE-Initiative/LIBERO
- **作用**: 提供 Libero 基准测试环境
- **安装**: `pip install -e LIBERO`

### openpi
- **来源**: https://github.com/Physical-Intelligence/openpi
- **作用**: 预训练的通用策略 π₀
- **安装**:
  ```bash
  pip install -e openpi
  pip install -e openpi/packages/openpi-client
  ```

---

## 训练流程总结

### 模拟环境训练
```
bash examples/scripts/run_libero.sh
     ↓
examples/launch_train_sim.py (参数解析)
     ↓
examples/train_sim.py (主程序)
     ↓
- 加载 π₀ 策略
- 创建环境 (Libero/Aloha)
- 初始化 PixelSACLearner
- 创建 ReplayBuffer
     ↓
examples/train_utils_sim.py::trajwise_alternating_training_loop()
     ↓
- 收集轨迹 (使用 π₀ + 噪声)
- 存储到 buffer
- 执行 UTD 梯度更新
- 记录日志到 W&B
```

### 真实机器人训练
```
[远程服务器] python scripts/serve_policy.py --env=DROID
                          ↓
             启动 π₀ 策略服务器

bash examples/scripts/run_real.sh
     ↓
examples/launch_train_real.py (参数解析)
     ↓
examples/train_real.py (主程序)
     ↓
- 连接远程 π₀ 服务器 (WebSocket)
- 创建 DROID 机器人环境
- 初始化 PixelSACLearner
- 创建 ReplayBuffer
     ↓
examples/train_utils_real.py::trajwise_alternating_training_loop()
     ↓
- 收集轨迹 (真实机器人交互)
- 存储到 buffer
- 执行 UTD 梯度更新
- 记录日志到 W&B
```

---

## 关键概念

### DSRL (Diffusion Steering via Reinforcement Learning)
- **目标**: 在潜在空间中使用强化学习引导预训练的扩散策略
- **方法**: 学习一个 Actor 在 π₀ 的噪声空间中输出扰动
- **算法**: Pixel-based SAC (Soft Actor-Critic)

### UTD (Update-to-Data) Ratio
- **定义**: 每个环境步执行的梯度更新次数
- **参数**: `--multi_grad_step`
- **典型值**:
  - Libero: 20
  - Aloha: 20
  - Real: 30

### Query Frequency
- **定义**: 每隔多少步查询一次 π₀ 策略
- **参数**: `--query_freq`
- **典型值**:
  - Libero: 20
  - Aloha: 50
  - Real: 10

---

## 日志和输出

### 训练日志
- **存储路径**: `./logs/<proj_name>/<exp_name>/`
- **内容**: 模型检查点、训练日志

### Weights & Biases
- **项目**: 由 `--wandb_project` 指定
- **示例日志**: https://wandb.ai/mitsuhiko/DSRL_pi0_public
- **记录指标**:
  - 训练损失（actor_loss, critic_loss, temp）
  - 环境交互（rewards, success_rate）
  - 在线样本数量（num_online_samples, num_online_trajs）

---

## 依赖版本要求

### 关键依赖
```
python=3.11.11
jax[cuda12]==0.5.0
torch==2.6.0 (CPU only, for Libero)
mujoco==3.3.1 (Libero) / 2.3.7 (Aloha)
```

### 安装顺序
```bash
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"
pip install -e openpi
pip install -e openpi/packages/openpi-client
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## 环境变量说明

### 模拟环境
```bash
DISPLAY=:0                  # X11 display
MUJOCO_GL=egl              # MuJoCo 渲染后端
PYOPENGL_PLATFORM=egl      # OpenGL 平台
MUJOCO_EGL_DEVICE_ID       # GPU 设备 ID
OPENPI_DATA_HOME           # OpenPI 数据目录
EXP                        # 实验输出目录
CUDA_VISIBLE_DEVICES       # 可见 CUDA 设备
XLA_PYTHON_CLIENT_PREALLOCATE=false  # 禁用 XLA 内存预分配
```

### 真实机器人
```bash
LEFT_CAMERA_ID             # 左侧相机 ID
RIGHT_CAMERA_ID            # 右侧相机 ID
WRIST_CAMERA_ID            # 手腕相机 ID
remote_host                # π₀ 远程服务器地址
remote_port                # π₀ 远程服务器端口
```

---

## 引用

```bibtex
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

---

## 联系方式

如有问题、Bug 反馈或改进建议，请联系:
- 邮箱: nakamoto[at]berkeley[dot]edu
- GitHub Issues: https://github.com/nakamotoo/dsrl_pi0/issues
