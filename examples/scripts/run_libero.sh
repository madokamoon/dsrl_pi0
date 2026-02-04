#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=7

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export PYTHONPATH=.:$PYTHONPATH
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name;
export CUDA_VISIBLE_DEVICES=$device_id
# 禁用 XLA (JAX 的编译器) 的内存预分配
# 这样可以避免一次性占用所有 GPU 内存，允许多个程序共享 GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 注意: Libero 需要特定版本的 MuJoCo (3.3.1)
# Aloha 环境则需要 MuJoCo 2.3.7
pip install mujoco==3.3.1

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \              # 使用的算法: Pixel-based SAC (Soft Actor-Critic)
--env libero \                       # 环境名称: libero (机器人操作任务)
--prefix dsrl_pi0_libero \           # 实验前缀，用于命名日志和检查点
--wandb_project ${proj_name} \       # W&B 项目名称，用于在线监控训练
--batch_size 256 \                   # 批次大小: 每次梯度更新使用的样本数
--discount 0.999 \                   # 折扣因子 (gamma): 未来奖励的衰减系数 (0.999 表示非常看重长期奖励)
--seed 0 \                           # 随机种子: 保证实验可重复
--max_steps 500000 \                 # 最大训练步数: 环境交互的总步数
--eval_interval 10000 \              # 评估间隔: 每 10000 步评估一次策略性能
--log_interval 500 \                 # 日志间隔: 每 500 步记录一次训练指标
--eval_episodes 10 \                 # 评估回合数: 每次评估运行 10 个 episode
--multi_grad_step 20 \               # UTD (Update-to-Data) Ratio: 每个环境步执行 20 次梯度更新
--start_online_updates 500 \         # 开始在线更新前的初始样本数: 先收集 500 步数据再开始训练
--resize_image 64 \                  # 图像缩放尺寸: 将相机图像缩放到 64x64 像素
--action_magnitude 1.0 \             # 动作幅度: 控制添加到 π₀ 策略的噪声强度
--query_freq 20 \                    # 查询频率: 每 20 步查询一次 π₀ 策略获取动作
--hidden_dims 128 \