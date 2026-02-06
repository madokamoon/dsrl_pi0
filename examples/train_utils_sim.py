from tqdm import tqdm
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL

def _quat2axisangle(quat):
    """
    将四元数转换为轴角表示。

    从robosuite复制: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # 裁剪四元数
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 这接近于零度旋转，直接返回
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    将原始观测转换为DSRL actor/critic使用的缩放图像
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0:
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    """
    将原始观测转换为π₀策略的输入格式。

    π₀策略需要特定格式的输入，包括：
    - 224x224的RGB图像（经过填充和缩放）
    - 手腕相机图像（用于精细操作）
    - 本体感知状态（关节位置、末端执行器位姿等）
    - 语言指令（prompt）

    参数:
        obs: 原始环境观测字典
        variant: 配置对象，包含环境类型和任务描述

    返回:
        obs_pi_zero: π₀策略输入字典
    """
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )

        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    """
    从原始观测中提取关节位置（qpos），用于状态输入。

    参数:
        obs: 原始环境观测字典
        variant: 配置对象，包含环境类型

    返回:
        qpos: 关节位置数组
    """
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    """
    DSRL（基于强化学习的扩散策略引导）的主训练循环。

    该函数实现轨迹级交替训练，智能体在以下三个步骤之间交替：
    1. 在环境中收集一条完整的轨迹
    2. 将轨迹添加到回放缓冲区
    3. 执行多次梯度更新（UTD比率）

    循环持续进行，直到达到最大训练步数。

    参数:
        variant: 包含超参数和设置的配置对象
        agent: 正在训练的SAC智能体（PixelSACLearner）
        env: 用于收集训练轨迹的环境
        eval_env: 用于评估的环境
        online_replay_buffer: 存储在线交互数据的缓冲区
        replay_buffer: 用于训练批次的回放缓冲区迭代器
        wandb_logger: 用于跟踪指标的Weights & Biases日志记录器
        perform_control_evals: 是否执行控制评估
        shard_fn: 用于多设备训练的数据分片函数
        agent_dp: 用于动作生成的扩散策略（π₀）
    """

    # 创建从回放缓冲区采样批次的迭代器
    # shard_fn用于多设备训练，在多个设备之间分发数据
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    # 初始化训练统计
    total_env_steps = 0  # 已执行的环境步数总数
    i = 0  # 当前训练步数计数器

    # 记录初始指标到W&B
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    # 主训练循环
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:

            # ============================================================================
            # 阶段1: 在环境中收集轨迹
            # ============================================================================
            # 使用当前策略与环境交互以收集一条轨迹
            # 策略由以下部分组成：π₀（扩散策略）+ SAC智能体噪声扰动
            traj = collect_traj(variant, agent, env, i, agent_dp)

            # 获取轨迹ID并将收集的数据添加到在线回放缓冲区
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)

            # 更新环境步数计数器
            total_env_steps += traj['env_steps']

            # 打印当前缓冲区统计信息
            print('online buffer timesteps length:', len(online_replay_buffer)) 
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)

            # ============================================================================
            # 阶段2: 计算梯度更新次数（UTD比率）
            # ============================================================================
            # 确定为该轨迹执行多少个梯度步数
            # 可选项：
            # 1. 每个批次的固定梯度步数（如果已指定）
            # 2. 动态：episode_length * multi_grad_step（UTD比率）
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                # 如果已指定，使用固定的梯度步数
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                # 使用动态UTD比率：较长的回合执行更多梯度步数
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step

            # ============================================================================
            # 阶段3: 执行多次梯度更新
            # ============================================================================
            # 只有在收集足够初始数据后才开始更新
            if len(online_replay_buffer) > variant.start_online_updates:

                # 为该轨迹执行多次梯度更新
                for _ in range(num_gradsteps):

                    # ---------------------------------------------------------------------
                    # 特殊情况：在任何更新之前进行初始评估
                    # ---------------------------------------------------------------------
                    # 在第一次梯度更新之前执行初始策略评估
                    # 这为我们提供了未训练（或预训练）策略的基线
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        # 评估控制性能
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        # 智能体特定评估（如果已实现）
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # ---------------------------------------------------------------------
                    # 采样批次并执行梯度更新
                    # ---------------------------------------------------------------------
                    # 从回放缓冲区采样一个批次并执行一次梯度更新
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    # 更新进度条和步数计数器
                    pbar.update()
                    i += 1


                    # ---------------------------------------------------------------------
                    # 日志记录：训练指标
                    # ---------------------------------------------------------------------
                    # 定期记录训练指标
                    if i % variant.log_interval == 0:
                        # 将指标从GPU移动到CPU以进行日志记录
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}

                        # 记录不同类型的指标
                        for k, v in update_info.items():
                            if v.ndim == 0:  # 标量指标
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:  # 直方图指标（向量、矩阵）
                                wandb_logger.log_histogram(f'training/{k}', v, i)

                        # 记录探索轨迹统计信息
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),  # 当前缓冲区大小
                            'episode_return (exploration)': traj['episode_return'],  # 刚收集轨迹的回报
                            'is_success (exploration)': int(traj['is_success']),  # 成功状态
                        }, i)

                    # ---------------------------------------------------------------------
                    # 评估：定期策略评估
                    # ---------------------------------------------------------------------
                    # 定期执行全面评估
                    if i % variant.eval_interval == 0:
                        # 记录当前训练统计信息
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)

                        # 评估控制性能
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)

                        # 智能体特定评估
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # ---------------------------------------------------------------------
                    # 检查点：保存模型权重
                    # ---------------------------------------------------------------------
                    # 定期保存模型检查点
                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    """
    将收集到的轨迹添加到在线回放缓冲区。

    该函数处理轨迹并将每个时间步作为单个状态转移添加到回放缓冲区。
    每个状态转移包含：
    - 当前观测
    - 下一观测
    - 动作（噪声扰动）
    - 下一动作
    - 奖励
    - 掩码（1表示非终止，0表示终止）
    - 折扣因子

    参数:
        variant: 配置对象
        traj: 收集的轨迹字典，包含：
            - 'actions': 动作列表 (T, chunk_size, action_dim)
            - 'observations': 观测列表
            - 'rewards': 奖励数组
            - 'masks': 掩码数组（终止标志）
        online_replay_buffer: 用于存储数据的回放缓冲区
    """

    # 折扣范围是查询频率（查询π₀的频率）
    discount_horizon = variant.query_freq

    # 将动作转换为numpy数组: (T, chunk_size, action_dim)
    actions = np.array(traj['actions'])
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    # 处理轨迹中的每个时间步
    for t in range(episode_len):
        # 获取当前和下一观测
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]

        # 从观测中移除批次维度
        # obs最初是批处理形式 {key: array[1, ...]}, 我们提取 array[0]
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}

        # 可选：如果不使用状态信息，则移除
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        # 创建回放缓冲区的状态转移字典
        insert_dict = dict(
            observations=obs,  # 当前观测
            next_observations=next_obs,  # 下一观测
            actions=actions[t],  # 当前动作（噪声扰动）
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],  # 下一动作
            rewards=rewards[t],  # 此状态转移的奖励
            masks=masks[t],  # 终止掩码（1表示未结束，0表示结束）
            discount=variant.discount ** discount_horizon  # 折扣因子
        )

        # 将状态转移插入回放缓冲区
        online_replay_buffer.insert(insert_dict)

    # 增加缓冲区中的轨迹计数器
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None):
    """
    在环境中收集一条完整的轨迹数据。

    该函数与环境交互，使用当前策略（π₀扩散策略 + SAC智能体噪声扰动）
    收集一条轨迹。轨迹在指定的查询频率下由π₀策略生成动作块。

    工作流程：
    1. 重置环境并初始化
    2. 对于每个时间步：
       - 获取当前观测
       - 每query_frequency步查询π₀策略生成动作块
       - SAC智能体预测噪声扰动
       - 执行动作并记录数据
    3. 处理稀疏奖励并返回轨迹

    Args:
        variant: 配置对象，包含超参数和设置
        agent: 正在训练的SAC智能体（PixelSACLearner）
        env: 用于收集轨迹的环境
        i: 当前训练步数（用于决定初始噪声采样）
        agent_dp: π₀扩散策略，用于生成底层动作

    Returns:
        trajectory: 轨迹字典，包含：
            - 'observations': 观测列表，形状为[(T+1,), ...]
            - 'actions': 动作（噪声扰动）列表，形状为[(T, chunk_size, action_dim), ...]
            - 'rewards': 稀疏奖励数组，形状为(T,)
            - 'masks': 终止掩码数组（1为未终止，0为已终止），形状为(T,)
            - 'is_success': 轨迹是否成功
            - 'episode_return': 回合总回报
            - 'images': 图像列表用于可视化
            - 'env_steps': 环境步数
    """

    # 从配置中获取参数
    query_frequency = variant.query_freq  # π₀查询频率
    max_timesteps = variant.max_timesteps  # 最大时间步数
    env_max_reward = variant.env_max_reward  # 环境最大奖励（用于判断成功）

    # 分割随机数生成器，用于环境重置和噪声采样
    agent._rng, rng = jax.random.split(agent._rng)

    # 重置环境
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()

    # 初始化数据收集列表
    image_list = []  # 用于可视化的图像列表
    rewards = []  # 奖励列表
    action_list = []  # 动作（噪声扰动）列表
    obs_list = []  # 观测列表

    # 主交互循环
    for t in tqdm(range(max_timesteps)):
        # 将原始观测转换为DSRL使用的图像格式
        curr_image = obs_to_img(obs, variant)

        # 将观测转换为关节位置（状态）
        qpos = obs_to_qpos(obs, variant)

        # 构建观测字典，用于SAC智能体
        if variant.add_states:
            # 同时使用视觉观测和状态信息
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],  # 添加批次和通道维度
                'state': qpos[np.newaxis, ..., np.newaxis],  # 添加批次和通道维度
            }
        else:
            # 只使用视觉观测
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],  # 添加批次和通道维度
            }

        # ============================================================================
        # 查询π₀策略（每query_frequency步查询一次）
        # ============================================================================
        if t % query_frequency == 0:
            # 确保扩散策略可用
            assert agent_dp is not None

            # 分割随机数生成器
            rng, key = jax.random.split(rng)

            # 将观测转换为π₀输入格式
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)

            # ---------------------------------------------------------------------
            # 生成噪声用于π₀策略
            # ---------------------------------------------------------------------
            if i == 0:
                # 在初始回合的数据收集中，从标准高斯分布采样噪声
                # 这用于评估预训练π₀策略的基线性能
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                # 将噪声填充到50步（π₀默认的动作块长度）
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                # 提取用于当前时间步的噪声
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                # SAC智能体预测用于扩散模型的噪声
                # SAC智能体学习在潜在空间中输出最优的噪声扰动
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                # 将噪声填充到50步
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

            # 使用π₀策略生成动作块
            # π₀策略接收噪声作为输入，输出最终动作
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

            # 记录噪声扰动和观测（用于训练SAC）
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        # 从动作块中选择当前时间步的动作
        action_t = actions[t % query_frequency]

        # 执行动作
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        # 记录奖励和图像
        rewards.append(reward)
        image_list.append(curr_image)

        # 如果回合结束，跳出循环
        if done:
            break

    # ============================================================================
    # 添加最后一个观测（用于最后的完整状态转移）
    # ============================================================================
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)

    # ============================================================================
    # 处理轨迹奖励和掩码
    # ============================================================================
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])  # 计算回合总回报
    is_success = (reward == env_max_reward)  # 判断是否成功
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')

    '''
    我们使用稀疏的-1/0奖励来训练SAC智能体。
    奖励设计：
    - 成功：前query_steps-1步为-1，最后一步为0
    - 失败：所有步都为-1
    这种稀疏奖励鼓励智能体尽快完成任务。
    '''
    if is_success:
        query_steps = len(action_list)
        # 成功轨迹：除了最后一步有奖励，前面所有步的奖励为-1
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])

        # 掩码：前query_steps-1步为1（继续），最后一步为0（终止）
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        # 失败轨迹：所有步的奖励为-1
        rewards = -np.ones(query_steps)
        # 掩码：所有步都为1（因为没成功，我们假设还会继续）
        masks = np.ones(query_steps)

    # 返回轨迹字典
    return {
        'observations': obs_list,  # 观测列表（包含最后一个观测）
        'actions': action_list,  # 动作（噪声扰动）列表
        'rewards': rewards,  # 稀疏奖励数组
        'masks': masks,  # 终止掩码数组
        'is_success': is_success,  # 是否成功
        'episode_return': episode_return,  # 回合总回报
        'images': image_list,  # 图像列表
        'env_steps': t + 1  # 环境步数
    }

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    """
    在评估环境中评估当前策略的性能。

    该函数在与训练环境独立的评估环境中，运行多个回合进行评估。
    评估使用确定性策略（均值动作）而非随机策略。

    评估指标包括：
    - 成功率：达到最大奖励的回合比例
    - 平均回报：回合奖励总和的平均值
    - 平均回合长度：成功的回合平均耗时
    - 回报分布：达到各个奖励阈值的回合比例

    Args:
        agent: 当前SAC智能体
        env: 评估环境
        i: 当前训练步数
        variant: 配置对象，包含超参数和设置
        wandb_logger: Weights & Biases日志记录器
        agent_dp: π₀扩散策略

    Returns:
        None（结果直接记录到W&B）
    """

    # 从配置中获取参数
    query_frequency = variant.query_freq  # π₀查询频率
    max_timesteps = variant.max_timesteps  # 最大时间步数
    env_max_reward = variant.env_max_reward  # 环境最大奖励

    # 初始化评估统计列表
    episode_returns = []  # 回合回报列表
    highest_rewards = []  # 最高奖励列表
    success_rates = []  # 成功率列表
    episode_lens = []  # 回合长度列表

    # 为评估创建独立的随机数生成器
    rng = jax.random.PRNGKey(variant.seed + 456)

    # 运行多个评估回合
    for rollout_id in range(variant.eval_episodes):
        # 重置评估环境
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()

        # 初始化回合数据收集列表
        image_list = []  # 用于可视化的图像列表
        rewards = []  # 奖励列表

        # 执行单个评估回合
        for t in tqdm(range(max_timesteps)):
            # 将原始观测转换为DSRL使用的图像格式
            curr_image = obs_to_img(obs, variant)

            # ============================================================================
            # 查询π₀策略（每query_frequency步查询一次）
            # ============================================================================
            if t % query_frequency == 0:
                # 将观测转换为关节位置（状态）
                qpos = obs_to_qpos(obs, variant)

                # 构建观测字典
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],  # 添加批次维度
                        'state': qpos[np.newaxis, ..., np.newaxis],  # 添加批次维度
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],  # 添加批次维度
                    }

                # 分割随机数生成器
                rng, key = jax.random.split(rng)

                # 确保扩散策略可用
                assert agent_dp is not None

                # 将观测转换为π₀输入格式
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)

                # ---------------------------------------------------------------------
                # 生成噪声用于π₀策略
                # ---------------------------------------------------------------------
                if i == 0:
                    # 在初始评估时，从标准高斯分布采样噪声
                    # 这用于评估基础π₀策略的性能（作为基线）
                    noise = jax.random.normal(rng, (1, 50, 32))
                else:
                    # SAC智能体预测用于扩散模型的噪声（确定性策略）
                    # 在评估时使用确定性策略（均值动作）
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    # 将噪声填充到50步
                    noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                # 使用π₀策略生成动作块
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

            # 从动作块中选择当前时间步的动作
            action_t = actions[t % query_frequency]

            # 执行动作
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated

            # 记录奖励和图像
            rewards.append(reward)
            image_list.append(curr_image)

            # 如果回合结束，跳出循环
            if done:
                break

        # ============================================================================
        # 回合统计和处理
        # ============================================================================
        # 记录回合长度
        episode_lens.append(t + 1)

        # 将奖励转换为数组
        rewards = np.array(rewards)

        # 计算回合统计
        episode_return = np.sum(rewards)  # 回合总回报
        episode_returns.append(episode_return)

        episode_highest_reward = np.max(rewards)  # 回合最高奖励
        highest_rewards.append(episode_highest_reward)

        is_success = (reward == env_max_reward)  # 是否成功
        success_rates.append(is_success)

        # 打印回合结果
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')

        # 将图像序列转换为视频并记录到W&B
        video = np.stack(image_list).transpose(0, 3, 1, 2)  # 转换格式为(T, C, H, W)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)

    # ============================================================================
    # 计算并记录总体评估结果
    # ============================================================================
    # 计算平均统计
    success_rate = np.mean(np.array(success_rates))  # 成功率
    avg_return = np.mean(episode_returns)  # 平均回报
    avg_episode_len = np.mean(episode_lens)  # 平均回合长度

    # 构建评估摘要字符串
    summary_str = f'\n成功率: {success_rate}\n平均回报: {avg_return}\n\n'

    # 记录关键指标到W&B
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)

    # 记录回报分布（达到每个奖励阈值的回合比例）
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()  # 达到至少r奖励的回合数
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes  # 比例
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'奖励 >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate * 100}%\n'

    # 打印评估摘要
    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    """
    生成多个价值函数-奖励可视化图像。

    从回放缓冲区中随机采样几条轨迹，使用智能体的可视化功能生成
    价值函数和奖励的热力图，并记录到W&B。

    参数:
        agent: 智能体（具有make_value_reward_visulization方法）
        variant: 配置对象
        i: 当前训练步数
        replay_buffer: 回放缓冲区
        wandb_logger: Weights & Biases日志记录器
    """
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
