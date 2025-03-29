"""
Các hàm huấn luyện và đánh giá cho DQNAgent.

Module này cung cấp các hàm tiện ích để huấn luyện và đánh giá DQNAgent.
"""

import numpy as np
import torch
import time
import os
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
from .utils import format_time, logger, plot_learning_curve
from .agent import DQNAgent

def train_dqn(agent: DQNAgent, 
             env, 
             n_episodes: int = 1000, 
             max_t: int = 1000, 
             eps_decay_type: str = 'exponential', 
             checkpoint_freq: int = 100,
             early_stopping: bool = True,
             patience: int = 50,
             min_improvement: float = 0.01,
             eval_interval: int = 10,
             render: bool = False,
             verbose: bool = True):
    """
    Huấn luyện DQNAgent
    
    Args:
        agent: DQNAgent cần huấn luyện
        env: Môi trường tương tác
        n_episodes: Số lượng episodes
        max_t: Số bước tối đa trong mỗi episode
        eps_decay_type: Loại suy giảm epsilon ('exponential', 'linear', hoặc 'custom')
        checkpoint_freq: Tần suất lưu checkpoint (số episodes)
        early_stopping: Sử dụng early stopping hay không
        patience: Số episodes chờ trước khi dừng sớm
        min_improvement: Ngưỡng cải thiện tối thiểu
        eval_interval: Tần suất đánh giá (số episodes)
        render: Có render môi trường hay không
        verbose: In thông tin chi tiết hay không
        
    Returns:
        Dict: Dictionary chứa kết quả huấn luyện (rewards, validation_rewards, best_reward, training_time)
    """
    train_start_time = time.time()
    rewards = []
    best_avg_reward = -float('inf')
    episodes_without_improvement = 0
    window_size = 100
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        done = False
        steps = 0
        
        while not done and steps < max_t:
            # Chọn hành động
            action = agent.act(state, agent.epsilon)
            
            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            
            # Lưu trữ kinh nghiệm và cập nhật policy
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            steps += 1
            
            if render:
                env.render()
        
        # Cập nhật epsilon
        agent.update_epsilon(eps_decay_type)
        
        # Thêm phần thưởng
        rewards.append(score)
        agent.training_rewards.append(score)
        
        # Tính trung bình động trong cửa sổ
        avg_reward = np.mean(rewards[-min(len(rewards), window_size):])
        
        # In thông tin tiến trình
        if verbose and (episode % 10 == 0 or episode == 1):
            elapsed_time = time.time() - train_start_time
            time_str = format_time(elapsed_time)
            
            remaining_episodes = n_episodes - episode
            if episode > 1:
                time_per_episode = elapsed_time / episode
                remaining_time = remaining_episodes * time_per_episode
                eta_str = format_time(remaining_time)
            else:
                eta_str = "N/A"
            
            logger.info(f"Episode {episode}/{n_episodes} | Score: {score:.2f} | Avg Score: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Time: {time_str} | ETA: {eta_str}")
        
        # Lưu checkpoint
        if episode % checkpoint_freq == 0:
            is_best = avg_reward > best_avg_reward
            if is_best:
                best_avg_reward = avg_reward
                episodes_without_improvement = 0
            agent.save_checkpoint(episode, avg_reward, is_best)
        
        # Đánh giá agent
        if episode % eval_interval == 0:
            eval_reward = evaluate_dqn(agent, env, 5, max_t, render=False)
            agent.validation_rewards.append(eval_reward)
            if verbose:
                logger.info(f"Evaluation at episode {episode}: Average reward = {eval_reward:.2f}")
            
            # Cập nhật best score và early stopping
            if eval_reward > best_avg_reward + min_improvement:
                best_avg_reward = eval_reward
                episodes_without_improvement = 0
                agent.save_checkpoint(episode, eval_reward, True)
            else:
                episodes_without_improvement += eval_interval
        
        # Early stopping
        if early_stopping and episodes_without_improvement >= patience:
            logger.info(f"Early stopping tại episode {episode}. Không có cải thiện sau {patience} episodes.")
            break
    
    # Kết thúc huấn luyện
    agent.load_best_model()  # Tải model tốt nhất
    total_time = time.time() - train_start_time
    logger.info(f"Huấn luyện hoàn tất sau {episode} episodes.")
    logger.info(f"Thời gian huấn luyện: {format_time(total_time)}")
    logger.info(f"Điểm tốt nhất: {best_avg_reward:.2f}")
    
    # Trả về kết quả
    return {
        'rewards': rewards,
        'validation_rewards': agent.validation_rewards,
        'best_reward': best_avg_reward,
        'training_time': total_time,
        'episodes': episode
    }

def evaluate_dqn(agent: DQNAgent, env, n_episodes: int = 5, max_t: int = 1000, render: bool = False):
    """
    Đánh giá DQNAgent
    
    Args:
        agent: DQNAgent cần đánh giá
        env: Môi trường tương tác
        n_episodes: Số lượng episodes để đánh giá
        max_t: Số bước tối đa trong mỗi episode
        render: Có render môi trường hay không
        
    Returns:
        float: Phần thưởng trung bình trên tất cả các episodes
    """
    rewards = []
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        done = False
        steps = 0
        
        while not done and steps < max_t:
            # Chọn hành động theo policy hiện tại (không có epsilon)
            action = agent.act(state, eps=0.0)
            
            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            score += reward
            steps += 1
            
            if render:
                env.render()
        
        rewards.append(score)
    
    avg_reward = np.mean(rewards)
    return avg_reward

def plot_dqn_rewards(rewards: List[float], 
                    val_rewards: Optional[List[float]] = None,
                    window_size: int = 20,
                    title: str = "DQN Training Rewards", 
                    save_path: Optional[str] = None):
    """
    Vẽ biểu đồ phần thưởng huấn luyện và đánh giá
    
    Args:
        rewards: Danh sách phần thưởng huấn luyện
        val_rewards: Danh sách phần thưởng đánh giá
        window_size: Kích thước cửa sổ để tính trung bình động
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ, nếu None thì hiển thị
    """
    plt.figure(figsize=(12, 6))
    
    # Vẽ phần thưởng từng episode
    plt.plot(rewards, label='Training Reward', alpha=0.3, color='blue')
    
    # Tính và vẽ trung bình động
    avg_rewards = []
    for i in range(len(rewards)):
        if i < window_size:
            avg_rewards.append(np.mean(rewards[:i+1]))
        else:
            avg_rewards.append(np.mean(rewards[i-window_size+1:i+1]))
    
    plt.plot(avg_rewards, label=f'Average Reward (window={window_size})', color='blue')
    
    # Vẽ phần thưởng validation nếu có
    if val_rewards is not None:
        # Xác định khoảng cách giữa các lần đánh giá
        eval_interval = len(rewards) // len(val_rewards)
        if eval_interval < 1:
            eval_interval = 1
        
        # Tạo trục x cho dữ liệu validation
        eval_indices = [i * eval_interval for i in range(len(val_rewards))]
        plt.plot(eval_indices, val_rewards, label='Validation Reward', color='red', marker='o')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Đồ thị đã được lưu tại: {save_path}")
    else:
        plt.show()

def compare_dqn_variants(env, variants: List[Dict[str, Any]], 
                         n_episodes: int = 500, 
                         max_t: int = 1000):
    """
    So sánh các biến thể DQN
    
    Args:
        env: Môi trường tương tác
        variants: Danh sách các từ điển chứa tham số cấu hình cho từng biến thể
        n_episodes: Số lượng episodes huấn luyện
        max_t: Số bước tối đa trong mỗi episode
        
    Returns:
        Dict: Dictionary chứa kết quả so sánh
    """
    results = {}
    
    for config in variants:
        name = config.pop('name', 'Unknown')
        logger.info(f"Huấn luyện biến thể: {name}")
        
        # Tạo agent
        agent = DQNAgent(**config)
        
        # Huấn luyện agent
        training_result = train_dqn(agent, env, n_episodes, max_t)
        
        # Đánh giá agent
        eval_reward = evaluate_dqn(agent, env, n_episodes=10)
        
        results[name] = {
            'agent': agent,
            'training_result': training_result,
            'eval_reward': eval_reward
        }
        
        logger.info(f"Kết quả biến thể {name}: {eval_reward:.2f}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    
    for name, result in results.items():
        rewards = result['training_result']['rewards']
        window_size = 20
        
        # Tính trung bình động
        avg_rewards = []
        for i in range(len(rewards)):
            if i < window_size:
                avg_rewards.append(np.mean(rewards[:i+1]))
            else:
                avg_rewards.append(np.mean(rewards[i-window_size+1:i+1]))
        
        plt.plot(avg_rewards, label=f'{name} (Final: {result["eval_reward"]:.2f})')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Comparison of DQN Variants')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_comparison.png')
    
    return results 