"""
Script thử nghiệm cho Rainbow DQN và Actor-Critic.

Script này thực hiện thử nghiệm Rainbow DQN và Actor-Critic Agent trong môi trường mạng blockchain.
"""

import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from qtrust.agents.dqn import DQNAgent, RainbowDQNAgent, ActorCriticAgent
from qtrust.agents.dqn.utils import plot_learning_curve, logger, format_time

def test_rainbow_dqn(env, num_episodes=10, max_steps=500, render=False):
    """
    Thử nghiệm Rainbow DQN trong môi trường cho trước.
    
    Args:
        env: Môi trường học tập
        num_episodes: Số episode thử nghiệm
        max_steps: Số bước tối đa mỗi episode
        render: Có hiển thị môi trường hay không
    """
    # Lấy kích thước không gian trạng thái và hành động
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Khởi tạo Rainbow DQN Agent
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        n_step=3,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_layers=[512, 256],
        learning_rate=5e-4,
        batch_size=128,
        buffer_size=100000
    )
    
    # Giám sát hiệu suất
    scores = []
    avg_scores = []
    timestamps = []
    start_time = time.time()
    
    # Huấn luyện agent qua các episode
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_steps):
            # Hiển thị môi trường nếu yêu cầu
            if render:
                env.render()
                
            # Chọn hành động
            action = agent.act(state)
            
            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            
            # Học từ kinh nghiệm
            agent.step(state, action, reward, next_state, done)
            
            # Cập nhật state và score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Lưu score
        scores.append(score)
        
        # Tính điểm trung bình
        if len(scores) < 100:
            avg_score = np.mean(scores)
        else:
            avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Ghi thời gian
        elapsed_time = time.time() - start_time
        timestamps.append(elapsed_time)
        
        # In thông tin
        logger.info(f"Episode {i_episode}/{num_episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Time: {format_time(elapsed_time)}")
        
        # Lưu model khi đạt được điểm cao nhất
        if avg_score >= agent.best_score:
            agent.best_score = avg_score
            agent.best_model_path = agent.save(episode=i_episode)
    
    # Vẽ đồ thị học tập
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_episodes+1), scores, alpha=0.6, label='Score')
    plt.plot(range(1, num_episodes+1), avg_scores, label='Avg Score (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Rainbow DQN Learning Curve')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(agent.loss_history, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Rainbow DQN Loss')
    
    plt.tight_layout()
    plt.savefig('rainbow_dqn_learning_curve.png')
    plt.show()
    
    return agent, scores, avg_scores

def test_actor_critic(env, num_episodes=10, max_steps=500, render=False):
    """
    Thử nghiệm Actor-Critic Agent trong môi trường cho trước.
    
    Args:
        env: Môi trường học tập
        num_episodes: Số episode thử nghiệm
        max_steps: Số bước tối đa mỗi episode
        render: Có hiển thị môi trường hay không
    """
    # Lấy kích thước không gian trạng thái và hành động
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Khởi tạo Actor-Critic Agent
    agent = ActorCriticAgent(
        state_size=state_size,
        action_size=action_size,
        use_n_step=True,
        n_step=3,
        hidden_layers=[512, 256],
        actor_lr=1e-4,
        critic_lr=5e-4,
        batch_size=128,
        buffer_size=100000,
        entropy_coef=0.01,
        use_noisy_nets=True,
        distributional=True,
        n_atoms=51
    )
    
    # Giám sát hiệu suất
    scores = []
    avg_scores = []
    timestamps = []
    start_time = time.time()
    
    # Huấn luyện agent qua các episode
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_steps):
            # Hiển thị môi trường nếu yêu cầu
            if render:
                env.render()
                
            # Chọn hành động
            action = agent.act(state)
            
            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            
            # Học từ kinh nghiệm
            agent.step(state, action, reward, next_state, done)
            
            # Cập nhật state và score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Lưu score
        scores.append(score)
        
        # Tính điểm trung bình
        if len(scores) < 100:
            avg_score = np.mean(scores)
        else:
            avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Ghi thời gian
        elapsed_time = time.time() - start_time
        timestamps.append(elapsed_time)
        
        # In thông tin
        logger.info(f"Episode {i_episode}/{num_episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Time: {format_time(elapsed_time)}")
        
        # Lưu model khi đạt được điểm cao nhất
        if avg_score >= agent.best_score:
            agent.best_score = avg_score
            agent.best_model_path = agent.save(episode=i_episode)
    
    # Vẽ đồ thị học tập
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(range(1, num_episodes+1), scores, alpha=0.6, label='Score')
    plt.plot(range(1, num_episodes+1), avg_scores, label='Avg Score (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Actor-Critic Learning Curve')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(agent.actor_loss_history, alpha=0.7, label='Actor Loss')
    plt.plot(agent.critic_loss_history, alpha=0.7, label='Critic Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Actor-Critic Loss')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(agent.entropy_history, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    
    plt.tight_layout()
    plt.savefig('actor_critic_learning_curve.png')
    plt.show()
    
    return agent, scores, avg_scores

def compare_methods(env, num_episodes=10, max_steps=500):
    """
    So sánh hiệu suất của DQN, Rainbow DQN và Actor-Critic.
    
    Args:
        env: Môi trường học tập
        num_episodes: Số episode thử nghiệm
        max_steps: Số bước tối đa mỗi episode
    """
    # Lấy kích thước không gian trạng thái và hành động
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Khởi tạo các agent
    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_layers=[256, 128],
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=50000
    )
    
    rainbow_agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        n_step=3,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_layers=[512, 256],
        learning_rate=5e-4,
        batch_size=128,
        buffer_size=100000
    )
    
    actor_critic_agent = ActorCriticAgent(
        state_size=state_size,
        action_size=action_size,
        use_n_step=True,
        n_step=3,
        hidden_layers=[512, 256],
        actor_lr=1e-4,
        critic_lr=5e-4,
        batch_size=128,
        buffer_size=100000,
        entropy_coef=0.01,
        use_noisy_nets=True
    )
    
    # Danh sách các agent và tên
    agents = [dqn_agent, rainbow_agent, actor_critic_agent]
    agent_names = ['DQN', 'Rainbow DQN', 'Actor-Critic']
    
    # Lưu trữ kết quả
    all_scores = []
    all_avg_scores = []
    
    # Huấn luyện và đánh giá từng agent
    for agent, name in zip(agents, agent_names):
        logger.info(f"=== Huấn luyện và đánh giá {name} ===")
        scores = []
        avg_scores = []
        
        for i_episode in range(1, num_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_steps):
                # Chọn hành động
                if name == 'Actor-Critic':
                    action = agent.act(state, deterministic=False)
                else:
                    action = agent.act(state)
                
                # Thực hiện hành động
                next_state, reward, done, _ = env.step(action)
                
                # Học từ kinh nghiệm
                agent.step(state, action, reward, next_state, done)
                
                # Cập nhật state và score
                state = next_state
                score += reward
                
                if done:
                    break
            
            # Lưu score
            scores.append(score)
            
            # Tính điểm trung bình
            if len(scores) < 100:
                avg_score = np.mean(scores)
            else:
                avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            # In thông tin
            logger.info(f"{name} - Episode {i_episode}/{num_episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f}")
        
        all_scores.append(scores)
        all_avg_scores.append(avg_scores)
    
    # Vẽ đồ thị so sánh
    plt.figure(figsize=(12, 8))
    
    # So sánh điểm trung bình
    plt.subplot(2, 1, 1)
    for i, name in enumerate(agent_names):
        plt.plot(range(1, num_episodes+1), all_scores[i], alpha=0.3, label=f'{name} Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Raw Scores Comparison')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i, name in enumerate(agent_names):
        plt.plot(range(1, num_episodes+1), all_avg_scores[i], linewidth=2, label=f'{name} Avg Score')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Scores Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rl_methods_comparison.png')
    plt.show()
    
    return agents, all_scores, all_avg_scores

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test Rainbow DQN and Actor-Critic')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--method', type=str, default='all', help='Method to test (rainbow, actor_critic, all, compare)')
    args = parser.parse_args()
    
    # Import gym only if needed
    try:
        import gym
        env = gym.make(args.env)
    except ImportError:
        logger.error("OpenAI Gym not found. Please install with: pip install gym")
        exit(1)
    except gym.error.Error:
        logger.error(f"Environment {args.env} not found. Try a standard environment like CartPole-v1.")
        exit(1)
    
    # Thực hiện thử nghiệm dựa trên phương pháp được chọn
    if args.method == 'rainbow' or args.method == 'all':
        rainbow_agent, rainbow_scores, rainbow_avg_scores = test_rainbow_dqn(
            env, num_episodes=args.episodes, max_steps=args.steps, render=args.render
        )
    
    if args.method == 'actor_critic' or args.method == 'all':
        actor_critic_agent, ac_scores, ac_avg_scores = test_actor_critic(
            env, num_episodes=args.episodes, max_steps=args.steps, render=args.render
        )
    
    if args.method == 'compare':
        agents, all_scores, all_avg_scores = compare_methods(
            env, num_episodes=args.episodes, max_steps=args.steps
        )
    
    # Đóng môi trường
    env.close() 