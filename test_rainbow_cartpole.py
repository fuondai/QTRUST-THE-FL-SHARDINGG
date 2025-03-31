import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from qtrust.agents.dqn.rainbow_agent import RainbowDQNAgent
import time

def test_rainbow_dqn(env_name='CartPole-v1', num_episodes=10, max_steps=500, render=False):
    """
    Thử nghiệm Rainbow DQN trong môi trường cho trước.
    
    Args:
        env_name: Tên môi trường gym
        num_episodes: Số episode thử nghiệm
        max_steps: Số bước tối đa mỗi episode
        render: Có hiển thị môi trường hay không
    """
    env = gym.make(env_name)
    
    # Lấy kích thước không gian trạng thái và hành động
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Môi trường: {env_name}")
    print(f"Kích thước state: {state_size}")
    print(f"Kích thước action: {action_size}")
    
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
        batch_size=32,
        buffer_size=10000
    )
    
    # Giám sát hiệu suất
    scores = []
    avg_scores = []
    start_time = time.time()
    
    # Huấn luyện agent qua các episode
    for i_episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_steps):
            # Hiển thị môi trường nếu yêu cầu
            if render:
                env.render()
                
            # Chọn hành động
            action = agent.act(state)
            
            # Thực hiện hành động
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Học từ kinh nghiệm
            agent.step(state, action, reward, next_state, done)
            
            # Cập nhật state và score
            state = next_state
            score += reward
            
            if done or truncated:
                break
        
        # Lưu score
        scores.append(score)
        
        # Tính điểm trung bình
        avg_score = np.mean(scores)
        avg_scores.append(avg_score)
        
        # In thông tin
        elapsed_time = time.time() - start_time
        print(f"Episode {i_episode}/{num_episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Time: {elapsed_time:.2f}s")
    
    # Vẽ đồ thị học tập
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_episodes+1), scores, alpha=0.6, label='Score')
    plt.plot(range(1, num_episodes+1), avg_scores, label='Avg Score')
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
    plt.savefig('rainbow_dqn_cartpole_learning_curve.png')
    plt.show()
    
    env.close()
    return agent, scores, avg_scores

if __name__ == "__main__":
    test_rainbow_dqn(num_episodes=20) 