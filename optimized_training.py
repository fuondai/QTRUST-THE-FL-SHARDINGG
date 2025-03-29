#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust - Ứng dụng blockchain sharding tối ưu với Deep Reinforcement Learning
Tệp này là phiên bản tối ưu cho việc đào tạo DQN Agent.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
import random
from pathlib import Path
import json

# Thêm thư mục hiện tại vào PYTHONPATH để đảm bảo các module có thể được import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.train import train_dqn, evaluate_dqn, plot_dqn_rewards
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.trust.htdcm import HTDCM
from qtrust.federated.federated_learning import FederatedLearning, FederatedModel, FederatedClient
from qtrust.utils.metrics import (
    calculate_throughput, 
    calculate_latency_metrics,
    calculate_energy_efficiency,
    calculate_security_metrics,
    calculate_cross_shard_transaction_ratio,
    plot_performance_metrics,
    plot_comparison_charts
)
from qtrust.utils.data_generation import (
    generate_network_topology,
    assign_nodes_to_shards,
    generate_transactions
)

# Thiết lập ngẫu nhiên cho khả năng tái tạo
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Phân tích tham số dòng lệnh với các giá trị tối ưu cho đào tạo."""
    parser = argparse.ArgumentParser(description='QTrust - Đào tạo tối ưu cho DQN Agent')
    
    parser.add_argument('--num-shards', type=int, default=12, 
                        help='Số lượng shard trong mạng (mặc định: 12)')
    parser.add_argument('--nodes-per-shard', type=int, default=16, 
                        help='Số lượng node trong mỗi shard (mặc định: 16)')
    parser.add_argument('--episodes', type=int, default=2000, 
                        help='Số lượng episode để huấn luyện (mặc định: 2000)')
    parser.add_argument('--max-steps', type=int, default=1000, 
                        help='Số bước tối đa trong mỗi episode (mặc định: 1000)')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Kích thước batch cho mô hình DQN (mặc định: 128)')
    parser.add_argument('--hidden-size', type=int, default=256, 
                        help='Kích thước lớp ẩn của mô hình DQN (mặc định: 256)')
    parser.add_argument('--lr', type=float, default=0.0005, 
                        help='Tốc độ học (mặc định: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Hệ số chiết khấu (mặc định: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, 
                        help='Giá trị epsilon khởi đầu (mặc định: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01, 
                        help='Giá trị epsilon cuối cùng (mặc định: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.998, 
                        help='Tốc độ giảm epsilon (mặc định: 0.998)')
    parser.add_argument('--memory-size', type=int, default=100000, 
                        help='Kích thước bộ nhớ replay (mặc định: 100000)')
    parser.add_argument('--target-update', type=int, default=10, 
                        help='Cập nhật mô hình target sau mỗi bao nhiêu episode (mặc định: 10)')
    parser.add_argument('--save-dir', type=str, default='models/optimized_dqn', 
                        help='Thư mục lưu mô hình')
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='Số episode giữa các lần in kết quả')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Số episode giữa các lần đánh giá')
    parser.add_argument('--patience', type=int, default=100,
                        help='Số episode không cải thiện để dừng sớm')
    parser.add_argument('--enable-federated', action='store_true', 
                        help='Bật chế độ Federated Learning')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Thiết bị để huấn luyện (cuda hoặc cpu)')
    parser.add_argument('--attack-scenario', type=str, 
                        choices=['51_percent', 'sybil', 'eclipse', 'selfish_mining', 'bribery', 'ddos', 'finney', 'mixed', 'none'], 
                        default='none', help='Kịch bản tấn công')
    
    return parser.parse_args()

def setup_environment(args):
    """Thiết lập môi trường blockchain."""
    print("Khởi tạo môi trường blockchain...")
    
    env = BlockchainEnvironment(
        num_shards=args.num_shards,
        num_nodes_per_shard=args.nodes_per_shard,
        max_steps=args.max_steps,
        latency_penalty=0.5,
        energy_penalty=0.3,
        throughput_reward=1.0,
        security_reward=0.8
    )
    
    return env

def setup_dqn_agent(env, args):
    """Thiết lập DQN Agent với cấu hình tối ưu."""
    print("Khởi tạo DQN Agent với cấu hình tối ưu...")
    
    # Lấy kích thước state từ môi trường
    state = env.reset()
    state_size = len(state)
    print(f"Kích thước thực tế của state: {state_size}")
    
    # Tính tổng số hành động có thể
    total_actions = env.num_shards * 3  # num_shards * num_consensus_protocols
    
    # Tạo agent với cấu hình tối ưu
    agent = DQNAgent(
        state_size=state_size,
        action_size=total_actions,
        seed=SEED,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=0.001,  # Hệ số cập nhật mạng target
        hidden_layers=[args.hidden_size, args.hidden_size//2],
        update_every=4,  # Cập nhật mạng sau mỗi 4 bước
        device=args.device,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.epsilon_end,
        buffer_size=args.memory_size,
        prioritized_replay=True,  # Sử dụng Prioritized Experience Replay 
        alpha=0.6,  # Alpha for prioritized replay
        beta_start=0.4,   # Beta for prioritized replay
        dueling=True,  # Sử dụng Dueling DQN cho hiệu quả học tập tốt hơn
        clip_gradients=True,  # Giới hạn gradient để ổn định quá trình học
        grad_clip_value=1.0  # Giá trị giới hạn gradient
    )
    
    print(f"Đã tạo DQN Agent với {state_size} trạng thái và {total_actions} hành động.")
    print(f"Sử dụng Prioritized Replay: {agent.prioritized_replay}")
    
    # Tạo wrapper cho agent để chuyển đổi từ hành động đơn lẻ sang hành động MultiDiscrete
    class DQNAgentWrapper:
        def __init__(self, agent, num_shards, num_consensus_protocols=3):
            self.agent = agent
            self.num_shards = num_shards
            self.num_consensus_protocols = num_consensus_protocols
            
        def act(self, state, eps=None):
            # Gọi hành động từ agent cơ bản
            action_idx = self.agent.act(state, eps)
            
            # Chuyển đổi action_idx thành hành động MultiDiscrete [shard_idx, consensus_idx]
            shard_idx = action_idx % self.num_shards
            consensus_idx = (action_idx // self.num_shards) % self.num_consensus_protocols
            
            return np.array([shard_idx, consensus_idx], dtype=np.int32)
            
        def step(self, state, action, reward, next_state, done):
            # Chuyển đổi hành động MultiDiscrete thành hành động đơn lẻ
            # Đảm bảo action là mảng
            if isinstance(action, np.ndarray) and len(action) >= 2:
                action_idx = action[0] + action[1] * self.num_shards
            else:
                # Nếu action là số nguyên, xử lý trực tiếp
                action_idx = action
            
            # Gọi step của agent cơ bản
            self.agent.step(state, action_idx, reward, next_state, done)
            
        def save(self, path):
            return self.agent.save(path)
            
        def load(self, path):
            return self.agent.load(path)
            
        # Thuộc tính chuyển tiếp
        @property
        def epsilon(self):
            return self.agent.epsilon
        
        @property
        def device(self):
            return self.agent.device
            
    # Bọc agent trong wrapper
    wrapped_agent = DQNAgentWrapper(agent, args.num_shards)
    
    return wrapped_agent, agent

# Hàm chính sửa đổi để tối ưu hoá quá trình đào tạo
def train_optimized_dqn(env, agent, base_agent, args):
    """
    Huấn luyện DQN Agent với tối ưu hóa và giám sát hiệu suất
    """
    print(f"Bắt đầu huấn luyện DQN Agent trong {args.episodes} episodes...")
    
    # Tạo thư mục lưu model nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu giám sát
    all_rewards = []
    episode_lengths = []
    val_rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    best_val_reward = -float('inf')
    episodes_without_improvement = 0
    
    # Thiết lập tqdm progress bar
    progress_bar = tqdm(range(args.episodes), desc="Training")
    
    # Thời gian bắt đầu đào tạo
    start_time = time.time()
    
    # Vòng lặp đào tạo chính
    for episode in progress_bar:
        # Reset môi trường
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Chạy một episode
        while not done and episode_steps < args.max_steps:
            # Chọn hành động với epsilon-greedy
            action = agent.act(state, base_agent.epsilon)
            
            # Thực hiện hành động trong môi trường
            next_state, reward, done, _ = env.step(action)
            
            # Cập nhật agent
            agent.step(state, action, reward, next_state, done)
            
            # Cập nhật trạng thái và thống kê
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # Cập nhật epsilon theo lịch trình
        base_agent.update_epsilon("exponential")
        
        # Lưu thông tin episode
        all_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Tính trung bình di động của phần thưởng
        window_size = min(50, len(all_rewards))
        avg_reward = np.mean(all_rewards[-window_size:])
        avg_rewards.append(avg_reward)
        
        # Cập nhật trạng thái tiến trình
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.2f}", 
            'avg_reward': f"{avg_reward:.2f}", 
            'epsilon': f"{base_agent.epsilon:.4f}"
        })
        
        # Đánh giá agent định kỳ
        if (episode + 1) % args.eval_interval == 0:
            # Đánh giá trong 10 episode
            val_reward = evaluate_agent_wrapper(agent, env, n_episodes=10, max_t=args.max_steps)
            val_rewards.append(val_reward)
            
            print(f"\nEpisode {episode+1}/{args.episodes} | Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | Val: {val_reward:.2f} | Epsilon: {base_agent.epsilon:.4f}")
            
            # Lưu model tốt nhất dựa trên val_reward
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                model_path = os.path.join(args.save_dir, "best_validation_model.pth")
                base_agent.save(model_path)
                print(f"Đã lưu model tốt nhất tại: {model_path} (Val reward: {val_reward:.2f})")
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += args.eval_interval
            
            # Lưu model dựa trên avg_reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                model_path = os.path.join(args.save_dir, "best_training_model.pth")
                base_agent.save(model_path)
                print(f"Đã lưu model training tốt nhất tại: {model_path} (Avg reward: {avg_reward:.2f})")
        
        # Lưu checkpoint định kỳ
        if (episode + 1) % 100 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_ep{episode+1}.pth")
            base_agent.save(checkpoint_path)
        
        # Early stopping nếu không có cải thiện trong một khoảng thời gian
        if episodes_without_improvement >= args.patience:
            print(f"\nEarly stopping tại episode {episode+1} do không có cải thiện sau {args.patience} episodes")
            break
    
    # Lưu model cuối cùng
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    base_agent.save(final_model_path)
    
    # Tính tổng thời gian đào tạo
    total_training_time = time.time() - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nHoàn thành đào tạo trong {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Lưu thông tin mô hình và kết quả đào tạo
    training_info = {
        "episodes": episode + 1,
        "best_validation_reward": float(best_val_reward),
        "best_avg_reward": float(best_avg_reward),
        "final_avg_reward": float(avg_rewards[-1]),
        "training_time_seconds": total_training_time,
        "hyperparameters": {
            "num_shards": args.num_shards,
            "nodes_per_shard": args.nodes_per_shard,
            "hidden_size": args.hidden_size,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "memory_size": args.memory_size,
            "prioritized_replay": base_agent.prioritized_replay
        }
    }
    
    # Lưu thông tin đào tạo
    with open(os.path.join(args.save_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=4)
    
    # Vẽ và lưu biểu đồ đào tạo
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(all_rewards, alpha=0.3, color='blue')
    plt.plot(avg_rewards, color='blue', linewidth=2)
    if val_rewards:
        plt.plot(np.arange(0, len(all_rewards), args.eval_interval), val_rewards, 'ro-')
    plt.title('Training & Validation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"))
    
    print(f"Biểu đồ đào tạo đã được lưu tại: {os.path.join(args.save_dir, 'training_curves.png')}")
    
    # Trả về model tốt nhất
    if os.path.exists(os.path.join(args.save_dir, "best_validation_model.pth")):
        base_agent.load(os.path.join(args.save_dir, "best_validation_model.pth"))
        print("Đã tải model tốt nhất cho đánh giá.")
    
    return {
        "rewards": all_rewards,
        "val_rewards": val_rewards,
        "best_reward": best_val_reward,
        "avg_rewards": avg_rewards,
        "episode_lengths": episode_lengths
    }

def evaluate_agent_wrapper(agent, env, n_episodes=10, max_t=1000):
    """
    Đánh giá hiệu suất của agent wrapper
    
    Args:
        agent: DQNAgentWrapper - Agent wrapper cần đánh giá
        env: BlockchainEnvironment - Môi trường blockchain
        n_episodes: int - Số lượng episode để đánh giá
        max_t: int - Số bước tối đa trong một episode
        
    Returns:
        float: Phần thưởng trung bình trong tất cả các episode
    """
    rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_t):
            # Lấy hành động từ agent (không sử dụng epsilon trong quá trình đánh giá)
            action = agent.act(state, 0)  # epsilon = 0 cho việc đánh giá
            
            # Thực hiện hành động trong môi trường
            next_state, reward, done, _ = env.step(action)
            
            # Cộng dồn phần thưởng
            episode_reward += reward
            
            # Cập nhật trạng thái
            state = next_state
            
            if done:
                break
                
        rewards.append(episode_reward)
    
    # Trả về phần thưởng trung bình
    return np.mean(rewards)

def main():
    """Hàm chính để chạy quá trình đào tạo tối ưu hóa."""
    # Phân tích tham số dòng lệnh
    args = parse_args()
    
    print("=== Bắt đầu đào tạo model Q-TRUST tối ưu ===")
    print(f"Thiết bị: {args.device}")
    print(f"Số shards: {args.num_shards}")
    print(f"Số nodes/shard: {args.nodes_per_shard}")
    print(f"Số episodes: {args.episodes}")
    print(f"Kích thước lớp ẩn: {args.hidden_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    
    # Thiết lập môi trường blockchain
    env = setup_environment(args)
    
    # Thiết lập DQN agent với cấu hình tối ưu
    agent_wrapper, base_agent = setup_dqn_agent(env, args)
    
    # Xóa tất cả các model cũ trong thư mục đích
    os.makedirs(args.save_dir, exist_ok=True)
    for file in os.listdir(args.save_dir):
        if file.endswith(".pth"):
            os.remove(os.path.join(args.save_dir, file))
    print(f"Đã xóa các model cũ trong thư mục {args.save_dir}")
    
    # Đào tạo DQN Agent
    train_results = train_optimized_dqn(env, agent_wrapper, base_agent, args)
    
    print("=== Đào tạo hoàn tất ===")
    print(f"Model tốt nhất đã được lưu tại: {os.path.join(args.save_dir, 'best_validation_model.pth')}")
    print(f"Điểm đánh giá tốt nhất: {train_results['best_reward']:.2f}")
    
    # Đóng môi trường
    env.close()

if __name__ == "__main__":
    main() 