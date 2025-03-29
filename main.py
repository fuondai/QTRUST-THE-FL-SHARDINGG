#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust - Ứng dụng blockchain sharding tối ưu với Deep Reinforcement Learning

Tệp này là điểm vào chính cho việc chạy các mô phỏng QTrust.
"""

import sys
import os
import locale

# Force UTF-8 encoding for console output
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Try to set locale to UTF-8
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        pass

# Add the current directory to Python path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
import random
from pathlib import Path
from gym import spaces
import json
from datetime import datetime

from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.agents.dqn.agent import DQNAgent
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
from qtrust.federated.manager import FederatedLearningManager

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
    """
    Phân tích tham số dòng lệnh.
    
    Returns:
        argparse.Namespace: Các tham số đã phân tích
    """
    parser = argparse.ArgumentParser(description='Q-TRUST: Blockchain Sharding tối ưu hóa với DRL')
    
    # Tham số cho môi trường blockchain
    parser.add_argument('--num-shards', type=int, default=4, help='Số lượng shard')
    parser.add_argument('--nodes-per-shard', type=int, default=6, help='Số lượng node trong mỗi shard')
    parser.add_argument('--max-steps', type=int, default=100, help='Số bước tối đa trong mỗi episode')
    parser.add_argument('--episodes', type=int, default=1, help='Số lượng episode')
    
    # Tham số cho DQN
    parser.add_argument('--batch-size', type=int, default=64, help='Kích thước batch trong huấn luyện')
    parser.add_argument('--hidden-size', type=int, default=128, help='Số neuron trong lớp ẩn')
    parser.add_argument('--memory-size', type=int, default=10000, help='Kích thước replay memory')
    parser.add_argument('--target-update', type=int, default=10, help='Số episode cập nhật target network')
    parser.add_argument('--gamma', type=float, default=0.99, help='Hệ số giảm giá (discount factor)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Epsilon khởi đầu cho ε-greedy')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Epsilon kết thúc cho ε-greedy')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Hệ số giảm epsilon')
    parser.add_argument('--lr', type=float, default=0.001, help='Tốc độ học')
    
    # Tham số chế độ đánh giá
    parser.add_argument('--eval', action='store_true', help='Chế độ đánh giá')
    parser.add_argument('--model-path', type=str, help='Đường dẫn tới mô hình đã huấn luyện')
    parser.add_argument('--attack-scenario', type=str, default='none', 
                      choices=['none', 'ddos', '51_percent', 'sybil', 'eclipse'], 
                      help='Kịch bản tấn công cho mô phỏng')
    
    # Lưu trữ
    parser.add_argument('--save-dir', type=str, default='models/simulation', help='Thư mục lưu kết quả')
    parser.add_argument('--load-model', type=str, help='Đường dẫn tới mô hình để tiếp tục huấn luyện')
    parser.add_argument('--eval-interval', type=int, default=10, help='Tần suất đánh giá (số episode)')
    
    args = parser.parse_args()
    
    # Tạo các thư mục nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)
    
    return args

def setup_simulation(args):
    """Thiết lập môi trường mô phỏng."""
    print("Khởi tạo môi trường mô phỏng...")
    
    # Khởi tạo môi trường blockchain
    env = BlockchainEnvironment(
        num_shards=args.num_shards,
        num_nodes_per_shard=args.nodes_per_shard,
        max_steps=args.max_steps
    )
    
    # Cài đặt router
    router = MADRAPIDRouter(
        network=env.network,
        shards=env.shards,
        congestion_weight=0.4,
        latency_weight=0.3,
        energy_weight=0.2,
        trust_weight=0.1
    )
    
    # Khởi tạo HTDCM
    htdcm = HTDCM(num_nodes=args.num_shards * args.nodes_per_shard)
    
    # Thiết lập quản lý đồng thuận thích ứng
    ac_manager = AdaptiveConsensus(
        transaction_threshold_low=10.0,
        transaction_threshold_high=50.0,
        congestion_threshold=0.7,
        min_trust_threshold=0.3
    )
    
    return env, router, htdcm, ac_manager

def setup_dqn_agent(env, args):
    """Thiết lập DQN Agent."""
    print("Khởi tạo DQN Agent...")
    
    # In kích thước không gian trạng thái của môi trường để debug
    print(f"Kích thước observation_space: {env.observation_space.shape}")
    print(f"Số lượng shard: {env.num_shards}")
    
    # Lấy kích thước state từ môi trường
    # Cần đảm bảo chính xác kích thước đầu vào
    state = env.reset()
    state_size = len(state)
    print(f"Kích thước thực tế của state: {state_size}")
    
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
            if isinstance(action, np.ndarray) and len(action) >= 2:
                action_idx = action[0] + action[1] * self.num_shards
            else:
                # Xử lý trường hợp action là số nguyên
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
        
        # Thêm các thuộc tính khác nếu cần
        @property
        def device(self):
            return self.agent.device
            
    # Tính tổng số hành động có thể
    total_actions = env.num_shards * 3  # num_shards * num_consensus_protocols
    
    # Khởi tạo agent cơ bản
    base_agent = DQNAgent(
        state_size=state_size,
        action_size=total_actions,  # Tổng số hành động có thể
        seed=SEED,
        buffer_size=args.memory_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.lr,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.epsilon_end,
        hidden_layers=[args.hidden_size, args.hidden_size//2],
        device=args.device,
        prioritized_replay=True,
        dueling=True,
        update_every=5
    )
    
    # Tạo wrapper
    agent = DQNAgentWrapper(base_agent, env.num_shards)
    
    # Nếu đường dẫn mô hình được cung cấp, tải mô hình đã huấn luyện
    if args.model_path and os.path.exists(args.model_path):
        print(f"Đang tải mô hình đã huấn luyện từ: {args.model_path}")
        agent.load(args.model_path)
        print("Đã tải mô hình DQN thành công")
        
        # Nếu đang đánh giá, đặt epsilon thành 0 để không có khám phá ngẫu nhiên
        if args.eval:
            base_agent.epsilon = 0.0
            print("Đã đặt epsilon = 0 cho chế độ đánh giá (không khám phá ngẫu nhiên)")
    else:
        if args.model_path:
            print(f"Cảnh báo: Không tìm thấy mô hình tại {args.model_path}, sẽ sử dụng mô hình mới")
        else:
            print("Khởi tạo mô hình DQN mới")
    
    return agent

def setup_federated_learning(env, args, htdcm):
    """Thiết lập hệ thống Federated Learning."""
    if not args.enable_federated:
        return None
    
    print("Khởi tạo hệ thống Federated Learning...")
    
    # Khởi tạo mô hình toàn cục
    # Lấy kích thước state thực tế thay vì từ observation_space
    state = env.reset()
    input_size = len(state)
    hidden_size = args.hidden_size
    
    # Xử lý trường hợp action_space là MultiDiscrete
    if hasattr(env.action_space, 'n'):
        output_size = env.action_space.n
    else:
        # Đối với MultiDiscrete, lấy tổng số hành động khả dụng
        output_size = env.action_space.nvec.sum()
    
    global_model = FederatedModel(input_size, hidden_size, output_size)
    
    # Khởi tạo hệ thống Federated Learning
    fl_system = FederatedLearning(
        global_model=global_model,
        aggregation_method='fedtrust',
        client_selection_method='trust_based',
        min_clients_per_round=3,
        device=args.device
    )
    
    # Tạo một client cho mỗi shard
    for shard_id in range(env.num_shards):
        # Lấy điểm tin cậy trung bình của shard
        shard_trust = htdcm.shard_trust_scores[shard_id]
        
        # Tạo client mới
        client = FederatedClient(
            client_id=shard_id,
            model=global_model,
            learning_rate=args.lr,
            local_epochs=5,
            batch_size=32,
            trust_score=shard_trust,
            device=args.device
        )
        
        # Thêm client vào hệ thống
        fl_system.add_client(client)
    
    return fl_system

def train_qtrust(env, agent, router, consensus, htdcm, fl_system, args):
    """Huấn luyện hệ thống QTrust."""
    print("Bắt đầu huấn luyện hệ thống QTrust...")
    
    # Tạo thư mục lưu mô hình nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Theo dõi hiệu suất
    episode_rewards = []
    avg_rewards = []
    transaction_throughputs = []
    latencies = []
    malicious_detections = []
    
    # Huấn luyện qua nhiều episode
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Lưu trữ dữ liệu cho federated learning
        shard_experiences = [[] for _ in range(env.num_shards)]
        
        # Chạy một episode
        while not done and steps < args.max_steps:
            # Chọn hành động từ DQN Agent
            action = agent.act(state)
            
            # Mô phỏng các hoạt động trong mạng blockchain
            
            # 1. Định tuyến các giao dịch
            transaction_routes = router.find_optimal_paths_for_transactions(env.transaction_pool)
            
            # 2. Thực hiện giao dịch với các giao thức đồng thuận thích ứng
            for tx in env.transaction_pool:
                # Chọn giao thức đồng thuận dựa trên giá trị giao dịch và tình trạng mạng
                trust_scores_dict = htdcm.get_node_trust_scores()
                network_congestion = env.get_congestion_level()
                protocol = consensus.select_protocol(
                    transaction_value=tx['value'],
                    congestion=network_congestion,
                    trust_scores=trust_scores_dict
                )
                
                # Thực hiện giao thức đồng thuận
                result, latency, energy = consensus.execute_consensus(
                    tx, env.network, protocol
                )
                
                # Ghi nhận kết quả vào hệ thống tin cậy
                for node_id in tx.get('validator_nodes', []):
                    htdcm.update_node_trust(
                        node_id=node_id,
                        tx_success=result,
                        response_time=latency,
                        is_validator=True
                    )
            
            # 3. Thực hiện bước trong môi trường với hành động đã chọn
            next_state, reward, done, info = env.step(action)
            
            # 4. Cập nhật Agent
            agent.step(state, action, reward, next_state, done)
            
            # Lưu trữ trải nghiệm cho từng shard
            for shard_id in range(env.num_shards):
                # Lấy các giao dịch liên quan đến shard này
                shard_txs = [tx for tx in env.transaction_pool 
                            if tx.get('shard_id') == shard_id or tx.get('destination_shard') == shard_id]
                
                if shard_txs:
                    # Lưu trữ trải nghiệm (state, action, reward, next_state)
                    shard_experience = (state, action, reward, next_state, done)
                    shard_experiences[shard_id].append(shard_experience)
            
            # Cập nhật cho bước tiếp theo
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Cập nhật trạng thái mạng cho router
            router.update_network_state(
                shard_congestion=env.get_congestion_data(),
                node_trust_scores=htdcm.get_node_trust_scores()
            )
        
        # Thực hiện Federated Learning nếu được bật
        if fl_system is not None and episode % 5 == 0:
            # Chuẩn bị dữ liệu huấn luyện cho mỗi client
            for shard_id in range(env.num_shards):
                if len(shard_experiences[shard_id]) > 0:
                    # Chuyển đổi các trải nghiệm thành dữ liệu huấn luyện
                    # Với wrapper DQNAgent, action là 2 phần tử
                    states = torch.FloatTensor([exp[0] for exp in shard_experiences[shard_id]])
                    
                    # Flattened actions cho dễ đào tạo
                    # Chuyển từ [shard_idx, consensus_idx] thành một giá trị đơn lẻ
                    if isinstance(shard_experiences[shard_id][0][1], np.ndarray):
                        actions = torch.FloatTensor([
                            exp[1][0] + exp[1][1] * env.num_shards 
                            for exp in shard_experiences[shard_id]
                        ]).unsqueeze(1)  # [batch, 1]
                    else:
                        actions = torch.FloatTensor([[exp[1]] for exp in shard_experiences[shard_id]])
                    
                    # Thiết lập dữ liệu cục bộ cho client
                    fl_system.clients[shard_id].set_local_data(
                        train_data=(states, actions),
                        val_data=None
                    )
                    
                    # Cập nhật điểm tin cậy cho client dựa trên điểm tin cậy của shard
                    fl_system.update_client_trust(
                        client_id=shard_id,
                        trust_score=htdcm.shard_trust_scores[shard_id]
                    )
            
            # Thực hiện một vòng huấn luyện Federated Learning
            round_metrics = fl_system.train_round(
                round_num=episode // 5,
                client_fraction=0.8
            )
            
            if round_metrics:
                print(f"Vòng Federated {episode//5}: "
                     f"Mất mát = {round_metrics['avg_train_loss']:.4f}")
        
        # Theo dõi hiệu suất
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        
        # Lưu các metrics
        transaction_throughputs.append(info.get('successful_transactions', 0))
        episode_latencies = [info.get('latency', 0) for tx in env.transaction_pool if tx['status'] == 'completed']
        if episode_latencies:
            latencies.append(np.mean(episode_latencies))
        else:
            latencies.append(0)
        malicious_nodes = htdcm.identify_malicious_nodes()
        malicious_detections.append(len(malicious_nodes))
        
        # In thông tin huấn luyện
        if episode % args.log_interval == 0:
            print(f"Episode {episode}/{args.episodes} - "
                 f"Reward: {episode_reward:.2f}, "
                 f"Avg Reward: {avg_reward:.2f}, "
                 f"Epsilon: {agent.epsilon:.4f}, "
                 f"Throughput: {transaction_throughputs[-1]}, "
                 f"Latency: {latencies[-1]:.2f}ms, "
                 f"Malicious Nodes: {len(malicious_nodes)}")
        
        # Lưu mô hình
        if episode % 100 == 0 or episode == args.episodes - 1:
            model_path = os.path.join(args.save_dir, f"dqn_model_ep{episode}.pth")
            agent.save(model_path)
            print(f"Đã lưu mô hình tại: {model_path}")
    
    # Lưu mô hình cuối cùng
    final_model_path = os.path.join(args.save_dir, "dqn_model_final.pth")
    agent.save(final_model_path)
    print(f"Đã lưu mô hình cuối cùng tại: {final_model_path}")
    
    # Lưu mô hình Federated Learning (nếu có)
    if fl_system is not None:
        fl_model_path = os.path.join(args.save_dir, "federated_model_final.pth")
        fl_system.save_global_model(fl_model_path)
        print(f"Đã lưu mô hình Federated Learning tại: {fl_model_path}")
    
    # Trả về dữ liệu hiệu suất
    return {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'transaction_throughputs': transaction_throughputs,
        'latencies': latencies,
        'malicious_detections': malicious_detections
    }

def evaluate_qtrust(env, agent, router, consensus, htdcm, fl_system, args):
    """Đánh giá hệ thống QTrust với mô hình đã huấn luyện."""
    num_episodes = 5  # Số lượng episode đánh giá cố định
    
    print(f"Bắt đầu đánh giá hệ thống QTrust...")
    
    rewards = []
    throughputs = []
    latencies = []
    energy_consumptions = []
    security_levels = []
    cross_shard_ratios = []
    
    # Tạo thanh tiến trình
    progress_bar = tqdm(total=num_episodes, desc="Đánh giá Tiến trình")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_latencies = []
        episode_energy = 0
        total_successful_txs = 0
        episode_cross_shard_txs = 0
        total_txs = 0
        
        # Khởi tạo lại các tham số cho episode mới
        htdcm.reset()
        if fl_system:
            fl_system.reset()
            
        # Khởi tạo metrics cho episode
        env.metrics = {
            'latency': [],
            'energy_consumption': [],
            'security_score': []
        }
        
        while not done:
            # Lấy trạng thái mạng hiện tại - sử dụng get_state() thay vì get_network_state()
            network_state = env.get_state()
            shard_congestion = env.shard_congestion
            node_trust_scores = htdcm.get_node_trust_scores()
            
            # Cập nhật router với thông tin mới
            router.update_network_state(shard_congestion, node_trust_scores)
            
            # Sử dụng phương thức select_protocol mà không cần vòng lặp for
            average_congestion = np.mean(shard_congestion) if isinstance(shard_congestion, np.ndarray) else 0.5
            tx_value = random.uniform(0, 100)  # Giả lập giá trị giao dịch
            protocol = consensus.select_protocol(
                transaction_value=tx_value,
                congestion=average_congestion,
                trust_scores=node_trust_scores
            )
            
            # Chọn hành động: sử dụng agent nếu có, ngược lại chọn ngẫu nhiên
            if isinstance(state, np.ndarray) and len(state) != agent.state_size:
                # Xử lý trường hợp kích thước state khác với mô hình
                # Trích xuất các thông tin cần thiết và điều chỉnh kích thước
                adapted_state = adapt_state_for_model(state, agent.state_size)
                action = agent.act(adapted_state)
            else:
                action = agent.act(state)
            
            # Xử lý định tuyến giao dịch
            txs_to_route = env.get_pending_transactions()
            for tx in txs_to_route:
                # Định tuyến giao dịch sử dụng router
                source_shard = env.get_node_shard(tx['source'])
                target_shard = env.get_node_shard(tx['target'])
                
                # Sử dụng router để tìm đường đi tốt nhất
                if source_shard != target_shard:
                    episode_cross_shard_txs += 1
                    path = router.find_path(source_shard, target_shard, tx)
                    tx['route'] = path
            
            # Cập nhật điểm tin cậy
            tx_results = env.get_recent_transactions()
            for tx_result in tx_results:
                if tx_result['status'] == 'successful':
                    # Tăng điểm tin cậy cho các node tham gia
                    participating_nodes = tx_result.get('participating_nodes', [])
                    for node in participating_nodes:
                        htdcm.update_node_trust(node, 0.01, 0, True)  # Tăng nhẹ điểm tin cậy
                else:
                    # Giảm điểm tin cậy cho các node gây lỗi
                    failing_nodes = tx_result.get('failing_nodes', [])
                    for node in failing_nodes:
                        htdcm.update_node_trust(node, -0.05, 0, True)  # Giảm đáng kể điểm tin cậy
            
            # Thực hiện bước mô phỏng với hành động đã chọn
            next_state, reward, done, info = env.step(action)
            
            # Cập nhật thông tin
            state = next_state
            episode_reward += reward
            
            # Thu thập metrics
            if 'successful_txs' in info:
                total_successful_txs += info['successful_txs']
            if 'total_txs' in info:
                total_txs += info['total_txs']
            if 'latency' in info:
                episode_latencies.append(info['latency'])
            if 'energy' in info:
                episode_energy += info['energy']
        
        # Tính toán metrics
        throughput = total_successful_txs / args.max_steps
        cross_shard_ratio = (episode_cross_shard_txs / max(1, total_txs)) * 100 if total_txs > 0 else 0
        
        # Lưu metrics
        rewards.append(episode_reward)
        throughputs.append(throughput)
        
        if env.metrics['latency']:
            avg_latency = np.mean(env.metrics['latency'])
            latencies.append(avg_latency)
        else:
            latencies.append(np.mean(episode_latencies) if episode_latencies else 0)
        
        if env.metrics['energy_consumption']:
            avg_energy = np.mean(env.metrics['energy_consumption'])
            energy_consumptions.append(avg_energy)
        else:
            energy_consumptions.append(episode_energy / max(1, total_successful_txs))
        
        if env.metrics['security_score']:
            avg_security = np.mean(env.metrics['security_score'])
            security_levels.append(avg_security)
        else:
            # Tính mức độ bảo mật dựa trên điểm tin cậy trung bình của mạng
            trust_scores = list(htdcm.get_node_trust_scores().values())
            security_level = np.mean(trust_scores) if trust_scores else 0
            security_levels.append(security_level)
        
        cross_shard_ratios.append(cross_shard_ratio)
        
        print(f"Đánh giá Episode {episode+1}/{num_episodes} - "
              f"Reward: {episode_reward:.2f}, "
              f"Throughput: {throughput:.2f} tx/s, "
              f"Latency: {avg_latency:.2f} ms, "
              f"Energy: {energy_consumptions[-1]:.2f} mJ/tx, "
              f"Security: {security_levels[-1]:.4f}, "
              f"Cross-Shard Ratio: {cross_shard_ratio:.2f}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Tổng hợp kết quả
    metrics = {
        'rewards': rewards,
        'throughputs': throughputs,
        'latencies': latencies,
        'energy_consumptions': energy_consumptions,
        'security_levels': security_levels,
        'cross_shard_ratios': cross_shard_ratios
    }
    
    return metrics

# Hàm hỗ trợ điều chỉnh kích thước state
def adapt_state_for_model(state, model_state_size):
    """Điều chỉnh kích thước state để phù hợp với model."""
    if len(state) > model_state_size:
        # Nếu state lớn hơn, lấy phần đầu tiên phù hợp với model
        return state[:model_state_size]
    elif len(state) < model_state_size:
        # Nếu state nhỏ hơn, mở rộng với các giá trị 0
        padded_state = np.zeros(model_state_size)
        padded_state[:len(state)] = state
        return padded_state
    else:
        return state

def plot_results(metrics, args, mode='train'):
    """
    Vẽ biểu đồ kết quả huấn luyện hoặc đánh giá.
    
    Args:
        metrics: Dictionary chứa các metrics
        args: Tham số dòng lệnh
        mode: 'train' hoặc 'eval'
    """
    print(f"Vẽ đồ thị kết quả {'huấn luyện' if mode == 'train' else 'đánh giá'}...")
    
    # Tạo thư mục cho các biểu đồ
    plots_dir = os.path.join(args.save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Đảm bảo sử dụng font tiếng Việt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Arial']
    
    # 1. Biểu đồ Reward theo thời gian
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['rewards'], label='Episode Reward', alpha=0.4, color='#1f77b4')
    
    # Đường trung bình động
    window_size = max(1, len(metrics['rewards']) // 5)
    if len(metrics['rewards']) > 1:
        smoothed_rewards = np.convolve(metrics['rewards'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(metrics['rewards'])), smoothed_rewards, 
                color='blue', label=f'Moving Avg (window={window_size})')
    
    plt.title('Reward theo Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{mode}_rewards.png'), dpi=300, bbox_inches='tight')
    
    if mode == 'train':
        # 2. Biểu đồ kết hợp Throughput, Latency và Energy Consumption
        plt.figure(figsize=(12, 8))
        
        if 'throughputs' in metrics and len(metrics['throughputs']) > 0:
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(metrics['throughputs'], color='green', alpha=0.6)
            ax1.set_title('Throughput (tx/s)')
            ax1.grid(alpha=0.3)
            
            if len(metrics['throughputs']) > 1:
                smoothed = np.convolve(metrics['throughputs'], np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(metrics['throughputs'])), smoothed, color='darkgreen')
        
        if 'latencies' in metrics and len(metrics['latencies']) > 0:
            ax2 = plt.subplot(3, 1, 2)
            ax2.plot(metrics['latencies'], color='red', alpha=0.6)
            ax2.set_title('Latency (ms)')
            ax2.grid(alpha=0.3)
            
            if len(metrics['latencies']) > 1:
                smoothed = np.convolve(metrics['latencies'], np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(metrics['latencies'])), smoothed, color='darkred')
        
        if 'energy_consumptions' in metrics and len(metrics['energy_consumptions']) > 0:
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(metrics['energy_consumptions'], color='orange', alpha=0.6)
            ax3.set_title('Energy Consumption (mJ/tx)')
            ax3.grid(alpha=0.3)
            
            if len(metrics['energy_consumptions']) > 1:
                smoothed = np.convolve(metrics['energy_consumptions'], np.ones(window_size)/window_size, mode='valid')
                ax3.plot(range(window_size-1, len(metrics['energy_consumptions'])), smoothed, color='darkorange')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{mode}_performance.png'), dpi=300, bbox_inches='tight')
        
        # 3. Biểu đồ Security và Cross-shard ratio
        plt.figure(figsize=(10, 8))
        
        if 'security_levels' in metrics and len(metrics['security_levels']) > 0:
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(metrics['security_levels'], color='purple', alpha=0.6)
            ax1.set_title('Security Score')
            ax1.set_ylim([0, 1])
            ax1.grid(alpha=0.3)
            
            if len(metrics['security_levels']) > 1:
                smoothed = np.convolve(metrics['security_levels'], np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(metrics['security_levels'])), smoothed, color='purple')
        
        if 'cross_shard_ratios' in metrics and len(metrics['cross_shard_ratios']) > 0:
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(metrics['cross_shard_ratios'], color='blue', alpha=0.6)
            ax2.set_title('Cross-shard Transaction Ratio')
            ax2.set_ylim([0, 1])
            ax2.grid(alpha=0.3)
            
            if len(metrics['cross_shard_ratios']) > 1:
                smoothed = np.convolve(metrics['cross_shard_ratios'], np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(metrics['cross_shard_ratios'])), smoothed, color='blue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{mode}_security.png'), dpi=300, bbox_inches='tight')
    
    # Đóng tất cả các biểu đồ để giải phóng bộ nhớ
    plt.close('all')
    
    print(f"Đã lưu biểu đồ kết quả vào {plots_dir}")

def create_blockchain_network(num_shards, nodes_per_shard):
    """
    Khởi tạo mạng blockchain với số lượng shard và node cụ thể.
    
    Args:
        num_shards: Số lượng shard
        nodes_per_shard: Số lượng node trong mỗi shard
        
    Returns:
        tuple: (đồ thị mạng, ánh xạ node tới shard, danh sách các shard)
    """
    # Khởi tạo đồ thị
    G = nx.Graph()
    total_nodes = num_shards * nodes_per_shard
    
    # Tạo các node
    for i in range(total_nodes):
        G.add_node(i, trust_score=0.7)
    
    # Tạo kết nối giữa các node
    # Tạo danh sách các shard, mỗi shard chứa các node ID
    shards = []
    node_to_shard = {}
    
    for shard_id in range(num_shards):
        shard_nodes = list(range(shard_id * nodes_per_shard, (shard_id + 1) * nodes_per_shard))
        shards.append(shard_nodes)
        
        # Ánh xạ node tới shard
        for node_id in shard_nodes:
            node_to_shard[node_id] = shard_id
        
        # Kết nối nội shard (mỗi node kết nối với tất cả các node khác trong shard)
        for i in range(len(shard_nodes)):
            for j in range(i + 1, len(shard_nodes)):
                G.add_edge(shard_nodes[i], shard_nodes[j])
    
    # Kết nối liên shard (mỗi shard có một số lượng nhất định kết nối với các shard khác)
    nodes_per_connection = min(3, nodes_per_shard // 2)  # Số lượng kết nối giữa các shard
    
    for i in range(num_shards):
        for j in range(i + 1, num_shards):
            for _ in range(nodes_per_connection):
                node_i = random.choice(shards[i])
                node_j = random.choice(shards[j])
                G.add_edge(node_i, node_j)
    
    # Thêm thông tin shard vào mỗi node
    for shard_id, shard in enumerate(shards):
        for node_id in shard:
            G.nodes[node_id]['shard_id'] = shard_id
    
    return G, node_to_shard, shards

def main():
    args = parse_args()
    
    # Tạo thư mục cho kết quả
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Hiển thị thông tin
    print("\n=== Q-TRUST: Hệ thống Blockchain Sharding tối ưu hóa với Deep Reinforcement Learning ===")
    print(f"Thiết bị: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Số shards: {args.num_shards}")
    print(f"Số nodes/shard: {args.nodes_per_shard}")
    
    # Chế độ đánh giá hoặc huấn luyện
    if args.eval:
        print(f"Chế độ: ĐÁNH GIÁ")
        if args.model_path:
            print(f"Sử dụng mô hình từ: {args.model_path}")
    else:
        print(f"Chế độ: HUẤN LUYỆN")
        if hasattr(args, 'load_model') and args.load_model:
            print(f"Tiếp tục huấn luyện từ: {args.load_model}")
    
    print("Khởi tạo môi trường mô phỏng...")
    
    # Khởi tạo môi trường blockchain
    env, router, htdcm, ac_manager = setup_simulation(args)
    
    # Đảm bảo tất cả các cạnh có thuộc tính latency và bandwidth
    for u, v in env.network.edges():
        if 'latency' not in env.network.edges[u, v]:
            env.network.edges[u, v]['latency'] = random.uniform(5, 50)  # ms
        if 'bandwidth' not in env.network.edges[u, v]:
            env.network.edges[u, v]['bandwidth'] = random.uniform(1, 10)  # Mbps
    
    # Nếu có kịch bản tấn công, hiển thị thông tin
    if args.attack_scenario != 'none':
        print(f"Kịch bản tấn công: {args.attack_scenario.upper()}")
        # Trong thực tế, cấu hình môi trường theo kịch bản tấn công ở đây
    
    print("Khởi tạo các thành phần của hệ thống QTrust...")
    
    # Thiết lập quản lý đồng thuận thích ứng
    ac_manager = AdaptiveConsensus(
        transaction_threshold_low=10.0,
        transaction_threshold_high=50.0,
        congestion_threshold=0.7,
        min_trust_threshold=0.3
    )
    
    # Khởi tạo Federated Learning (FL)
    fl_manager = FederatedLearningManager(
        num_shards=args.num_shards,
        nodes_per_shard=args.nodes_per_shard,
        aggregation_method='weighted_average'
    )
    
    # Khởi tạo DQN Agent nếu cần
    dqn_agent = None
    if args.model_path:
        print(f"Đang tải mô hình DQN từ: {args.model_path}")
        # Tính kích thước state và action spaces
        state_size = env.observation_space.shape[0]
        if isinstance(env.action_space, spaces.Discrete):
            action_size = env.action_space.n
        else:
            action_size = np.prod(env.action_space.nvec)  # Tổng số hành động có thể
        
        # Điều chỉnh state_size để đảm bảo tương thích với số shard
        adjusted_state_size = state_size  # Sử dụng kích thước state từ môi trường
        
        # Khởi tạo DQNAgent
        dqn_agent = DQNAgent(
            state_size=adjusted_state_size,
            action_size=action_size,
            seed=SEED,
            buffer_size=args.memory_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_rate=args.lr,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.epsilon_end,
            hidden_layers=[args.hidden_size, args.hidden_size//2],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            prioritized_replay=True,
            dueling=True,
            update_every=5
        )
        
        try:
            dqn_agent.load(args.model_path)
            print("Đã tải mô hình DQN thành công")
            dqn_agent.epsilon = 0.0  # Không khám phá ngẫu nhiên trong chế độ đánh giá
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            print("Tiếp tục với mô hình được khởi tạo ngẫu nhiên")
    
    print("Tất cả các thành phần đã được kích hoạt thành công!")
    print(f"\n=== THÀNH PHẦN ĐÃ KÍCH HOẠT ===")
    print(f"1. Hierarchical Trust-based Data Center Mechanism (HTDCM)")
    print(f"2. Adaptive Consensus (AC)")
    print(f"3. Federated Learning Manager (FL)")
    if dqn_agent:
        print(f"4. DQN Agent (Đã tải từ {args.model_path})")
    print(f"5. MAD-RAPID Router")
    print(f"6. Blockchain Environment ({args.num_shards} shards, {args.nodes_per_shard} nodes/shard)")
    
    print("\nBắt đầu mô phỏng đầy đủ...")
    
    # Tiến hành mô phỏng với đầy đủ các thành phần
    total_episodes = args.episodes if args.episodes else 1
    all_rewards = []
    all_throughputs = []
    all_latencies = []
    all_energy_consumptions = []
    all_security_levels = []
    all_cross_shard_ratios = []
    
    # Tạo wrapper cho agent nếu có
    agent_wrapper = None
    if dqn_agent:
        class DQNAgentWrapper:
            def __init__(self, agent, num_shards):
                self.agent = agent
                self.num_shards = num_shards
                self.state_size = agent.state_size
            
            def act(self, state, eps=None):
                # Gọi action từ agent
                action_idx = self.agent.act(state, eps)
                
                # Chuyển đổi action_idx thành định dạng cho MultiDiscrete
                shard_idx = action_idx % self.num_shards  # Chọn shard
                consensus_idx = action_idx // self.num_shards  # Chọn consensus
                
                return np.array([shard_idx, consensus_idx % 3], dtype=np.int32)
        
        agent_wrapper = DQNAgentWrapper(dqn_agent, args.num_shards)
    
    # Tiến hành các episode
    for episode in range(total_episodes):
        print(f"\nEpisode {episode+1}/{total_episodes}")
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Thu thập metrics cho episode
        episode_throughputs = []
        episode_latencies = []
        episode_energy_consumptions = []
        episode_security_levels = []
        episode_cross_shard_txs = 0
        episode_total_txs = 0
        
        # Progress bar cho episode
        with tqdm(total=args.max_steps, desc=f"Episode {episode+1}") as pbar:
            while not done and step < args.max_steps:
                # Lấy trạng thái mạng hiện tại
                state = env.get_state()
                shard_congestion = env.shard_congestion
                node_trust_scores = htdcm.get_node_trust_scores()
                
                # Cập nhật router với thông tin mới
                router.update_network_state(shard_congestion, node_trust_scores)
                
                # Sử dụng phương thức select_protocol mà không cần vòng lặp for
                average_congestion = np.mean(shard_congestion) if isinstance(shard_congestion, np.ndarray) else 0.5
                tx_value = random.uniform(0, 100)  # Giả lập giá trị giao dịch
                protocol = ac_manager.select_protocol(
                    transaction_value=tx_value,
                    congestion=average_congestion,
                    trust_scores=node_trust_scores
                )
                
                # Chọn hành động: sử dụng agent nếu có, ngược lại chọn ngẫu nhiên
                if agent_wrapper:
                    # Đã có state của môi trường, giữ nguyên
                    action = agent_wrapper.act(state)
                else:
                    # Chọn ngẫu nhiên
                    action = env.action_space.sample()
                
                # Thực hiện bước mô phỏng với hành động đã chọn
                next_state, reward, done, info = env.step(action)
                
                # Thu thập metrics từ thông tin
                txs_completed = info.get('successful_txs', 0)
                latency = info.get('latency', 0)
                energy = info.get('energy_consumption', 0)
                security = info.get('security_level', 0)
                cross_shard_txs = info.get('cross_shard_txs', 0)
                total_txs = info.get('total_txs', 0)
                
                if txs_completed > 0:
                    # Thêm thông tin vào danh sách metrics
                    episode_throughputs.append(txs_completed)
                    if latency > 0:
                        episode_latencies.append(latency)
                    if energy > 0:
                        episode_energy_consumptions.append(energy)
                    if security > 0:
                        episode_security_levels.append(security)
                
                episode_cross_shard_txs += cross_shard_txs
                episode_total_txs += total_txs
                
                # Cập nhật HTDCM với kết quả giao dịch
                for tx in env.transaction_pool:
                    if tx.get('status') == 'completed':
                        # Tăng điểm tin cậy cho các node tham gia
                        nodes_involved = [tx.get('source_node'), tx.get('destination_node')]
                        for node in nodes_involved:
                            if node is not None:
                                response_time = random.uniform(5, 50)  # ms
                                htdcm.update_node_trust(node, True, response_time, True)
                    elif tx.get('status') == 'failed':
                        # Giảm điểm tin cậy cho các node lỗi
                        for node in tx.get('failed_nodes', []):
                            htdcm.update_node_trust(node, False, 0, True)
                
                # Cập nhật FL
                if step % 10 == 0:  # Định kỳ cập nhật mô hình FL
                    # Giả lập dữ liệu cục bộ từ các node
                    local_data = {}
                    for i in range(args.num_shards):
                        for j in range(args.nodes_per_shard):
                            node_id = i * args.nodes_per_shard + j
                            if random.random() > 0.3:  # 70% node tham gia
                                trust_score = node_trust_scores.get(node_id, 0.5)
                                # Mô hình đơn giản: một số đại diện cho trọng số
                                local_data[node_id] = {
                                    'model': random.uniform(0.1, 0.9),
                                    'quality': trust_score
                                }
                    
                    # Tổng hợp mô hình
                    fl_manager.round += 1
                    global_model = fl_manager.aggregate_models(local_data)
                    if global_model:
                        print(f" FL Round {fl_manager.round}: Mô hình toàn cục = {global_model:.4f}")
                
                # Cập nhật trạng thái và reward
                state = next_state
                episode_reward += reward
                step += 1
                
                # Cập nhật progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f"{episode_reward:.2f}", 
                    'txs': info.get('total_txs', 0)
                })
        
        # Tính toán metrics trung bình cho episode
        avg_throughput = np.mean(episode_throughputs) if episode_throughputs else 0
        avg_latency = np.mean(episode_latencies) if episode_latencies else 0
        avg_energy = np.mean(episode_energy_consumptions) if episode_energy_consumptions else 0
        avg_security = np.mean(episode_security_levels) if episode_security_levels else 0
        cross_shard_ratio = episode_cross_shard_txs / max(1, episode_total_txs)
        
        # Lưu metrics
        all_rewards.append(episode_reward)
        all_throughputs.append(avg_throughput)
        all_latencies.append(avg_latency)
        all_energy_consumptions.append(avg_energy)
        all_security_levels.append(avg_security)
        all_cross_shard_ratios.append(cross_shard_ratio)
        
        # Hiển thị kết quả của episode
        print(f"\nKết quả Episode {episode+1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Throughput: {avg_throughput:.2f} tx/s")
        print(f"  Latency: {avg_latency:.2f} ms")
        print(f"  Energy: {avg_energy:.2f} mJ/tx")
        print(f"  Security: {avg_security:.2f}")
        print(f"  Cross-shard ratio: {cross_shard_ratio:.2f}")
    
    # Hiển thị kết quả tổng kết
    print("\n=== KẾT QUẢ TỔNG KẾT ===")
    print(f"Reward trung bình: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Throughput trung bình: {np.mean(all_throughputs):.2f} tx/s ± {np.std(all_throughputs):.2f}")
    print(f"Latency trung bình: {np.mean(all_latencies):.2f} ms ± {np.std(all_latencies):.2f}")
    print(f"Energy trung bình: {np.mean(all_energy_consumptions):.2f} mJ/tx ± {np.std(all_energy_consumptions):.2f}")
    print(f"Security trung bình: {np.mean(all_security_levels):.2f} ± {np.std(all_security_levels):.2f}")
    print(f"Cross-shard ratio: {np.mean(all_cross_shard_ratios):.2f} ± {np.std(all_cross_shard_ratios):.2f}")
    
    # Lưu kết quả
    results = {
        'rewards': all_rewards,
        'throughputs': all_throughputs,
        'latencies': all_latencies,
        'energy_consumptions': all_energy_consumptions,
        'security_levels': all_security_levels,
        'cross_shard_ratios': all_cross_shard_ratios
    }
    
    # Lưu kết quả vào file
    with open(os.path.join(args.save_dir, 'simulation_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== KẾT QUẢ MÔ PHỎNG Q-TRUST ===\n")
        f.write(f"Số shards: {args.num_shards}\n")
        f.write(f"Số nodes/shard: {args.nodes_per_shard}\n")
        f.write(f"Số episodes: {total_episodes}\n")
        if args.attack_scenario != 'none':
            f.write(f"Kịch bản tấn công: {args.attack_scenario.upper()}\n")
        if args.model_path:
            f.write(f"Mô hình DQN: {args.model_path}\n")
        f.write("\n")
        
        f.write(f"Reward trung bình: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n")
        f.write(f"Throughput trung bình: {np.mean(all_throughputs):.2f} tx/s ± {np.std(all_throughputs):.2f}\n")
        f.write(f"Latency trung bình: {np.mean(all_latencies):.2f} ms ± {np.std(all_latencies):.2f}\n")
        f.write(f"Energy trung bình: {np.mean(all_energy_consumptions):.2f} mJ/tx ± {np.std(all_energy_consumptions):.2f}\n")
        f.write(f"Security trung bình: {np.mean(all_security_levels):.2f} ± {np.std(all_security_levels):.2f}\n")
        f.write(f"Cross-shard ratio: {np.mean(all_cross_shard_ratios):.2f} ± {np.std(all_cross_shard_ratios):.2f}\n")
    
    print(f"Đã lưu kết quả vào {args.save_dir}/simulation_results.txt")
    
    # Vẽ biểu đồ kết quả nếu có nhiều hơn 1 episode
    if total_episodes > 1:
        plot_results(results, args)
        print(f"Đã lưu biểu đồ kết quả vào thư mục {args.save_dir}")
    
    print("\nHoàn thành mô phỏng!")

if __name__ == "__main__":
    main() 