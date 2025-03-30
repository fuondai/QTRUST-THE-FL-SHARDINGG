import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import copy
import random

from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.networks import QNetwork
from qtrust.federated.federated_learning import FederatedLearning, FederatedClient
from qtrust.federated.model_aggregation import ModelAggregationManager

class FRLClient(FederatedClient):
    """
    Đại diện cho một client trong Federated Reinforcement Learning.
    Mở rộng từ FederatedClient để hỗ trợ các tác vụ RL.
    """
    def __init__(self, 
                 client_id: int, 
                 agent: DQNAgent,
                 model: nn.Module, 
                 optimizer_class: torch.optim.Optimizer = optim.Adam,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 trust_score: float = 0.7,
                 device: str = 'cpu'):
        """
        Khởi tạo client cho Federated Reinforcement Learning.
        
        Args:
            client_id: ID duy nhất của client
            agent: DQNAgent sử dụng cho quá trình học
            model: Mô hình cần học
            optimizer_class: Lớp optimizer sử dụng cho quá trình học
            learning_rate: Tốc độ học
            local_epochs: Số epoch huấn luyện cục bộ
            batch_size: Kích thước batch
            trust_score: Điểm tin cậy của client
            device: Thiết bị sử dụng (CPU hoặc GPU)
        """
        super(FRLClient, self).__init__(
            client_id, model, optimizer_class, learning_rate, 
            local_epochs, batch_size, trust_score, device
        )
        
        self.agent = agent
        
        # Dữ liệu kinh nghiệm cục bộ
        self.local_experiences = []
        self.environment = None
        
    def set_environment(self, env):
        """
        Thiết lập môi trường cho client.
        
        Args:
            env: Môi trường RL
        """
        self.environment = env
    
    def collect_experiences(self, num_steps: int = 1000, epsilon: float = 0.1):
        """
        Thu thập kinh nghiệm từ môi trường.
        
        Args:
            num_steps: Số bước tương tác với môi trường
            epsilon: Giá trị epsilon cho chính sách epsilon-greedy
            
        Returns:
            List: Các kinh nghiệm được thu thập
        """
        if self.environment is None:
            raise ValueError("Môi trường chưa được thiết lập cho client!")
        
        experiences = []
        state = self.environment.reset()
        
        for _ in range(num_steps):
            action = self.agent.act(state, eps=epsilon)
            next_state, reward, done, _ = self.environment.step(action)
            
            # Lưu trữ kinh nghiệm
            experiences.append((state, action, reward, next_state, done))
            
            # Cập nhật trạng thái
            state = next_state
            
            if done:
                state = self.environment.reset()
        
        self.local_experiences.extend(experiences)
        return experiences
    
    def update_agent_model(self):
        """
        Cập nhật mô hình của agent với mô hình đã học.
        """
        # Cập nhật qnetwork_local của agent từ mô hình của client
        self.agent.qnetwork_local.load_state_dict(self.model.state_dict())
        
        # Cập nhật target network
        if not self.agent.tau or self.agent.tau > 0:
            self.agent.qnetwork_target.load_state_dict(self.agent.qnetwork_local.state_dict())
    
    def train_local_model(self, loss_fn: Callable = nn.MSELoss()):
        """
        Huấn luyện mô hình cục bộ với dữ liệu của client.
        
        Args:
            loss_fn: Hàm mất mát sử dụng cho huấn luyện
            
        Returns:
            Dict: Dictionary chứa lịch sử mất mát và số mẫu được huấn luyện
        """
        if not self.local_experiences:
            raise ValueError("Client không có kinh nghiệm cục bộ để huấn luyện!")
        
        # Đưa kinh nghiệm vào replay buffer của agent
        for exp in self.local_experiences:
            state, action, reward, next_state, done = exp
            self.agent.step(state, action, reward, next_state, done)
        
        # Thực hiện các bước cập nhật
        losses = []
        for _ in range(self.local_epochs * len(self.local_experiences) // self.batch_size):
            if self.agent.prioritized_replay:
                self.agent._learn_prioritized()
            else:
                experiences = self.agent._sample_from_memory()
                if experiences:
                    self.agent._learn(experiences)
            
            if self.agent.loss_history:
                losses.append(self.agent.loss_history[-1])
        
        # Cập nhật mô hình client từ agent
        self.model.load_state_dict(self.agent.qnetwork_local.state_dict())
        
        return {
            'client_id': self.client_id,
            'train_loss': losses,
            'val_loss': None,
            'samples': len(self.local_experiences),
            'trust_score': self.trust_score
        }

class FederatedRL(FederatedLearning):
    """
    Quản lý quá trình huấn luyện Federated Reinforcement Learning.
    """
    def __init__(self, 
                 global_model: QNetwork,
                 aggregation_method: str = 'fedavg',
                 client_selection_method: str = 'random',
                 min_clients_per_round: int = 2,
                 min_samples_per_client: int = 10,
                 device: str = 'cpu',
                 personalized: bool = True,
                 personalization_alpha: float = 0.3,
                 privacy_preserving: bool = False,
                 privacy_epsilon: float = 0.1,
                 optimized_aggregation: bool = False):
        """
        Khởi tạo hệ thống Federated Reinforcement Learning.
        
        Args:
            global_model: Mô hình Q Network toàn cục
            aggregation_method: Phương pháp tổng hợp ('fedavg', 'fedtrust', 'fedadam')
            client_selection_method: Phương pháp chọn client ('random', 'trust_based', 'performance_based')
            min_clients_per_round: Số lượng client tối thiểu cần thiết cho mỗi vòng
            min_samples_per_client: Số lượng mẫu tối thiểu mỗi client cần có
            device: Thiết bị sử dụng
            personalized: Có sử dụng cá nhân hóa cho mỗi client hay không 
            personalization_alpha: Trọng số cho cá nhân hóa (0-1)
            privacy_preserving: Bật tính năng bảo vệ quyền riêng tư
            privacy_epsilon: Tham số epsilon cho differential privacy
            optimized_aggregation: Sử dụng tối ưu hóa tổng hợp mô hình
        """
        super(FederatedRL, self).__init__(
            global_model, aggregation_method, client_selection_method,
            min_clients_per_round, min_samples_per_client, device,
            personalized, personalization_alpha
        )
        
        # Thêm các tham số cho RL liên bang
        self.privacy_preserving = privacy_preserving
        self.privacy_epsilon = privacy_epsilon
        
        # Lưu trữ thông tin thêm cho RL
        self.client_rewards = defaultdict(list)
        self.global_environment = None
        
        # Tối ưu hóa tổng hợp mô hình
        self.optimized_aggregation = optimized_aggregation
        if optimized_aggregation:
            self.aggregation_manager = ModelAggregationManager(default_method='weighted_average')
            
            # Ánh xạ tên phương pháp cũ sang tên phương pháp mới
            self.method_mapping = {
                'fedavg': 'weighted_average',
                'fedtrust': 'adaptive_fedavg',
                'fedadam': 'fedprox'
            }
        
    def add_client(self, client: FRLClient):
        """
        Thêm client vào hệ thống Federated RL.
        
        Args:
            client: FRLClient cần thêm vào
        """
        if not isinstance(client, FRLClient):
            raise TypeError("Client phải là kiểu FRLClient")
            
        super().add_client(client)
    
    def set_global_environment(self, env):
        """
        Thiết lập môi trường toàn cục để đánh giá.
        
        Args:
            env: Môi trường RL toàn cục
        """
        self.global_environment = env
    
    def _apply_differential_privacy(self, model_updates):
        """
        Áp dụng differential privacy cho các cập nhật mô hình.
        
        Args:
            model_updates: Cập nhật tham số mô hình
            
        Returns:
            Tensor: Cập nhật tham số đã được bổ sung nhiễu
        """
        if not self.privacy_preserving:
            return model_updates
            
        # Thêm nhiễu Laplace để bảo vệ quyền riêng tư
        sensitivity = 2.0  # Độ nhạy cảm tối đa của cập nhật gradient
        scale = sensitivity / self.privacy_epsilon
        
        noise = torch.distributions.laplace.Laplace(
            torch.zeros_like(model_updates),
            torch.ones_like(model_updates) * scale
        ).sample()
        
        return model_updates + noise
    
    def train_round(self, 
                   round_num: int, 
                   client_fraction: float = 0.5,
                   steps_per_client: int = 1000,
                   exploration_epsilon: float = 0.1,
                   global_eval_episodes: int = 5) -> Dict:
        """
        Thực hiện một vòng huấn luyện trong Federated RL.
        
        Args:
            round_num: Số thứ tự vòng huấn luyện
            client_fraction: Phần trăm client tham gia
            steps_per_client: Số bước mỗi client tương tác với môi trường
            exploration_epsilon: Epsilon cho chính sách exploration
            global_eval_episodes: Số episode đánh giá mô hình toàn cục
            
        Returns:
            Dict: Thông tin và metrics của vòng huấn luyện
        """
        self.round_counter = round_num
        
        # Chọn các client tham gia vào vòng này
        selected_clients = self.select_clients(client_fraction)
        
        if len(selected_clients) < self.min_clients_per_round:
            print(f"Không đủ clients: {len(selected_clients)} < {self.min_clients_per_round}")
            return {'round': round_num, 'error': 'Không đủ clients'}
        
        client_updates = {}
        
        # Gửi mô hình toàn cục đến các client đã chọn
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            if self.personalized:
                # Nếu sử dụng cá nhân hóa, kết hợp mô hình toàn cục với mô hình cá nhân
                self._personalize_client_model(client)
            else:
                # Cập nhật mô hình của client với mô hình toàn cục
                client.set_model_params(self.global_model.state_dict())
            
            # Cập nhật mô hình của agent
            client.update_agent_model()
            
            # Thu thập kinh nghiệm từ môi trường
            client.collect_experiences(steps_per_client, exploration_epsilon)
            
            # Huấn luyện mô hình cục bộ
            client_metrics = client.train_local_model()
            
            # Nếu client có đủ dữ liệu, thu thập cập nhật
            if client_metrics['samples'] >= self.min_samples_per_client:
                client_updates[client_id] = {
                    'params': client.get_model_params(),
                    'metrics': client_metrics
                }
                
                # Áp dụng differential privacy nếu được bật
                if self.privacy_preserving:
                    for key in client_updates[client_id]['params']:
                        client_updates[client_id]['params'][key] = self._apply_differential_privacy(
                            client_updates[client_id]['params'][key]
                        )
            else:
                print(f"Bỏ qua client {client_id} vì không đủ dữ liệu: "
                     f"{client_metrics['samples']} < {self.min_samples_per_client}")
        
        # Tổng hợp cập nhật từ các client
        if client_updates:
            self.aggregate_updates(client_updates)
        
        # Đánh giá mô hình toàn cục nếu có môi trường toàn cục
        global_reward = None
        if self.global_environment is not None:
            global_reward = self._evaluate_global_model(global_eval_episodes)
            self.global_val_loss.append(-global_reward)  # Sử dụng âm reward làm mất mát
        
        # Tính mất mát huấn luyện trung bình trên các client
        avg_train_loss = np.mean([
            np.mean(update['metrics']['train_loss']) if update['metrics']['train_loss'] else 0
            for _, update in client_updates.items()
        ])
        self.global_train_loss.append(avg_train_loss)
        
        # Lưu các chỉ số đánh giá cho vòng hiện tại
        round_metrics = {
            'round': round_num,
            'clients': selected_clients,
            'avg_train_loss': avg_train_loss,
            'global_reward': global_reward,
            'client_metrics': {
                cid: update['metrics'] for cid, update in client_updates.items()
            }
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def _evaluate_global_model(self, num_episodes: int = 5) -> float:
        """
        Đánh giá mô hình toàn cục trên môi trường toàn cục.
        
        Args:
            num_episodes: Số episode đánh giá
            
        Returns:
            float: Phần thưởng trung bình
        """
        if self.global_environment is None:
            return None
        
        # Tạo agent tạm thời với mô hình toàn cục
        state_size = next(iter(self.global_model.parameters())).size(1)
        action_size = self.global_model.action_dim[0]
        
        temp_agent = DQNAgent(state_size, action_size, device=self.device)
        temp_agent.qnetwork_local.load_state_dict(self.global_model.state_dict())
        temp_agent.qnetwork_target.load_state_dict(self.global_model.state_dict())
        
        # Đánh giá 
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.global_environment.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = temp_agent.act(state, eps=0.0)  # Không exploration trong đánh giá
                next_state, reward, done, _ = self.global_environment.step(action)
                episode_reward += reward
                state = next_state
                
            total_rewards.append(episode_reward)
            
        return np.mean(total_rewards)
    
    def train(self, 
             num_rounds: int,
             client_fraction: float = 0.5,
             steps_per_client: int = 1000,
             exploration_schedule = lambda r: max(0.05, 0.5 * (0.99 ** r)),
             global_eval_episodes: int = 5,
             save_path: Optional[str] = None,
             early_stopping_rounds: int = 10,
             early_stopping_tolerance: float = 0.001,
             verbose: bool = True):
        """
        Thực hiện quá trình huấn luyện Federated RL qua nhiều vòng.
        
        Args:
            num_rounds: Số vòng huấn luyện
            client_fraction: Tỷ lệ client tham gia mỗi vòng
            steps_per_client: Số bước mỗi client tương tác với môi trường
            exploration_schedule: Hàm lấy epsilon từ số vòng
            global_eval_episodes: Số episode đánh giá mô hình toàn cục
            save_path: Đường dẫn để lưu mô hình tốt nhất
            early_stopping_rounds: Số vòng chờ trước khi dừng sớm
            early_stopping_tolerance: Ngưỡng cải thiện tối thiểu
            verbose: Hiển thị thông tin chi tiết trong quá trình huấn luyện
            
        Returns:
            Dict: Dictionary chứa lịch sử huấn luyện
        """
        best_reward = float('-inf')
        rounds_without_improvement = 0
        
        for round_idx in range(1, num_rounds + 1):
            # Lấy epsilon từ lịch trình
            exploration_epsilon = exploration_schedule(round_idx)
            
            # Thực hiện một vòng huấn luyện
            round_metrics = self.train_round(
                round_idx, 
                client_fraction, 
                steps_per_client,
                exploration_epsilon,
                global_eval_episodes
            )
            
            global_reward = round_metrics.get('global_reward')
            
            # Kiểm tra cải thiện
            if global_reward is not None:
                if global_reward > best_reward + early_stopping_tolerance:
                    best_reward = global_reward
                    rounds_without_improvement = 0
                    
                    if save_path:
                        self.save_global_model(save_path)
                else:
                    rounds_without_improvement += 1
            
            # In thông tin nếu được yêu cầu
            if verbose:
                print(f"Round {round_idx}/{num_rounds} | "
                     f"Loss: {round_metrics['avg_train_loss']:.4f} | "
                     f"Reward: {global_reward if global_reward is not None else 'N/A':.2f} | "
                     f"Epsilon: {exploration_epsilon:.3f} | "
                     f"Clients: {len(round_metrics['clients'])}")
            
            # Early stopping
            if rounds_without_improvement >= early_stopping_rounds:
                print(f"Early stopping triggered after {round_idx} rounds")
                break
        
        # Đọc lại mô hình tốt nhất nếu có
        if save_path:
            try:
                self.load_global_model(save_path)
                print(f"Đã đọc lại mô hình tốt nhất từ {save_path}")
            except:
                print(f"Không thể đọc lại mô hình từ {save_path}, sử dụng mô hình cuối cùng")
        
        return {
            'rounds_completed': min(num_rounds, round_idx),
            'best_reward': best_reward,
            'train_loss_history': self.global_train_loss,
            'reward_history': [-loss for loss in self.global_val_loss] if self.global_val_loss else None,
            'round_metrics': self.round_metrics
        }
    
    def aggregate_updates(self, client_updates: Dict) -> None:
        """
        Tổng hợp cập nhật tham số từ các client theo phương pháp đã chọn.
        
        Args:
            client_updates: Dictionary chứa cập nhật từ mỗi client
        """
        if not client_updates:
            return
        
        # Sử dụng tối ưu hóa tổng hợp mô hình nếu được bật
        if self.optimized_aggregation:
            # Lấy danh sách tham số mô hình
            params_list = [update['params'] for _, update in client_updates.items()]
            
            # Lấy trọng số dựa trên số lượng mẫu
            sample_counts = [update['metrics']['samples'] for _, update in client_updates.items()]
            weights = [count / sum(sample_counts) if sum(sample_counts) > 0 else 1.0 / len(sample_counts) 
                      for count in sample_counts]
            
            # Các tham số bổ sung dựa trên phương pháp tổng hợp
            kwargs = {'weights': weights}
            
            if self.aggregation_method == 'fedtrust':
                # Lấy điểm tin cậy và hiệu suất cho adaptive_fedavg
                trust_scores = [update['metrics']['trust_score'] for _, update in client_updates.items()]
                
                # Tính điểm hiệu suất (nghịch đảo của loss)
                performance_scores = []
                for _, update in client_updates.items():
                    if update['metrics']['val_loss'] is not None:
                        perf = 1.0 / (update['metrics']['val_loss'] + 1e-10)
                    else:
                        perf = 1.0 / (update['metrics']['train_loss'][-1] if update['metrics']['train_loss'] else 1.0 + 1e-10)
                    performance_scores.append(perf)
                
                kwargs.update({
                    'trust_scores': trust_scores,
                    'performance_scores': performance_scores
                })
            
            elif self.aggregation_method == 'fedadam':
                # Thêm tham số mô hình toàn cục cho FedProx
                kwargs.update({
                    'global_params': self.global_model.state_dict(),
                    'mu': 0.01  # Hệ số regularization mặc định
                })
            
            # Kiểm tra yêu cầu bảo mật
            suspected_byzantine = any(client.trust_score < 0.3 for client in self.clients.values())
            
            # Đề xuất phương pháp tổng hợp tốt nhất nếu không chỉ định
            if self.aggregation_method == 'auto':
                method = self.aggregation_manager.recommend_method(
                    num_clients=len(params_list),
                    has_trust_scores=(self.aggregation_method == 'fedtrust'),
                    suspected_byzantine=suspected_byzantine
                )
            else:
                # Ánh xạ tên phương pháp cũ sang tên phương pháp mới
                method = self.method_mapping.get(self.aggregation_method, 'weighted_average')
            
            # Tổng hợp mô hình với phương pháp đã chọn
            aggregated_params = self.aggregation_manager.aggregate(method, params_list, **kwargs)
            
            # Cập nhật mô hình toàn cục
            self.global_model.load_state_dict(aggregated_params)
            
            # Cập nhật hiệu suất client
            for client_id, update in client_updates.items():
                # Lưu phần thưởng trung bình
                if 'rewards' in update['metrics']:
                    self.client_rewards[client_id].append(np.mean(update['metrics']['rewards']))
        else:
            # Sử dụng phương pháp tổng hợp mặc định nếu không bật tối ưu hóa
            super().aggregate_updates(client_updates) 