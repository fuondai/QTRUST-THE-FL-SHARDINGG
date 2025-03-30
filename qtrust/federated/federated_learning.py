import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import copy
import math
import random
from .model_aggregation import ModelAggregationManager
from .privacy import PrivacyManager, SecureAggregator

class FederatedModel(nn.Module):
    """
    Mô hình cơ sở cho Federated Learning.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Khởi tạo mô hình với các tham số cơ bản.
        
        Args:
            input_size: Kích thước đầu vào
            hidden_size: Kích thước lớp ẩn
            output_size: Kích thước đầu ra
        """
        super(FederatedModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Truyền dữ liệu qua mô hình.
        
        Args:
            x: Dữ liệu đầu vào
            
        Returns:
            Dữ liệu đầu ra từ mô hình
        """
        return self.layers(x)

class FederatedClient:
    """
    Đại diện cho một client trong hệ thống Federated Learning.
    """
    def __init__(self, 
                 client_id: int, 
                 model: nn.Module, 
                 optimizer_class: torch.optim.Optimizer = optim.Adam,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 trust_score: float = 0.7,
                 device: str = 'cpu'):
        """
        Khởi tạo client Federated Learning.
        
        Args:
            client_id: ID duy nhất của client
            model: Mô hình PyTorch được dùng bởi client
            optimizer_class: Lớp optimizer sử dụng cho huấn luyện
            learning_rate: Tốc độ học
            local_epochs: Số epoch huấn luyện cục bộ
            batch_size: Kích thước batch
            trust_score: Điểm tin cậy của client
            device: Thiết bị sử dụng (CPU hoặc GPU)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.trust_score = trust_score
        self.device = device
        
        # Dữ liệu cục bộ của client
        self.local_data = []
        
        # Mô hình của client sau khi được cá nhân hóa
        self.personalized_model = None
    
    def set_data(self, data: List):
        """
        Thiết lập dữ liệu cho client.
        
        Args:
            data: Dữ liệu dành cho client này
        """
        self.local_data = data
    
    def set_model_params(self, model_params: Dict[str, torch.Tensor]):
        """
        Thiết lập tham số mô hình từ mô hình toàn cục.
        
        Args:
            model_params: Tham số mô hình toàn cục
        """
        self.model.load_state_dict(model_params)
    
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        Lấy tham số mô hình hiện tại của client.
        
        Returns:
            Dict: Tham số mô hình
        """
        return copy.deepcopy(self.model.state_dict())
    
    def train_local_model(self) -> Dict:
        """
        Huấn luyện mô hình cục bộ với dữ liệu của client.
        
        Returns:
            Dict: Dictionary chứa lịch sử mất mát và số mẫu được huấn luyện
        """
        # Method này cần được ghi đè trong lớp con tùy vào loại dữ liệu
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")
    
    def set_personalized_model(self, alpha: float, global_model_params: Dict[str, torch.Tensor]):
        """
        Thiết lập mô hình cá nhân hóa bằng cách kết hợp mô hình toàn cục với mô hình cục bộ.
        
        Args:
            alpha: Trọng số cho mô hình cục bộ (0-1)
            global_model_params: Tham số mô hình toàn cục
        """
        if not self.personalized_model:
            self.personalized_model = copy.deepcopy(self.model)
        
        local_params = self.model.state_dict()
        
        # Kết hợp mô hình toàn cục và cục bộ với trọng số alpha
        for key in local_params:
            local_params[key] = alpha * local_params[key] + (1 - alpha) * global_model_params[key]
        
        self.personalized_model.load_state_dict(local_params)

class FederatedLearning:
    """
    Quản lý quá trình huấn luyện Federated Learning.
    """
    def __init__(self, 
                 global_model: nn.Module,
                 aggregation_method: str = 'fedavg',
                 client_selection_method: str = 'random',
                 min_clients_per_round: int = 2,
                 min_samples_per_client: int = 10,
                 device: str = 'cpu',
                 personalized: bool = False,
                 personalization_alpha: float = 0.3,
                 optimized_aggregation: bool = True,
                 privacy_preserving: bool = True,
                 privacy_epsilon: float = 1.0,
                 privacy_delta: float = 1e-5,
                 secure_aggregation: bool = True):
        """
        Khởi tạo hệ thống Federated Learning.
        
        Args:
            global_model: Mô hình PyTorch toàn cục
            aggregation_method: Phương pháp tổng hợp ('fedavg', 'fedtrust', 'fedadam', 'auto')
            client_selection_method: Phương pháp chọn client ('random', 'trust_based', 'performance_based')
            min_clients_per_round: Số lượng client tối thiểu cần thiết cho mỗi vòng
            min_samples_per_client: Số lượng mẫu tối thiểu mỗi client cần có
            device: Thiết bị sử dụng
            personalized: Có sử dụng cá nhân hóa cho mỗi client hay không 
            personalization_alpha: Trọng số cho cá nhân hóa (0-1)
            optimized_aggregation: Sử dụng tối ưu hóa tổng hợp mô hình
            privacy_preserving: Bật tính năng bảo vệ quyền riêng tư
            privacy_epsilon: Privacy budget (epsilon)
            privacy_delta: Xác suất thất bại (delta)
            secure_aggregation: Bật tính năng tổng hợp an toàn
        """
        self.global_model = global_model.to(device)
        self.aggregation_method = aggregation_method
        self.client_selection_method = client_selection_method
        self.min_clients_per_round = min_clients_per_round
        self.min_samples_per_client = min_samples_per_client
        self.device = device
        self.personalized = personalized
        self.personalization_alpha = personalization_alpha
        
        # Thông tin về clients
        self.clients = {}
        self.client_performance = defaultdict(list)
        
        # Thông tin về huấn luyện
        self.round_counter = 0
        self.global_train_loss = []
        self.global_val_loss = []
        self.round_metrics = []
        
        # Khởi tạo ModelAggregationManager
        self.aggregation_manager = ModelAggregationManager(
            default_method='weighted_average' if aggregation_method == 'fedavg' else aggregation_method
        )
        
        # Ánh xạ tên phương pháp cũ sang tên phương pháp mới
        self.method_mapping = {
            'fedavg': 'weighted_average',
            'fedtrust': 'adaptive_fedavg',
            'fedadam': 'fedprox',
            'auto': 'auto'
        }
        
        # Khởi tạo PrivacyManager và SecureAggregator nếu được yêu cầu
        self.privacy_preserving = privacy_preserving
        if privacy_preserving:
            self.privacy_manager = PrivacyManager(
                epsilon=privacy_epsilon,
                delta=privacy_delta
            )
            
            if secure_aggregation:
                self.secure_aggregator = SecureAggregator(
                    privacy_manager=self.privacy_manager,
                    threshold=min_clients_per_round
                )
            else:
                self.secure_aggregator = None
        else:
            self.privacy_manager = None
            self.secure_aggregator = None
    
    def add_client(self, client: FederatedClient):
        """
        Thêm client vào hệ thống Federated Learning.
        
        Args:
            client: FederatedClient cần thêm vào
        """
        if not isinstance(client, FederatedClient):
            raise TypeError("Client phải là kiểu FederatedClient")
            
        self.clients[client.client_id] = client
    
    def select_clients(self, fraction: float) -> List[int]:
        """
        Chọn các client tham gia vào vòng huấn luyện hiện tại.
        
        Args:
            fraction: Tỷ lệ client được chọn
            
        Returns:
            List: Danh sách ID client được chọn
        """
        num_clients = max(self.min_clients_per_round, int(fraction * len(self.clients)))
        num_clients = min(num_clients, len(self.clients))
        
        if self.client_selection_method == 'random':
            # Chọn ngẫu nhiên
            selected_clients = random.sample(list(self.clients.keys()), num_clients)
            
        elif self.client_selection_method == 'trust_based':
            # Chọn dựa trên điểm tin cậy
            clients_by_trust = sorted(
                self.clients.items(), 
                key=lambda x: x[1].trust_score, 
                reverse=True
            )
            selected_clients = [c[0] for c in clients_by_trust[:num_clients]]
            
        elif self.client_selection_method == 'performance_based':
            # Chọn dựa trên hiệu suất gần đây
            if not self.client_performance or len(self.client_performance) < len(self.clients) / 2:
                # Nếu chưa có đủ dữ liệu hiệu suất, chọn ngẫu nhiên
                selected_clients = random.sample(list(self.clients.keys()), num_clients)
            else:
                # Tính hiệu suất trung bình gần đây cho mỗi client
                avg_performance = {}
                for client_id, performances in self.client_performance.items():
                    # Lấy tối đa 5 vòng gần nhất
                    recent_perf = performances[-5:] 
                    avg_performance[client_id] = sum(recent_perf) / len(recent_perf)
                
                # Chọn các client có hiệu suất cao nhất
                top_clients = sorted(
                    avg_performance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                selected_clients = [c[0] for c in top_clients[:num_clients]]
        else:
            raise ValueError(f"Phương pháp chọn client không hợp lệ: {self.client_selection_method}")
        
        return selected_clients
    
    def aggregate_updates(self, client_updates: Dict) -> None:
        """
        Tổng hợp cập nhật tham số từ các client theo phương pháp đã chọn.
        
        Args:
            client_updates: Dictionary chứa cập nhật từ mỗi client
        """
        if not client_updates:
            return
            
        # Lấy danh sách tham số mô hình
        params_list = [update['params'] for _, update in client_updates.items()]
        
        # Lấy trọng số dựa trên số lượng mẫu
        sample_counts = [update['metrics']['samples'] for _, update in client_updates.items()]
        weights = [count / sum(sample_counts) if sum(sample_counts) > 0 else 1.0 / len(sample_counts) 
                  for count in sample_counts]
        
        # Nếu sử dụng secure aggregation
        if self.secure_aggregator is not None:
            try:
                aggregated_params = self.secure_aggregator.aggregate_secure(
                    client_updates, weights
                )
                self.global_model.load_state_dict(aggregated_params)
                return
            except Exception as e:
                print(f"Lỗi khi thực hiện secure aggregation: {e}")
                print("Chuyển sang phương pháp tổng hợp thông thường...")
        
        # Các tham số bổ sung dựa trên phương pháp tổng hợp
        kwargs = {'weights': weights}
        
        if self.aggregation_method == 'fedtrust':
            # Lấy điểm tin cậy và hiệu suất cho adaptive_fedavg
            trust_scores = [update['metrics'].get('trust_score', 0.5) for _, update in client_updates.items()]
            
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
        suspected_byzantine = any(
            update['metrics'].get('trust_score', 1.0) < 0.3 
            for _, update in client_updates.items()
        )
        
        # Đề xuất phương pháp tổng hợp tốt nhất nếu đang ở chế độ tự động
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
        
        # Thêm nhiễu nếu sử dụng privacy preserving nhưng không có secure aggregation
        if self.privacy_preserving and self.secure_aggregator is None:
            total_samples = sum(sample_counts)
            aggregated_params = self.privacy_manager.add_noise_to_model(
                aggregated_params, total_samples
            )
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(aggregated_params)
        
        # Cập nhật metrics hiệu suất
        avg_loss = np.mean([
            update['metrics']['val_loss'] if update['metrics']['val_loss'] is not None 
            else update['metrics']['train_loss'][-1]
            for _, update in client_updates.items()
        ])
        
        self.aggregation_manager.update_performance_metrics(method, {
            'loss': avg_loss,
            'score': 1.0 / (avg_loss + 1e-10),
            'num_clients': len(params_list),
            'suspected_byzantine': suspected_byzantine
        })
        
        # Cập nhật privacy report nếu có
        if self.privacy_preserving:
            privacy_report = self.privacy_manager.get_privacy_report()
            print("\nPrivacy Report:")
            print(f"Status: {privacy_report['status']}")
            print(f"Consumed Budget: {privacy_report['consumed_budget']:.4f}")
            print(f"Remaining Budget: {privacy_report['remaining_budget']:.4f}")
            if privacy_report['status'] == 'Privacy budget exceeded':
                print("Warning: Privacy budget đã vượt quá giới hạn!")
    
    def _personalize_client_model(self, client: FederatedClient) -> None:
        """
        Cá nhân hóa mô hình cho client bằng cách kết hợp mô hình toàn cục và cục bộ.
        
        Args:
            client: Client cần cá nhân hóa mô hình
        """
        client.set_personalized_model(
            self.personalization_alpha, 
            self.global_model.state_dict()
        )
    
    def train_round(self, round_num: int, client_fraction: float = 0.5) -> Dict:
        """
        Thực hiện một vòng huấn luyện Federated Learning.
        
        Args:
            round_num: Số thứ tự vòng huấn luyện
            client_fraction: Phần trăm client tham gia
            
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
            
            # Huấn luyện mô hình cục bộ
            client_metrics = client.train_local_model()
            
            # Nếu client có đủ dữ liệu, thu thập cập nhật
            if client_metrics['samples'] >= self.min_samples_per_client:
                client_updates[client_id] = {
                    'params': client.get_model_params(),
                    'metrics': client_metrics
                }
            else:
                print(f"Bỏ qua client {client_id} vì không đủ dữ liệu: "
                     f"{client_metrics['samples']} < {self.min_samples_per_client}")
        
        # Tổng hợp cập nhật từ các client
        if client_updates:
            self.aggregate_updates(client_updates)
        
        # Tính mất mát huấn luyện và validation trung bình trên các client
        avg_train_loss = np.mean([
            np.mean(update['metrics']['train_loss']) if update['metrics']['train_loss'] else 0
            for _, update in client_updates.items()
        ])
        
        avg_val_loss = None
        if all(update['metrics']['val_loss'] is not None for _, update in client_updates.items()):
            avg_val_loss = np.mean([
                update['metrics']['val_loss'] for _, update in client_updates.items()
            ])
        
        self.global_train_loss.append(avg_train_loss)
        if avg_val_loss is not None:
            self.global_val_loss.append(avg_val_loss)
        
        # Lưu các chỉ số đánh giá cho vòng hiện tại
        round_metrics = {
            'round': round_num,
            'clients': selected_clients,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'client_metrics': {
                cid: update['metrics'] for cid, update in client_updates.items()
            }
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def train(self, 
             num_rounds: int,
             client_fraction: float = 0.5,
             early_stopping_rounds: int = 10,
             early_stopping_tolerance: float = 0.001,
             save_path: Optional[str] = None,
             verbose: bool = True) -> Dict:
        """
        Thực hiện quá trình huấn luyện Federated Learning qua nhiều vòng.
        
        Args:
            num_rounds: Số vòng huấn luyện
            client_fraction: Tỷ lệ client tham gia mỗi vòng
            early_stopping_rounds: Số vòng chờ trước khi dừng sớm
            early_stopping_tolerance: Ngưỡng cải thiện tối thiểu
            save_path: Đường dẫn để lưu mô hình tốt nhất
            verbose: Hiển thị thông tin chi tiết trong quá trình huấn luyện
            
        Returns:
            Dict: Dictionary chứa lịch sử huấn luyện
        """
        best_val_loss = float('inf')
        rounds_without_improvement = 0
        
        for round_idx in range(1, num_rounds + 1):
            # Thực hiện một vòng huấn luyện
            round_metrics = self.train_round(round_idx, client_fraction)
            
            val_loss = round_metrics.get('avg_val_loss')
            
            # Kiểm tra cải thiện
            if val_loss is not None:
                if val_loss < best_val_loss - early_stopping_tolerance:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                    
                    if save_path:
                        self.save_global_model(save_path)
                else:
                    rounds_without_improvement += 1
            
            # In thông tin nếu được yêu cầu
            if verbose:
                val_str = f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Round {round_idx}/{num_rounds} | Train Loss: {round_metrics['avg_train_loss']:.4f}{val_str} | "
                     f"Clients: {len(round_metrics['clients'])}")
            
            # Early stopping
            if rounds_without_improvement >= early_stopping_rounds and val_loss is not None:
                print(f"Early stopping triggered after {round_idx} rounds")
                break
        
        # Đọc lại mô hình tốt nhất nếu có
        if save_path and val_loss is not None:
            try:
                self.load_global_model(save_path)
                print(f"Đã đọc lại mô hình tốt nhất từ {save_path}")
            except:
                print(f"Không thể đọc lại mô hình từ {save_path}, sử dụng mô hình cuối cùng")
        
        return {
            'rounds_completed': min(num_rounds, round_idx),
            'best_val_loss': best_val_loss if val_loss is not None else None,
            'train_loss_history': self.global_train_loss,
            'val_loss_history': self.global_val_loss,
            'round_metrics': self.round_metrics
        }
    
    def save_global_model(self, path: str) -> None:
        """
        Lưu mô hình toàn cục vào file.
        
        Args:
            path: Đường dẫn đến file lưu trữ
        """
        torch.save(self.global_model.state_dict(), path)
    
    def load_global_model(self, path: str) -> None:
        """
        Đọc mô hình toàn cục từ file.
        
        Args:
            path: Đường dẫn đến file lưu trữ
        """
        self.global_model.load_state_dict(torch.load(path, map_location=self.device)) 