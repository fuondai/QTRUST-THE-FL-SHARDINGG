import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import copy
import math
import random

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
                 personalization_alpha: float = 0.3):
        """
        Khởi tạo hệ thống Federated Learning.
        
        Args:
            global_model: Mô hình PyTorch toàn cục
            aggregation_method: Phương pháp tổng hợp ('fedavg', 'fedtrust', 'fedadam')
            client_selection_method: Phương pháp chọn client ('random', 'trust_based', 'performance_based')
            min_clients_per_round: Số lượng client tối thiểu cần thiết cho mỗi vòng
            min_samples_per_client: Số lượng mẫu tối thiểu mỗi client cần có
            device: Thiết bị sử dụng
            personalized: Có sử dụng cá nhân hóa cho mỗi client hay không 
            personalization_alpha: Trọng số cho cá nhân hóa (0-1)
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
        
        # Tham số cho FedAdam
        self.adam_m = None
        self.adam_v = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_lr = 0.01
    
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
    
    def _aggregate_fedavg(self, client_updates) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp cập nhật mô hình từ các client sử dụng FedAvg.
        
        Args:
            client_updates: Dictionary chứa cập nhật từ mỗi client
            
        Returns:
            Dict: Tham số mô hình đã được tổng hợp
        """
        # Lấy tham số và số mẫu từ mỗi client
        client_params = []
        sample_counts = []
        
        for client_id, update in client_updates.items():
            client_params.append(update['params'])
            sample_counts.append(update['metrics']['samples'])
        
        total_samples = sum(sample_counts)
        
        # Tính trọng số cho mỗi client
        if total_samples == 0:
            weights = [1.0 / len(client_params)] * len(client_params)
        else:
            weights = [count / total_samples for count in sample_counts]
        
        # Khởi tạo tham số tổng hợp từ khung của client đầu tiên
        aggregated_params = copy.deepcopy(client_params[0])
        
        # Thực hiện tổng hợp có trọng số
        for key in aggregated_params:
            aggregated_params[key] = torch.zeros_like(aggregated_params[key])
            
            for i, params in enumerate(client_params):
                aggregated_params[key] += weights[i] * params[key]
        
        return aggregated_params
    
    def _aggregate_fedtrust(self, client_updates) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp cập nhật mô hình từ các client sử dụng FedTrust (cập nhật tham số dựa trên điểm tin cậy).
        
        Args:
            client_updates: Dictionary chứa cập nhật từ mỗi client
            
        Returns:
            Dict: Tham số mô hình đã được tổng hợp
        """
        # Lấy tham số, số mẫu và điểm tin cậy từ mỗi client
        client_params = []
        sample_counts = []
        trust_scores = []
        
        for client_id, update in client_updates.items():
            client_params.append(update['params'])
            sample_counts.append(update['metrics']['samples'])
            trust_scores.append(update['metrics']['trust_score'])
        
        # Tính trọng số kết hợp giữa số mẫu và điểm tin cậy
        trust_weights = [score / sum(trust_scores) for score in trust_scores]
        sample_weights = [count / sum(sample_counts) for count in sample_counts] if sum(sample_counts) > 0 else [1.0 / len(sample_counts)] * len(sample_counts)
        
        # Kết hợp hai loại trọng số (0.7 cho điểm tin cậy, 0.3 cho số mẫu)
        weights = [0.7 * tw + 0.3 * sw for tw, sw in zip(trust_weights, sample_weights)]
        
        # Chuẩn hóa trọng số
        weights = [w / sum(weights) for w in weights]
        
        # Khởi tạo tham số tổng hợp từ khung của client đầu tiên
        aggregated_params = copy.deepcopy(client_params[0])
        
        # Thực hiện tổng hợp có trọng số
        for key in aggregated_params:
            aggregated_params[key] = torch.zeros_like(aggregated_params[key])
            
            for i, params in enumerate(client_params):
                aggregated_params[key] += weights[i] * params[key]
        
        return aggregated_params
    
    def _aggregate_fedadam(self, client_updates) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp cập nhật mô hình từ các client sử dụng FedAdam (với cập nhật adaptive dựa trên Adam).
        
        Args:
            client_updates: Dictionary chứa cập nhật từ mỗi client
            
        Returns:
            Dict: Tham số mô hình đã được tổng hợp
        """
        # Tổng hợp các tham số mô hình bằng FedAvg
        avg_params = self._aggregate_fedavg(client_updates)
        current_params = self.global_model.state_dict()
        
        # Tính gradient là hiệu giữa tham số trung bình mới và tham số hiện tại
        grad = {}
        for key in avg_params:
            grad[key] = current_params[key] - avg_params[key]
        
        # Khởi tạo m và v nếu đây là vòng đầu tiên
        if self.adam_m is None or self.adam_v is None:
            self.adam_m = {}
            self.adam_v = {}
            for key in grad:
                self.adam_m[key] = torch.zeros_like(grad[key])
                self.adam_v[key] = torch.zeros_like(grad[key])
        
        # Cập nhật tham số sử dụng Adam
        t = self.round_counter + 1
        lr_t = self.adam_lr * (np.sqrt(1 - self.adam_beta2**t) / (1 - self.adam_beta1**t))
        
        for key in grad:
            # Cập nhật m và v
            self.adam_m[key] = self.adam_beta1 * self.adam_m[key] + (1 - self.adam_beta1) * grad[key]
            self.adam_v[key] = self.adam_beta2 * self.adam_v[key] + (1 - self.adam_beta2) * grad[key]**2
            
            # Cập nhật tham số
            current_params[key] = current_params[key] - lr_t * self.adam_m[key] / (torch.sqrt(self.adam_v[key]) + self.adam_eps)
        
        return current_params
    
    def aggregate_updates(self, client_updates: Dict) -> None:
        """
        Tổng hợp cập nhật tham số từ các client theo phương pháp đã chọn.
        
        Args:
            client_updates: Dictionary chứa cập nhật từ mỗi client
        """
        if not client_updates:
            return
        
        # Chọn phương pháp tổng hợp thích hợp
        if self.aggregation_method == 'fedavg':
            aggregated_params = self._aggregate_fedavg(client_updates)
        elif self.aggregation_method == 'fedtrust':
            aggregated_params = self._aggregate_fedtrust(client_updates)
        elif self.aggregation_method == 'fedadam':
            aggregated_params = self._aggregate_fedadam(client_updates)
        else:
            raise ValueError(f"Phương pháp tổng hợp không hợp lệ: {self.aggregation_method}")
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(aggregated_params)
        
        # Cập nhật hiệu suất client
        for client_id, update in client_updates.items():
            # Sử dụng loss làm thước đo hiệu suất (nghịch đảo)
            if update['metrics']['val_loss'] is not None:
                performance = 1.0 / (update['metrics']['val_loss'] + 1e-10)  # Tránh chia cho 0
            else:
                performance = 1.0 / (update['metrics']['train_loss'][-1] + 1e-10)
                
            self.client_performance[client_id].append(performance)
    
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