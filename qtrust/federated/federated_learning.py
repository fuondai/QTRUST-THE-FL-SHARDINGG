import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import copy
import math

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
    Đại diện cho một client tham gia vào quá trình Federated Learning.
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
        Khởi tạo client cho Federated Learning.
        
        Args:
            client_id: ID duy nhất của client
            model: Mô hình cần học
            optimizer_class: Lớp optimizer sử dụng cho quá trình học
            learning_rate: Tốc độ học
            local_epochs: Số epoch huấn luyện cục bộ
            batch_size: Kích thước batch
            trust_score: Điểm tin cậy của client
            device: Thiết bị sử dụng (CPU hoặc GPU)
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.trust_score = trust_score
        self.device = device
        
        # Lịch sử huấn luyện
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Dữ liệu cục bộ
        self.local_train_data = None
        self.local_val_data = None
    
    def set_local_data(self, train_data: Tuple[torch.Tensor, torch.Tensor], 
                       val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Thiết lập dữ liệu cục bộ cho client.
        
        Args:
            train_data: Tuple (features, labels) cho tập huấn luyện
            val_data: Tuple (features, labels) cho tập validation
        """
        self.local_train_data = train_data
        self.local_val_data = val_data
    
    def train_local_model(self, loss_fn: Callable = nn.MSELoss()):
        """
        Huấn luyện mô hình cục bộ với dữ liệu của client.
        
        Args:
            loss_fn: Hàm mất mát sử dụng cho huấn luyện
            
        Returns:
            Dict: Dictionary chứa lịch sử mất mát và số mẫu được huấn luyện
        """
        if self.local_train_data is None:
            raise ValueError("Client không có dữ liệu cục bộ để huấn luyện!")
        
        x_train, y_train = self.local_train_data
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        
        dataset_size = len(x_train)
        indices = list(range(dataset_size))
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            # Shuffle dữ liệu ở mỗi epoch
            np.random.shuffle(indices)
            
            running_loss = 0.0
            batches = 0
            
            # Huấn luyện theo batch
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass và tối ưu hóa
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batches += 1
            
            epoch_loss = running_loss / batches if batches > 0 else 0
            epoch_losses.append(epoch_loss)
            self.train_loss_history.append(epoch_loss)
        
        # Tính toán mất mát trên tập validation nếu có
        val_loss = None
        if self.local_val_data is not None:
            val_loss = self.evaluate()
            self.val_loss_history.append(val_loss)
        
        return {
            'client_id': self.client_id,
            'train_loss': epoch_losses,
            'val_loss': val_loss,
            'samples': dataset_size,
            'trust_score': self.trust_score
        }
    
    def evaluate(self, loss_fn: Callable = nn.MSELoss()):
        """
        Đánh giá mô hình trên tập validation cục bộ.
        
        Args:
            loss_fn: Hàm mất mát sử dụng cho đánh giá
            
        Returns:
            float: Mất mát trên tập validation
        """
        if self.local_val_data is None:
            return None
        
        x_val, y_val = self.local_val_data
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_val)
            loss = loss_fn(outputs, y_val).item()
        
        return loss
    
    def get_model_params(self):
        """
        Lấy tham số của mô hình cục bộ.
        
        Returns:
            OrderedDict: Tham số của mô hình
        """
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params):
        """
        Cập nhật tham số của mô hình cục bộ.
        
        Args:
            params: Tham số mới cho mô hình
        """
        self.model.load_state_dict(copy.deepcopy(params))

class FederatedLearning:
    """
    Quản lý quá trình huấn luyện Federated Learning giữa nhiều client.
    """
    def __init__(self, 
                 global_model: nn.Module,
                 aggregation_method: str = 'fedavg',
                 client_selection_method: str = 'random',
                 min_clients_per_round: int = 2,
                 min_samples_per_client: int = 10,
                 device: str = 'cpu',
                 personalized: bool = False,
                 personalization_alpha: float = 0.2,
                 adaptive_aggregation: bool = False,
                 momentum: float = 0.9):
        """
        Khởi tạo hệ thống Federated Learning.
        
        Args:
            global_model: Mô hình toàn cục
            aggregation_method: Phương pháp tổng hợp ('fedavg', 'fedtrust', 'fedadam')
            client_selection_method: Phương pháp chọn client ('random', 'trust_based')
            min_clients_per_round: Số lượng client tối thiểu cần thiết cho mỗi vòng
            min_samples_per_client: Số lượng mẫu tối thiểu mỗi client cần có
            device: Thiết bị sử dụng
            personalized: Có sử dụng cá nhân hóa cho mỗi client hay không 
            personalization_alpha: Trọng số cho cá nhân hóa (0-1)
            adaptive_aggregation: Sử dụng FedAdam adaptive aggregation 
            momentum: Hệ số động lượng cho adaptive aggregation
        """
        self.global_model = global_model
        self.aggregation_method = aggregation_method
        self.client_selection_method = client_selection_method
        self.min_clients_per_round = min_clients_per_round
        self.min_samples_per_client = min_samples_per_client
        self.device = device
        
        # Thêm các tham số mới cho cải tiến
        self.personalized = personalized
        self.personalization_alpha = personalization_alpha
        self.adaptive_aggregation = adaptive_aggregation
        self.momentum = momentum
        
        # Dictionary lưu các client
        self.clients = {}
        
        # Thêm các biến cho FedAdam
        if self.adaptive_aggregation:
            self.m = None  # Momentum
            self.v = None  # Variance
            self.beta1 = 0.9  # Beta1 cho FedAdam
            self.beta2 = 0.99  # Beta2 cho FedAdam
            self.epsilon = 1e-8  # Epsilon để tránh chia cho 0
            self.round_counter = 0  # Đếm số vòng đã huấn luyện
        
        # Lưu thông tin quá trình huấn luyện
        self.global_train_loss = []
        self.global_val_loss = []
        self.round_metrics = []
    
    def add_client(self, client: FederatedClient):
        """
        Thêm client vào hệ thống.
        
        Args:
            client: FederatedClient cần thêm
        """
        self.clients[client.client_id] = client
        
        # Cài đặt mô hình cá nhân hóa nếu được kích hoạt
        if self.personalized:
            # Tạo một bản sao của mô hình toàn cục cho client
            client_model = copy.deepcopy(self.global_model).to(self.device)
            client.model = client_model
    
    def select_clients(self, fraction: float = 0.5) -> List[int]:
        """
        Chọn subset của clients để tham gia vào vòng huấn luyện.
        
        Args:
            fraction: Tỷ lệ client tham gia
            
        Returns:
            List[int]: Danh sách client_id được chọn
        """
        available_clients = list(self.clients.keys())
        num_clients = max(self.min_clients_per_round, int(fraction * len(available_clients)))
        num_clients = min(num_clients, len(available_clients))
        
        if self.client_selection_method == 'random':
            return np.random.choice(available_clients, num_clients, replace=False).tolist()
        
        elif self.client_selection_method == 'trust_based':
            # Chọn client dựa trên điểm tin cậy
            trust_scores = {cid: client.trust_score for cid, client in self.clients.items()}
            
            # Chuẩn hóa điểm tin cậy thành xác suất
            total_trust = sum(trust_scores.values())
            if total_trust == 0:
                # Nếu tổng điểm tin cậy là 0, chọn ngẫu nhiên
                return np.random.choice(available_clients, num_clients, replace=False).tolist()
            
            # Tính xác suất lựa chọn
            probabilities = [trust_scores[cid] / total_trust for cid in available_clients]
            
            # Chọn các client dựa trên xác suất
            return np.random.choice(
                available_clients, 
                num_clients, 
                replace=False, 
                p=probabilities
            ).tolist()
        
        else:
            raise ValueError(f"Phương pháp chọn client không hợp lệ: {self.client_selection_method}")

    def train_round(self, 
                   round_num: int, 
                   client_fraction: float = 0.5,
                   loss_fn: Callable = nn.MSELoss(),
                   global_val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict:
        """
        Thực hiện một vòng huấn luyện Federated Learning.
        
        Args:
            round_num: Số thứ tự vòng huấn luyện
            client_fraction: Tỷ lệ client tham gia
            loss_fn: Hàm mất mát
            global_val_data: Dữ liệu validation toàn cục
            
        Returns:
            Dict: Thông tin về vòng huấn luyện
        """
        # Chọn clients tham gia vòng huấn luyện
        selected_clients = self.select_clients(client_fraction)
        
        # Kiểm tra số lượng client
        if len(selected_clients) < self.min_clients_per_round:
            print(f"Cảnh báo: Chỉ có {len(selected_clients)} client tham gia, "
                 f"nhỏ hơn yêu cầu tối thiểu {self.min_clients_per_round}")
            if len(selected_clients) == 0:
                return {
                    'round': round_num,
                    'clients': [],
                    'avg_train_loss': None,
                    'val_loss': None,
                    'client_metrics': {}
                }
        
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
            
            # Huấn luyện mô hình cục bộ trên client
            client_metrics = client.train_local_model(loss_fn)
            
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
        
        # Đánh giá mô hình toàn cục nếu có dữ liệu validation
        val_loss = None
        if global_val_data is not None:
            val_loss = self._evaluate_global_model(global_val_data, loss_fn)
            self.global_val_loss.append(val_loss)
        
        # Tính mất mát huấn luyện trung bình trên các client
        avg_train_loss = np.mean([
            np.mean(update['metrics']['train_loss']) 
            for _, update in client_updates.items()
        ])
        self.global_train_loss.append(avg_train_loss)
        
        # Lưu các chỉ số đánh giá cho vòng hiện tại
        round_metrics = {
            'round': round_num,
            'clients': selected_clients,
            'avg_train_loss': avg_train_loss,
            'val_loss': val_loss,
            'client_metrics': {
                cid: update['metrics'] for cid, update in client_updates.items()
            }
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def _personalize_client_model(self, client: FederatedClient):
        """
        Kết hợp mô hình toàn cục với mô hình cá nhân cho client.
        
        Args:
            client: Client cần cá nhân hóa
        """
        global_params = self.global_model.state_dict()
        client_params = client.get_model_params()
        
        personalized_params = {}
        for key in global_params.keys():
            if key in client_params:
                # Kết hợp mô hình toàn cục và cá nhân với trọng số alpha
                personalized_params[key] = (
                    self.personalization_alpha * client_params[key] + 
                    (1 - self.personalization_alpha) * global_params[key]
                )
            else:
                # Nếu tham số không có trong mô hình cá nhân, sử dụng mô hình toàn cục
                personalized_params[key] = global_params[key]
        
        client.set_model_params(personalized_params)
    
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật từ các client.
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        if not client_updates:
            return
        
        if self.aggregation_method == 'fedavg':
            self._aggregate_fedavg(client_updates)
        elif self.aggregation_method == 'fedtrust':
            self._aggregate_fedtrust(client_updates)
        elif self.aggregation_method == 'fedadam':
            self._aggregate_fedadam(client_updates)
        else:
            raise ValueError(f"Phương pháp tổng hợp không hợp lệ: {self.aggregation_method}")
    
    def _aggregate_fedavg(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật theo thuật toán FedAvg.
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        # Lấy trọng số dựa trên số lượng mẫu từ mỗi client
        total_samples = sum(update['metrics']['samples'] for _, update in client_updates.items())
        
        # Khởi tạo tham số tổng hợp
        global_params = self.global_model.state_dict()
        
        # Tính tổng tham số có trọng số
        for client_id, update in client_updates.items():
            client_params = update['params']
            weight = update['metrics']['samples'] / total_samples
            
            for key in global_params.keys():
                if client_id == list(client_updates.keys())[0]:
                    # Nếu là client đầu tiên, khởi tạo tham số với 0
                    global_params[key] = torch.zeros_like(global_params[key])
                
                # Thêm tham số của client với trọng số
                global_params[key] += client_params[key] * weight
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(global_params)
    
    def _aggregate_fedtrust(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật theo thuật toán FedTrust (tính đến mức độ tin cậy).
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        # Tính trọng số dựa trên số lượng mẫu và điểm tin cậy
        total_weight = sum(
            update['metrics']['samples'] * update['metrics']['trust_score']
            for _, update in client_updates.items()
        )
        
        if total_weight == 0:
            # Fallback về FedAvg nếu tổng trọng số là 0
            return self._aggregate_fedavg(client_updates)
        
        # Khởi tạo tham số tổng hợp
        global_params = self.global_model.state_dict()
        
        # Tính tổng tham số có trọng số
        for client_id, update in client_updates.items():
            client_params = update['params']
            weight = (update['metrics']['samples'] * update['metrics']['trust_score']) / total_weight
            
            for key in global_params.keys():
                if client_id == list(client_updates.keys())[0]:
                    # Nếu là client đầu tiên, khởi tạo tham số với 0
                    global_params[key] = torch.zeros_like(global_params[key])
                
                # Thêm tham số của client với trọng số
                global_params[key] += client_params[key] * weight
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(global_params)
    
    def _aggregate_fedadam(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật theo thuật toán FedAdam (adaptive aggregation).
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        self.round_counter += 1
        
        # Lấy trọng số dựa trên số lượng mẫu từ mỗi client
        total_samples = sum(update['metrics']['samples'] for _, update in client_updates.items())
        
        # Khởi tạo tham số tổng hợp
        global_params = self.global_model.state_dict()
        
        # Tính delta (sự khác biệt) của pseudogradient
        pseudo_gradient = {}
        for key in global_params.keys():
            pseudo_gradient[key] = torch.zeros_like(global_params[key])
        
        # Tính pseudogradient có trọng số
        for client_id, update in client_updates.items():
            client_params = update['params']
            weight = update['metrics']['samples'] / total_samples
            
            for key in global_params.keys():
                # Pseudogradient = (global_params - client_params)
                # Đảo dấu vì trong tối ưu, chúng ta di chuyển theo hướng ngược của gradient
                pseudo_gradient[key] -= weight * (global_params[key] - client_params[key])
        
        # Khởi tạo momentum và variance nếu chưa có
        if self.m is None:
            self.m = {key: torch.zeros_like(value) for key, value in pseudo_gradient.items()}
        if self.v is None:
            self.v = {key: torch.zeros_like(value) for key, value in pseudo_gradient.items()}
        
        # Cập nhật momentum và variance
        for key in pseudo_gradient.keys():
            # Cập nhật momentum (m_t = beta1 * m_{t-1} + (1 - beta1) * g_t)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * pseudo_gradient[key]
            # Cập nhật variance (v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (pseudo_gradient[key] ** 2)
            
            # Hiệu chỉnh bias
            m_hat = self.m[key] / (1 - self.beta1 ** self.round_counter)
            v_hat = self.v[key] / (1 - self.beta2 ** self.round_counter)
            
            # Cập nhật tham số toàn cục
            # theta_t = theta_{t-1} + alpha * m_hat / (sqrt(v_hat) + epsilon)
            global_params[key] += self.momentum * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(global_params)
    
    def _evaluate_global_model(self, global_val_data: Tuple[torch.Tensor, torch.Tensor], loss_fn: Callable):
        """
        Đánh giá mô hình toàn cục trên tập validation.
        
        Args:
            global_val_data: Dữ liệu validation toàn cục
            loss_fn: Hàm mất mát sử dụng cho đánh giá
            
        Returns:
            float: Mất mát trên tập validation
        """
        if global_val_data is None:
            return None
        
        x_val, y_val = global_val_data
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        
        self.global_model.eval()
        with torch.no_grad():
            outputs = self.global_model(x_val)
            val_loss = loss_fn(outputs, y_val).item()
        
        return val_loss
    
    def get_global_model(self):
        """
        Lấy mô hình toàn cục hiện tại.
        
        Returns:
            nn.Module: Mô hình toàn cục
        """
        return self.global_model
    
    def update_client_trust(self, client_id: int, trust_score: float):
        """
        Cập nhật điểm tin cậy cho một client.
        
        Args:
            client_id: ID của client
            trust_score: Điểm tin cậy mới
        """
        if client_id in self.clients:
            self.clients[client_id].trust_score = trust_score
    
    def save_global_model(self, path: str):
        """
        Lưu mô hình toàn cục vào file.
        
        Args:
            path: Đường dẫn để lưu mô hình
        """
        torch.save(self.global_model.state_dict(), path)
    
    def load_global_model(self, path: str):
        """
        Tải mô hình toàn cục từ file.
        
        Args:
            path: Đường dẫn đến file mô hình
        """
        self.global_model.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, 
             num_rounds: int,
             client_fraction: float = 0.5,
             loss_fn: Callable = nn.MSELoss(),
             global_val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             save_path: Optional[str] = None,
             early_stopping_rounds: int = 10,
             early_stopping_tolerance: float = 0.001,
             verbose: bool = True):
        """
        Thực hiện quá trình huấn luyện Federated Learning qua nhiều vòng.
        
        Args:
            num_rounds: Số vòng huấn luyện
            client_fraction: Tỷ lệ client tham gia mỗi vòng
            loss_fn: Hàm mất mát
            global_val_data: Dữ liệu validation toàn cục
            save_path: Đường dẫn để lưu mô hình tốt nhất
            early_stopping_rounds: Số vòng chờ trước khi dừng sớm
            early_stopping_tolerance: Ngưỡng cải thiện tối thiểu
            verbose: Hiển thị thông tin chi tiết trong quá trình huấn luyện
            
        Returns:
            Dict: Dictionary chứa lịch sử huấn luyện
        """
        best_val_loss = float('inf')
        rounds_without_improvement = 0
        
        for round_num in range(1, num_rounds + 1):
            round_metrics = self.train_round(
                round_num, client_fraction, loss_fn, global_val_data
            )
            
            # In kết quả
            if verbose:
                print(f"Vòng {round_num}: "
                    f"Mất mát huấn luyện = {round_metrics['avg_train_loss']:.4f}, "
                    f"Mất mát validation = {round_metrics['val_loss']:.4f if round_metrics['val_loss'] is not None else 'N/A'}")
            
            # Kiểm tra điều kiện dừng sớm
            if global_val_data is not None and round_metrics['val_loss'] is not None:
                if round_metrics['val_loss'] < best_val_loss - early_stopping_tolerance:
                    best_val_loss = round_metrics['val_loss']
                    rounds_without_improvement = 0
                    
                    # Lưu mô hình tốt nhất
                    if save_path:
                        self.save_global_model(save_path)
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= early_stopping_rounds:
                    if verbose:
                        print(f"Dừng sớm tại vòng {round_num} do không có cải thiện sau {early_stopping_rounds} vòng")
                    break
        
        # Tải mô hình tốt nhất nếu đã lưu
        if save_path and global_val_data is not None:
            self.load_global_model(save_path)
        
        # Trả về lịch sử huấn luyện
        return {
            'train_loss': self.global_train_loss,
            'val_loss': self.global_val_loss,
            'round_metrics': self.round_metrics
        } 