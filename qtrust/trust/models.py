import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional

class QNetwork(nn.Module):
    """
    Mạng Deep Q-Network với kiến trúc Dueling Network tùy chọn
    """
    def __init__(self, state_size: int, action_dims: List[int], seed: int = 42, 
                 hidden_layers: List[int] = [128, 128], dueling: bool = False):
        """
        Khởi tạo mạng Q.

        Args:
            state_size: Số chiều của không gian trạng thái
            action_dims: Danh sách kích thước cho mỗi chiều của không gian hành động
            seed: Seed cho việc tạo ngẫu nhiên
            hidden_layers: Kích thước các lớp ẩn
            dueling: Sử dụng kiến trúc Dueling Network hay không
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_dims = action_dims
        self.dueling = dueling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Số lượng đầu ra tổng cộng
        self.total_actions = action_dims[0] if len(action_dims) == 1 else sum(action_dims)
        
        # Lớp đặc trưng (feature layer)
        layers = []
        in_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.feature_layer = nn.Sequential(*layers)
        
        if dueling:
            # Kiến trúc Dueling Network - tách biệt ước lượng giá trị trạng thái và lợi thế của hành động
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_layers[-1] // 2, 1)
            )
            
            # Tạo mạng lợi thế (advantage network) cho mỗi chiều hành động
            self.advantage_streams = nn.ModuleList()
            for action_dim in action_dims:
                self.advantage_streams.append(nn.Sequential(
                    nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[-1] // 2, action_dim)
                ))
        else:
            # Kiến trúc chuẩn - mỗi chiều hành động có một đầu ra riêng
            self.output_layers = nn.ModuleList()
            for action_dim in action_dims:
                self.output_layers.append(nn.Linear(hidden_layers[-1], action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Truyền xuôi qua mạng.

        Args:
            state: Trạng thái đầu vào, shape (batch_size, state_size)

        Returns:
            Tuple[List[torch.Tensor], Optional[torch.Tensor]]: 
                - Danh sách các tensor Q-values, mỗi tensor có shape (batch_size, action_dim)
                - Giá trị trạng thái (nếu sử dụng dueling network)
        """
        x = self.feature_layer(state)
        
        if self.dueling:
            # Tính giá trị trạng thái (state value)
            value = self.value_stream(x)
            
            # Tính lợi thế hành động (action advantage) cho mỗi chiều
            advantages = [advantage_stream(x) for advantage_stream in self.advantage_streams]
            
            # Kết hợp giá trị và lợi thế để có Q-values
            q_values = []
            for advantage in advantages:
                # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))) để đảm bảo tính ổn định
                q = value + (advantage - advantage.mean(dim=1, keepdim=True))
                q_values.append(q)
            
            return q_values, value
        else:
            # Đầu ra trực tiếp là Q-values cho mỗi chiều hành động
            q_values = [output_layer(x) for output_layer in self.output_layers]
            return q_values, None

class ActorCriticNetwork(nn.Module):
    """
    Mạng Actor-Critic cho phương pháp học Advantage Actor-Critic (A2C/A3C)
    """
    def __init__(self, state_size: int, action_dims: List[int], 
                 hidden_layers: List[int] = [128, 128], seed: int = 42):
        """
        Khởi tạo mạng Actor-Critic.

        Args:
            state_size: Số chiều của không gian trạng thái
            action_dims: Danh sách kích thước cho mỗi chiều của không gian hành động
            hidden_layers: Kích thước các lớp ẩn
            seed: Seed cho việc tạo ngẫu nhiên
        """
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_dims = action_dims
        
        # Lớp đặc trưng chung
        layers = []
        in_size = state_size
        
        for hidden_size in hidden_layers[:-1]:  # Tất cả trừ lớp cuối cùng
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Lớp Actor - tạo chính sách (policy)
        self.actor_hidden = nn.Sequential(
            nn.Linear(in_size, hidden_layers[-1]),
            nn.ReLU()
        )
        
        # Đầu ra chính sách cho mỗi chiều hành động
        self.actor_heads = nn.ModuleList()
        for action_dim in action_dims:
            self.actor_heads.append(nn.Sequential(
                nn.Linear(hidden_layers[-1], action_dim),
                nn.Softmax(dim=1)
            ))
        
        # Lớp Critic - đánh giá giá trị trạng thái
        self.critic = nn.Sequential(
            nn.Linear(in_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Truyền xuôi qua mạng.

        Args:
            state: Trạng thái đầu vào, shape (batch_size, state_size)

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
                - Danh sách các xác suất hành động cho mỗi chiều
                - Giá trị trạng thái
        """
        shared_features = self.shared_layers(state)
        
        # Actor - tính xác suất hành động
        actor_features = self.actor_hidden(shared_features)
        action_probs = [head(actor_features) for head in self.actor_heads]
        
        # Critic - tính giá trị trạng thái
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class TrustNetwork(nn.Module):
    """
    Mạng neural đánh giá mức độ tin cậy của các node trong blockchain
    """
    def __init__(self, input_size: int, hidden_layers: List[int] = [64, 32], seed: int = 42):
        """
        Khởi tạo mạng đánh giá tin cậy.

        Args:
            input_size: Số chiều của đầu vào (thường là các đặc trưng của node)
            hidden_layers: Kích thước các lớp ẩn
            seed: Seed cho việc tạo ngẫu nhiên
        """
        super(TrustNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Xây dựng mạng
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        # Lớp đầu ra - điểm tin cậy trong khoảng [0, 1]
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Truyền xuôi qua mạng.

        Args:
            x: Đặc trưng đầu vào, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Điểm tin cậy, shape (batch_size, 1)
        """
        return self.model(x)
        
    def calculate_trust(self, features: torch.Tensor) -> torch.Tensor:
        """
        Tính điểm tin cậy dựa trên các đặc trưng.

        Args:
            features: Đặc trưng của node, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Điểm tin cậy, shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            trust_scores = self.forward(features)
        return trust_scores 