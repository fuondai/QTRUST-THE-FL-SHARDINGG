"""
Các mạng neural sử dụng trong DQN Agent.

File này chứa các mạng neural được sử dụng trong DQN Agent:
- NoisyLinear: Lớp tuyến tính với nhiễu cho việc exploration
- ResidualBlock: Khối phần dư với normalization
- QNetwork: Mạng Q cho DQN với kiến trúc Dueling
- CategoricalQNetwork: Mạng Q cho Distributional DQN
- ActorNetwork: Mạng Actor cho Actor-Critic
- CriticNetwork: Mạng Critic cho Actor-Critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Union

class NoisyLinear(nn.Module):
    """
    Lớp NoisyLinear cho việc exploration hiệu quả.
    Thay thế epsilon-greedy bằng tham số nhiễu.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Khởi tạo lớp NoisyLinear.
        
        Args:
            in_features: Số đầu vào
            out_features: Số đầu ra
            std_init: Độ lệch chuẩn ban đầu cho tham số nhiễu
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Khởi tạo tham số của lớp
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # Khởi tạo tham số
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Khởi tạo giá trị tham số."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Tạo nhiễu theo phân phối factorized Gaussian."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        """Tạo lại nhiễu."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """Lan truyền xuôi."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class ResidualBlock(nn.Module):
    """
    Khối phần dư với normalization để cải thiện khả năng hội tụ.
    """
    def __init__(self, in_features: int, out_features: int, noisy: bool = False, use_layer_norm: bool = False):
        super(ResidualBlock, self).__init__()
        
        # Chọn lớp dựa trên tham số noisy
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Lớp chính
        self.main = nn.Sequential(
            linear_layer(in_features, out_features),
            nn.LayerNorm(out_features) if use_layer_norm else nn.BatchNorm1d(out_features),
            nn.ReLU(),
            linear_layer(out_features, out_features),
            nn.LayerNorm(out_features) if use_layer_norm else nn.BatchNorm1d(out_features)
        )
        
        # Kết nối tắt (shortcut connection)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                linear_layer(in_features, out_features),
                nn.LayerNorm(out_features) if use_layer_norm else nn.BatchNorm1d(out_features)
            )
        
        self.relu = nn.ReLU()
        self.noisy = noisy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi qua khối phần dư."""
        identity = self.shortcut(x)
        out = self.main(x)
        out += identity
        out = self.relu(out)
        return out
    
    def reset_noise(self):
        """Reset noise cho các lớp NoisyLinear trong khối."""
        if not self.noisy:
            return
            
        # Reset noise trong main block
        for module in self.main:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise trong shortcut
        for module in self.shortcut:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class QNetwork(nn.Module):
    """
    Mạng neural cho Deep Q-Network với kết nối phần dư (residual connections).
    """
    def __init__(self, state_size: int, action_dim: List[int], hidden_sizes: List[int] = [256, 256], noisy: bool = False):
        """
        Khởi tạo mạng Q.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_dim: Danh sách các kích thước của không gian hành động
            hidden_sizes: Kích thước các lớp ẩn
            noisy: Sử dụng NoisyLinear thay vì Linear
        """
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_dim_product = np.prod(action_dim)
        self.noisy = noisy
        
        # Chọn lớp dựa trên tham số noisy
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Lớp nhúng đầu vào để mã hóa trạng thái
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        
        # Thay thế BatchNorm1d bằng LayerNorm cho khả năng làm việc với batch size = 1
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Các khối phần dư - cập nhật ResidualBlock bên dưới
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Lớp đầu ra cho từng chiều hành động
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, dim)
            ) for dim in action_dim
        ])
        
        # Đánh giá giá trị trạng thái (V(s))
        self.value_stream = nn.Sequential(
            linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            linear_layer(hidden_sizes[-1] // 2, 1)
        )
        
        # Lớp chú ý (attention) cho các đặc trưng quan trọng
        self.attention = nn.Sequential(
            linear_layer(hidden_sizes[-1], hidden_sizes[-1]),
            nn.Sigmoid()
        )
        
        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Lan truyền xuôi qua mạng.
        
        Args:
            state: Tensor trạng thái đầu vào
            
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
                - Danh sách các Q values cho từng chiều hành động
                - Giá trị trạng thái (V(s))
        """
        # Mã hóa đầu vào - sử dụng LayerNorm thay vì BatchNorm
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Truyền qua các khối phần dư
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Áp dụng cơ chế chú ý
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Áp dụng dropout - tắt khi đánh giá
        if self.training:
            x = self.dropout(x)
        
        # Tính giá trị trạng thái (V(s))
        state_value = self.value_stream(x)
        
        # Tính lợi thế của từng hành động (A(s,a))
        action_advantages = [layer(x) for layer in self.output_layers]
        
        # Kết hợp theo kiến trúc Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        action_values = []
        for advantages in action_advantages:
            # Trừ đi giá trị trung bình để tăng tính ổn định
            values = state_value + advantages - advantages.mean(dim=1, keepdim=True)
            action_values.append(values)
        
        return action_values, state_value
    
    def reset_noise(self):
        """Reset noise cho tất cả các lớp NoisyLinear."""
        if not self.noisy:
            return
            
        # Reset noise trong lớp đầu vào nếu là NoisyLinear
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
        
        # Reset noise trong các khối phần dư
        for res_block in self.res_blocks:
            res_block.reset_noise()
        
        # Reset noise trong các lớp đầu ra
        for layer in self.output_layers:
            for module in layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
        
        # Reset noise trong value stream
        for module in self.value_stream:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise trong lớp attention
        for module in self.attention:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class CategoricalQNetwork(nn.Module):
    """
    Mạng neural cho Distributional DQN (C51) với kết nối phần dư.
    Sử dụng phân phối xác suất thay vì giá trị Q duy nhất.
    """
    def __init__(self, state_size: int, action_dim: List[int], n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0, 
                 hidden_sizes: List[int] = [256, 256], noisy: bool = False):
        """
        Khởi tạo mạng Distributional Q.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_dim: Danh sách các kích thước của không gian hành động
            n_atoms: Số atom trong phân phối
            v_min: Giá trị Q tối thiểu
            v_max: Giá trị Q tối đa
            hidden_sizes: Kích thước các lớp ẩn
            noisy: Sử dụng NoisyLinear thay vì Linear
        """
        super(CategoricalQNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_dim_product = np.prod(action_dim)
        self.noisy = noisy
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Chọn lớp dựa trên tham số noisy
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Lớp nhúng đầu vào để mã hóa trạng thái
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Các khối phần dư
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Đánh giá giá trị trạng thái (V(s))
        self.value_stream = nn.Sequential(
            linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            linear_layer(hidden_sizes[-1] // 2, n_atoms)
        )
        
        # Lớp đầu ra cho từng hành động x số atom
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, dim * n_atoms)
            ) for dim in action_dim
        ])
        
        # Lớp chú ý
        self.attention = nn.Sequential(
            linear_layer(hidden_sizes[-1], hidden_sizes[-1]),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Lan truyền xuôi qua mạng.
        
        Args:
            state: Tensor trạng thái đầu vào
            
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
                - Danh sách các phân phối xác suất cho từng hành động
                - Giá trị trạng thái (V(s))
        """
        batch_size = state.size(0)
        
        # Mã hóa đầu vào - sử dụng LayerNorm
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Truyền qua các khối phần dư
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Áp dụng cơ chế chú ý
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Áp dụng dropout - tắt khi đánh giá
        if self.training:
            x = self.dropout(x)
            
        # Tính giá trị trạng thái (V(s))
        state_value = self.value_stream(x)  # [batch_size, n_atoms]
        
        # Tính lợi thế của từng hành động (A(s,a))
        action_distributions = []
        
        for i, layer in enumerate(self.output_layers):
            advantages = layer(x)  # [batch_size, action_dim * n_atoms]
            advantages = advantages.view(batch_size, self.action_dim[i], self.n_atoms)
            
            # Zero-center advantages within each atom
            advantages = advantages - advantages.mean(dim=1, keepdim=True)
            
            # Combine value and advantages for each atom
            q_atoms = state_value.unsqueeze(1) + advantages
            
            # Apply softmax over atoms to get probabilities
            q_dist = F.softmax(q_atoms, dim=2)
            
            action_distributions.append(q_dist)
            
        return action_distributions, state_value
    
    def reset_noise(self):
        """Reset noise cho tất cả các lớp NoisyLinear."""
        if not self.noisy:
            return
            
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
            
        for res_block in self.res_blocks:
            res_block.reset_noise()
            
        for output_layer in self.output_layers:
            for module in output_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
                    
        for module in self.value_stream:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
        for module in self.attention:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ActorNetwork(nn.Module):
    """
    Mạng Actor cho Actor-Critic architecture.
    Đầu ra là phân phối xác suất trên không gian hành động.
    """
    def __init__(self, state_size: int, action_dim: List[int], 
                 hidden_sizes: List[int] = [256, 256], noisy: bool = False):
        """
        Khởi tạo mạng Actor.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_dim: Danh sách các kích thước của không gian hành động
            hidden_sizes: Kích thước các lớp ẩn
            noisy: Sử dụng NoisyLinear thay vì Linear
        """
        super(ActorNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.noisy = noisy
        
        # Chọn lớp dựa trên tham số noisy
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Lớp nhúng đầu vào
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Các khối phần dư
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Đầu ra policy cho từng chiều hành động
        self.policy_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, dim)
            ) for dim in action_dim
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """
        Lan truyền xuôi qua mạng Actor.
        
        Args:
            state: Tensor trạng thái đầu vào
            
        Returns:
            List[torch.Tensor]: Danh sách các phân phối xác suất hành động
        """
        # Mã hóa đầu vào
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Truyền qua các khối phần dư
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Áp dụng dropout
        if self.training:
            x = self.dropout(x)
        
        # Tính toán phân phối xác suất hành động
        action_probs = []
        for layer in self.policy_layers:
            logits = layer(x)
            # Áp dụng softmax để có phân phối xác suất
            probs = F.softmax(logits, dim=-1)
            action_probs.append(probs)
        
        return action_probs
    
    def reset_noise(self):
        """Reset noise cho tất cả các lớp NoisyLinear."""
        if not self.noisy:
            return
            
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
            
        for res_block in self.res_blocks:
            res_block.reset_noise()
            
        for policy_layer in self.policy_layers:
            for module in policy_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class CriticNetwork(nn.Module):
    """
    Mạng Critic cho Actor-Critic architecture.
    Đánh giá cặp trạng thái-hành động.
    """
    def __init__(self, state_size: int, action_dim: List[int],
                 hidden_sizes: List[int] = [256, 256], noisy: bool = False,
                 n_atoms: int = 1, v_min: float = -10.0, v_max: float = 10.0):
        """
        Khởi tạo mạng Critic.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_dim: Danh sách các kích thước của không gian hành động
            hidden_sizes: Kích thước các lớp ẩn
            noisy: Sử dụng NoisyLinear thay vì Linear
            n_atoms: Số atom trong phân phối
            v_min: Giá trị Q tối thiểu
            v_max: Giá trị Q tối đa
        """
        super(CriticNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.noisy = noisy
        self.distributional = n_atoms > 1
        self.n_atoms = n_atoms
        
        if self.distributional:
            self.v_min = v_min
            self.v_max = v_max
            self.support = torch.linspace(v_min, v_max, n_atoms)
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Chọn lớp dựa trên tham số noisy
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Lớp nhúng đầu vào
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Các khối phần dư
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Đầu ra giá trị Q cho từng cặp trạng thái-hành động
        self.q_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1] + dim, hidden_sizes[-1] // 2),  # Concatenate state features và action
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, n_atoms)
            ) for dim in action_dim
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor, actions: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Lan truyền xuôi qua mạng Critic.
        
        Args:
            state: Tensor trạng thái đầu vào
            actions: Danh sách các tensor hành động đã thực hiện
            
        Returns:
            Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]: 
                - Nếu distributional: (danh sách phân phối xác suất, danh sách giá trị trung bình)
                - Nếu không: danh sách giá trị Q
        """
        # Mã hóa đầu vào
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Truyền qua các khối phần dư
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Áp dụng dropout
        if self.training:
            x = self.dropout(x)
        
        # Tính toán giá trị Q
        q_values = []
        q_distributions = []
        
        for i, (layer, action) in enumerate(zip(self.q_layers, actions)):
            # Kết hợp đặc trưng trạng thái và hành động
            sa_features = torch.cat([x, action], dim=-1)
            q_output = layer(sa_features)
            
            if self.distributional:
                # Áp dụng softmax để có phân phối xác suất
                q_dist = F.softmax(q_output, dim=-1)
                q_distributions.append(q_dist)
                
                # Tính giá trị Q trung bình
                with torch.no_grad():
                    support = self.support.to(state.device)
                    q_val = torch.sum(q_dist * support, dim=-1, keepdim=True)
                q_values.append(q_val)
            else:
                q_values.append(q_output)
        
        if self.distributional:
            return q_distributions, q_values
        else:
            return q_values
    
    def reset_noise(self):
        """Reset noise cho tất cả các lớp NoisyLinear."""
        if not self.noisy:
            return
            
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
            
        for res_block in self.res_blocks:
            res_block.reset_noise()
            
        for q_layer in self.q_layers:
            for module in q_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise() 