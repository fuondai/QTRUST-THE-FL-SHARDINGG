import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import copy

class PrivacyManager:
    """
    Quản lý các cơ chế bảo vệ quyền riêng tư trong Federated Learning.
    """
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 clip_norm: float = 1.0,
                 noise_multiplier: float = 1.1):
        """
        Khởi tạo Privacy Manager.
        
        Args:
            epsilon: Tham số privacy budget
            delta: Tham số xác suất thất bại
            clip_norm: Ngưỡng clip gradient
            noise_multiplier: Hệ số nhân nhiễu
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        
        # Theo dõi privacy budget
        self.consumed_budget = 0.0
        self.privacy_metrics = []
    
    def add_noise_to_gradients(self, 
                             gradients: torch.Tensor,
                             num_samples: int) -> torch.Tensor:
        """
        Thêm nhiễu Gaussian vào gradients để bảo vệ quyền riêng tư.
        
        Args:
            gradients: Tensor chứa gradients
            num_samples: Số lượng mẫu trong batch
            
        Returns:
            Tensor: Gradients đã được thêm nhiễu
        """
        # Clip gradients
        total_norm = torch.norm(gradients)
        scale = torch.min(torch.tensor(1.0), self.clip_norm / (total_norm + 1e-10))
        gradients = gradients * scale
        
        # Tính độ lệch chuẩn của nhiễu
        noise_scale = self.clip_norm * self.noise_multiplier / np.sqrt(num_samples)
        
        # Thêm nhiễu Gaussian
        noise = torch.normal(0, noise_scale, size=gradients.shape)
        noisy_gradients = gradients + noise
        
        # Cập nhật privacy budget đã sử dụng
        self._update_privacy_accounting(num_samples)
        
        return noisy_gradients
    
    def add_noise_to_model(self, 
                          model_params: Dict[str, torch.Tensor],
                          num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Thêm nhiễu vào tham số mô hình để bảo vệ quyền riêng tư.
        
        Args:
            model_params: Dictionary chứa tham số mô hình
            num_samples: Số lượng mẫu được sử dụng
            
        Returns:
            Dict: Tham số mô hình đã được thêm nhiễu
        """
        noisy_params = {}
        for key, param in model_params.items():
            noisy_params[key] = self.add_noise_to_gradients(param, num_samples)
        return noisy_params
    
    def _update_privacy_accounting(self, num_samples: int) -> None:
        """
        Cập nhật privacy budget đã sử dụng.
        
        Args:
            num_samples: Số lượng mẫu trong batch
        """
        # Tính privacy loss cho lần cập nhật này
        q = 1.0 / num_samples  # Tỷ lệ sampling
        steps = 1
        
        # Sử dụng RDP accountant để tính privacy loss
        rdp = self._compute_rdp(q, self.noise_multiplier, steps)
        
        # Chuyển đổi RDP sang (ε, δ)-DP
        eps = self._rdp_to_dp(rdp, self.delta)
        
        self.consumed_budget += eps
        
        # Lưu metrics
        self.privacy_metrics.append({
            'epsilon': eps,
            'total_budget': self.consumed_budget,
            'remaining_budget': max(0, self.epsilon - self.consumed_budget),
            'num_samples': num_samples
        })
    
    def _compute_rdp(self, q: float, noise_multiplier: float, steps: int) -> float:
        """
        Tính Rényi Differential Privacy (RDP) cho Gaussian mechanism.
        
        Args:
            q: Tỷ lệ sampling
            noise_multiplier: Hệ số nhân nhiễu
            steps: Số bước
            
        Returns:
            float: Giá trị RDP
        """
        # Tính RDP cho Gaussian mechanism với subsampling
        c = noise_multiplier
        alpha = 10  # Order của RDP
        
        # Tính RDP cho một bước
        rdp_step = q**2 * alpha / (2 * c**2)
        
        # Tính tổng RDP cho tất cả các bước
        return rdp_step * steps
    
    def _rdp_to_dp(self, rdp: float, delta: float) -> float:
        """
        Chuyển đổi RDP sang (ε, δ)-DP.
        
        Args:
            rdp: Giá trị RDP
            delta: Tham số δ mong muốn
            
        Returns:
            float: Giá trị ε tương ứng
        """
        # Chuyển đổi theo công thức từ định lý 3.1 trong paper RDP
        alpha = 10  # Order của RDP
        return rdp + np.sqrt(2 * np.log(1/delta) / alpha)
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Tạo báo cáo về tình trạng privacy.
        
        Returns:
            Dict: Báo cáo chi tiết về privacy
        """
        if not self.privacy_metrics:
            return {
                'status': 'No privacy metrics available',
                'consumed_budget': 0.0,
                'remaining_budget': self.epsilon
            }
            
        latest = self.privacy_metrics[-1]
        return {
            'status': 'Privacy budget exceeded' if self.consumed_budget > self.epsilon else 'Active',
            'consumed_budget': self.consumed_budget,
            'remaining_budget': max(0, self.epsilon - self.consumed_budget),
            'noise_multiplier': self.noise_multiplier,
            'clip_norm': self.clip_norm,
            'last_update': {
                'epsilon': latest['epsilon'],
                'num_samples': latest['num_samples']
            },
            'total_updates': len(self.privacy_metrics)
        }

class SecureAggregator:
    """
    Thực hiện tổng hợp mô hình an toàn với các cơ chế bảo mật.
    """
    def __init__(self, 
                 privacy_manager: PrivacyManager,
                 secure_communication: bool = True,
                 threshold: int = 3):
        """
        Khởi tạo Secure Aggregator.
        
        Args:
            privacy_manager: PrivacyManager để quản lý quyền riêng tư
            secure_communication: Bật/tắt giao tiếp an toàn
            threshold: Số lượng client tối thiểu để tổng hợp
        """
        self.privacy_manager = privacy_manager
        self.secure_communication = secure_communication
        self.threshold = threshold
        
        # Lưu trữ khóa và share
        self.key_shares = {}
        self.masked_models = {}
    
    def generate_key_shares(self, num_clients: int) -> List[bytes]:
        """
        Tạo các phần của khóa cho Secure Aggregation.
        
        Args:
            num_clients: Số lượng client tham gia
            
        Returns:
            List[bytes]: Danh sách các phần khóa
        """
        # TODO: Implement Shamir's Secret Sharing
        pass
    
    def aggregate_secure(self,
                        client_updates: Dict[int, Dict[str, torch.Tensor]],
                        weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình an toàn.
        
        Args:
            client_updates: Dictionary chứa cập nhật từ các client
            weights: Trọng số cho mỗi client
            
        Returns:
            Dict: Tham số mô hình đã được tổng hợp an toàn
        """
        if len(client_updates) < self.threshold:
            raise ValueError(f"Không đủ clients: {len(client_updates)} < {self.threshold}")
            
        # Chuẩn hóa trọng số
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Thêm nhiễu vào mỗi cập nhật client
        noisy_updates = {}
        for client_id, update in client_updates.items():
            num_samples = update.get('num_samples', 1)
            noisy_updates[client_id] = self.privacy_manager.add_noise_to_model(
                update['params'], num_samples
            )
        
        # Tổng hợp các cập nhật đã được bảo vệ
        result = {}
        for key in noisy_updates[list(noisy_updates.keys())[0]].keys():
            result[key] = sum(
                weights[i] * updates[key]
                for i, (_, updates) in enumerate(noisy_updates.items())
            )
            
        return result 