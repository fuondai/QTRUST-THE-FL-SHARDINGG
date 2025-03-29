import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import random

class ConsensusProtocol:
    """
    Lớp cơ sở cho các giao thức đồng thuận.
    """
    def __init__(self, name: str, latency_factor: float, energy_factor: float, security_factor: float):
        """
        Khởi tạo giao thức đồng thuận.
        
        Args:
            name: Tên giao thức
            latency_factor: Hệ số độ trễ (1.0 = độ trễ cơ sở)
            energy_factor: Hệ số tiêu thụ năng lượng (1.0 = năng lượng cơ sở)
            security_factor: Hệ số bảo mật (1.0 = bảo mật tối đa)
        """
        self.name = name
        self.latency_factor = latency_factor
        self.energy_factor = energy_factor
        self.security_factor = security_factor
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Thực hiện giao thức đồng thuận trên một giao dịch.
        
        Args:
            transaction_value: Giá trị của giao dịch
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")

class FastBFT(ConsensusProtocol):
    """
    Fast Byzantine Fault Tolerance - Giao thức đồng thuận nhanh với độ bảo mật thấp hơn.
    Thích hợp cho các giao dịch có giá trị thấp và yêu cầu xử lý nhanh.
    """
    def __init__(self, latency_factor: float = 0.2, energy_factor: float = 0.3, security_factor: float = 0.7):
        super().__init__(
            name="FastBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Thực hiện Fast BFT.
        
        Args:
            transaction_value: Giá trị của giao dịch
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Tính điểm tin cậy trung bình của các node tham gia
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Xác suất thành công dựa trên điểm tin cậy trung bình
        success_prob = min(0.98, avg_trust * 0.9 + 0.1)
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = bool(random.random() < success_prob)
        
        # Tính độ trễ
        latency = self.latency_factor * (5.0 + 0.1 * transaction_value)
        
        # Tính năng lượng tiêu thụ
        energy = self.energy_factor * (10.0 + 0.2 * transaction_value)
        
        return consensus_achieved, latency, energy

class PBFT(ConsensusProtocol):
    """
    Practical Byzantine Fault Tolerance - Giao thức đồng thuận cân bằng giữa hiệu suất và bảo mật.
    Phù hợp với hầu hết các giao dịch thông thường.
    """
    def __init__(self, latency_factor: float = 0.5, energy_factor: float = 0.6, security_factor: float = 0.85):
        super().__init__(
            name="PBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Thực hiện PBFT.
        
        Args:
            transaction_value: Giá trị của giao dịch
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Tính điểm tin cậy trung bình của các node tham gia
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Xác suất thành công dựa trên điểm tin cậy trung bình
        success_prob = min(0.99, avg_trust * 0.95 + 0.05)
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = bool(random.random() < success_prob)
        
        # Tính độ trễ
        latency = self.latency_factor * (10.0 + 0.2 * transaction_value)
        
        # Tính năng lượng tiêu thụ
        energy = self.energy_factor * (20.0 + 0.4 * transaction_value)
        
        return consensus_achieved, latency, energy

class RobustBFT(ConsensusProtocol):
    """
    Robust Byzantine Fault Tolerance - Giao thức đồng thuận với độ bảo mật cao nhất.
    Thích hợp cho các giao dịch có giá trị cao và yêu cầu bảo mật tối đa.
    """
    def __init__(self, latency_factor: float = 0.8, energy_factor: float = 0.8, security_factor: float = 0.95):
        super().__init__(
            name="RobustBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Thực hiện Robust BFT.
        
        Args:
            transaction_value: Giá trị của giao dịch
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Tính điểm tin cậy trung bình của các node tham gia
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Xác suất thành công dựa trên điểm tin cậy trung bình
        success_prob = min(0.995, avg_trust * 0.98 + 0.02)
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = bool(random.random() < success_prob)
        
        # Tính độ trễ
        latency = self.latency_factor * (20.0 + 0.3 * transaction_value)
        
        # Tính năng lượng tiêu thụ
        energy = self.energy_factor * (30.0 + 0.5 * transaction_value)
        
        return consensus_achieved, latency, energy

class AdaptiveConsensus:
    """
    Lớp chọn giao thức đồng thuận thích ứng dựa trên các yếu tố khác nhau.
    """
    def __init__(self, 
                transaction_threshold_low: float = 10.0, 
                transaction_threshold_high: float = 50.0,
                congestion_threshold: float = 0.7,
                min_trust_threshold: float = 0.3):
        """
        Khởi tạo AdaptiveConsensus.
        
        Args:
            transaction_threshold_low: Ngưỡng giá trị giao dịch thấp
            transaction_threshold_high: Ngưỡng giá trị giao dịch cao
            congestion_threshold: Ngưỡng tắc nghẽn
            min_trust_threshold: Ngưỡng tin cậy tối thiểu
        """
        self.transaction_threshold_low = transaction_threshold_low
        self.transaction_threshold_high = transaction_threshold_high
        self.congestion_threshold = congestion_threshold
        self.min_trust_threshold = min_trust_threshold
        
        # Khởi tạo các giao thức đồng thuận
        self.fast_bft = FastBFT(0.2, 0.3, 0.7)
        self.pbft = PBFT(0.5, 0.6, 0.85)
        self.robust_bft = RobustBFT(0.8, 0.8, 0.95)
        
        # Trạng thái hiện tại
        self.current_protocol = "PBFT"  # Mặc định
        self.shard_protocols = {}  # Ánh xạ từ shard_id đến giao thức
    
    def update_consensus_mechanism(self, congestion_levels: Dict[int, float], trust_scores: Dict[int, float]) -> Dict[str, Any]:
        """
        Cập nhật cơ chế đồng thuận dựa trên mức độ tắc nghẽn và điểm tin cậy.
        
        Args:
            congestion_levels: Dict ánh xạ từ shard_id đến mức độ tắc nghẽn
            trust_scores: Dict ánh xạ từ node_id đến điểm tin cậy
            
        Returns:
            Dict[str, Any]: Các tham số đồng thuận đã cập nhật
        """
        # Tính điểm tin cậy trung bình
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Cập nhật giao thức cho từng shard
        for shard_id, congestion in congestion_levels.items():
            # Chọn giao thức dựa trên tắc nghẽn và tin cậy
            if avg_trust < self.min_trust_threshold:
                # Nếu tin cậy thấp, ưu tiên bảo mật bằng RobustBFT
                protocol = "RobustBFT"
            elif congestion > self.congestion_threshold:
                # Nếu tắc nghẽn cao, ưu tiên hiệu suất bằng FastBFT
                protocol = "FastBFT"
            else:
                # Mặc định là PBFT
                protocol = "PBFT"
            
            # Cập nhật protocol cho shard
            self.shard_protocols[shard_id] = protocol
        
        # Trả về các tham số đồng thuận
        consensus_params = {
            'shard_protocols': self.shard_protocols,
            'avg_trust': avg_trust,
            'fast_bft_params': {
                'latency_factor': self.fast_bft.latency_factor,
                'energy_factor': self.fast_bft.energy_factor,
                'security_factor': self.fast_bft.security_factor
            },
            'pbft_params': {
                'latency_factor': self.pbft.latency_factor,
                'energy_factor': self.pbft.energy_factor,
                'security_factor': self.pbft.security_factor
            },
            'robust_bft_params': {
                'latency_factor': self.robust_bft.latency_factor,
                'energy_factor': self.robust_bft.energy_factor,
                'security_factor': self.robust_bft.security_factor
            }
        }
        
        return consensus_params
    
    def select_protocol(self, transaction_value: float, congestion: float, trust_scores: Dict[int, float]) -> ConsensusProtocol:
        """
        Chọn giao thức đồng thuận dựa trên các yếu tố.
        
        Args:
            transaction_value: Giá trị giao dịch
            congestion: Mức độ tắc nghẽn (0.0-1.0)
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            ConsensusProtocol: Giao thức đồng thuận được chọn
        """
        # Tính điểm tin cậy trung bình
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.0
        
        # Nếu tin cậy thấp, ưu tiên bảo mật bằng RobustBFT
        if avg_trust < self.min_trust_threshold:
            return self.robust_bft
        
        # Nếu tắc nghẽn cao, ưu tiên hiệu suất bằng FastBFT
        if congestion > self.congestion_threshold:
            return self.fast_bft
        
        # Dựa vào giá trị giao dịch
        if transaction_value < self.transaction_threshold_low:
            return self.fast_bft
        elif transaction_value > self.transaction_threshold_high:
            return self.robust_bft
        else:
            return self.pbft
    
    def execute_consensus(self, transaction_value: float, congestion: float, trust_scores: Dict[int, float]) -> Tuple[bool, str, float, float]:
        """
        Thực hiện đồng thuận với giao thức được chọn.
        
        Args:
            transaction_value: Giá trị giao dịch
            congestion: Mức độ tắc nghẽn
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, str, float, float]: (Kết quả, tên giao thức, độ trễ, năng lượng)
        """
        # Chọn giao thức thích hợp
        protocol = self.select_protocol(transaction_value, congestion, trust_scores)
        
        # Thực hiện đồng thuận
        result, latency, energy = protocol.execute(transaction_value, trust_scores)
        
        # Đảm bảo result là boolean
        result = bool(result)
        
        return result, protocol.name, latency, energy
    
    def get_protocol_factors(self, protocol_name: str) -> Tuple[float, float, float]:
        """
        Lấy các hệ số của giao thức.
        
        Args:
            protocol_name: Tên của giao thức
            
        Returns:
            Tuple[float, float, float]: (Hệ số độ trễ, hệ số năng lượng, hệ số bảo mật)
            
        Raises:
            ValueError: Nếu protocol_name không tồn tại
        """
        protocol_map = {
            "FastBFT": self.fast_bft,
            "PBFT": self.pbft,
            "RobustBFT": self.robust_bft
        }
        
        if protocol_name in protocol_map:
            protocol = protocol_map[protocol_name]
            return protocol.latency_factor, protocol.energy_factor, protocol.security_factor
        else:
            # Nếu không tìm thấy giao thức, ném ngoại lệ
            raise ValueError(f"Không tìm thấy giao thức '{protocol_name}'")