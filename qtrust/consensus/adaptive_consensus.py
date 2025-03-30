import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import random
from .bls_signatures import BLSBasedConsensus
from .adaptive_pos import AdaptivePoSManager
from qtrust.consensus.lightweight_crypto import AdaptiveCryptoManager

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

class LightBFT(ConsensusProtocol):
    """
    Light Byzantine Fault Tolerance - Giao thức đồng thuận nhẹ cho mạng ổn định.
    Giảm thiểu tiêu thụ năng lượng và độ trễ cho các mạng có độ ổn định cao.
    """
    def __init__(self, latency_factor: float = 0.15, energy_factor: float = 0.2, security_factor: float = 0.75):
        super().__init__(
            name="LightBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Thực hiện Light BFT.
        
        Args:
            transaction_value: Giá trị của giao dịch
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Tính điểm tin cậy trung bình của các node tham gia
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # LightBFT chỉ nên được sử dụng trong môi trường tin cậy cao
        # Nếu tin cậy thấp, xác suất thành công sẽ giảm đáng kể
        if avg_trust < 0.7:
            success_prob = 0.6 + avg_trust * 0.3  # Giảm xác suất thành công nếu tin cậy thấp
        else:
            # Xác suất thành công cao cho môi trường tin cậy cao
            success_prob = min(0.99, 0.9 + avg_trust * 0.09)
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = bool(random.random() < success_prob)
        
        # Tính độ trễ - LightBFT có độ trễ thấp hơn FastBFT
        latency = self.latency_factor * (4.0 + 0.05 * transaction_value)
        
        # Tính năng lượng tiêu thụ - LightBFT tiêu thụ ít năng lượng hơn FastBFT
        energy = self.energy_factor * (8.0 + 0.1 * transaction_value)
        
        return consensus_achieved, latency, energy

class AdaptiveConsensus:
    """
    Lớp chọn giao thức đồng thuận thích ứng dựa trên các yếu tố khác nhau.
    """
    def __init__(self, 
                transaction_threshold_low: float = 10.0, 
                transaction_threshold_high: float = 50.0,
                congestion_threshold: float = 0.7,
                min_trust_threshold: float = 0.3,
                transaction_history_size: int = 100,
                network_stability_weight: float = 0.3,
                transaction_value_weight: float = 0.3,
                congestion_weight: float = 0.2,
                trust_weight: float = 0.2,
                high_stability_threshold: float = 0.8,
                high_trust_threshold: float = 0.7,
                enable_bls: bool = True,
                num_validators_per_shard: int = 10,
                enable_adaptive_pos: bool = True,
                enable_lightweight_crypto: bool = True,
                active_validator_ratio: float = 0.7,
                rotation_period: int = 50):
        """
        Khởi tạo AdaptiveConsensus.
        
        Args:
            transaction_threshold_low: Ngưỡng giá trị giao dịch thấp
            transaction_threshold_high: Ngưỡng giá trị giao dịch cao
            congestion_threshold: Ngưỡng tắc nghẽn
            min_trust_threshold: Ngưỡng tin cậy tối thiểu
            transaction_history_size: Kích thước lịch sử giao dịch để phân tích
            network_stability_weight: Trọng số cho độ ổn định mạng
            transaction_value_weight: Trọng số cho giá trị giao dịch
            congestion_weight: Trọng số cho mức tắc nghẽn
            trust_weight: Trọng số cho độ tin cậy
            high_stability_threshold: Ngưỡng ổn định cao để sử dụng LightBFT
            high_trust_threshold: Ngưỡng tin cậy cao để sử dụng LightBFT
            enable_bls: Có sử dụng BLS signature aggregation hay không
            num_validators_per_shard: Số lượng validator mỗi shard
            enable_adaptive_pos: Có sử dụng Adaptive PoS hay không
            enable_lightweight_crypto: Bật/tắt Lightweight Cryptography
            active_validator_ratio: Tỷ lệ validator hoạt động (0.0-1.0)
            rotation_period: Số vòng trước khi xem xét luân chuyển
        """
        self.transaction_threshold_low = transaction_threshold_low
        self.transaction_threshold_high = transaction_threshold_high
        self.congestion_threshold = congestion_threshold
        self.min_trust_threshold = min_trust_threshold
        self.transaction_history_size = transaction_history_size
        self.high_stability_threshold = high_stability_threshold
        self.high_trust_threshold = high_trust_threshold
        self.enable_bls = enable_bls
        self.num_validators_per_shard = num_validators_per_shard
        self.enable_adaptive_pos = enable_adaptive_pos
        self.enable_lightweight_crypto = enable_lightweight_crypto
        
        # Thiết lập trọng số cho các yếu tố
        self.network_stability_weight = network_stability_weight
        self.transaction_value_weight = transaction_value_weight
        self.congestion_weight = congestion_weight
        self.trust_weight = trust_weight
        
        # Khởi tạo các giao thức đồng thuận
        self.consensus_protocols = {
            "FastBFT": FastBFT(),
            "PBFT": PBFT(),
            "RobustBFT": RobustBFT(),
            "LightBFT": LightBFT()
        }
        
        # Thêm BLS-based Consensus nếu được kích hoạt
        if self.enable_bls:
            self.consensus_protocols["BLS_Consensus"] = BLSBasedConsensus(
                num_validators=self.num_validators_per_shard,
                threshold_percent=0.7,
                latency_factor=0.4,
                energy_factor=0.5,
                security_factor=0.9
            )
        
        # Khởi tạo Adaptive PoS Manager nếu được kích hoạt
        self.pos_managers = {}
        if self.enable_adaptive_pos:
            # Tạo một PoS Manager cho mỗi shard
            for shard_id in range(10):  # Giả định tối đa 10 shard
                self.pos_managers[shard_id] = AdaptivePoSManager(
                    num_validators=num_validators_per_shard,
                    active_validator_ratio=active_validator_ratio,
                    rotation_period=rotation_period,
                    min_stake=10.0,
                    energy_threshold=30.0,
                    performance_threshold=0.3,
                    seed=42 + shard_id  # Seed khác nhau cho mỗi shard
                )
        
        # Danh sách lịch sử hiệu suất giao thức
        self.protocol_performance = {}
        for name in self.consensus_protocols.keys():
            self.protocol_performance[name] = {
                "total_count": 10,       # Số lượng giao dịch tổng cộng
                "success_count": 8,      # Số lượng giao dịch thành công
                "latency_sum": 500.0,    # Tổng độ trễ (ms)
                "energy_sum": 250.0      # Tổng năng lượng tiêu thụ
            }
        
        # Danh sách lịch sử giao dịch
        self.transaction_history = []
        
        # Ánh xạ giao thức cho từng shard
        self.shard_protocols = {}
        
        # Thống kê sử dụng giao thức
        self.protocol_usage = {name: 0 for name in self.consensus_protocols.keys()}
        
        # Thống kê energy tiết kiệm được nhờ Adaptive PoS
        self.total_energy_saved = 0.0
        self.total_rotations = 0
        
        # Khởi tạo crypto manager
        if enable_lightweight_crypto:
            self.crypto_manager = AdaptiveCryptoManager()
        else:
            self.crypto_manager = None
        
        # Thống kê tối ưu năng lượng
        self.energy_optimization_stats = {
            "total_energy_saved_crypto": 0.0,
            "total_operations": 0,
            "security_level_distribution": {"low": 0, "medium": 0, "high": 0}
        }
    
    def update_consensus_mechanism(self, congestion_levels: Dict[int, float], trust_scores: Dict[int, float], 
                              network_stability: float = 0.5, cross_shard_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Cập nhật cơ chế đồng thuận cho mỗi shard dựa trên điều kiện mạng hiện tại.
        
        Args:
            congestion_levels: Dict ánh xạ từ shard ID sang mức độ tắc nghẽn
            trust_scores: Dict ánh xạ từ node ID sang điểm tin cậy
            network_stability: Độ ổn định mạng tổng thể (0-1)
            cross_shard_ratio: Tỉ lệ giao dịch xuyên shard
            
        Returns:
            Dict[str, Any]: Thông tin về việc cập nhật cơ chế đồng thuận
        """
        protocol_assignments = {}
        changes_made = 0
        
        # Phân tích hiệu suất của các giao thức đồng thuận hiện tại
        protocol_metrics = self._analyze_protocol_performance()
        
        # Tính toán điểm tin cậy trung bình cho mỗi shard
        shard_trust_scores = {}
        for node_id, trust in trust_scores.items():
            shard_id = node_id // self.num_validators_per_shard
            if shard_id not in shard_trust_scores:
                shard_trust_scores[shard_id] = []
            shard_trust_scores[shard_id].append(trust)
        
        avg_shard_trust = {shard_id: sum(scores)/len(scores) if scores else 0.5 
                          for shard_id, scores in shard_trust_scores.items()}
        
        # Lựa chọn giao thức phù hợp nhất cho mỗi shard
        for shard_id, congestion in congestion_levels.items():
            # Lấy điểm tin cậy trung bình của shard
            trust = avg_shard_trust.get(shard_id, 0.5)
            
            # Xác định độ quan trọng của shard dựa trên cross-shard ratio
            is_important_shard = congestion > 0.5 or cross_shard_ratio > 0.4
            
            # Lựa chọn giao thức phù hợp nhất
            if network_stability > self.high_stability_threshold and trust > self.high_trust_threshold:
                # Môi trường ổn định và tin cậy cao -> sử dụng giao thức nhẹ
                if "LightBFT" in self.consensus_protocols:
                    selected_protocol = "LightBFT"
                else:
                    selected_protocol = "FastBFT"
            elif is_important_shard and trust < self.min_trust_threshold:
                # Shard quan trọng nhưng tin cậy thấp -> sử dụng giao thức mạnh nhất
                selected_protocol = "RobustBFT"
            elif congestion > self.congestion_threshold:
                # Môi trường tắc nghẽn cao -> sử dụng BLS nếu được bật hoặc FastBFT
                if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
                    selected_protocol = "BLS_Consensus"
                else:
                    selected_protocol = "FastBFT"
            elif cross_shard_ratio > 0.4:
                # Tỉ lệ giao dịch xuyên shard cao -> sử dụng BLS nếu có
                if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
                    selected_protocol = "BLS_Consensus"
                else:
                    selected_protocol = "PBFT"
            else:
                # Trường hợp thông thường -> sử dụng PBFT
                selected_protocol = "PBFT"
            
            # Kiểm tra xem có thay đổi so với giao thức hiện tại không
            current_protocol = self.shard_protocols.get(shard_id, "PBFT")
            if current_protocol != selected_protocol:
                changes_made += 1
            
            # Cập nhật giao thức cho shard
            self.shard_protocols[shard_id] = selected_protocol
            protocol_assignments[shard_id] = {
                "protocol": selected_protocol,
                "congestion": congestion,
                "trust": trust,
                "changed": current_protocol != selected_protocol
            }
            
            # Cập nhật thống kê sử dụng
            self.protocol_usage[selected_protocol] = self.protocol_usage.get(selected_protocol, 0) + 1
        
        # Thống kê sử dụng các giao thức
        protocol_distribution = {name: count/sum(self.protocol_usage.values()) 
                               for name, count in self.protocol_usage.items() if count > 0}
        
        return {
            "assignments": protocol_assignments,
            "changes_made": changes_made,
            "protocol_distribution": protocol_distribution,
            "protocol_metrics": protocol_metrics,
            "bls_enabled": self.enable_bls and "BLS_Consensus" in self.consensus_protocols
        }
    
    def _analyze_protocol_performance(self) -> Dict[str, float]:
        """
        Phân tích hiệu suất của các giao thức dựa trên lịch sử.
        
        Returns:
            Dict[str, float]: Điểm hiệu suất của mỗi giao thức (0.0-1.0)
        """
        scores = {}
        
        for protocol, stats in self.protocol_performance.items():
            if stats["total_count"] == 0:
                scores[protocol] = 0.33  # Điểm mặc định
                continue
                
            # Tính tỷ lệ thành công
            success_rate = stats["success_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # Tính độ trễ trung bình và năng lượng trung bình
            avg_latency = stats["latency_sum"] / stats["total_count"] if stats["total_count"] > 0 else 0
            avg_energy = stats["energy_sum"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # Chuẩn hóa độ trễ và năng lượng (giá trị thấp hơn tốt hơn)
            # Giả sử giá trị tối đa là 100ms và 100mJ
            norm_latency = 1.0 - min(1.0, avg_latency / 100.0)
            norm_energy = 1.0 - min(1.0, avg_energy / 100.0)
            
            # Tính điểm tổng hợp
            # Trọng số: success_rate (0.5), latency (0.3), energy (0.2)
            scores[protocol] = 0.5 * success_rate + 0.3 * norm_latency + 0.2 * norm_energy
        
        return scores
    
    def select_protocol(self, transaction_value: float, congestion: float, trust_scores: Dict[int, float],
                      network_stability: float = 0.5, cross_shard: bool = False) -> ConsensusProtocol:
        """
        Chọn giao thức đồng thuận phù hợp nhất dựa trên các yếu tố.
        
        Args:
            transaction_value: Giá trị giao dịch
            congestion: Mức độ tắc nghẽn (0-1)
            trust_scores: Điểm tin cậy của các node
            network_stability: Độ ổn định mạng (0-1)
            cross_shard: Có phải giao dịch xuyên shard không
            
        Returns:
            ConsensusProtocol: Giao thức đồng thuận được chọn
        """
        # Tính toán điểm số cho mỗi giao thức
        protocol_scores = {}
        
        # Tính điểm tin cậy trung bình
        avg_trust = sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0.5
        
        # Kiểm tra các điều kiện để lựa chọn giao thức
        for name, protocol in self.consensus_protocols.items():
            # Bắt đầu với điểm cơ bản
            score = 5.0
            
            # Điều chỉnh dựa trên giá trị giao dịch
            if transaction_value <= self.transaction_threshold_low:
                # Giao dịch giá trị thấp ưu tiên tốc độ
                if name in ["FastBFT", "LightBFT"]:
                    score += 2.0
                elif name == "BLS_Consensus":
                    score += 1.5
                elif name == "PBFT":
                    score += 1.0
                # RobustBFT không phù hợp với giao dịch giá trị thấp
            elif transaction_value >= self.transaction_threshold_high:
                # Giao dịch giá trị cao ưu tiên bảo mật
                if name == "RobustBFT":
                    score += 2.5
                elif name == "PBFT":
                    score += 1.5
                elif name == "BLS_Consensus":
                    score += 1.0
                # FastBFT/LightBFT không phù hợp với giao dịch giá trị cao
            else:
                # Giao dịch giá trị trung bình
                if name == "PBFT":
                    score += 1.5
                elif name == "BLS_Consensus":
                    score += 1.2
                else:
                    score += 0.8
                
            # Điều chỉnh dựa trên mức độ tắc nghẽn
            if congestion > self.congestion_threshold:
                # Tắc nghẽn cao ưu tiên hiệu suất
                if name in ["FastBFT", "BLS_Consensus"]:
                    score += 1.5
                elif name == "LightBFT":
                    score += 1.0
                # Các giao thức khác kém hiệu quả trong tắc nghẽn cao
            else:
                # Tắc nghẽn thấp, cân nhắc bảo mật hơn
                if name == "RobustBFT":
                    score += 0.8
                elif name == "PBFT":
                    score += 0.5
                    
            # Điều chỉnh dựa trên độ tin cậy
            if avg_trust < self.min_trust_threshold:
                # Tin cậy thấp ưu tiên bảo mật
                if name == "RobustBFT":
                    score += 2.0
                elif name == "PBFT":
                    score += 1.0
                # Các giao thức khác kém an toàn trong môi trường tin cậy thấp
            elif avg_trust > self.high_trust_threshold:
                # Tin cậy cao cho phép dùng giao thức nhẹ hơn
                if name == "LightBFT":
                    score += 1.5
                elif name == "FastBFT":
                    score += 1.0
                elif name == "BLS_Consensus":
                    score += 0.8
                    
            # Điều chỉnh dựa trên độ ổn định mạng
            if network_stability > self.high_stability_threshold:
                # Mạng ổn định ưu tiên hiệu suất
                if name == "LightBFT":
                    score += 1.5
                elif name in ["FastBFT", "BLS_Consensus"]:
                    score += 1.0
            elif network_stability < 0.3:
                # Mạng không ổn định ưu tiên bảo mật
                if name == "RobustBFT":
                    score += 1.5
                elif name == "PBFT":
                    score += 0.8
                    
            # Điều chỉnh cho giao dịch xuyên shard
            if cross_shard:
                # Giao dịch xuyên shard thường phức tạp hơn
                if name == "BLS_Consensus":
                    score += 2.0  # BLS phù hợp nhất cho xuyên shard do giảm overhead
                elif name == "RobustBFT":
                    score += 1.0  # Bảo mật tốt cho xuyên shard
                elif name == "PBFT":
                    score += 0.5  # Cân bằng cho xuyên shard
                # FastBFT/LightBFT có thể không đủ bảo mật cho xuyên shard
                    
            # Lưu điểm số
            protocol_scores[name] = score
        
        # Chọn giao thức có điểm cao nhất
        selected_protocol_name = max(protocol_scores.items(), key=lambda x: x[1])[0]
        selected_protocol = self.consensus_protocols[selected_protocol_name]
        
        # Cập nhật thống kê sử dụng
        self.protocol_usage[selected_protocol_name] = self.protocol_usage.get(selected_protocol_name, 0) + 1
        
        return selected_protocol
    
    def execute_consensus(self, transaction_value: float, congestion: float, trust_scores: Dict[int, float],
                         network_stability: float = 0.5, cross_shard: bool = False, 
                         shard_id: int = 0) -> Tuple[bool, str, float, float]:
        """
        Thực hiện đồng thuận với giao thức được chọn.
        
        Args:
            transaction_value: Giá trị giao dịch
            congestion: Mức độ tắc nghẽn
            trust_scores: Điểm tin cậy của các node
            network_stability: Mức độ ổn định của mạng
            cross_shard: Là giao dịch xuyên shard hay không
            shard_id: ID của shard đang xử lý giao dịch
            
        Returns:
            Tuple[bool, str, float, float]: (Kết quả, tên giao thức, độ trễ, năng lượng)
        """
        # Chọn giao thức thích hợp
        protocol = self.select_protocol(transaction_value, congestion, trust_scores, network_stability, cross_shard)
        
        # Chọn validator từ AdaptivePoS (nếu được bật)
        selected_validator = None
        pos_result = None
        
        if self.enable_adaptive_pos and shard_id in self.pos_managers:
            pos_manager = self.pos_managers[shard_id]
            
            # Chọn validator để thực hiện đồng thuận
            selected_validator = pos_manager.select_validator_for_block(trust_scores)
            
            # Mô phỏng một round của PoS
            pos_result = pos_manager.simulate_round(trust_scores, transaction_value)
            
            # Cập nhật thống kê tiết kiệm năng lượng
            if pos_result['rotations'] > 0:
                self.total_rotations += pos_result['rotations']
                self.total_energy_saved += pos_result['energy_saved']
        
        # Thực hiện đồng thuận với lightweight cryptography (nếu được bật)
        if self.enable_lightweight_crypto and self.crypto_manager is not None:
            # Lấy mức năng lượng của validator để cân nhắc chọn mức bảo mật
            remaining_energy = 100.0
            if selected_validator is not None and self.enable_adaptive_pos:
                remaining_energy = self.pos_managers[shard_id].validators[selected_validator].current_energy
            
            # Xác định mức độ quan trọng của giao dịch
            is_critical = cross_shard or transaction_value > self.transaction_threshold_high
            
            # Áp dụng lightweight cryptography (mô phỏng thông qua hash message)
            message = f"tx_{transaction_value}_{time.time()}"
            crypto_params = {"message": message}
            
            # Thực hiện hash với mức bảo mật thích ứng
            crypto_result = self.crypto_manager.execute_crypto_operation(
                "hash", crypto_params, transaction_value, congestion, 
                remaining_energy, is_critical)
            
            # Cập nhật thống kê
            self.energy_optimization_stats["total_energy_saved_crypto"] += crypto_result["energy_saved"]
            self.energy_optimization_stats["total_operations"] += 1
            self.energy_optimization_stats["security_level_distribution"][crypto_result["security_level"]] += 1
            
            # Điều chỉnh năng lượng tiêu thụ của giao thức dựa trên kết quả lightweight crypto
            energy_adjustment_factor = 0.7 if crypto_result["security_level"] == "low" else \
                                      0.85 if crypto_result["security_level"] == "medium" else 1.0
        else:
            energy_adjustment_factor = 1.0  # Không có điều chỉnh
            
        # Thực hiện đồng thuận
        result, latency, energy = protocol.execute(transaction_value, trust_scores)
        
        # Áp dụng điều chỉnh năng lượng nếu sử dụng lightweight crypto
        energy = energy * energy_adjustment_factor
        
        # Đảm bảo result là boolean
        result = bool(result)
        
        # Cập nhật thống kê hiệu suất của giao thức
        self._update_protocol_performance(protocol.name, result, latency, energy)
        
        # Cập nhật năng lượng của validator (nếu có)
        if selected_validator is not None and self.enable_adaptive_pos:
            pos_manager.update_validator_energy(selected_validator, energy, result)
        
        return result, protocol.name, latency, energy
    
    def _update_protocol_performance(self, protocol_name: str, success: bool, latency: float, energy: float):
        """
        Cập nhật thống kê hiệu suất của giao thức.
        
        Args:
            protocol_name: Tên của giao thức
            success: Giao dịch có thành công không
            latency: Độ trễ của giao dịch
            energy: Năng lượng tiêu thụ của giao dịch
        """
        if protocol_name in self.protocol_performance:
            stats = self.protocol_performance[protocol_name]
            stats["total_count"] += 1
            if success:
                stats["success_count"] += 1
            stats["latency_sum"] += latency
            stats["energy_sum"] += energy
            
            # Giữ thống kê trong khoảng lịch sử gần đây
            if stats["total_count"] > self.transaction_history_size:
                # Giảm tất cả các giá trị theo tỷ lệ
                ratio = self.transaction_history_size / stats["total_count"]
                stats["total_count"] = self.transaction_history_size
                stats["success_count"] = int(stats["success_count"] * ratio)
                stats["latency_sum"] *= ratio
                stats["energy_sum"] *= ratio
    
    def get_protocol_factors(self, protocol_name: str) -> Tuple[float, float, float]:
        """
        Lấy các hệ số hiệu suất của một giao thức.
        
        Args:
            protocol_name: Tên giao thức
            
        Returns:
            Tuple[float, float, float]: (latency_factor, energy_factor, security_factor)
        """
        if protocol_name in self.consensus_protocols:
            protocol = self.consensus_protocols[protocol_name]
            return protocol.latency_factor, protocol.energy_factor, protocol.security_factor
        else:
            # Giá trị mặc định nếu không tìm thấy giao thức
            return 0.5, 0.6, 0.8
    
    def get_bls_metrics(self) -> Dict[str, float]:
        """
        Lấy các số liệu hiệu suất của BLS signature aggregation.
        
        Returns:
            Dict[str, float]: Các số liệu hiệu suất của BLS hoặc None nếu không được kích hoạt
        """
        if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
            return self.consensus_protocols["BLS_Consensus"].get_performance_metrics()
        return None
    
    def get_pos_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về hiệu suất của Adaptive PoS.
        
        Returns:
            Dict[str, Any]: Thống kê về Adaptive PoS
        """
        if not self.enable_adaptive_pos:
            return {"enabled": False}
        
        result = {
            "enabled": True,
            "total_energy_saved": self.total_energy_saved,
            "total_rotations": self.total_rotations,
            "shard_stats": {}
        }
        
        # Thu thập thống kê từ mỗi shard
        for shard_id, pos_manager in self.pos_managers.items():
            result["shard_stats"][shard_id] = {
                "energy": pos_manager.get_energy_statistics(),
                "validators": pos_manager.get_validator_statistics()
            }
        
        return result
    
    def select_committee_for_shard(self, shard_id: int, committee_size: int, 
                                  trust_scores: Dict[int, float] = None) -> List[int]:
        """
        Chọn ủy ban validator cho một shard sử dụng Adaptive PoS.
        
        Args:
            shard_id: ID của shard
            committee_size: Kích thước ủy ban
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            List[int]: Danh sách ID của các validator được chọn
        """
        if not self.enable_adaptive_pos or shard_id not in self.pos_managers:
            # Nếu không dùng Adaptive PoS, trả về danh sách ngẫu nhiên
            return list(range(1, min(committee_size + 1, self.num_validators_per_shard + 1)))
        
        # Sử dụng Adaptive PoS để chọn ủy ban
        pos_manager = self.pos_managers[shard_id]
        return pos_manager.select_validators_for_committee(committee_size, trust_scores)

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về các cơ chế tối ưu hóa năng lượng.
        
        Returns:
            Dict[str, Any]: Thống kê về tối ưu hóa
        """
        stats = {
            "adaptive_pos": {
                "enabled": self.enable_adaptive_pos,
                "total_energy_saved": self.total_energy_saved,
                "total_rotations": self.total_rotations
            },
            "lightweight_crypto": {
                "enabled": self.enable_lightweight_crypto,
                "total_energy_saved": self.energy_optimization_stats["total_energy_saved_crypto"],
                "total_operations": self.energy_optimization_stats["total_operations"],
                "security_distribution": self.energy_optimization_stats["security_level_distribution"]
            }
        }
        
        # Thêm thống kê chi tiết từ crypto manager nếu được bật
        if self.enable_lightweight_crypto and self.crypto_manager is not None:
            crypto_detailed_stats = self.crypto_manager.get_crypto_statistics()
            stats["lightweight_crypto"]["detailed"] = crypto_detailed_stats
        
        # Thêm thống kê về BLS
        if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
            bls_stats = self.get_bls_metrics()
            stats["bls_signature_aggregation"] = {
                "enabled": self.enable_bls,
                "metrics": bls_stats
            }
        
        # Tính tổng tiết kiệm năng lượng từ tất cả các cơ chế
        total_savings = self.total_energy_saved
        if self.enable_lightweight_crypto:
            total_savings += self.energy_optimization_stats["total_energy_saved_crypto"]
        
        stats["total_energy_saved"] = total_savings
        
        return stats