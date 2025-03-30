import random
from typing import Dict, List, Tuple, Set, Optional
import hashlib
import time

class BLSSignatureManager:
    """
    Lớp quản lý BLS signature aggregation.
    
    BLS (Boneh-Lynn-Shacham) cho phép tổng hợp nhiều chữ ký thành một chữ ký duy nhất,
    giúp giảm đáng kể kích thước dữ liệu và thời gian xác minh trong các giao thức đồng thuận.
    
    Lưu ý: Đây là phiên bản mô phỏng để dùng trong môi trường thử nghiệm, không sử dụng trong production.
    """
    
    def __init__(self, num_validators: int = 10, threshold: int = 7, seed: int = 42):
        """
        Khởi tạo BLS signature manager.
        
        Args:
            num_validators: Số lượng validator
            threshold: Số lượng validator tối thiểu cần thiết để tạo chữ ký hợp lệ
            seed: Giá trị seed cho tính toán ngẫu nhiên có thể tái tạo
        """
        self.num_validators = num_validators
        self.threshold = threshold
        self.seed = seed
        random.seed(seed)
        
        # Mô phỏng các khóa và ID
        self.validator_ids = list(range(1, num_validators + 1))
        self.validator_keys = {vid: hashlib.sha256(f"key_{vid}_{seed}".encode()).hexdigest() for vid in self.validator_ids}
        
        # Lưu trữ thông tin hiệu suất
        self.verification_times = []
        self.signature_sizes = []
        
    def sign_message(self, message: str, validator_id: int) -> str:
        """
        Mô phỏng quá trình ký một tin nhắn bởi một validator.
        
        Args:
            message: Tin nhắn cần ký
            validator_id: ID của validator thực hiện ký
            
        Returns:
            str: Chữ ký mô phỏng
        """
        if validator_id not in self.validator_ids:
            raise ValueError(f"Validator ID {validator_id} không hợp lệ")
            
        key = self.validator_keys[validator_id]
        # Mô phỏng chữ ký bằng cách kết hợp khóa, ID và tin nhắn
        signature = hashlib.sha256(f"{key}_{message}".encode()).hexdigest()
        
        # Mô phỏng độ trễ ký
        time.sleep(0.001)
        
        return signature
    
    def aggregate_signatures(self, message: str, signatures: Dict[int, str]) -> Tuple[str, int, float]:
        """
        Tổng hợp nhiều chữ ký thành một chữ ký duy nhất.
        
        Args:
            message: Tin nhắn gốc
            signatures: Dict ánh xạ từ validator ID đến chữ ký
            
        Returns:
            Tuple[str, int, float]: (Chữ ký tổng hợp, kích thước, thời gian tổng hợp)
        """
        start_time = time.time()
        
        # Kiểm tra xem có đủ chữ ký không
        if len(signatures) < self.threshold:
            raise ValueError(f"Không đủ chữ ký: {len(signatures)}/{self.threshold}")
        
        # Mô phỏng quá trình tổng hợp
        combined = "_".join([f"{vid}:{sig}" for vid, sig in sorted(signatures.items())])
        aggregated_signature = hashlib.sha256(combined.encode()).hexdigest()
        
        # Tính thời gian và kích thước
        aggregate_time = time.time() - start_time
        
        # Kích thước thực của nhiều chữ ký riêng biệt (giả sử mỗi chữ ký 64 byte)
        original_size = len(signatures) * 64
        
        # Kích thước của chữ ký tổng hợp (64 byte) + thông tin validator (2 byte/validator)
        aggregated_size = 64 + len(signatures) * 2
        
        # Tiết kiệm kích thước
        size_reduction = original_size - aggregated_size
        
        # Lưu thông tin hiệu suất
        self.signature_sizes.append((original_size, aggregated_size))
        
        return aggregated_signature, size_reduction, aggregate_time
    
    def verify_aggregated_signature(self, message: str, aggregated_signature: str, signer_ids: Set[int]) -> Tuple[bool, float]:
        """
        Xác minh chữ ký tổng hợp.
        
        Args:
            message: Tin nhắn gốc
            aggregated_signature: Chữ ký tổng hợp
            signer_ids: Tập hợp ID của các validator đã ký
            
        Returns:
            Tuple[bool, float]: (Kết quả xác minh, thời gian xác minh)
        """
        start_time = time.time()
        
        # Trong mô phỏng, ta giả định chữ ký luôn hợp lệ nếu số lượng người ký >= threshold
        if len(signer_ids) < self.threshold:
            verification_time = time.time() - start_time
            self.verification_times.append(verification_time)
            return False, verification_time
            
        # Mô phỏng thời gian xác minh (tăng theo logarithm của số lượng validator)
        verification_delay = 0.001 * (1 + 0.2 * (len(signer_ids) / self.num_validators))
        time.sleep(verification_delay)
        
        verification_time = time.time() - start_time
        self.verification_times.append(verification_time)
        
        return True, verification_time
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Lấy các số liệu hiệu suất của BLS signature aggregation.
        
        Returns:
            Dict[str, float]: Các số liệu hiệu suất
        """
        if not self.verification_times:
            return {
                "avg_verification_time": 0.0,
                "avg_size_reduction_percent": 0.0,
                "verification_speedup": 0.0
            }
            
        avg_verification_time = sum(self.verification_times) / len(self.verification_times)
        
        # Tính toán trung bình giảm kích thước (%)
        if self.signature_sizes:
            size_reductions = [(original - aggregated) / original * 100 
                               for original, aggregated in self.signature_sizes]
            avg_size_reduction = sum(size_reductions) / len(size_reductions)
        else:
            avg_size_reduction = 0.0
            
        # Ước tính tăng tốc xác minh so với xác minh riêng biệt
        # Giả sử xác minh riêng biệt mất O(n) thời gian, trong khi tổng hợp là O(1)
        verification_speedup = self.num_validators / 2.0  # Giả định
        
        return {
            "avg_verification_time": avg_verification_time,
            "avg_size_reduction_percent": avg_size_reduction,
            "verification_speedup": verification_speedup
        }

class BLSBasedConsensus:
    """
    Giao thức đồng thuận sử dụng BLS signature aggregation.
    """
    
    def __init__(self, 
                 num_validators: int = 10, 
                 threshold_percent: float = 0.7,
                 latency_factor: float = 0.4,
                 energy_factor: float = 0.5,
                 security_factor: float = 0.9,
                 seed: int = 42):
        """
        Khởi tạo giao thức đồng thuận dựa trên BLS.
        
        Args:
            num_validators: Số lượng validator
            threshold_percent: Phần trăm validator cần thiết để đạt đồng thuận
            latency_factor: Hệ số độ trễ
            energy_factor: Hệ số năng lượng
            security_factor: Hệ số bảo mật
            seed: Seed cho tính toán ngẫu nhiên
        """
        self.name = "BLS_Consensus"
        self.latency_factor = latency_factor
        self.energy_factor = energy_factor
        self.security_factor = security_factor
        
        self.num_validators = num_validators
        self.threshold = max(1, int(num_validators * threshold_percent))
        
        # Khởi tạo BLS signature manager
        self.bls_manager = BLSSignatureManager(
            num_validators=num_validators,
            threshold=self.threshold,
            seed=seed
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Thực hiện đồng thuận sử dụng BLS signature aggregation.
        
        Args:
            transaction_value: Giá trị của giao dịch
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        message = f"tx_{random.randint(0, 1000000)}_{transaction_value}"
        
        # Chọn validator dựa trên điểm tin cậy
        selected_validators = set()
        for vid in range(1, self.num_validators + 1):
            trust = trust_scores.get(vid, 0.5)
            if random.random() < trust:
                selected_validators.add(vid)
        
        # Nếu không đủ validator tin cậy, đồng thuận thất bại
        if len(selected_validators) < self.threshold:
            latency = self.latency_factor * (5.0 + 0.1 * transaction_value)
            energy = self.energy_factor * (10.0 + 0.1 * transaction_value)
            return False, latency, energy
        
        # Thu thập chữ ký từ các validator đã chọn
        signatures = {}
        for vid in selected_validators:
            signatures[vid] = self.bls_manager.sign_message(message, vid)
        
        # Tổng hợp chữ ký
        try:
            aggregated_signature, size_reduction, aggregate_time = self.bls_manager.aggregate_signatures(message, signatures)
            
            # Xác minh chữ ký tổng hợp
            consensus_achieved, verification_time = self.bls_manager.verify_aggregated_signature(
                message, aggregated_signature, selected_validators
            )
            
            # Tính độ trễ: thời gian tổng hợp + xác minh
            latency = self.latency_factor * (verification_time + aggregate_time) * 1000  # Chuyển đổi sang ms
            
            # Tính năng lượng tiêu thụ (giả định tiết kiệm năng lượng tỷ lệ với giảm kích thước)
            base_energy = (15.0 + 0.2 * transaction_value)
            energy_reduction_factor = 1.0 - (size_reduction / (self.num_validators * 64)) * 0.5
            energy = self.energy_factor * base_energy * energy_reduction_factor
            
            return consensus_achieved, latency, energy
            
        except ValueError:
            # Nếu không thể tổng hợp chữ ký
            latency = self.latency_factor * (10.0 + 0.2 * transaction_value)
            energy = self.energy_factor * (20.0 + 0.3 * transaction_value)
            return False, latency, energy
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Lấy các số liệu hiệu suất của giao thức.
        
        Returns:
            Dict[str, float]: Các số liệu hiệu suất
        """
        return self.bls_manager.get_performance_metrics() 