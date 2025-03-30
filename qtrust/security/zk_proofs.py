import numpy as np
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

class ProofType(Enum):
    """Các loại bằng chứng ZK được hỗ trợ."""
    TRANSACTION_VALIDITY = "tx_validity"
    OWNERSHIP = "ownership"
    RANGE_PROOF = "range_proof"
    SET_MEMBERSHIP = "set_membership"
    CUSTOM = "custom"

class SecurityLevel(Enum):
    """Các mức độ bảo mật được hỗ trợ."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ZKProofSystem:
    """
    Hệ thống Zero-Knowledge Proofs nhẹ cho blockchain.
    
    Cung cấp các phương thức để tạo và xác minh các bằng chứng zero-knowledge
    với nhiều mức độ bảo mật khác nhau để tối ưu hóa năng lượng.
    """
    
    def __init__(self, security_level: str = "medium", 
                energy_optimization: bool = True,
                verification_speedup: bool = True):
        """
        Khởi tạo hệ thống ZK Proof.
        
        Args:
            security_level: Mức độ bảo mật ("low", "medium", "high")
            energy_optimization: Bật tính năng tối ưu hóa năng lượng
            verification_speedup: Bật tính năng tăng tốc xác minh
        """
        self.security_level = SecurityLevel(security_level)
        self.energy_optimization = energy_optimization
        self.verification_speedup = verification_speedup
        
        # Cấu hình các tham số dựa trên mức độ bảo mật
        self._configure_parameters()
        
        # Thống kê sử dụng
        self.stats = {
            "proofs_generated": 0,
            "proofs_verified": 0,
            "verification_success": 0,
            "verification_failure": 0,
            "energy_saved": 0.0,
            "avg_proof_time": 0.0,
            "avg_verify_time": 0.0,
            "total_proof_time": 0.0,
            "total_verify_time": 0.0
        }
        
        # Cache các bằng chứng/xác minh gần đây để tối ưu
        self.proof_cache = {}
        self.verification_cache = {}
        
    def _configure_parameters(self):
        """Cấu hình các tham số dựa trên mức độ bảo mật."""
        # Số lượng vòng lặp cho các thuật toán
        if self.security_level == SecurityLevel.LOW:
            self.iterations = 8
            self.hash_iterations = 100
            self.prime_bits = 512
            self.base_energy = 10
        elif self.security_level == SecurityLevel.MEDIUM:
            self.iterations = 16
            self.hash_iterations = 1000
            self.prime_bits = 1024
            self.base_energy = 30
        else:  # HIGH
            self.iterations = 32
            self.hash_iterations = 10000
            self.prime_bits = 2048
            self.base_energy = 100
        
        # Điều chỉnh tham số nếu tối ưu hóa năng lượng được bật
        if self.energy_optimization:
            self.iterations = max(4, int(self.iterations * 0.75))
            self.hash_iterations = max(50, int(self.hash_iterations * 0.8))
    
    def generate_proof(self, data: Dict[str, Any], proof_type: ProofType, 
                      custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tạo một bằng chứng zero-knowledge.
        
        Args:
            data: Dữ liệu cần tạo bằng chứng
            proof_type: Loại bằng chứng
            custom_params: Các tham số tùy chỉnh (tùy chọn)
            
        Returns:
            Dict[str, Any]: Bằng chứng được tạo
        """
        start_time = time.time()
        
        # Kiểm tra cache nếu có bằng chứng tương tự gần đây
        cache_key = self._generate_cache_key(data, proof_type)
        if cache_key in self.proof_cache:
            proof = self.proof_cache[cache_key].copy()
            proof["from_cache"] = True
            end_time = time.time()
            proof["generation_time"] = end_time - start_time
            
            # Cập nhật thống kê
            self.stats["proofs_generated"] += 1
            self.stats["energy_saved"] += self.base_energy * 0.9  # Tiết kiệm 90% năng lượng khi dùng cache
            
            return proof
        
        # Tạo bằng chứng tùy theo loại
        if proof_type == ProofType.TRANSACTION_VALIDITY:
            proof = self._generate_tx_validity_proof(data)
        elif proof_type == ProofType.OWNERSHIP:
            proof = self._generate_ownership_proof(data)
        elif proof_type == ProofType.RANGE_PROOF:
            proof = self._generate_range_proof(data)
        elif proof_type == ProofType.SET_MEMBERSHIP:
            proof = self._generate_set_membership_proof(data)
        else:  # CUSTOM
            proof = self._generate_custom_proof(data, custom_params)
        
        # Thêm metadata
        end_time = time.time()
        generation_time = end_time - start_time
        
        proof.update({
            "proof_type": proof_type.value,
            "security_level": self.security_level.value,
            "timestamp": time.time(),
            "iterations": self.iterations,
            "generation_time": generation_time,
            "from_cache": False
        })
        
        # Lưu vào cache
        self.proof_cache[cache_key] = proof.copy()
        
        # Cập nhật thống kê
        self.stats["proofs_generated"] += 1
        self.stats["total_proof_time"] += generation_time
        self.stats["avg_proof_time"] = self.stats["total_proof_time"] / self.stats["proofs_generated"]
        
        return proof
    
    def verify_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """
        Xác minh một bằng chứng zero-knowledge.
        
        Args:
            data: Dữ liệu gốc
            proof: Bằng chứng cần xác minh
            
        Returns:
            bool: True nếu bằng chứng hợp lệ, False nếu không
        """
        start_time = time.time()
        
        # Kiểm tra cache nếu có kết quả xác minh gần đây
        cache_key = self._generate_cache_key(data, ProofType(proof["proof_type"]))
        verification_key = f"{cache_key}_verify"
        
        if verification_key in self.verification_cache:
            result = self.verification_cache[verification_key]
            end_time = time.time()
            
            # Cập nhật thống kê
            self.stats["proofs_verified"] += 1
            if result:
                self.stats["verification_success"] += 1
            else:
                self.stats["verification_failure"] += 1
            
            self.stats["energy_saved"] += self.base_energy * 0.95  # Tiết kiệm 95% năng lượng khi dùng cache
            
            return result
        
        # Xác minh bằng chứng tùy theo loại
        proof_type = ProofType(proof["proof_type"])
        
        if proof_type == ProofType.TRANSACTION_VALIDITY:
            result = self._verify_tx_validity_proof(data, proof)
        elif proof_type == ProofType.OWNERSHIP:
            result = self._verify_ownership_proof(data, proof)
        elif proof_type == ProofType.RANGE_PROOF:
            result = self._verify_range_proof(data, proof)
        elif proof_type == ProofType.SET_MEMBERSHIP:
            result = self._verify_set_membership_proof(data, proof)
        else:  # CUSTOM
            result = self._verify_custom_proof(data, proof)
        
        # Cập nhật thống kê
        end_time = time.time()
        verification_time = end_time - start_time
        
        self.stats["proofs_verified"] += 1
        if result:
            self.stats["verification_success"] += 1
        else:
            self.stats["verification_failure"] += 1
        
        self.stats["total_verify_time"] += verification_time
        self.stats["avg_verify_time"] = self.stats["total_verify_time"] / self.stats["proofs_verified"]
        
        # Lưu kết quả vào cache
        self.verification_cache[verification_key] = result
        
        return result
    
    def _generate_cache_key(self, data: Dict[str, Any], proof_type: ProofType) -> str:
        """Tạo khóa cache từ dữ liệu và loại bằng chứng."""
        # Tạo chuỗi đại diện cho dữ liệu
        data_str = str(sorted([(k, str(v)) for k, v in data.items()]))
        
        # Tạo hash từ dữ liệu và loại bằng chứng
        key = f"{data_str}_{proof_type.value}_{self.security_level.value}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _simulate_proof_generation(self, complexity: float = 1.0) -> Tuple[Dict[str, Any], float]:
        """
        Mô phỏng việc tạo bằng chứng và tính toán chi phí năng lượng.
        
        Args:
            complexity: Độ phức tạp của bằng chứng (1.0 = mặc định)
            
        Returns:
            Tuple[Dict[str, Any], float]: (Dữ liệu mô phỏng, chi phí năng lượng)
        """
        # Mô phỏng tính toán bằng cách thực hiện một số phép hash
        start_energy = time.time()
        
        # Số lượng phép hash phụ thuộc vào mức độ bảo mật
        for _ in range(int(self.hash_iterations * complexity)):
            h = hashlib.sha256()
            h.update(str(np.random.random()).encode())
            h_val = h.hexdigest()
        
        # Mô phỏng tạo số ngẫu nhiên lớn
        rand_prime = self._simulate_prime_generation(self.prime_bits)
        
        # Ước tính chi phí năng lượng
        end_energy = time.time()
        time_taken = end_energy - start_energy
        
        # Chi phí tỷ lệ với thời gian tính toán * mức bảo mật
        if self.security_level == SecurityLevel.LOW:
            energy_cost = time_taken * self.base_energy * 0.5
        elif self.security_level == SecurityLevel.MEDIUM:
            energy_cost = time_taken * self.base_energy * 1.0
        else:  # HIGH
            energy_cost = time_taken * self.base_energy * 2.0
        
        # Dữ liệu mô phỏng
        sim_data = {
            "random_value": h_val,
            "time_taken": time_taken,
            "prime": rand_prime,
            "complexity": complexity
        }
        
        return sim_data, energy_cost
    
    def _simulate_prime_generation(self, bits: int):
        """Mô phỏng việc tạo số nguyên tố lớn."""
        # Thực tế chỉ mô phỏng bằng cách tạo số lớn
        if bits > 30:  # Tránh lỗi int32 overflow
            # Sử dụng số nhỏ hơn khi bits lớn
            return np.random.randint(1000000, 9999999)
        return np.random.randint(2**(bits-1), 2**bits)
    
    def _generate_tx_validity_proof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo bằng chứng xác nhận tính hợp lệ của giao dịch."""
        sim_data, energy_cost = self._simulate_proof_generation(1.0)
        
        # Mô phỏng tạo bằng chứng
        tx_hash = hashlib.sha256(str(data).encode()).hexdigest()
        
        # Cập nhật năng lượng tiết kiệm được
        if self.energy_optimization:
            # Đảm bảo energy_saved > 0
            energy_saved = max(0.1, energy_cost * 0.3)  # Tiết kiệm tối thiểu 0.1
            self.stats["energy_saved"] += energy_saved
            energy_cost -= energy_saved
        
        return {
            "tx_hash": tx_hash,
            "witness": sim_data["random_value"],
            "energy_cost": energy_cost
        }
    
    def _generate_ownership_proof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo bằng chứng xác nhận quyền sở hữu."""
        sim_data, energy_cost = self._simulate_proof_generation(1.2)  # Phức tạp hơn
        
        # Mô phỏng tạo bằng chứng
        if "public_key" in data and "signature" in data:
            ownership_hash = hashlib.sha256(f"{data['public_key']}:{data['signature']}".encode()).hexdigest()
        else:
            # Tạo dữ liệu mô phỏng nếu không có dữ liệu thực
            ownership_hash = hashlib.sha256(str(np.random.random()).encode()).hexdigest()
        
        # Cập nhật năng lượng tiết kiệm được
        if self.energy_optimization:
            energy_saved = energy_cost * 0.25  # Tiết kiệm 25%
            self.stats["energy_saved"] += energy_saved
            energy_cost -= energy_saved
        
        return {
            "ownership_hash": ownership_hash,
            "commitment": sim_data["random_value"],
            "energy_cost": energy_cost
        }
    
    def _generate_range_proof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo bằng chứng xác nhận một giá trị nằm trong khoảng."""
        sim_data, energy_cost = self._simulate_proof_generation(1.5)  # Phức tạp hơn nhiều
        
        # Mô phỏng tạo bằng chứng
        value = data.get("value", np.random.randint(0, 1000))
        min_range = data.get("min", 0)
        max_range = data.get("max", 1000)
        
        # Tạo bằng chứng mô phỏng
        range_hash = hashlib.sha256(f"{value}:{min_range}:{max_range}".encode()).hexdigest()
        
        # Cập nhật năng lượng tiết kiệm được
        if self.energy_optimization:
            energy_saved = energy_cost * 0.35  # Tiết kiệm 35%
            self.stats["energy_saved"] += energy_saved
            energy_cost -= energy_saved
        
        return {
            "range_hash": range_hash,
            "commitments": [sim_data["random_value"][:16], sim_data["random_value"][16:32]],
            "energy_cost": energy_cost
        }
    
    def _generate_set_membership_proof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo bằng chứng xác nhận một giá trị nằm trong tập hợp."""
        sim_data, energy_cost = self._simulate_proof_generation(1.3)
        
        # Mô phỏng tạo bằng chứng
        element = data.get("element", "element1")
        set_elements = data.get("set", ["element1", "element2", "element3"])
        
        # Tạo bằng chứng mô phỏng
        set_hash = hashlib.sha256(f"{element}:{','.join(set_elements)}".encode()).hexdigest()
        
        # Cập nhật năng lượng tiết kiệm được
        if self.energy_optimization:
            energy_saved = energy_cost * 0.3  # Tiết kiệm 30%
            self.stats["energy_saved"] += energy_saved
            energy_cost -= energy_saved
        
        return {
            "set_hash": set_hash,
            "witness": sim_data["random_value"],
            "energy_cost": energy_cost
        }
    
    def _generate_custom_proof(self, data: Dict[str, Any], custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Tạo bằng chứng tùy chỉnh."""
        complexity = custom_params.get("complexity", 1.0) if custom_params else 1.0
        sim_data, energy_cost = self._simulate_proof_generation(complexity)
        
        # Mô phỏng tạo bằng chứng
        custom_hash = hashlib.sha256(str(data).encode()).hexdigest()
        
        # Cập nhật năng lượng tiết kiệm được
        if self.energy_optimization:
            energy_saved = energy_cost * 0.2  # Tiết kiệm 20%
            self.stats["energy_saved"] += energy_saved
            energy_cost -= energy_saved
        
        return {
            "custom_hash": custom_hash,
            "witness": sim_data["random_value"],
            "parameters": custom_params,
            "energy_cost": energy_cost
        }
    
    def _verify_tx_validity_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """Xác minh bằng chứng tính hợp lệ của giao dịch."""
        # Tính toán lại hash từ dữ liệu
        tx_hash = hashlib.sha256(str(data).encode()).hexdigest()
        
        # So sánh với hash trong bằng chứng
        return tx_hash == proof.get("tx_hash", "")
    
    def _verify_ownership_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """Xác minh bằng chứng quyền sở hữu."""
        # Tính toán lại hash từ dữ liệu
        if "public_key" in data and "signature" in data:
            ownership_hash = hashlib.sha256(f"{data['public_key']}:{data['signature']}".encode()).hexdigest()
            
            # So sánh với hash trong bằng chứng
            return ownership_hash == proof.get("ownership_hash", "")
        
        return False
    
    def _verify_range_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """Xác minh bằng chứng khoảng."""
        # Tính toán lại hash từ dữ liệu
        value = data.get("value", 0)
        min_range = data.get("min", 0)
        max_range = data.get("max", 1000)
        
        # Kiểm tra xem giá trị có nằm trong khoảng không
        if not (min_range <= value <= max_range):
            return False
        
        # Tính toán lại hash
        range_hash = hashlib.sha256(f"{value}:{min_range}:{max_range}".encode()).hexdigest()
        
        # So sánh với hash trong bằng chứng
        return range_hash == proof.get("range_hash", "")
    
    def _verify_set_membership_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """Xác minh bằng chứng tư cách thành viên tập hợp."""
        # Tính toán lại hash từ dữ liệu
        element = data.get("element", "")
        set_elements = data.get("set", [])
        
        # Kiểm tra xem phần tử có trong tập hợp không
        if element not in set_elements:
            return False
        
        # Tính toán lại hash
        set_hash = hashlib.sha256(f"{element}:{','.join(set_elements)}".encode()).hexdigest()
        
        # So sánh với hash trong bằng chứng
        return set_hash == proof.get("set_hash", "")
    
    def _verify_custom_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """Xác minh bằng chứng tùy chỉnh."""
        # Tính toán lại hash từ dữ liệu
        custom_hash = hashlib.sha256(str(data).encode()).hexdigest()
        
        # So sánh với hash trong bằng chứng
        return custom_hash == proof.get("custom_hash", "")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê về hệ thống ZK proof.
        
        Returns:
            Dict[str, Any]: Thống kê về việc sử dụng
        """
        # Tính toán thêm một số thống kê
        if self.stats["proofs_verified"] > 0:
            success_rate = self.stats["verification_success"] / self.stats["proofs_verified"]
        else:
            success_rate = 0
            
        stats = self.stats.copy()
        stats.update({
            "security_level": self.security_level.value,
            "energy_optimization": self.energy_optimization,
            "verification_speedup": self.verification_speedup,
            "success_rate": success_rate,
            "proof_cache_size": len(self.proof_cache),
            "verification_cache_size": len(self.verification_cache)
        })
        
        return stats
    
    def update_security_level(self, security_level: str):
        """
        Cập nhật mức độ bảo mật của hệ thống.
        
        Args:
            security_level: Mức độ bảo mật mới ("low", "medium", "high")
        """
        self.security_level = SecurityLevel(security_level)
        
        # Cấu hình lại các tham số
        self._configure_parameters()
        
        # Xóa cache khi thay đổi mức độ bảo mật
        self.proof_cache.clear()
        self.verification_cache.clear()
    
    def clear_caches(self):
        """Xóa tất cả các cache."""
        self.proof_cache.clear()
        self.verification_cache.clear() 