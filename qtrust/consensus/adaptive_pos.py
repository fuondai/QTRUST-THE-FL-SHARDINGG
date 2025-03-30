import random
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import math


class ValidatorStakeInfo:
    """
    Lưu trữ thông tin về stake và trạng thái của một validator.
    """
    def __init__(self, id: int, initial_stake: float = 100.0, max_energy: float = 100.0):
        """
        Khởi tạo thông tin validator.
        
        Args:
            id: ID của validator
            initial_stake: Số lượng stake ban đầu
            max_energy: Năng lượng tối đa (dùng cho mô phỏng năng lượng pin)
        """
        self.id = id
        self.stake = initial_stake
        self.max_energy = max_energy
        self.current_energy = max_energy
        self.active = True
        self.last_active_time = time.time()
        self.active_rounds = 0
        self.total_rounds = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.energy_consumption_history = []
        self.performance_score = 0.5  # điểm hiệu suất (0.0-1.0)
        self.participation_rate = 1.0  # tỷ lệ tham gia (0.0-1.0)
        self.last_rotation_time = time.time()
        
    def update_stake(self, delta: float):
        """Cập nhật stake."""
        self.stake = max(0.0, self.stake + delta)
        
    def consume_energy(self, amount: float) -> bool:
        """
        Tiêu thụ năng lượng và trả về True nếu có đủ năng lượng.
        """
        if self.current_energy >= amount:
            self.current_energy -= amount
            self.energy_consumption_history.append(amount)
            if len(self.energy_consumption_history) > 100:
                self.energy_consumption_history.pop(0)
            return True
        return False
    
    def recharge_energy(self, amount: float = None):
        """
        Nạp lại năng lượng cho validator.
        
        Args:
            amount: Lượng năng lượng nạp lại, nếu None, nạp đầy
        """
        if amount is None:
            self.current_energy = self.max_energy
        else:
            self.current_energy = min(self.max_energy, self.current_energy + amount)
    
    def update_performance(self, success: bool):
        """
        Cập nhật điểm hiệu suất dựa trên kết quả xác thực.
        """
        self.total_rounds += 1
        alpha = 0.1  # hệ số học tập
        
        if success:
            self.successful_validations += 1
            self.performance_score = (1 - alpha) * self.performance_score + alpha * 1.0
        else:
            self.failed_validations += 1
            self.performance_score = (1 - alpha) * self.performance_score
        
        if self.active:
            self.active_rounds += 1
        
        self.participation_rate = self.active_rounds / max(1, self.total_rounds)
    
    def get_average_energy_consumption(self) -> float:
        """Trả về mức tiêu thụ năng lượng trung bình."""
        if not self.energy_consumption_history:
            return 0.0
        return sum(self.energy_consumption_history) / len(self.energy_consumption_history)


class AdaptivePoSManager:
    """
    Quản lý cơ chế Proof of Stake (PoS) thích ứng có luân chuyển validator.
    """
    def __init__(self, 
                 num_validators: int = 20,
                 active_validator_ratio: float = 0.7,
                 rotation_period: int = 50,
                 min_stake: float = 10.0,
                 energy_threshold: float = 30.0,
                 performance_threshold: float = 0.3,
                 energy_optimization_level: str = "balanced",
                 enable_smart_energy_management: bool = True,
                 seed: int = 42):
        """
        Khởi tạo AdaptivePoSManager.
        
        Args:
            num_validators: Tổng số validator
            active_validator_ratio: Tỷ lệ validator hoạt động (0.0-1.0)
            rotation_period: Số vòng trước khi xem xét luân chuyển
            min_stake: Stake tối thiểu để được chọn làm validator
            energy_threshold: Ngưỡng năng lượng thấp cần thay thế
            performance_threshold: Ngưỡng hiệu suất tối thiểu
            energy_optimization_level: Mức độ tối ưu năng lượng ("low", "balanced", "aggressive")
            enable_smart_energy_management: Bật/tắt quản lý năng lượng thông minh
            seed: Giá trị seed cho tính toán ngẫu nhiên
        """
        self.num_validators = num_validators
        self.active_validator_ratio = active_validator_ratio
        self.num_active_validators = max(1, int(num_validators * active_validator_ratio))
        self.rotation_period = rotation_period
        self.min_stake = min_stake
        self.energy_threshold = energy_threshold
        self.performance_threshold = performance_threshold
        self.energy_optimization_level = energy_optimization_level
        self.enable_smart_energy_management = enable_smart_energy_management
        self.seed = seed
        random.seed(seed)
        
        # Khởi tạo validators
        self.validators = {i: ValidatorStakeInfo(id=i) for i in range(1, num_validators + 1)}
        
        # Tập hợp các validator đang hoạt động
        self.active_validators = set()
        
        # Thực hiện chọn lựa ban đầu
        self._select_initial_validators()
        
        # Thống kê
        self.rounds_since_rotation = 0
        self.total_rotations = 0
        self.total_rounds = 0
        self.energy_saved = 0.0
        
        # Thông tin năng lượng và dự đoán
        self.energy_prediction_model = {}  # validator_id -> predicted_energy
        self.energy_efficiency_rankings = {}  # validator_id -> efficiency_rank
        self.historical_energy_usage = []  # lịch sử sử dụng năng lượng
        self.energy_optimization_weights = self._get_optimization_weights()
        
    def _get_optimization_weights(self) -> Dict[str, float]:
        """
        Lấy trọng số tối ưu năng lượng dựa trên cấu hình.
        """
        if self.energy_optimization_level == "low":
            return {
                "energy_weight": 0.2,
                "performance_weight": 0.5,
                "stake_weight": 0.3,
                "rotation_aggressiveness": 0.3
            }
        elif self.energy_optimization_level == "aggressive":
            return {
                "energy_weight": 0.6,
                "performance_weight": 0.2,
                "stake_weight": 0.2,
                "rotation_aggressiveness": 0.8
            }
        else:  # balanced
            return {
                "energy_weight": 0.4,
                "performance_weight": 0.3,
                "stake_weight": 0.3,
                "rotation_aggressiveness": 0.5
            }

    def _select_initial_validators(self):
        """
        Chọn validators ban đầu dựa trên stake.
        """
        # Sắp xếp theo stake giảm dần
        sorted_validators = sorted(self.validators.items(), 
                                  key=lambda x: x[1].stake, reverse=True)
        
        # Chọn những validator có stake cao nhất
        self.active_validators = set()
        for i in range(min(self.num_active_validators, len(sorted_validators))):
            validator_id = sorted_validators[i][0]
            self.validators[validator_id].active = True
            self.active_validators.add(validator_id)
    
    def select_validator_for_block(self, trust_scores: Dict[int, float] = None) -> int:
        """
        Chọn validator cho việc tạo block tiếp theo dựa trên stake, điểm tin cậy và năng lượng.
        
        Args:
            trust_scores: Điểm tin cậy của các validator (có thể None)
            
        Returns:
            int: ID của validator được chọn
        """
        if not self.active_validators:
            self._select_initial_validators()
            if not self.active_validators:
                return None  # Không có validator nào hợp lệ
        
        # Tính xác suất chọn dựa trên stake, điểm tin cậy và hiệu quả năng lượng
        selection_weights = {}
        total_weight = 0.0
        
        for validator_id in self.active_validators:
            validator = self.validators[validator_id]
            # Kiểm tra năng lượng tối thiểu
            if validator.current_energy < self.energy_threshold / 2:
                continue  # Bỏ qua validator có năng lượng quá thấp
                
            # Kết hợp stake với điểm tin cậy (nếu có)
            trust_factor = trust_scores.get(validator_id, 0.5) if trust_scores else 0.5
            
            # Lấy hiệu quả năng lượng
            energy_rank = self.energy_efficiency_rankings.get(validator_id, self.num_validators)
            energy_efficiency = 1.0 - (energy_rank / self.num_validators)  # 0.0-1.0, cao hơn là tốt hơn
            
            # Tính trọng số dựa trên các yếu tố
            stake_component = validator.stake * self.energy_optimization_weights["stake_weight"]
            performance_component = validator.performance_score * self.energy_optimization_weights["performance_weight"]
            energy_component = energy_efficiency * self.energy_optimization_weights["energy_weight"]
            trust_component = trust_factor * 0.1  # Trọng số nhỏ cho trust
            
            weight = stake_component + performance_component + energy_component + trust_component
            
            selection_weights[validator_id] = weight
            total_weight += weight
        
        # Chọn validator theo cơ chế xác suất có trọng số
        if total_weight <= 0 or not selection_weights:
            # Nếu tổng trọng số là 0 hoặc không có validator hợp lệ, chọn ngẫu nhiên
            return random.choice(list(self.active_validators))
        
        # Chọn dựa trên xác suất có trọng số
        selection_point = random.uniform(0, total_weight)
        current_sum = 0.0
        
        for validator_id, weight in selection_weights.items():
            current_sum += weight
            if current_sum >= selection_point:
                return validator_id
        
        # Fallback nếu có lỗi trong tính toán
        return random.choice(list(self.active_validators))
    
    def select_validators_for_committee(self, 
                                      committee_size: int, 
                                      trust_scores: Dict[int, float] = None) -> List[int]:
        """
        Chọn một ủy ban validator cho consensus.
        
        Args:
            committee_size: Kích thước của ủy ban
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            List[int]: Danh sách ID của các validator được chọn
        """
        actual_committee_size = min(committee_size, len(self.active_validators))
        
        if actual_committee_size <= 0:
            return []
            
        # Tạo danh sách các validator có trọng số
        weighted_validators = []
        for validator_id in self.active_validators:
            validator = self.validators[validator_id]
            
            # Tính trọng số dựa trên stake, performance và trust score
            trust_factor = trust_scores.get(validator_id, 0.5) if trust_scores else 0.5
            weight = validator.stake * validator.performance_score * (0.5 + 0.5 * trust_factor)
            
            # Kiểm tra năng lượng còn lại
            if validator.current_energy >= self.energy_threshold:
                weighted_validators.append((validator_id, weight))
        
        # Nếu không có đủ validator có năng lượng, thêm validator có năng lượng thấp
        if len(weighted_validators) < actual_committee_size:
            for validator_id in self.active_validators:
                if validator_id not in [v[0] for v in weighted_validators]:
                    validator = self.validators[validator_id]
                    trust_factor = trust_scores.get(validator_id, 0.5) if trust_scores else 0.5
                    weight = validator.stake * validator.performance_score * (0.5 + 0.5 * trust_factor)
                    weighted_validators.append((validator_id, weight))
        
        # Sắp xếp theo trọng số và chọn
        weighted_validators.sort(key=lambda x: x[1], reverse=True)
        committee = [validator_id for validator_id, _ in weighted_validators[:actual_committee_size]]
        
        # Đảm bảo chọn đủ, nếu cần thiết chọn ngẫu nhiên từ các validator còn lại
        remaining = actual_committee_size - len(committee)
        if remaining > 0:
            remaining_validators = [v_id for v_id in self.active_validators if v_id not in committee]
            if remaining_validators:
                committee.extend(random.sample(remaining_validators, min(remaining, len(remaining_validators))))
        
        return committee
    
    def update_validator_energy(self, 
                               validator_id: int, 
                               energy_consumed: float, 
                               transaction_success: bool):
        """
        Cập nhật năng lượng của validator và đánh giá hiệu suất.
        
        Args:
            validator_id: ID của validator
            energy_consumed: Lượng năng lượng tiêu thụ
            transaction_success: Giao dịch có thành công không
        """
        if validator_id not in self.validators:
            return
            
        validator = self.validators[validator_id]
        
        # Tiêu thụ năng lượng
        sufficient_energy = validator.consume_energy(energy_consumed)
        
        # Dự đoán mức tiêu thụ năng lượng trong tương lai
        self._update_energy_prediction(validator_id, energy_consumed)
        
        # Cập nhật điểm hiệu suất
        validator.update_performance(transaction_success)
        
        # Thưởng stake khi giao dịch thành công
        if transaction_success:
            reward = 0.1  # Reward nhỏ cho mỗi giao dịch thành công
            validator.update_stake(reward)
        
        # Cập nhật thống kê năng lượng
        self.historical_energy_usage.append({
            "validator_id": validator_id,
            "energy_consumed": energy_consumed,
            "success": transaction_success,
            "remaining_energy": validator.current_energy
        })
        
        # Giới hạn kích thước lịch sử
        if len(self.historical_energy_usage) > 1000:
            self.historical_energy_usage = self.historical_energy_usage[-1000:]
        
        # Đánh giá lại hiệu quả năng lượng
        self._recalculate_energy_efficiency()
        
        # Xem xét quản lý năng lượng thông minh
        if self.enable_smart_energy_management:
            self._apply_smart_energy_management(validator_id)
    
    def _update_energy_prediction(self, validator_id: int, energy_consumed: float):
        """
        Cập nhật mô hình dự đoán năng lượng cho validator.
        """
        if validator_id not in self.energy_prediction_model:
            self.energy_prediction_model[validator_id] = energy_consumed
        else:
            # Cập nhật với trọng số 0.2 cho giá trị mới
            current_prediction = self.energy_prediction_model[validator_id]
            self.energy_prediction_model[validator_id] = 0.8 * current_prediction + 0.2 * energy_consumed
    
    def _recalculate_energy_efficiency(self):
        """
        Tính toán lại hiệu quả năng lượng của tất cả validator.
        """
        efficiency_scores = {}
        
        for validator_id, validator in self.validators.items():
            # Lấy lịch sử năng lượng của validator này
            history = [entry for entry in self.historical_energy_usage 
                      if entry["validator_id"] == validator_id]
            
            if not history:
                efficiency_scores[validator_id] = 0.5  # Giá trị mặc định
                continue
                
            # Tính tỷ lệ thành công/năng lượng
            total_energy = sum(entry["energy_consumed"] for entry in history)
            successful_txs = sum(1 for entry in history if entry["success"])
            
            if total_energy > 0:
                # Điểm hiệu quả = số giao dịch thành công / tổng năng lượng * hiệu suất
                efficiency = (successful_txs / total_energy) * validator.performance_score
            else:
                efficiency = 0.0
                
            efficiency_scores[validator_id] = efficiency
        
        # Xếp hạng các validator theo hiệu quả năng lượng
        sorted_validators = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Gán xếp hạng
        for rank, (validator_id, _) in enumerate(sorted_validators):
            self.energy_efficiency_rankings[validator_id] = rank + 1
    
    def _apply_smart_energy_management(self, validator_id: int):
        """
        Áp dụng quản lý năng lượng thông minh cho validator.
        """
        validator = self.validators[validator_id]
        
        # Kiểm tra nếu năng lượng thấp
        if validator.current_energy < self.energy_threshold:
            # Nếu validator đang hoạt động, cân nhắc tạm thời nghỉ để tiết kiệm năng lượng
            if validator_id in self.active_validators and len(self.active_validators) > self.num_active_validators / 2:
                self.active_validators.remove(validator_id)
                validator.active = False
                
                # Tính lượng năng lượng tiết kiệm được
                predicted_energy = self.energy_prediction_model.get(validator_id, 10.0)
                self.energy_saved += predicted_energy
                
                # Thay thế bằng validator khác có năng lượng cao hơn
                self._add_replacement_validator()
    
    def _add_replacement_validator(self):
        """
        Thêm validator thay thế dựa trên hiệu quả năng lượng.
        """
        # Tìm validator không hoạt động có hiệu quả năng lượng tốt và đủ năng lượng
        candidates = []
        
        for validator_id, validator in self.validators.items():
            if validator_id not in self.active_validators:
                if validator.stake >= self.min_stake and validator.current_energy >= 2 * self.energy_threshold:  # Đảm bảo đủ năng lượng
                    efficiency_rank = self.energy_efficiency_rankings.get(validator_id, float('inf'))
                    candidates.append((validator_id, efficiency_rank, validator.current_energy))
        
        if not candidates:
            return
            
        # Chọn validator dựa trên xếp hạng hiệu quả và năng lượng còn lại
        candidates.sort(key=lambda x: (x[1], -x[2]))  # Sắp xếp theo xếp hạng, sau đó là năng lượng (cao nhất)
        
        if candidates:
            new_validator_id = candidates[0][0]
            self.active_validators.add(new_validator_id)
            self.validators[new_validator_id].active = True

    def rotate_validators(self, trust_scores: Dict[int, float] = None) -> int:
        """
        Luân chuyển các validator để cân bằng năng lượng và hiệu suất.
        
        Args:
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            int: Số lượng validator đã luân chuyển
        """
        self.rounds_since_rotation += 1
        
        # Kiểm tra xem có cần luân chuyển không
        if self.rounds_since_rotation < self.rotation_period:
            return 0
            
        self.rounds_since_rotation = 0
        rotations = 0
        
        # Xác định validator cần thay thế
        validators_to_replace = []
        
        # Bước 1: Xác định validator có năng lượng thấp
        for validator_id in list(self.active_validators):
            validator = self.validators[validator_id]
            
            # Các tiêu chí để thay thế:
            # 1. Năng lượng thấp
            # 2. Hiệu suất kém
            # 3. Xếp hạng hiệu quả năng lượng thấp
            
            energy_criteria = validator.current_energy < self.energy_threshold
            performance_criteria = validator.performance_score < self.performance_threshold
            
            energy_rank = self.energy_efficiency_rankings.get(validator_id, float('inf'))
            efficiency_criteria = energy_rank > 0.7 * self.num_validators  # Bottom 30%
            
            # Trọng số quyết định dựa trên mức độ tối ưu năng lượng
            if energy_criteria:
                replace_score = 0.6
            else:
                replace_score = 0
                
            if performance_criteria:
                replace_score += 0.3
                
            if efficiency_criteria:
                replace_score += 0.3 * self.energy_optimization_weights["rotation_aggressiveness"]
                
            # Quyết định thay thế nếu điểm vượt ngưỡng
            if replace_score >= 0.5:
                validators_to_replace.append(validator_id)
        
        # Giới hạn số lượng validator thay thế trong một lần
        max_rotations = max(1, int(self.num_active_validators * 0.3))  # Tối đa 30%
        if len(validators_to_replace) > max_rotations:
            # Ưu tiên validator có năng lượng thấp nhất
            validators_to_replace.sort(
                key=lambda v_id: (
                    self.validators[v_id].current_energy,
                    self.validators[v_id].performance_score
                )
            )
            validators_to_replace = validators_to_replace[:max_rotations]
        
        # Bước 2: Thực hiện thay thế
        for validator_id in validators_to_replace:
            # Xóa khỏi danh sách active
            self.active_validators.remove(validator_id)
            self.validators[validator_id].active = False
            
            # Tìm validator thay thế
            # Ưu tiên validator có hiệu quả năng lượng tốt
            replacement_found = self._find_replacement_validator()
            
            if replacement_found:
                rotations += 1
                # Tính lượng năng lượng tiết kiệm được
                predicted_energy = self.energy_prediction_model.get(validator_id, 10.0)
                self.energy_saved += predicted_energy
        
        self.total_rotations += rotations
        return rotations
    
    def _find_replacement_validator(self) -> bool:
        """
        Tìm validator thay thế phù hợp.
        
        Returns:
            bool: True nếu tìm thấy validator thay thế
        """
        candidates = []
        
        for validator_id, validator in self.validators.items():
            # Kiểm tra validator không đang hoạt động
            if validator_id not in self.active_validators:
                # Kiểm tra đủ stake và năng lượng
                if validator.stake >= self.min_stake and validator.current_energy >= 2 * self.energy_threshold:
                    # Lấy các thông tin cần thiết
                    energy_score = validator.current_energy / 100.0  # 0.0-1.0
                    energy_rank = self.energy_efficiency_rankings.get(validator_id, self.num_validators)
                    rank_score = 1.0 - (energy_rank / self.num_validators)  # 0.0-1.0
                    performance_score = validator.performance_score
                    
                    # Tính điểm ứng viên
                    candidate_score = (
                        energy_score * 0.4 +
                        rank_score * 0.4 +
                        performance_score * 0.1
                    )
                    
                    candidates.append((validator_id, candidate_score))
        
        if candidates:
            # Sắp xếp theo điểm giảm dần
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Chọn validator có điểm cao nhất
            best_candidate_id = candidates[0][0]
            self.active_validators.add(best_candidate_id)
            self.validators[best_candidate_id].active = True
            return True
            
        return False

    def update_energy_recharge(self, recharge_rate: float = 0.02):
        """
        Cập nhật tốc độ nạp năng lượng cho validators không hoạt động.
        
        Args:
            recharge_rate: Tỷ lệ năng lượng tối đa được nạp lại mỗi lần gọi
        """
        for validator_id, validator in self.validators.items():
            if not validator.active:
                # Nạp lại năng lượng cho validator không hoạt động
                recharge_amount = validator.max_energy * recharge_rate
                validator.recharge_energy(recharge_amount)
                
                # Kiểm tra xem validator đã đủ điều kiện để tham gia lại chưa
                if (validator.current_energy >= 0.9 * validator.max_energy and 
                    time.time() - validator.last_rotation_time >= self.rotation_period):
                    # Đánh dấu validator là ứng cử viên sẵn sàng tham gia lại
                    validator.performance_score = max(0.4, validator.performance_score)
    
    def get_energy_statistics(self) -> Dict[str, float]:
        """
        Lấy thống kê về năng lượng của các validator.
        
        Returns:
            Dict[str, float]: Thống kê năng lượng
        """
        total_energy = 0.0
        active_energy = 0.0
        inactive_energy = 0.0
        energy_levels = []
        active_energy_levels = []
        
        for validator_id, validator in self.validators.items():
            energy = validator.current_energy
            total_energy += energy
            energy_levels.append(energy)
            
            if validator_id in self.active_validators:
                active_energy += energy
                active_energy_levels.append(energy)
            else:
                inactive_energy += energy
        
        # Tính toán các chỉ số năng lượng bổ sung
        avg_energy = total_energy / len(self.validators) if self.validators else 0
        avg_active_energy = active_energy / len(self.active_validators) if self.active_validators else 0
        
        # Dự đoán mức tiêu thụ tương lai
        predicted_consumption_rate = 0.0
        if self.historical_energy_usage:
            recent_usage = self.historical_energy_usage[-min(10, len(self.historical_energy_usage)):]
            if recent_usage:
                predicted_consumption_rate = sum(entry["energy_consumed"] for entry in recent_usage) / len(recent_usage)
        
        # Tính toán chênh lệch năng lượng giữa các validator
        energy_std_dev = np.std(energy_levels) if energy_levels else 0.0
        
        return {
            "total_energy": total_energy,
            "active_energy": active_energy,
            "inactive_energy": inactive_energy,
            "avg_energy": avg_energy,
            "avg_active_energy": avg_active_energy,
            "energy_saved": self.energy_saved,
            "predicted_consumption_rate": predicted_consumption_rate,
            "energy_distribution_std": energy_std_dev,
            "optimization_level": self.energy_optimization_level
        }
    
    def get_validator_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về validator.
        
        Returns:
            Dict[str, Any]: Thống kê validator
        """
        active_count = len(self.active_validators)
        inactive_count = self.num_validators - active_count
        
        # Tính toán hiệu suất trung bình
        avg_performance = sum(v.performance_score for v in self.validators.values()) / self.num_validators
        
        # Tính toán chỉ số hiệu quả năng lượng trung bình
        avg_energy_efficiency = 0.0
        if self.energy_efficiency_rankings:
            efficiency_scores = [1.0 - (rank / self.num_validators) 
                               for rank in self.energy_efficiency_rankings.values()]
            avg_energy_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        
        # Tổng hợp thông tin của top validators về hiệu quả năng lượng
        top_efficient_validators = sorted(
            [(v_id, 1.0 - (self.energy_efficiency_rankings.get(v_id, float('inf')) / self.num_validators))
             for v_id in self.validators.keys()],
            key=lambda x: x[1], reverse=True
        )[:5]  # Top 5
        
        return {
            "total_validators": self.num_validators,
            "active_validators": active_count,
            "inactive_validators": inactive_count,
            "avg_performance_score": avg_performance,
            "avg_energy_efficiency": avg_energy_efficiency,
            "total_rotations": self.total_rotations,
            "top_energy_efficient": top_efficient_validators,
            "energy_optimization_level": self.energy_optimization_level
        }
    
    def simulate_round(self, trust_scores: Dict[int, float] = None, transaction_value: float = 10.0) -> Dict[str, Any]:
        """
        Mô phỏng một round của PoS.
        
        Args:
            trust_scores: Điểm tin cậy của các validator
            transaction_value: Giá trị của giao dịch
            
        Returns:
            Dict[str, Any]: Kết quả mô phỏng
        """
        # Cập nhật số round
        self.total_rounds += 1
        
        # Chọn validator
        validator_id = self.select_validator_for_block(trust_scores)
        
        if validator_id is None:
            return {
                "success": False,
                "error": "No suitable validator found",
                "rotations": 0,
                "energy_saved": 0.0
            }
            
        validator = self.validators[validator_id]
        
        # Tính toán mức tiêu thụ năng lượng cho giao dịch này
        base_energy = 5.0 + 0.1 * transaction_value
        # Giảm năng lượng dựa trên hiệu quả
        energy_rank = self.energy_efficiency_rankings.get(validator_id, self.num_validators)
        energy_efficiency_factor = 1.0 - 0.3 * (1.0 - energy_rank / self.num_validators)
        energy_consumed = base_energy * energy_efficiency_factor
        
        # Tiêu thụ năng lượng
        sufficient_energy = validator.consume_energy(energy_consumed)
        
        # Tỷ lệ thành công
        success_probability = 0.95
        
        # Nếu năng lượng không đủ, giảm tỷ lệ thành công
        if not sufficient_energy:
            success_probability *= 0.5
        
        # Mô phỏng kết quả giao dịch
        success = random.random() < success_probability
        
        # Cập nhật performance
        validator.update_performance(success)
        
        # Cập nhật mô hình dự đoán năng lượng
        self._update_energy_prediction(validator_id, energy_consumed)
        
        # Xem xét luân chuyển validator
        rotations = self.rotate_validators(trust_scores)
        
        # Nạp lại năng lượng cho validator không hoạt động
        self.update_energy_recharge(0.02)  # 2% mỗi round
        
        # Cập nhật thống kê sử dụng năng lượng
        self.historical_energy_usage.append({
            "validator_id": validator_id,
            "energy_consumed": energy_consumed,
            "success": success,
            "remaining_energy": validator.current_energy
        })
        
        # Thưởng/phạt stake
        if success:
            reward = 0.1 * transaction_value
            validator.update_stake(reward)
        
        # Cập nhật hiệu quả năng lượng định kỳ
        if self.total_rounds % 10 == 0:
            self._recalculate_energy_efficiency()
        
        return {
            "success": success,
            "validator": validator_id,
            "stake": validator.stake,
            "energy_consumed": energy_consumed,
            "remaining_energy": validator.current_energy,
            "performance_score": validator.performance_score,
            "rotations": rotations,
            "energy_saved": self.energy_saved,
        } 