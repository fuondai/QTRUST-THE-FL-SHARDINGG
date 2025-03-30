import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from .zk_proofs import ZKProofSystem, ProofType

class ValidatorSelectionPolicy:
    """Chính sách lựa chọn validator."""
    RANDOM = "random"  # Lựa chọn ngẫu nhiên
    REPUTATION = "reputation"  # Dựa trên danh tiếng
    STAKE_WEIGHTED = "stake_weighted"  # Dựa trên stake
    PERFORMANCE = "performance"  # Dựa trên hiệu suất
    HYBRID = "hybrid"  # Kết hợp các tiêu chí

class ReputationBasedValidatorSelection:
    """
    Hệ thống lựa chọn validator dựa trên reputation và bảo mật.
    
    Kết hợp với Zero-Knowledge Proofs để chọn validator một cách
    công bằng và bảo mật.
    """
    
    def __init__(self, 
                trust_manager, 
                policy: str = ValidatorSelectionPolicy.HYBRID,
                zk_enabled: bool = True,
                security_level: str = "medium",
                use_rotation: bool = True,
                rotation_period: int = 20):
        """
        Khởi tạo hệ thống lựa chọn validator.
        
        Args:
            trust_manager: Trình quản lý tin cậy của hệ thống
            policy: Chính sách lựa chọn validator
            zk_enabled: Kích hoạt ZK Proofs
            security_level: Mức độ bảo mật ("low", "medium", "high")
            use_rotation: Bật chế độ luân chuyển validator
            rotation_period: Chu kỳ luân chuyển (số lượng blocks)
        """
        self.trust_manager = trust_manager
        self.policy = policy
        self.zk_enabled = zk_enabled
        self.use_rotation = use_rotation
        self.rotation_period = rotation_period
        
        # Khởi tạo hệ thống ZK proof nếu được bật
        if zk_enabled:
            self.zk_system = ZKProofSystem(security_level=security_level)
        else:
            self.zk_system = None
        
        # Theo dõi validators hiện tại và lịch sử
        self.active_validators = {}  # Shard ID -> List of validator IDs
        self.validator_history = defaultdict(list)  # Shard ID -> List of (round, validators) tuples
        self.current_round = 0
        
        # Thống kê hiệu suất
        self.stats = {
            "selections": 0,
            "rotations": 0,
            "energy_saved": 0.0,
            "validator_diversity": 0.0,
            "avg_selection_time": 0.0,
            "total_selection_time": 0.0
        }
    
    def select_validators(self, shard_id: int, block_num: int, num_validators: int = 3) -> List[int]:
        """
        Chọn ra các validator cho một block/shard cụ thể.
        
        Args:
            shard_id: ID của shard
            block_num: Số thứ tự block
            num_validators: Số lượng validator cần chọn
            
        Returns:
            List[int]: Danh sách ID của các validator được chọn
        """
        start_time = time.time()
        self.current_round = block_num
        
        # Kiểm tra xem có cần luân chuyển validator không
        needs_rotation = self.use_rotation and block_num > 0 and block_num % self.rotation_period == 0
        
        # Nếu đã có validators và không cần luân chuyển, giữ nguyên
        if shard_id in self.active_validators and not needs_rotation:
            selected_validators = self.active_validators[shard_id]
            
            # Ghi lại lịch sử
            self.validator_history[shard_id].append((block_num, selected_validators.copy()))
            
            # Cập nhật thống kê
            end_time = time.time()
            selection_time = end_time - start_time
            self.stats["selections"] += 1
            self.stats["total_selection_time"] += selection_time
            self.stats["avg_selection_time"] = self.stats["total_selection_time"] / self.stats["selections"]
            
            return selected_validators
        
        # Chọn validators mới (luân chuyển hoặc lần đầu)
        if needs_rotation:
            self.stats["rotations"] += 1
        
        # Lấy danh sách các node đáng tin cậy từ trust manager
        trusted_nodes = self.trust_manager.recommend_trusted_validators(
            shard_id, count=min(10, num_validators * 3)
        )
        
        # Áp dụng chính sách lựa chọn
        if not trusted_nodes:
            # Nếu không có thông tin tin cậy, chọn ngẫu nhiên từ tất cả các node trong shard
            if hasattr(self.trust_manager, "shards") and self.trust_manager.shards:
                all_shard_nodes = self.trust_manager.shards[shard_id]
                selected_validators = np.random.choice(
                    all_shard_nodes, 
                    size=min(num_validators, len(all_shard_nodes)), 
                    replace=False
                ).tolist()
            else:
                # Không có thông tin về shards, trả về danh sách rỗng
                selected_validators = []
        else:
            # Lọc các node theo chính sách
            filtered_nodes = self._apply_selection_policy(trusted_nodes, block_num)
            
            # Nếu cần luân chuyển, thêm yếu tố ngẫu nhiên mạnh hơn
            if needs_rotation and len(filtered_nodes) > num_validators:
                # Lưu validators hiện tại để đảm bảo chọn được validators mới khác
                current_validators = set()
                if shard_id in self.active_validators:
                    current_validators = set(self.active_validators[shard_id])
                
                # Khi luân chuyển, ưu tiên chọn các node chưa từng làm validator gần đây
                if shard_id in self.validator_history and len(self.validator_history[shard_id]) > 0:
                    recent_validators = set()
                    for _, validators in self.validator_history[shard_id][-3:]:  # 3 block gần nhất
                        recent_validators.update(validators)
                    
                    # Ưu tiên node chưa làm validator gần đây
                    non_recent = [node for node in filtered_nodes if node["node_id"] not in recent_validators]
                    
                    # Nếu có đủ node mới, ưu tiên chọn chúng
                    if len(non_recent) >= num_validators:
                        filtered_nodes = non_recent
                
                # Thêm yếu tố ngẫu nhiên mạnh hơn
                randomness = np.random.random(len(filtered_nodes)) * 0.3  # Thêm tối đa 30% ngẫu nhiên
                adjusted_scores = [node["composite_score"] * (1 + rand) for node, rand in zip(filtered_nodes, randomness)]
                nodes_with_scores = list(zip(filtered_nodes, adjusted_scores))
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                filtered_nodes = [node for node, _ in nodes_with_scores]
            
            # Chọn các node tốt nhất
            selected_validators = [node["node_id"] for node in filtered_nodes[:num_validators]]
            
            # Đảm bảo validators có sự thay đổi khi luân chuyển
            if needs_rotation and shard_id in self.active_validators:
                current_set = set(self.active_validators[shard_id])
                new_set = set(selected_validators)
                
                # Nếu không có sự thay đổi, buộc thay đổi ít nhất 1 validator
                if current_set == new_set and len(filtered_nodes) > num_validators:
                    # Chọn một validator hiện tại để thay thế
                    validator_to_replace = np.random.choice(list(current_set))
                    
                    # Tìm validator mới không nằm trong tập hiện tại
                    new_candidates = [node["node_id"] for node in filtered_nodes[num_validators:] 
                                     if node["node_id"] not in current_set]
                    
                    if new_candidates:
                        # Thay thế validator cũ bằng validator mới
                        selected_validators = list(new_set)
                        selected_validators.remove(validator_to_replace)
                        selected_validators.append(np.random.choice(new_candidates))
            
            # Tạo bằng chứng ZK nếu được bật
            if self.zk_enabled and self.zk_system:
                self._generate_selection_proof(shard_id, block_num, selected_validators)
        
        # Cập nhật validators hiện tại và lịch sử
        self.active_validators[shard_id] = selected_validators
        self.validator_history[shard_id].append((block_num, selected_validators.copy()))
        
        # Cập nhật thống kê
        end_time = time.time()
        selection_time = end_time - start_time
        self.stats["selections"] += 1
        self.stats["total_selection_time"] += selection_time
        self.stats["avg_selection_time"] = self.stats["total_selection_time"] / self.stats["selections"]
        
        # Tính toán đa dạng của validators
        if len(self.validator_history[shard_id]) >= 2:
            self._calculate_validator_diversity(shard_id)
        
        return selected_validators
    
    def _apply_selection_policy(self, trusted_nodes: List[Dict[str, Any]], block_num: int) -> List[Dict[str, Any]]:
        """
        Áp dụng chính sách lựa chọn validator.
        
        Args:
            trusted_nodes: Danh sách các node đáng tin cậy với thông tin chi tiết
            block_num: Số thứ tự block
            
        Returns:
            List[Dict[str, Any]]: Danh sách các node được lọc theo chính sách
        """
        if self.policy == ValidatorSelectionPolicy.RANDOM:
            # Chọn ngẫu nhiên, không quan tâm đến điểm
            np.random.shuffle(trusted_nodes)
            return trusted_nodes
            
        elif self.policy == ValidatorSelectionPolicy.REPUTATION:
            # Sắp xếp theo điểm tin cậy
            return sorted(trusted_nodes, key=lambda x: x["trust_score"], reverse=True)
            
        elif self.policy == ValidatorSelectionPolicy.STAKE_WEIGHTED:
            # Nếu có thông tin stake, sắp xếp theo đó
            if all("stake" in node for node in trusted_nodes):
                return sorted(trusted_nodes, key=lambda x: x["stake"], reverse=True)
            else:
                # Nếu không, dùng điểm tổng hợp
                return sorted(trusted_nodes, key=lambda x: x["composite_score"], reverse=True)
                
        elif self.policy == ValidatorSelectionPolicy.PERFORMANCE:
            # Sắp xếp theo tỷ lệ thành công và thời gian phản hồi
            return sorted(trusted_nodes, 
                         key=lambda x: (x["success_rate"], -x["response_time"]), 
                         reverse=True)
                
        elif self.policy == ValidatorSelectionPolicy.HYBRID:
            # Kết hợp nhiều tiêu chí với hệ số khác nhau
            # Sử dụng điểm tổng hợp đã tính từ trust manager
            nodes = sorted(trusted_nodes, key=lambda x: x["composite_score"], reverse=True)
            
            # Thêm yếu tố ngẫu nhiên nhỏ để tránh trường hợp luôn chọn các node cố định
            # Điều này chỉ áp dụng khi có nhiều node hơn cần chọn
            if len(nodes) > 3:
                randomness = np.random.random(len(nodes)) * 0.05  # Thêm tối đa 5% ngẫu nhiên
                adjusted_scores = [node["composite_score"] * (1 + rand) for node, rand in zip(nodes, randomness)]
                nodes_with_scores = list(zip(nodes, adjusted_scores))
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                nodes = [node for node, _ in nodes_with_scores]
            
            return nodes
        
        # Mặc định, trả về danh sách gốc
        return trusted_nodes
    
    def _generate_selection_proof(self, shard_id: int, block_num: int, selected_validators: List[int]):
        """
        Tạo bằng chứng ZK cho việc lựa chọn validator.
        
        Args:
            shard_id: ID của shard
            block_num: Số thứ tự block
            selected_validators: Danh sách validator được chọn
        """
        if not self.zk_system:
            return
            
        # Tạo dữ liệu cho bằng chứng
        proof_data = {
            "shard_id": shard_id,
            "block_num": block_num,
            "validators": selected_validators,
            "policy": self.policy,
            "timestamp": time.time()
        }
        
        # Tạo bằng chứng
        proof = self.zk_system.generate_proof(
            data=proof_data,
            proof_type=ProofType.SET_MEMBERSHIP
        )
        
        # Cập nhật năng lượng tiết kiệm
        if "energy_cost" in proof:
            self.stats["energy_saved"] += proof.get("energy_cost", 0) * 0.3
    
    def _calculate_validator_diversity(self, shard_id: int):
        """
        Tính toán mức độ đa dạng của validators qua các vòng.
        
        Args:
            shard_id: ID của shard
        """
        if shard_id not in self.validator_history or len(self.validator_history[shard_id]) < 2:
            return
            
        # Lấy các validator set gần đây
        recent_history = self.validator_history[shard_id][-10:]  # Tối đa 10 vòng gần nhất
        
        # Tính tỷ lệ validators khác nhau
        unique_validators = set()
        for _, validators in recent_history:
            unique_validators.update(validators)
        
        total_slots = sum(len(validators) for _, validators in recent_history)
        diversity = len(unique_validators) / total_slots if total_slots > 0 else 0
        
        # Cập nhật thống kê
        self.stats["validator_diversity"] = diversity
    
    def verify_selection(self, shard_id: int, block_num: int, validators: List[int]) -> bool:
        """
        Xác minh tính hợp lệ của một bộ validators đã chọn.
        
        Args:
            shard_id: ID của shard
            block_num: Số thứ tự block
            validators: Danh sách validator cần xác minh
            
        Returns:
            bool: True nếu bộ validators hợp lệ
        """
        # Kiểm tra xem có khớp với lịch sử không
        if shard_id in self.validator_history:
            for round_num, round_validators in self.validator_history[shard_id]:
                if round_num == block_num:
                    return set(validators) == set(round_validators)
        
        # Nếu không tìm thấy trong lịch sử, thực hiện kiểm tra hợp lệ
        # Nguyên tắc: validators phải có điểm tin cậy cao
        trusted_nodes = self.trust_manager.recommend_trusted_validators(
            shard_id, count=len(validators) * 2
        )
        
        if not trusted_nodes:
            return False
            
        trusted_ids = {node["node_id"] for node in trusted_nodes}
        return all(validator in trusted_ids for validator in validators)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về hệ thống lựa chọn validator.
        
        Returns:
            Dict[str, Any]: Thống kê về việc sử dụng
        """
        stats = self.stats.copy()
        
        # Thêm thông tin từ ZK system nếu có
        if self.zk_enabled and self.zk_system:
            zk_stats = self.zk_system.get_statistics()
            stats["zk_proofs_generated"] = zk_stats["proofs_generated"]
            stats["zk_proofs_verified"] = zk_stats["proofs_verified"]
            stats["zk_energy_saved"] = zk_stats["energy_saved"]
        
        # Thêm thông tin về cấu hình
        stats.update({
            "policy": self.policy,
            "zk_enabled": self.zk_enabled,
            "use_rotation": self.use_rotation,
            "rotation_period": self.rotation_period,
            "active_shards": len(self.active_validators),
            "total_rounds": self.current_round
        })
        
        return stats 