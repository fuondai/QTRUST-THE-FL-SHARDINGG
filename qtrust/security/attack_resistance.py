import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

class AttackType:
    """Các loại tấn công được hỗ trợ phát hiện."""
    ECLIPSE = "eclipse"
    SYBIL = "sybil"
    MAJORITY = "majority"
    DOS = "dos"
    REPLAY = "replay"
    MALLEABILITY = "malleability"
    PVT_KEY_COMPROMISE = "private_key_compromise"
    TIMEJACKING = "timejacking"
    SMURFING = "smurfing"
    SELFISH_MINING = "selfish_mining"

class AttackResistanceSystem:
    """
    Hệ thống chống tấn công cho blockchain.
    
    Phát hiện và phản ứng với các tấn công bằng cách sử dụng
    kết hợp quy tắc cứng và machine learning.
    """
    
    def __init__(self, 
                trust_manager, 
                validator_selector = None,
                network = None,
                detection_threshold: float = 0.65,
                auto_response: bool = True,
                collect_evidence: bool = True):
        """
        Khởi tạo hệ thống chống tấn công.
        
        Args:
            trust_manager: Trình quản lý tin cậy của hệ thống
            validator_selector: Hệ thống chọn validator (tùy chọn)
            network: Đồ thị mạng blockchain (tùy chọn)
            detection_threshold: Ngưỡng phát hiện tấn công
            auto_response: Tự động phản ứng với tấn công
            collect_evidence: Thu thập bằng chứng về tấn công
        """
        self.trust_manager = trust_manager
        self.validator_selector = validator_selector
        self.network = network
        self.detection_threshold = detection_threshold
        self.auto_response = auto_response
        self.collect_evidence = collect_evidence
        
        # Trạng thái hệ thống
        self.under_attack = False
        self.active_attacks = {}
        self.attack_evidence = defaultdict(list)
        self.attack_history = []
        
        # Lưu trữ các quy tắc phát hiện tùy chỉnh
        self.custom_detection_rules = []
        
        # Thống kê hoạt động
        self.stats = {
            "total_scans": 0,
            "attacks_detected": 0,
            "false_positives": 0,
            "mitigations_applied": 0
        }
    
    def scan_for_attacks(self, transaction_history: List[Dict[str, Any]], 
                        network_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Quét để phát hiện các tấn công tiềm năng.
        
        Args:
            transaction_history: Lịch sử giao dịch gần đây
            network_state: Trạng thái mạng hiện tại (tùy chọn)
            
        Returns:
            Dict[str, Any]: Kết quả quét với các tấn công được phát hiện
        """
        # Cập nhật số lần quét
        self.stats["total_scans"] += 1
        
        # Kết quả mặc định
        result = {
            "under_attack": False,
            "new_attacks_detected": [],
            "active_attacks": self.active_attacks.copy(),
            "confidence": 0.0,
            "recommendations": []
        }
        
        # Gọi phương thức phát hiện tấn công từ trust manager
        if hasattr(self.trust_manager, "detect_advanced_attacks"):
            attack_detection = self.trust_manager.detect_advanced_attacks(transaction_history)
            
            # Cập nhật kết quả từ phát hiện
            if attack_detection["under_attack"]:
                result["under_attack"] = True
                result["confidence"] = max(result["confidence"], attack_detection["confidence"])
                
                # Thêm thông tin về các tấn công
                for attack_type in attack_detection["attack_types"]:
                    if attack_type not in self.active_attacks:
                        result["new_attacks_detected"].append(attack_type)
                        
                    # Cập nhật hoặc thêm vào active_attacks
                    self.active_attacks[attack_type] = {
                        "first_detected": time.time() if attack_type not in self.active_attacks else 
                                        self.active_attacks[attack_type]["first_detected"],
                        "last_detected": time.time(),
                        "confidence": attack_detection["confidence"],
                        "suspect_nodes": attack_detection["suspect_nodes"],
                        "recommended_actions": attack_detection.get("recommended_actions", [])
                    }
                
                # Cập nhật hành động được đề xuất
                result["recommendations"].extend(attack_detection.get("recommended_actions", []))
                
                # Cập nhật số lượng tấn công phát hiện được
                if result["new_attacks_detected"]:
                    self.stats["attacks_detected"] += len(result["new_attacks_detected"])
                
                # Thu thập bằng chứng nếu được bật
                if self.collect_evidence:
                    for attack_type in attack_detection["attack_types"]:
                        evidence = {
                            "time": time.time(),
                            "attack_type": attack_type,
                            "confidence": attack_detection["confidence"],
                            "suspect_nodes": attack_detection["suspect_nodes"].copy(),
                            "transaction_sample": transaction_history[:5] if transaction_history else []
                        }
                        self.attack_evidence[attack_type].append(evidence)
        
        # Bổ sung kiểm tra thêm nếu có network state
        if network_state and self.network:
            # Kiểm tra tấn công Sybil
            sybil_confidence = self._check_for_sybil_attack(network_state)
            if sybil_confidence > self.detection_threshold:
                attack_type = AttackType.SYBIL
                result["under_attack"] = True
                result["confidence"] = max(result["confidence"], sybil_confidence)
                
                if attack_type not in self.active_attacks:
                    result["new_attacks_detected"].append(attack_type)
                
                self.active_attacks[attack_type] = {
                    "first_detected": time.time() if attack_type not in self.active_attacks else 
                                    self.active_attacks[attack_type]["first_detected"],
                    "last_detected": time.time(),
                    "confidence": sybil_confidence,
                    "suspect_nodes": self._identify_sybil_suspects(network_state),
                    "recommended_actions": [
                        "Tăng yêu cầu stake cho validators",
                        "Áp dụng luân chuyển validator",
                        "Kích hoạt tường lửa nhận dạng Sybil"
                    ]
                }
                
                # Cập nhật hành động được đề xuất
                result["recommendations"].extend(self.active_attacks[attack_type]["recommended_actions"])
            
            # Kiểm tra tấn công Eclipse
            eclipse_confidence = self._check_for_eclipse_attack(network_state)
            if eclipse_confidence > self.detection_threshold:
                attack_type = AttackType.ECLIPSE
                result["under_attack"] = True
                result["confidence"] = max(result["confidence"], eclipse_confidence)
                
                if attack_type not in self.active_attacks:
                    result["new_attacks_detected"].append(attack_type)
                
                self.active_attacks[attack_type] = {
                    "first_detected": time.time() if attack_type not in self.active_attacks else 
                                    self.active_attacks[attack_type]["first_detected"],
                    "last_detected": time.time(),
                    "confidence": eclipse_confidence,
                    "suspect_nodes": self._identify_eclipse_suspects(network_state),
                    "recommended_actions": [
                        "Mở rộng kết nối mesh giữa các node",
                        "Thêm seed nodes tin cậy",
                        "Áp dụng thay đổi vòng đời kết nối"
                    ]
                }
                
                # Cập nhật hành động được đề xuất
                result["recommendations"].extend(self.active_attacks[attack_type]["recommended_actions"])
        
        # Cập nhật trạng thái hệ thống
        self.under_attack = result["under_attack"]
        
        # Tự động phản ứng nếu được bật
        if self.under_attack and self.auto_response:
            self._apply_attack_mitigations(result)
        
        return result
    
    def _check_for_sybil_attack(self, network_state: Dict[str, Any]) -> float:
        """
        Phát hiện tấn công Sybil.
        
        Args:
            network_state: Trạng thái mạng hiện tại
            
        Returns:
            float: Độ tin cậy của phát hiện (0.0-1.0)
        """
        # Mô phỏng phát hiện Sybil dựa trên thống kê mạng
        if not network_state or not self.network:
            return 0.0
        
        # Phân tích cấu trúc mạng
        total_nodes = len(self.network.nodes)
        if total_nodes < 5:
            return 0.0
            
        # Kiểm tra các node có kết nối bất thường
        suspected_count = 0
        connection_counts = {}
        
        for node in self.network.nodes:
            connection_counts[node] = len(list(self.network.neighbors(node)))
        
        # Tính trung bình và độ lệch chuẩn
        avg_connections = np.mean(list(connection_counts.values()))
        std_connections = np.std(list(connection_counts.values()))
        
        # Node có quá nhiều kết nối có thể là điều khiển nhiều Sybil
        for node, count in connection_counts.items():
            if count > avg_connections + 2 * std_connections:
                suspected_count += 1
        
        # Tính độ tin cậy dựa trên tỷ lệ node đáng ngờ
        sybil_confidence = min(0.95, suspected_count / (total_nodes * 0.1))
        
        return sybil_confidence
    
    def _check_for_eclipse_attack(self, network_state: Dict[str, Any]) -> float:
        """
        Phát hiện tấn công Eclipse.
        
        Args:
            network_state: Trạng thái mạng hiện tại
            
        Returns:
            float: Độ tin cậy của phát hiện (0.0-1.0)
        """
        # Mô phỏng phát hiện Eclipse dựa trên thống kê mạng
        if not network_state or not self.network:
            return 0.0
        
        # Phân tích cấu trúc mạng
        total_nodes = len(self.network.nodes)
        if total_nodes < 5:
            return 0.0
            
        # Kiểm tra các node có kết nối thấp (có thể bị cô lập)
        isolated_count = 0
        connection_counts = {}
        
        for node in self.network.nodes:
            connection_counts[node] = len(list(self.network.neighbors(node)))
        
        # Tính trung bình và độ lệch chuẩn
        avg_connections = np.mean(list(connection_counts.values()))
        
        # Node có ít kết nối có thể đang bị tấn công Eclipse
        for node, count in connection_counts.items():
            if count < avg_connections * 0.3:  # Ít hơn 30% trung bình
                isolated_count += 1
        
        # Tính độ tin cậy dựa trên tỷ lệ node bị cô lập
        eclipse_confidence = min(0.95, isolated_count / (total_nodes * 0.1))
        
        return eclipse_confidence
    
    def _identify_sybil_suspects(self, network_state: Dict[str, Any]) -> List[int]:
        """Xác định các node đáng ngờ trong tấn công Sybil."""
        suspects = []
        
        if not network_state or not self.network:
            return suspects
            
        # Các node có quá nhiều kết nối có thể là điều khiển Sybil
        connection_counts = {}
        for node in self.network.nodes:
            connection_counts[node] = len(list(self.network.neighbors(node)))
        
        # Tính trung bình và độ lệch chuẩn
        avg_connections = np.mean(list(connection_counts.values()))
        std_connections = np.std(list(connection_counts.values()))
        
        # Node có quá nhiều kết nối
        for node, count in connection_counts.items():
            if count > avg_connections + 2.5 * std_connections:
                suspects.append(node)
        
        return suspects
    
    def _identify_eclipse_suspects(self, network_state: Dict[str, Any]) -> List[int]:
        """Xác định các node đáng ngờ trong tấn công Eclipse."""
        suspects = []
        
        if not network_state or not self.network:
            return suspects
            
        # Phân tích cấu trúc kết nối
        for node in self.network.nodes:
            neighbors = list(self.network.neighbors(node))
            
            # Kiểm tra nếu các kết nối của node đều đến từ một nhóm nhỏ node
            if len(neighbors) >= 3:
                # Kiểm tra độ liên kết giữa các neighbors
                interconnections = 0
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        if self.network.has_edge(neighbors[i], neighbors[j]):
                            interconnections += 1
                
                max_interconnections = (len(neighbors) * (len(neighbors) - 1)) / 2
                
                # Nếu các neighbors có liên kết cao với nhau, có thể là một nhóm tấn công
                if interconnections / max_interconnections > 0.7:
                    suspects.extend(neighbors)
        
        return list(set(suspects))  # Loại bỏ trùng lặp
    
    def _apply_attack_mitigations(self, attack_result: Dict[str, Any]):
        """
        Áp dụng các biện pháp giảm thiểu tấn công.
        
        Args:
            attack_result: Kết quả quét tấn công
        """
        # Không làm gì nếu không phát hiện tấn công
        if not attack_result["under_attack"]:
            return
            
        # Các biện pháp giảm thiểu áp dụng
        mitigations_applied = []
        
        # 1. Tăng cường hệ thống tin cậy
        if hasattr(self.trust_manager, "enhance_security_posture"):
            self.trust_manager.enhance_security_posture(attack_result)
            mitigations_applied.append("trust_system_enhanced")
        
        # 2. Điều chỉnh validator selection nếu có
        if self.validator_selector and hasattr(self.validator_selector, "update_security_level"):
            # Tăng mức độ bảo mật và buộc luân chuyển validator
            if attack_result["confidence"] > 0.8:
                self.validator_selector.update_security_level("high")
                mitigations_applied.append("validator_security_level_increased")
            
            # Kích hoạt luân chuyển validator
            if hasattr(self.validator_selector, "force_rotation"):
                self.validator_selector.force_rotation = True
                mitigations_applied.append("validator_rotation_forced")
        
        # 3. Đánh dấu các node đáng ngờ
        all_suspects = []
        for attack_info in self.active_attacks.values():
            all_suspects.extend(attack_info.get("suspect_nodes", []))
        
        # Loại bỏ trùng lặp
        all_suspects = list(set(all_suspects))
        
        # Ghi lại việc áp dụng biện pháp giảm thiểu
        self.stats["mitigations_applied"] += len(mitigations_applied)
        
        # Ghi lại vào lịch sử
        mitigation_record = {
            "time": time.time(),
            "attacks": list(self.active_attacks.keys()),
            "confidence": attack_result["confidence"],
            "mitigations_applied": mitigations_applied,
            "suspect_nodes_count": len(all_suspects)
        }
        
        self.attack_history.append(mitigation_record)
    
    def add_custom_detection_rule(self, rule_function):
        """
        Thêm quy tắc phát hiện tùy chỉnh.
        
        Args:
            rule_function: Hàm kiểm tra tùy chỉnh 
                        (nhận network_state, trả về (attack_type, confidence, suspects))
        """
        self.custom_detection_rules.append(rule_function)
    
    def clear_attack_history(self, older_than_hours: float = 24.0):
        """
        Xóa lịch sử tấn công cũ.
        
        Args:
            older_than_hours: Số giờ để xác định lịch sử cũ
        """
        if not self.attack_history:
            return
            
        current_time = time.time()
        cutoff_time = current_time - older_than_hours * 3600
        
        # Lọc lịch sử tấn công
        self.attack_history = [record for record in self.attack_history 
                             if record["time"] >= cutoff_time]
        
        # Lọc bằng chứng
        for attack_type in list(self.attack_evidence.keys()):
            self.attack_evidence[attack_type] = [evidence for evidence in self.attack_evidence[attack_type]
                                               if evidence["time"] >= cutoff_time]
            
            # Xóa khóa nếu không còn bằng chứng
            if not self.attack_evidence[attack_type]:
                del self.attack_evidence[attack_type]
        
        # Xóa các cuộc tấn công không hoạt động
        for attack_type in list(self.active_attacks.keys()):
            if self.active_attacks[attack_type]["last_detected"] < cutoff_time:
                del self.active_attacks[attack_type]
    
    def get_attack_report(self) -> Dict[str, Any]:
        """
        Tạo báo cáo về các cuộc tấn công.
        
        Returns:
            Dict[str, Any]: Báo cáo chi tiết về tấn công
        """
        report = {
            "under_attack": self.under_attack,
            "active_attacks": len(self.active_attacks),
            "attack_types": list(self.active_attacks.keys()),
            "attack_history_count": len(self.attack_history),
            "stats": self.stats.copy(),
            "current_recommendations": []
        }
        
        # Thêm các đề xuất từ các tấn công hiện tại
        for attack_info in self.active_attacks.values():
            report["current_recommendations"].extend(attack_info.get("recommended_actions", []))
        
        # Loại bỏ các đề xuất trùng lặp
        report["current_recommendations"] = list(set(report["current_recommendations"]))
        
        return report 