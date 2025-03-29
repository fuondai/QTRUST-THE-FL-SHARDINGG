import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

class HTDCMNode:
    """
    Đại diện cho thông tin tin cậy của một node trong hệ thống.
    """
    def __init__(self, node_id: int, shard_id: int, initial_trust: float = 0.7):
        """
        Khởi tạo thông tin tin cậy cho một node.
        
        Args:
            node_id: ID của node
            shard_id: ID của shard mà node thuộc về
            initial_trust: Điểm tin cậy ban đầu (0.0-1.0)
        """
        self.node_id = node_id
        self.shard_id = shard_id
        self.trust_score = initial_trust
        
        # Lưu trữ lịch sử hoạt động
        self.successful_txs = 0
        self.failed_txs = 0
        self.malicious_activities = 0
        self.response_times = []
        
        # Các tham số cho việc tính toán tin cậy
        self.alpha = 0.8  # Trọng số cho lịch sử tin cậy
        self.beta = 0.2   # Trọng số cho đánh giá mới
        
        # Lưu trữ đánh giá từ các node khác
        self.peer_ratings = defaultdict(lambda: 0.5)  # Node ID -> Rating
        
        # Lịch sử hành vi
        self.activity_history = deque(maxlen=100)  # Lưu trữ 100 hoạt động gần nhất
    
    def update_trust_score(self, new_rating: float):
        """
        Cập nhật điểm tin cậy dựa trên đánh giá mới.
        
        Args:
            new_rating: Đánh giá mới (0.0-1.0)
        """
        # Xử lý trường hợp cực đoan ngay từ đầu
        if new_rating >= 1.0:
            self.trust_score = 1.0
            return
        elif new_rating <= 0.0:
            self.trust_score = 0.0
            return
            
        # Cập nhật điểm tin cậy theo hàm trung bình có trọng số
        new_trust = self.alpha * self.trust_score + self.beta * new_rating
        
        # Đảm bảo điểm tin cậy nằm trong khoảng [0.0, 1.0]
        self.trust_score = max(0.0, min(1.0, new_trust))
    
    def record_transaction_result(self, success: bool, response_time: float, is_validator: bool):
        """
        Ghi lại kết quả một giao dịch mà node tham gia.
        
        Args:
            success: Giao dịch thành công hay thất bại
            response_time: Thời gian phản hồi (ms)
            is_validator: Node có là validator cho giao dịch hay không
        """
        if success:
            self.successful_txs += 1
            self.activity_history.append(('success', response_time, is_validator))
        else:
            self.failed_txs += 1
            self.activity_history.append(('fail', response_time, is_validator))
        
        self.response_times.append(response_time)
        
        # Giới hạn số lượng phản hồi lưu trữ
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_peer_rating(self, peer_id: int, rating: float):
        """
        Ghi lại đánh giá từ một node khác.
        
        Args:
            peer_id: ID của node đánh giá
            rating: Đánh giá (0.0-1.0)
        """
        self.peer_ratings[peer_id] = rating
    
    def record_malicious_activity(self, activity_type: str):
        """
        Ghi lại một hoạt động độc hại được phát hiện.
        
        Args:
            activity_type: Loại hoạt động độc hại
        """
        self.malicious_activities += 1
        self.activity_history.append(('malicious', activity_type, True))
        
        # Áp dụng hình phạt nghiêm khắc cho hoạt động độc hại - đặt trực tiếp thành 0
        self.trust_score = 0.0
    
    def get_average_response_time(self) -> float:
        """
        Lấy thời gian phản hồi trung bình gần đây.
        
        Returns:
            float: Thời gian phản hồi trung bình
        """
        if not self.response_times:
            return 0.0
        return np.mean(self.response_times)
    
    def get_success_rate(self) -> float:
        """
        Lấy tỷ lệ thành công của các giao dịch.
        
        Returns:
            float: Tỷ lệ thành công
        """
        total = self.successful_txs + self.failed_txs
        if total == 0:
            return 0.0
        return self.successful_txs / total
    
    def get_peer_trust(self) -> float:
        """
        Lấy điểm tin cậy trung bình từ các node khác.
        
        Returns:
            float: Điểm tin cậy trung bình từ các peers
        """
        if not self.peer_ratings:
            return 0.5
        return np.mean(list(self.peer_ratings.values()))

class HTDCM:
    """
    Hierarchical Trust-based Data Center Mechanism (HTDCM).
    Cơ chế đánh giá tin cậy đa cấp cho mạng blockchain.
    """
    
    def __init__(self, 
                 network = None,
                 shards = None,
                 num_nodes = None,
                 tx_success_weight: float = 0.4,
                 response_time_weight: float = 0.2,
                 peer_rating_weight: float = 0.3,
                 history_weight: float = 0.1,
                 malicious_threshold: float = 0.25,
                 suspicious_pattern_window: int = 8):
        """
        Khởi tạo hệ thống đánh giá tin cậy HTDCM.
        
        Args:
            network: Đồ thị mạng blockchain (tùy chọn)
            shards: Danh sách các shard và node trong mỗi shard (tùy chọn)
            num_nodes: Tổng số node trong mạng (tùy chọn)
            tx_success_weight: Trọng số cho tỷ lệ giao dịch thành công
            response_time_weight: Trọng số cho thời gian phản hồi
            peer_rating_weight: Trọng số cho đánh giá từ các node khác
            history_weight: Trọng số cho lịch sử hành vi
            malicious_threshold: Ngưỡng điểm tin cậy để coi là độc hại
            suspicious_pattern_window: Kích thước cửa sổ để phát hiện mẫu đáng ngờ
        """
        # Trọng số cho các yếu tố khác nhau trong đánh giá tin cậy
        self.tx_success_weight = tx_success_weight
        self.response_time_weight = response_time_weight
        self.peer_rating_weight = peer_rating_weight
        self.history_weight = history_weight
        
        # Ngưỡng và tham số phát hiện độc hại
        self.malicious_threshold = malicious_threshold
        self.suspicious_pattern_window = suspicious_pattern_window
        
        # Nếu dùng constructor đơn giản với num_nodes
        if num_nodes is not None and (network is None or shards is None):
            self.network = None
            self.num_shards = 1  # Mặc định
            
            # Điểm tin cậy của các shard
            self.shard_trust_scores = np.ones(self.num_shards) * 0.7
            
            # Khởi tạo thông tin tin cậy cho mỗi node
            self.nodes = {}
            for node_id in range(num_nodes):
                shard_id = node_id % self.num_shards
                self.nodes[node_id] = HTDCMNode(node_id, shard_id, 0.7)
            
            # Giả định các shard có số lượng node bằng nhau
            self.shards = []
            for i in range(self.num_shards):
                self.shards.append([node_id for node_id in range(num_nodes) if node_id % self.num_shards == i])
        else:
            # Constructor gốc
            self.network = network
            self.shards = shards
            self.num_shards = len(shards) if shards else 1
            
            # Điểm tin cậy của các shard
            self.shard_trust_scores = np.ones(self.num_shards) * 0.7
            
            # Khởi tạo thông tin tin cậy cho mỗi node
            self.nodes = {}
            if network and shards:
                for shard_id, shard_nodes in enumerate(shards):
                    for node_id in shard_nodes:
                        initial_trust = self.network.nodes[node_id].get('trust_score', 0.7)
                        self.nodes[node_id] = HTDCMNode(node_id, shard_id, initial_trust)
            
        # Lịch sử đánh giá toàn cục
        self.global_ratings_history = []
        
        # Theo dõi toàn cục cho sự phát hiện tấn công phối hợp
        self.suspected_nodes = set()  
        self.attack_patterns = {} # Lưu trữ các mẫu tấn công đã được phát hiện
        
        # Cờ xác định trạng thái tấn công toàn cục
        self.under_attack = False
    
    def update_node_trust(self, 
                        node_id: int, 
                        tx_success: bool, 
                        response_time: float, 
                        is_validator: bool):
        """
        Cập nhật điểm tin cậy của một node dựa trên kết quả giao dịch.
        
        Args:
            node_id: ID của node
            tx_success: Kết quả giao dịch (True nếu thành công, False nếu thất bại)
            response_time: Thời gian phản hồi (ms)
            is_validator: Node có là validator cho giao dịch hay không
        """
        # Kiểm tra node có tồn tại trong hệ thống không
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Ghi lại kết quả giao dịch
        node.record_transaction_result(tx_success, response_time, is_validator)
        
        # Tính toán rating dựa trên nhiều yếu tố - tối ưu hóa quy trình tính toán
        # Sử dụng công thức trực tiếp thay vì gọi _calculate_node_rating
        success_rate = node.get_success_rate()
        avg_response_time = node.get_average_response_time()
        normalized_response_time = 1.0 - min(1.0, avg_response_time / 100.0)
        peer_trust = node.get_peer_trust()
        history_score = 1.0 - min(1.0, node.malicious_activities / 10.0)
        
        rating = (self.tx_success_weight * success_rate +
                 self.response_time_weight * normalized_response_time +
                 self.peer_rating_weight * peer_trust +
                 self.history_weight * history_score)
        
        # Cập nhật điểm tin cậy
        node.update_trust_score(rating)
        
        # Cập nhật điểm tin cậy trong mạng (nếu có)
        if self.network is not None:
            self.network.nodes[node_id]['trust_score'] = node.trust_score
        
        # Cập nhật điểm tin cậy của shard
        self._update_shard_trust(node.shard_id)
        
        # Lưu lịch sử đánh giá toàn cục
        self.global_ratings_history.append((node_id, node.trust_score))
        
        # Phát hiện hoạt động đáng ngờ - chỉ kiểm tra nếu node có điểm tin cậy thấp hoặc
        # lịch sử đủ dài để phát hiện mẫu
        if (node.trust_score < 0.4 or 
            (len(node.activity_history) >= self.suspicious_pattern_window and 
             node.successful_txs + node.failed_txs > 15)):
            self._detect_suspicious_behavior(node_id)
    
    def _calculate_node_rating(self, node: HTDCMNode) -> float:
        """
        Tính toán đánh giá cho một node dựa trên nhiều yếu tố.
        
        Args:
            node: Đối tượng HTDCMNode
            
        Returns:
            float: Đánh giá (0.0-1.0)
        """
        # Tỷ lệ giao dịch thành công
        success_rate = node.get_success_rate()
        
        # Thời gian phản hồi (chuẩn hóa: thấp hơn = tốt hơn)
        avg_response_time = node.get_average_response_time()
        normalized_response_time = 1.0 - min(1.0, avg_response_time / 100.0)  # Giả sử 100ms là tối đa
        
        # Đánh giá từ các peer
        peer_trust = node.get_peer_trust()
        
        # Điểm lịch sử (các hoạt động độc hại trước đây)
        history_score = 1.0 - min(1.0, node.malicious_activities / 10.0)  # Giả sử 10 hoạt động là tối đa
        
        # Tính điểm tổng hợp
        rating = (self.tx_success_weight * success_rate +
                 self.response_time_weight * normalized_response_time +
                 self.peer_rating_weight * peer_trust +
                 self.history_weight * history_score)
        
        return rating
    
    def _update_shard_trust(self, shard_id: int):
        """
        Cập nhật điểm tin cậy cho một shard dựa trên các node trong shard đó.
        
        Args:
            shard_id: ID của shard
        """
        # Lấy tất cả các node trong shard
        shard_nodes = [self.nodes[node_id] for node_id in self.shards[shard_id]]
        
        # Tính điểm tin cậy trung bình
        avg_trust = np.mean([node.trust_score for node in shard_nodes])
        
        # Cập nhật điểm tin cậy của shard
        self.shard_trust_scores[shard_id] = avg_trust
    
    def _detect_suspicious_behavior(self, node_id: int):
        """
        Phát hiện các hành vi đáng ngờ của một node.
        
        Args:
            node_id: ID của node cần kiểm tra
            
        Returns:
            bool: True nếu phát hiện hoạt động đáng ngờ, False nếu không
        """
        node = self.nodes[node_id]
        
        # Nếu điểm tin cậy dưới ngưỡng, coi là độc hại - tăng hiệu suất bằng cách kiểm tra ngay
        if node.trust_score < self.malicious_threshold:
            node.record_malicious_activity('low_trust')
            self.suspected_nodes.add(node_id)
            self._check_for_coordinated_attack()
            return True
        
        # Sử dụng cache để không phải kiểm tra quá thường xuyên - kiểm tra mỗi 3-5 giao dịch
        recent_tx_count = node.successful_txs + node.failed_txs
        last_check = getattr(node, 'last_suspicion_check', 0)
        if recent_tx_count - last_check < 4 and node.trust_score > 0.4:
            return False
        
        node.last_suspicion_check = recent_tx_count
        
        # Phát hiện mẫu đáng ngờ trong lịch sử hoạt động
        if len(node.activity_history) >= self.suspicious_pattern_window:
            recent_activities = list(node.activity_history)[-self.suspicious_pattern_window:]
            
            # 1. Kiểm tra nếu node liên tục thất bại trong các giao dịch - sử dụng counter để tối ưu
            fail_count = sum(1 for act in recent_activities if act[0] == 'fail')
            if fail_count >= self.suspicious_pattern_window * 0.65:  # Giảm ngưỡng từ 0.7 xuống 0.65
                node.record_malicious_activity('consistent_failure')
                self.suspected_nodes.add(node_id)
                self._check_for_coordinated_attack()
                return True
            
            # 2. Kiểm tra nếu node có thời gian phản hồi bất thường - chỉ khi fail_count dưới ngưỡng
            if fail_count < self.suspicious_pattern_window * 0.5:
                # Lấy thời gian phản hồi từ lịch sử hoạt động
                response_times = [act[1] for act in recent_activities if isinstance(act[1], (int, float))]
                if response_times and len(response_times) >= 6:
                    # Tính toán hiệu quả hơn bằng cách lưu trữ giá trị trung bình và độ lệch chuẩn
                    mean_time = np.mean(response_times)
                    std_time = np.std(response_times)
                    if std_time > 2.0 * mean_time:
                        node.record_malicious_activity('erratic_response_time')
                        self.suspected_nodes.add(node_id)
                        self._check_for_coordinated_attack()
                        return True
            
            # 3. Phát hiện mẫu hoạt động dao động - chỉ khi chưa phát hiện các mẫu khác
            success_fail_pattern = [1 if act[0] == 'success' else 0 for act in recent_activities]
            if len(success_fail_pattern) >= 6:
                # Tối ưu hóa tính toán tương quan
                if self._check_alternating_pattern(success_fail_pattern, 0.5):
                    node.record_malicious_activity('oscillating_behavior')
                    self.suspected_nodes.add(node_id)
                    self._check_for_coordinated_attack()
                    return True
            
            # 4 & 5. Bỏ qua một số kiểm tra phức tạp hơn nếu node có điểm tin cậy cao
            if node.trust_score < 0.65:
                # Lấy thời gian phản hồi từ lịch sử hoạt động (nếu chưa lấy ở bước trước)
                if not 'response_times' in locals():
                    response_times = [act[1] for act in recent_activities if isinstance(act[1], (int, float))]
                
                # 4. Phát hiện hoạt động bất thường dựa trên phân phối thời gian phản hồi
                if len(response_times) >= 6:  # Giảm từ 8 xuống 6
                    # Kiểm tra khoảng cách tối đa giữa các thời gian
                    sorted_times = sorted(response_times)
                    differences = np.diff(sorted_times)
                    if differences.size > 0 and np.max(differences) > 4 * np.mean(differences):
                        node.record_malicious_activity('bimodal_response_times')
                        self.suspected_nodes.add(node_id)
                        self._check_for_coordinated_attack()
                        return True
                
                # 5. Phát hiện hành vi "sleeper agent" - node hoạt động tốt ở đầu rồi xấu đi
                if len(response_times) >= 8:  # Giảm từ 10 xuống 8
                    n = len(recent_activities) // 2
                    early_success_rate = sum(1 for act in recent_activities[:n] if act[0] == 'success') / n
                    late_success_rate = sum(1 for act in recent_activities[n:] if act[0] == 'success') / n
                    
                    if early_success_rate > 0.7 and late_success_rate < 0.5 and early_success_rate - late_success_rate > 0.3:
                        node.record_malicious_activity('sleeper_agent')
                        self.suspected_nodes.add(node_id)
                        self._check_for_coordinated_attack()
                        return True
        
        # 6. Phân tích so với các node khác - chỉ kiểm tra nếu node có điểm tin cậy dưới ngưỡng
        if node.trust_score < 0.6:
            shard_id = node.shard_id
            shard_nodes = [self.nodes[n_id] for n_id in self.shards[shard_id] if n_id != node_id]
            
            if shard_nodes:
                other_nodes_avg_trust = np.mean([n.trust_score for n in shard_nodes])
                if node.trust_score < other_nodes_avg_trust * 0.65 and other_nodes_avg_trust > 0.45:
                    node.record_malicious_activity('significant_trust_deviation')
                    self.suspected_nodes.add(node_id)
                    self._check_for_coordinated_attack()
                    return True
                    
        # 7. Thêm: Phát hiện các node có điểm tin cậy biến động bất thường - chỉ khi có lịch sử dài
        if len(self.global_ratings_history) > 10:
            # Lọc lịch sử đánh giá cho node hiện tại - chỉ lấy 5 điểm dữ liệu gần nhất
            node_history = [(idx, rating) for idx, (rated_node, rating) in enumerate(self.global_ratings_history[-20:]) 
                           if rated_node == node_id]
            
            if len(node_history) >= 5:
                # Tính tốc độ thay đổi điểm tin cậy
                ratings = [r for (_, r) in node_history[-5:]]
                changes = np.abs(np.diff(ratings))
                
                # Nếu có thay đổi đột ngột
                if np.max(changes) > 0.25:
                    node.record_malicious_activity('trust_score_volatility')
                    self.suspected_nodes.add(node_id)
                    self._check_for_coordinated_attack()
                    return True
        
        return False
    
    def _check_alternating_pattern(self, pattern, threshold=0.5):
        """
        Kiểm tra mẫu luân phiên 0/1 trong một dãy.
        
        Args:
            pattern: Danh sách cần kiểm tra
            threshold: Ngưỡng xác định mẫu (0.0-1.0)
            
        Returns:
            bool: True nếu phát hiện mẫu luân phiên
        """
        if len(pattern) < 4:
            return False
            
        # Đếm số lần chuyển đổi giữa 0 và 1
        alternations = sum(1 for i in range(1, len(pattern)) if pattern[i] != pattern[i-1])
        max_alternations = len(pattern) - 1
        
        # Nếu tỷ lệ luân phiên cao, có thể là mẫu đáng ngờ
        return alternations / max_alternations >= threshold
    
    def _check_for_coordinated_attack(self):
        """
        Phát hiện các cuộc tấn công phối hợp dựa trên mẫu hoạt động đáng ngờ từ nhiều node.
        """
        # Nếu có quá nhiều node bị nghi ngờ, có thể đang có tấn công phối hợp
        suspected_ratio = len(self.suspected_nodes) / sum(len(shard) for shard in self.shards)
        
        if suspected_ratio > 0.15:  # Hơn 15% node bị nghi ngờ
            self.under_attack = True
            
            # Phân tích phân bố của các node bị nghi ngờ
            nodes_per_shard = defaultdict(int)
            for node_id in self.suspected_nodes:
                shard_id = self.nodes[node_id].shard_id
                nodes_per_shard[shard_id] += 1
            
            # Xác định các shard bị ảnh hưởng nặng nề nhất
            shard_sizes = {i: len(shard) for i, shard in enumerate(self.shards)}
            shard_ratios = {shard_id: count / shard_sizes[shard_id] 
                          for shard_id, count in nodes_per_shard.items()}
            
            # Lưu lại mẫu tấn công để theo dõi
            self.attack_patterns['affected_shards'] = [shard_id for shard_id, ratio in shard_ratios.items() 
                                                   if ratio > 0.3]  # Shard có > 30% node bị nghi ngờ
            
            # Đánh dấu tất cả các node trong các shard bị ảnh hưởng nặng
            for shard_id in self.attack_patterns['affected_shards']:
                for node_id in self.shards[shard_id]:
                    # Giảm score của tất cả các node trong shard bị ảnh hưởng
                    if node_id not in self.suspected_nodes:
                        self.nodes[node_id].trust_score = max(0.4, self.nodes[node_id].trust_score * 0.8)
                    # Cập nhật điểm trust_score trong network
                    if self.network is not None:
                        self.network.nodes[node_id]['trust_score'] = self.nodes[node_id].trust_score
                    
            # Cập nhật điểm tin cậy của shard bị ảnh hưởng
            for shard_id in self.attack_patterns['affected_shards']:
                self._update_shard_trust(shard_id)
                
            # Ghi lại thời điểm phát hiện tấn công
            self.attack_patterns['detection_time'] = len(self.global_ratings_history)
            
            # Thực hiện các biện pháp ứng phó
            self._attack_response()
    
    def _attack_response(self):
        """
        Thực hiện các biện pháp ứng phó khi phát hiện tấn công.
        """
        if not self.under_attack:
            return
            
        # 1. Tăng giám sát - giảm cỡ cửa sổ phát hiện
        self.suspicious_pattern_window = max(4, self.suspicious_pattern_window - 2)
        
        # 2. Điều chỉnh trọng số đánh giá để tin cậy hơn vào lịch sử
        self.history_weight = min(0.4, self.history_weight * 2)
        self.tx_success_weight = max(0.3, self.tx_success_weight * 0.8)
        
        # 3. Xác định các node đáng tin cậy nhất trong mỗi shard
        trusted_nodes_per_shard = {}
        for shard_id in range(self.num_shards):
            trusted_nodes = self.recommend_trusted_validators(shard_id, count=max(3, len(self.shards[shard_id]) // 3))
            trusted_nodes_per_shard[shard_id] = trusted_nodes
            
            # Tăng cường điểm tin cậy cho các node đáng tin cậy
            for node_id in trusted_nodes:
                self.nodes[node_id].trust_score = min(1.0, self.nodes[node_id].trust_score * 1.2)
                if self.network is not None:
                    self.network.nodes[node_id]['trust_score'] = self.nodes[node_id].trust_score
        
        # 4. Lưu danh sách các node tin cậy để tham khảo
        self.attack_patterns['trusted_nodes'] = trusted_nodes_per_shard
    
    def rate_peers(self, observer_id: int, transactions: List[Dict[str, Any]]):
        """
        Cho phép một node đánh giá các node khác dựa trên các giao dịch chung.
        
        Args:
            observer_id: ID của node quan sát
            transactions: Danh sách các giao dịch mà node quan sát tham gia
        """
        if observer_id not in self.nodes:
            return
        
        # Điểm tin cậy của node quan sát
        observer_trust = self.nodes[observer_id].trust_score
        
        # Tạo dictionary để theo dõi các node được quan sát
        observed_nodes = defaultdict(list)
        
        for tx in transactions:
            # Lấy danh sách các node tham gia trong giao dịch (ngoài observer)
            participant_nodes = []
            if 'source_node' in tx and tx['source_node'] != observer_id:
                participant_nodes.append(tx['source_node'])
            if 'destination_node' in tx and tx['destination_node'] != observer_id:
                participant_nodes.append(tx['destination_node'])
            
            # Thêm thông tin về sự tham gia của mỗi node trong giao dịch này
            for node_id in participant_nodes:
                if node_id in self.nodes:
                    observed_nodes[node_id].append({
                        'success': tx['status'] == 'completed',
                        'response_time': tx.get('completion_time', 0) - tx.get('created_at', 0)
                    })
        
        # Đánh giá từng node dựa trên hiệu suất trong các giao dịch chung
        for node_id, observations in observed_nodes.items():
            if not observations:
                continue
            
            # Tính tỷ lệ thành công và thời gian phản hồi trung bình
            success_rate = sum(1 for obs in observations if obs['success']) / len(observations)
            avg_response_time = np.mean([obs['response_time'] for obs in observations])
            
            # Chuẩn hóa thời gian phản hồi
            normalized_response_time = 1.0 - min(1.0, avg_response_time / 100.0)
            
            # Tính rating tổng hợp
            rating = 0.7 * success_rate + 0.3 * normalized_response_time
            
            # Lưu đánh giá vào node được quan sát
            self.nodes[node_id].record_peer_rating(observer_id, rating)
            
            # Cập nhật điểm tin cậy của node được quan sát (với trọng số thấp hơn)
            peer_influence = min(0.1, observer_trust * 0.2)  # Trọng số của đánh giá từ peer
            self.nodes[node_id].update_trust_score(
                self.nodes[node_id].trust_score * (1 - peer_influence) + rating * peer_influence
            )
    
    def get_node_trust_scores(self) -> Dict[int, float]:
        """
        Lấy danh sách điểm tin cậy của tất cả các node.
        
        Returns:
            Dict[int, float]: Dictionary ánh xạ node ID đến điểm tin cậy
        """
        return {node_id: node.trust_score for node_id, node in self.nodes.items()}
    
    def get_shard_trust_scores(self) -> np.ndarray:
        """
        Lấy danh sách điểm tin cậy của tất cả các shard.
        
        Returns:
            np.ndarray: Mảng chứa điểm tin cậy của các shard
        """
        return self.shard_trust_scores
    
    def identify_malicious_nodes(self, min_malicious_activities: int = 2, advanced_filtering: bool = True) -> List[int]:
        """
        Nhận diện các node độc hại trong mạng.
        
        Args:
            min_malicious_activities: Số lượng tối thiểu hoạt động độc hại được ghi nhận
                                    để xác định một node là độc hại
            advanced_filtering: Nếu True, áp dụng các bộ lọc nâng cao để giảm dương tính giả
        
        Returns:
            List[int]: Danh sách ID của các node được xác định là độc hại
        """
        malicious_nodes = []
        
        for node_id, node in self.nodes.items():
            # Điều kiện cơ bản: Điểm tin cậy thấp hơn ngưỡng
            trust_below_threshold = node.trust_score < self.malicious_threshold
            
            if not trust_below_threshold:
                continue  # Bỏ qua node nếu điểm tin cậy cao
            
            # Điều kiện 2: Có đủ số lượng hoạt động độc hại tối thiểu
            enough_malicious_activities = node.malicious_activities >= min_malicious_activities
            
            # Điều kiện 3: Tỉ lệ thành công quá thấp (nếu đã có đủ giao dịch)
            total_txs = node.successful_txs + node.failed_txs
            low_success_rate = False
            if total_txs >= 5:  # Chỉ tính khi có đủ dữ liệu
                success_rate = node.successful_txs / total_txs if total_txs > 0 else 0
                low_success_rate = success_rate < 0.4  # Tỉ lệ thành công dưới 40%
            
            if not advanced_filtering:
                # Phiên bản đơn giản: chỉ xét điểm tin cậy và số hoạt động độc hại
                if trust_below_threshold and enough_malicious_activities:
                    malicious_nodes.append(node_id)
            else:
                # Phiên bản nâng cao:
                
                # Kiểm tra thời gian phản hồi
                high_response_time = False
                if node.response_times and len(node.response_times) >= 3:
                    avg_response_time = np.mean(node.response_times)
                    high_response_time = avg_response_time > 20  # Thời gian phản hồi cao
                
                # Kiểm tra feedback từ peer
                poor_peer_rating = False
                if node.peer_ratings:
                    avg_peer_rating = np.mean(list(node.peer_ratings.values()))
                    poor_peer_rating = avg_peer_rating < 0.4  # Đánh giá từ peer thấp
                
                # Điều kiện kết hợp:
                # 1. Điểm tin cậy thấp VÀ ít nhất 2 trong số các điều kiện sau:
                #    - Đủ hoạt động độc hại
                #    - Tỉ lệ thành công thấp
                #    - Thời gian phản hồi cao
                #    - Đánh giá từ peer thấp
                evidence_count = sum([
                    enough_malicious_activities,
                    low_success_rate,
                    high_response_time,
                    poor_peer_rating
                ])
                
                if evidence_count >= 2:
                    malicious_nodes.append(node_id)
        
        return malicious_nodes
    
    def recommend_trusted_validators(self, shard_id: int, count: int = 3) -> List[int]:
        """
        Đề xuất các validator đáng tin cậy nhất cho một shard.
        
        Args:
            shard_id: ID của shard
            count: Số lượng validator cần đề xuất
            
        Returns:
            List[int]: Danh sách ID của các node được đề xuất làm validator
        """
        # Lấy tất cả các node trong shard đã cho
        shard_node_ids = self.shards[shard_id]
        
        # Sắp xếp các node theo điểm tin cậy giảm dần
        trusted_nodes = sorted(
            [(node_id, self.nodes[node_id].trust_score) for node_id in shard_node_ids],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Trả về danh sách ID của các node tin cậy nhất
        return [node_id for node_id, _ in trusted_nodes[:count]]
    
    def reset(self):
        """Khởi tạo lại điểm tin cậy cho tất cả các node về giá trị mặc định."""
        for node_id, node in self.nodes.items():
            node.trust_score = 0.7
            node.successful_txs = 0
            node.failed_txs = 0
            node.malicious_activities = 0
            node.response_times = []
            node.peer_ratings = defaultdict(lambda: 0.5)
            node.activity_history.clear()
        
        # Reset thông tin shard
        self.shard_trust_scores = np.ones(self.num_shards) * 0.7
        self.under_attack = False
        self.suspected_nodes.clear()
        self.attack_patterns.clear() 