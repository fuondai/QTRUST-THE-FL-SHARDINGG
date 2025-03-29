import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import heapq
import random

class MADRAPIDRouter:
    """
    Multi-Agent Dynamic Routing and Adaptive Path Intelligence Distribution (MAD-RAPID).
    Thuật toán định tuyến thông minh cho giao dịch xuyên shard trong mạng blockchain.
    """
    
    def __init__(self, 
                 network: nx.Graph,
                 shards: List[List[int]],
                 congestion_weight: float = 0.5,
                 latency_weight: float = 0.3,
                 energy_weight: float = 0.1,
                 trust_weight: float = 0.1,
                 prediction_horizon: int = 3,
                 congestion_threshold: float = 0.7):
        """
        Khởi tạo bộ định tuyến MAD-RAPID.
        
        Args:
            network: Đồ thị mạng blockchain
            shards: Danh sách các shard và node trong mỗi shard
            congestion_weight: Trọng số cho mức độ tắc nghẽn
            latency_weight: Trọng số cho độ trễ
            energy_weight: Trọng số cho tiêu thụ năng lượng
            trust_weight: Trọng số cho điểm tin cậy
            prediction_horizon: Số bước dự đoán tắc nghẽn trong tương lai
            congestion_threshold: Ngưỡng tắc nghẽn để coi là tắc nghẽn
        """
        self.network = network
        self.shards = shards
        self.num_shards = len(shards)
        
        # Trọng số cho các yếu tố khác nhau trong quyết định định tuyến
        self.congestion_weight = congestion_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.trust_weight = trust_weight
        
        # Tham số cho dự đoán tắc nghẽn
        self.prediction_horizon = prediction_horizon
        self.congestion_threshold = congestion_threshold
        
        # Lịch sử tắc nghẽn để dự đoán tắc nghẽn tương lai
        self.congestion_history = [np.zeros(self.num_shards) for _ in range(prediction_horizon)]
        
        # Cache đường dẫn và thời gian hết hạn cache
        self.path_cache = {}
        self.cache_expire_time = 10  # Số bước trước khi cache hết hạn
        self.current_step = 0
        
        # Thêm biến để theo dõi tỷ lệ giao dịch cùng shard
        self.same_shard_ratio = 0.8  # Mục tiêu tỷ lệ giao dịch nội shard
        
        # Xây dựng đồ thị shard level từ network
        self.shard_graph = self._build_shard_graph()
        
        # Thêm ma trận gần cận giữa các shard
        self.shard_affinity = np.ones((self.num_shards, self.num_shards)) - np.eye(self.num_shards)
    
    def _build_shard_graph(self) -> nx.Graph:
        """
        Xây dựng đồ thị cấp shard từ đồ thị mạng.
        
        Returns:
            nx.Graph: Đồ thị cấp shard
        """
        shard_graph = nx.Graph()
        
        # Thêm các shard như là node
        for shard_id in range(self.num_shards):
            shard_graph.add_node(shard_id, 
                                congestion=0.0,
                                trust_score=0.0)
        
        # Tìm các kết nối giữa các shard và tính toán độ trễ/băng thông trung bình
        for i in range(self.num_shards):
            for j in range(i + 1, self.num_shards):
                cross_shard_edges = []
                
                # Tìm tất cả các cạnh kết nối node giữa hai shard
                for node_i in self.shards[i]:
                    for node_j in self.shards[j]:
                        if self.network.has_edge(node_i, node_j):
                            cross_shard_edges.append((node_i, node_j))
                
                if cross_shard_edges:
                    # Thêm thuộc tính latency và bandwidth nếu chưa có
                    for u, v in cross_shard_edges:
                        if 'latency' not in self.network.edges[u, v]:
                            self.network.edges[u, v]['latency'] = random.uniform(5, 50)  # ms
                        if 'bandwidth' not in self.network.edges[u, v]:
                            self.network.edges[u, v]['bandwidth'] = random.uniform(1, 10)  # Mbps
                    
                    # Tính độ trễ và băng thông trung bình của các kết nối
                    avg_latency = np.mean([self.network.edges[u, v]['latency'] for u, v in cross_shard_edges])
                    avg_bandwidth = np.mean([self.network.edges[u, v]['bandwidth'] for u, v in cross_shard_edges])
                    
                    # Thêm cạnh giữa hai shard với độ trễ và băng thông trung bình
                    shard_graph.add_edge(i, j, 
                                        latency=avg_latency,
                                        bandwidth=avg_bandwidth,
                                        connection_count=len(cross_shard_edges))
        
        return shard_graph
    
    def update_network_state(self, shard_congestion: np.ndarray, node_trust_scores: Dict[int, float]):
        """
        Cập nhật trạng thái mạng với dữ liệu mới.
        
        Args:
            shard_congestion: Mảng chứa mức độ tắc nghẽn của từng shard
            node_trust_scores: Dictionary ánh xạ từ node ID đến điểm tin cậy
        """
        # Cập nhật lịch sử tắc nghẽn
        self.congestion_history.pop(0)
        self.congestion_history.append(shard_congestion.copy())
        
        # Cập nhật mức độ tắc nghẽn hiện tại cho mỗi shard trong shard_graph
        for shard_id in range(self.num_shards):
            self.shard_graph.nodes[shard_id]['congestion'] = shard_congestion[shard_id]
            
            # Tính điểm tin cậy trung bình cho shard
            shard_nodes = self.shards[shard_id]
            avg_trust = np.mean([node_trust_scores.get(node, 0.5) for node in shard_nodes])
            self.shard_graph.nodes[shard_id]['trust_score'] = avg_trust
        
        # Xóa cache đường dẫn khi trạng thái mạng thay đổi
        self.path_cache = {}
    
    def _predict_congestion(self, shard_id: int) -> float:
        """
        Dự đoán mức độ tắc nghẽn tương lai của một shard dựa trên lịch sử.
        
        Args:
            shard_id: ID của shard cần dự đoán
            
        Returns:
            float: Mức độ tắc nghẽn dự đoán
        """
        # Lấy lịch sử tắc nghẽn cho shard cụ thể
        congestion_values = [history[shard_id] for history in self.congestion_history]
        
        if not congestion_values:
            return 0.0
        
        # Dự đoán nâng cao với đa mô hình
        
        # Mô hình 1: Trung bình có trọng số theo thời gian gần đây
        # Tính trọng số giảm dần theo thời gian (càng gần hiện tại càng quan trọng)
        weights = np.exp(np.linspace(0, 2, len(congestion_values)))  # Tăng độ dốc từ 1 lên 2
        weights = weights / np.sum(weights)
        predicted_congestion_1 = np.sum(weights * congestion_values)
        
        # Mô hình 2: Dự đoán dựa trên xu hướng tuyến tính
        if len(congestion_values) >= 3:
            # Lấy chênh lệch giữa các giá trị liên tiếp
            diffs = np.diff(congestion_values[-3:])
            # Tính xu hướng trung bình
            avg_trend = np.mean(diffs)
            # Dự đoán tiếp theo dựa trên giá trị cuối cùng và xu hướng
            predicted_congestion_2 = congestion_values[-1] + avg_trend
            predicted_congestion_2 = np.clip(predicted_congestion_2, 0.0, 1.0)
        else:
            predicted_congestion_2 = congestion_values[-1]
        
        # Mô hình 3: ARMA(1,1) đơn giản hóa
        if len(congestion_values) >= 3:
            # Tham số AR và MA
            ar_param = 0.7
            ma_param = 0.3
            error = congestion_values[-1] - congestion_values[-2] if len(congestion_values) >= 2 else 0
            predicted_congestion_3 = ar_param * congestion_values[-1] + ma_param * error
            predicted_congestion_3 = np.clip(predicted_congestion_3, 0.0, 1.0)
        else:
            predicted_congestion_3 = congestion_values[-1]
        
        # Kết hợp các dự đoán với các trọng số dựa trên độ dài lịch sử
        # Lịch sử càng dài, càng tin cậy vào mô hình phức tạp hơn
        if len(congestion_values) >= 5:
            # Đủ dữ liệu cho mô hình phức tạp
            final_prediction = 0.3 * predicted_congestion_1 + 0.4 * predicted_congestion_2 + 0.3 * predicted_congestion_3
        elif len(congestion_values) >= 3:
            # Dữ liệu trung bình
            final_prediction = 0.4 * predicted_congestion_1 + 0.4 * predicted_congestion_2 + 0.2 * predicted_congestion_3
        else:
            # Dữ liệu ít
            final_prediction = 0.7 * predicted_congestion_1 + 0.3 * predicted_congestion_2
        
        # Điều chỉnh dựa trên mức độ biến động
        if len(congestion_values) >= 3:
            variance = np.var(congestion_values[-3:])
            # Nếu biến động cao, thêm hệ số an toàn
            if variance > 0.05:
                final_prediction += 0.1 * variance
        
        # Thêm ảnh hưởng của tình trạng kết nối 
        # Kiểm tra số lượng kết nối đến shard này
        num_connections = sum(1 for u, v, data in self.shard_graph.edges(data=True) if u == shard_id or v == shard_id)
        # Nếu ít kết nối, shard dễ bị tắc nghẽn hơn
        connection_factor = max(0.0, 0.1 * (1.0 - num_connections / max(1, self.num_shards)))
        final_prediction += connection_factor
        
        # Đảm bảo giá trị nằm trong khoảng [0, 1]
        return np.clip(final_prediction, 0.0, 1.0)
    
    def _calculate_path_cost(self, path: List[int], transaction: Dict[str, Any]) -> float:
        """
        Tính chi phí đường dẫn dựa trên các yếu tố hiệu suất.
        
        Args:
            path: Đường dẫn là danh sách các shard ID
            transaction: Giao dịch cần định tuyến
            
        Returns:
            float: Chi phí tổng hợp của đường dẫn
        """
        if len(path) < 2:
            return 0.0
        
        # Phạt đường dẫn quá dài để giảm thiểu giao dịch cross-shard
        # Tăng mạnh phạt cho đường dẫn dài để ưu tiên các đường dẫn ngắn hơn
        path_length_penalty = (len(path) - 2) * 0.5  # Tăng từ 0.3 lên 0.5
        
        total_latency = 0.0
        total_energy = 0.0
        total_congestion = 0.0
        total_trust = 0.0
        
        for i in range(len(path) - 1):
            shard_from = path[i]
            shard_to = path[i + 1]
            
            # Nếu không có kết nối trực tiếp giữa hai shard, trả về chi phí cao
            if not self.shard_graph.has_edge(shard_from, shard_to):
                return float('inf')
            
            # Lấy thông tin cạnh
            edge_data = self.shard_graph.edges[shard_from, shard_to]
            
            # Tính độ trễ của cạnh
            latency = edge_data['latency']
            total_latency += latency
            
            # Tính năng lượng tiêu thụ (dựa trên độ trễ và băng thông)
            # Giả định: Tiêu thụ năng lượng tỷ lệ thuận với độ trễ và tỷ lệ nghịch với băng thông
            energy = latency * (1.0 / edge_data['bandwidth']) * 10.0  # Hệ số 10.0 để chuẩn hóa
            total_energy += energy
            
            # Dự đoán tắc nghẽn của shard đích với quyền số cao hơn cho xu hướng gần đây
            predicted_congestion = self._predict_congestion(shard_to)
            
            # Áp dụng hàm phi tuyến cho mức tắc nghẽn để phạt mạnh hơn các shard tắc nghẽn cao
            # y = x^2 sẽ phạt mạnh hơn khi tắc nghẽn > 0.5
            nonlinear_congestion = predicted_congestion ** 2
            total_congestion += nonlinear_congestion
            
            # Lấy điểm tin cậy trung bình của shard đích
            trust_score = self.shard_graph.nodes[shard_to]['trust_score']
            total_trust += trust_score
            
            # Thêm phạt cho giao dịch cross-shard dựa trên ma trận gần cận
            if i == 0 and len(path) > 2:  # Chỉ áp dụng cho bước đầu tiên của giao dịch cross-shard
                affinity_penalty = (1 - self.shard_affinity[shard_from, shard_to]) * 0.4  # Tăng từ 0.2 lên 0.4
                total_congestion += affinity_penalty
        
        # Tính chi phí tổng hợp dựa trên các trọng số
        # Chi phí thấp hơn = đường dẫn tốt hơn
        
        # Nâng cao trọng số cho tiêu thụ năng lượng trong tính toán chi phí
        energy_weight_adjusted = self.energy_weight * 1.5  # Tăng trọng số năng lượng 50%
        
        # Điều chỉnh các trọng số khác để tổng vẫn là 1.0
        total_weight = self.congestion_weight + self.latency_weight + energy_weight_adjusted + self.trust_weight
        
        congestion_weight_normalized = self.congestion_weight / total_weight
        latency_weight_normalized = self.latency_weight / total_weight
        energy_weight_normalized = energy_weight_adjusted / total_weight
        trust_weight_normalized = self.trust_weight / total_weight
        
        cost = (congestion_weight_normalized * total_congestion / (len(path) - 1) + 
                latency_weight_normalized * total_latency / 100.0 +  # Chuẩn hóa độ trễ (giả sử tối đa 100ms)
                energy_weight_normalized * total_energy / 20.0 -     # Chuẩn hóa năng lượng (giả sử tối đa 20 đơn vị)
                trust_weight_normalized * total_trust / (len(path) - 1) +  # Điểm tin cậy cao = chi phí thấp
                path_length_penalty)  # Phạt đường dẫn dài
        
        # Ưu đãi giao dịch trong cùng shard (source = destination)
        if len(path) == 2 and path[0] == transaction.get('shard_id'):
            cost *= 0.6  # Giảm 40% chi phí cho giao dịch nội shard (từ 20% lên 40%)
        
        # Thêm ưu đãi cho giao dịch giá trị thấp trong cùng shard
        if len(path) == 2 and transaction.get('value', 100) < 30:
            cost *= 0.8  # Giảm thêm 20% chi phí cho giao dịch giá trị thấp
        
        # Thêm ưu đãi cho giao dịch đơn giản và bảo mật thấp
        if transaction.get('complexity', 'high') == 'low' and len(path) == 2:
            cost *= 0.9  # Giảm thêm 10% chi phí
            
        return cost
    
    def _dijkstra(self, source_shard: int, dest_shard: int, transaction: Dict[str, Any]) -> List[int]:
        """
        Thuật toán Dijkstra sửa đổi để tìm đường dẫn tối ưu giữa các shard.
        
        Args:
            source_shard: Shard nguồn
            dest_shard: Shard đích
            transaction: Giao dịch cần định tuyến
            
        Returns:
            List[int]: Đường dẫn tối ưu là danh sách các shard ID
        """
        # Kiểm tra cache
        cache_key = (source_shard, dest_shard, transaction['value'])
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Khởi tạo
        distances = {shard: float('inf') for shard in range(self.num_shards)}
        distances[source_shard] = 0
        previous = {shard: None for shard in range(self.num_shards)}
        priority_queue = [(0, source_shard)]
        
        while priority_queue:
            current_distance, current_shard = heapq.heappop(priority_queue)
            
            # Nếu đã đến đích, thoát khỏi vòng lặp
            if current_shard == dest_shard:
                break
            
            # Nếu khoảng cách hiện tại lớn hơn khoảng cách đã biết, bỏ qua
            if current_distance > distances[current_shard]:
                continue
            
            # Duyệt các shard kề
            for neighbor in self.shard_graph.neighbors(current_shard):
                # Xây dựng đường dẫn tạm thời đến neighbor
                temp_path = self._reconstruct_path(previous, current_shard)
                temp_path.append(neighbor)
                
                # Tính chi phí đường dẫn
                path_cost = self._calculate_path_cost(temp_path, transaction)
                
                # Nếu tìm thấy đường dẫn tốt hơn
                if path_cost < distances[neighbor]:
                    distances[neighbor] = path_cost
                    previous[neighbor] = current_shard
                    heapq.heappush(priority_queue, (path_cost, neighbor))
        
        # Xây dựng đường dẫn từ source đến dest
        path = self._reconstruct_path(previous, dest_shard)
        
        # Lưu vào cache
        self.path_cache[cache_key] = path
        
        return path
    
    def _reconstruct_path(self, previous: Dict[int, int], end: int) -> List[int]:
        """
        Xây dựng đường dẫn từ dict previous.
        
        Args:
            previous: Dictionary ánh xạ từ node đến node trước đó trong đường dẫn
            end: Node cuối cùng trong đường dẫn
            
        Returns:
            List[int]: Đường dẫn hoàn chỉnh
        """
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous[current]
        
        # Đảo ngược đường dẫn để được từ nguồn đến đích
        return path[::-1]
    
    def find_optimal_path(self, 
                         transaction: Dict[str, Any], 
                         source_shard: int, 
                         destination_shard: int,
                         prioritize_security: bool = False) -> List[int]:
        """
        Tìm đường dẫn tối ưu cho một giao dịch giữa hai shard.
        
        Args:
            transaction: Thông tin giao dịch
            source_shard: Shard nguồn
            destination_shard: Shard đích
            prioritize_security: Ưu tiên bảo mật hơn hiệu suất khi True
            
        Returns:
            List[int]: Danh sách shard IDs tạo thành đường dẫn tối ưu
        """
        # Kiểm tra cache trước
        cache_key = (source_shard, destination_shard, transaction.get('value', 0) > 50, prioritize_security)
        if hasattr(self, 'path_cache') and cache_key in self.path_cache:
            return self.path_cache.get(cache_key)
            
        # Nếu cùng một shard, không cần tìm đường
        if source_shard == destination_shard:
            return [source_shard]
        
        # Điều chỉnh trọng số nếu ưu tiên bảo mật
        original_weights = None
        if prioritize_security:
            # Lưu trữ trọng số gốc
            original_weights = {
                'congestion': self.congestion_weight,
                'latency': self.latency_weight,
                'energy': self.energy_weight,
                'trust': self.trust_weight
            }
            
            # Tăng trọng số cho tin cậy khi ưu tiên bảo mật
            self.trust_weight = min(0.6, self.trust_weight * 2)
            
            # Điều chỉnh các trọng số khác
            total_other = self.congestion_weight + self.latency_weight + self.energy_weight
            scale_factor = (1.0 - self.trust_weight) / total_other
            
            self.congestion_weight *= scale_factor
            self.latency_weight *= scale_factor
            self.energy_weight *= scale_factor
        
        # Tìm đường dẫn tối ưu bằng thuật toán Dijkstra sửa đổi
        path = self._dijkstra(source_shard, destination_shard, transaction)
        
        # Khôi phục trọng số gốc nếu đã thay đổi
        if original_weights:
            self.congestion_weight = original_weights['congestion']
            self.latency_weight = original_weights['latency']
            self.energy_weight = original_weights['energy']
            self.trust_weight = original_weights['trust']
        
        # Lưu kết quả vào cache
        if not hasattr(self, 'path_cache'):
            self.path_cache = {}
        self.path_cache[cache_key] = path
        
        return path
    
    def detect_congestion_hotspots(self) -> List[int]:
        """
        Phát hiện các điểm nóng tắc nghẽn trong mạng.
        
        Returns:
            List[int]: Danh sách các shard ID đang bị tắc nghẽn
        """
        hotspots = []
        
        for shard_id in range(self.num_shards):
            # Dự đoán tắc nghẽn
            predicted_congestion = self._predict_congestion(shard_id)
            
            # Nếu tắc nghẽn dự đoán vượt ngưỡng, coi là điểm nóng
            if predicted_congestion > self.congestion_threshold:
                hotspots.append(shard_id)
        
        return hotspots
    
    def find_optimal_paths_for_transactions(self, transaction_pool: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Tìm đường dẫn tối ưu cho nhiều giao dịch cùng một lúc.
        
        Args:
            transaction_pool: Danh sách các giao dịch cần định tuyến
            
        Returns:
            Dict[int, List[int]]: Dictionary ánh xạ từ ID giao dịch đến đường dẫn tối ưu
        """
        # Tăng bước hiện tại
        self.current_step += 1
        
        # Cập nhật tỷ lệ mục tiêu giao dịch cùng shard dựa trên tình trạng mạng
        congestion_hotspots = self.detect_congestion_hotspots()
        if len(congestion_hotspots) > self.num_shards // 3:
            # Nếu có nhiều điểm nóng tắc nghẽn, tăng tỷ lệ giao dịch nội shard
            self.same_shard_ratio = min(0.9, self.same_shard_ratio + 0.05)
        else:
            # Nếu ít điểm nóng, giảm nhẹ tỷ lệ giao dịch nội shard
            self.same_shard_ratio = max(0.7, self.same_shard_ratio - 0.02)
        
        optimal_paths = {}
        pending_transactions = [tx for tx in transaction_pool if tx.get('status') == 'pending']
        
        # Sắp xếp giao dịch theo độ ưu tiên
        # Ưu tiên giao dịch không xuyên shard và giao dịch có giá trị thấp
        sorted_transactions = sorted(
            pending_transactions, 
            key=lambda tx: (
                1 if tx.get('type') == 'cross_shard' else 0,  # Ưu tiên giao dịch không xuyên shard
                tx.get('value', 0)  # Sau đó, xác định ưu tiên theo giá trị
            )
        )
        
        # Cập nhật ma trận gần cận giữa các shard dựa trên giao dịch gần đây
        # Tăng gần cận cho các cặp shard thường xuyên giao dịch với nhau
        shard_transaction_counts = np.zeros((self.num_shards, self.num_shards))
        
        for tx in pending_transactions:
            source = tx.get('shard_id', 0)
            dest = tx.get('destination_shard', source)
            if source != dest:
                shard_transaction_counts[source, dest] += 1
                shard_transaction_counts[dest, source] += 1  # Cập nhật theo cả hai chiều
        
        # Chuẩn hóa và cập nhật ma trận gần cận
        if np.sum(shard_transaction_counts) > 0:
            normalized_counts = shard_transaction_counts / np.max(shard_transaction_counts)
            # Cập nhật ma trận gần cận với trọng số 0.2 cho dữ liệu mới
            self.shard_affinity = 0.8 * self.shard_affinity + 0.2 * normalized_counts
        
        # Tìm đường dẫn tối ưu cho từng giao dịch
        for tx in sorted_transactions:
            tx_id = tx.get('id', 0)
            source_shard = tx.get('shard_id', 0)
            
            # Đối với giao dịch trong cùng shard
            if 'destination_shard' not in tx or random.random() < self.same_shard_ratio:
                tx['destination_shard'] = source_shard
                tx['type'] = 'same_shard'
                optimal_paths[tx_id] = [source_shard]
            else:
                # Đối với giao dịch xuyên shard
                destination_shard = tx.get('destination_shard', random.randint(0, self.num_shards - 1))
                if destination_shard == source_shard:
                    # Đảm bảo là shard khác
                    destination_shard = (destination_shard + 1) % self.num_shards
                
                tx['destination_shard'] = destination_shard
                tx['type'] = 'cross_shard'
                
                path = self.find_optimal_path(tx, source_shard, destination_shard)
                optimal_paths[tx_id] = path
                
                # Cập nhật ma trận gần cận cho cặp shard này
                if len(path) >= 2:
                    self.shard_affinity[path[0], path[-1]] += 0.01
                    self.shard_affinity[path[-1], path[0]] += 0.01
        
        return optimal_paths
    
    def optimize_routing_weights(self, 
                               recent_metrics: Dict[str, List[float]], 
                               target_latency: float = 0.0, 
                               target_energy: float = 0.0):
        """
        Tối ưu hóa các trọng số định tuyến dựa trên các metrics gần đây và mục tiêu.
        
        Args:
            recent_metrics: Dictionary chứa các metrics hiệu suất gần đây
            target_latency: Mục tiêu độ trễ (0.0 = không giới hạn)
            target_energy: Mục tiêu tiêu thụ năng lượng (0.0 = không giới hạn)
        """
        # Nếu độ trễ gần đây cao và mục tiêu latency > 0
        if target_latency > 0 and 'latency' in recent_metrics:
            avg_latency = np.mean(recent_metrics['latency'])
            if avg_latency > target_latency:
                # Tăng trọng số cho độ trễ
                self.latency_weight = min(0.6, self.latency_weight * 1.2)
                
                # Giảm các trọng số khác để tổng = 1.0
                total_other = self.congestion_weight + self.energy_weight + self.trust_weight
                scale_factor = (1.0 - self.latency_weight) / total_other
                
                self.congestion_weight *= scale_factor
                self.energy_weight *= scale_factor
                self.trust_weight *= scale_factor
        
        # Nếu tiêu thụ năng lượng gần đây cao và mục tiêu energy > 0
        if target_energy > 0 and 'energy_consumption' in recent_metrics:
            avg_energy = np.mean(recent_metrics['energy_consumption'])
            if avg_energy > target_energy:
                # Tăng trọng số cho năng lượng
                self.energy_weight = min(0.5, self.energy_weight * 1.2)
                
                # Giảm các trọng số khác để tổng = 1.0
                total_other = self.congestion_weight + self.latency_weight + self.trust_weight
                scale_factor = (1.0 - self.energy_weight) / total_other
                
                self.congestion_weight *= scale_factor
                self.latency_weight *= scale_factor
                self.trust_weight *= scale_factor
        
        # Đảm bảo tổng các trọng số = 1.0
        total_weight = self.congestion_weight + self.latency_weight + self.energy_weight + self.trust_weight
        if abs(total_weight - 1.0) > 1e-6:
            scale = 1.0 / total_weight
            self.congestion_weight *= scale
            self.latency_weight *= scale
            self.energy_weight *= scale
            self.trust_weight *= scale
        
        # Xóa cache khi thay đổi trọng số
        self.path_cache = {} 