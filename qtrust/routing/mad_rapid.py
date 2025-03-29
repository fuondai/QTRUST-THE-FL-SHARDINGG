import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
import heapq
import random
import time
from collections import defaultdict

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
                 prediction_horizon: int = 5,  # Tăng từ 3 lên 5
                 congestion_threshold: float = 0.7,
                 proximity_weight: float = 0.2,  # Thêm trọng số yếu tố gần cận
                 use_dynamic_mesh: bool = True,  # Thêm cờ cho phép sử dụng dynamic mesh
                 predictive_window: int = 10,  # Cửa sổ dự đoán cho giao dịch tương lai
                 max_cache_size: int = 1000):  # Giới hạn kích thước cache
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
            proximity_weight: Trọng số cho yếu tố gần cận địa lý/logic
            use_dynamic_mesh: Có sử dụng dynamic mesh connections hay không
            predictive_window: Số bước giao dịch để xây dựng mô hình dự đoán
            max_cache_size: Kích thước tối đa của cache
        """
        self.network = network
        self.shards = shards
        self.num_shards = len(shards)
        
        # Trọng số cho các yếu tố khác nhau trong quyết định định tuyến
        self.congestion_weight = congestion_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.trust_weight = trust_weight
        self.proximity_weight = proximity_weight
        
        # Tham số cho dự đoán tắc nghẽn
        self.prediction_horizon = prediction_horizon
        self.congestion_threshold = congestion_threshold
        self.use_dynamic_mesh = use_dynamic_mesh
        self.predictive_window = predictive_window
        
        # Lịch sử tắc nghẽn để dự đoán tắc nghẽn tương lai
        self.congestion_history = [np.zeros(self.num_shards) for _ in range(prediction_horizon)]
        
        # Cache đường dẫn và thời gian hết hạn cache
        self.path_cache = {}
        self.cache_expire_time = 10  # Số bước trước khi cache hết hạn
        self.max_cache_size = max_cache_size
        self.current_step = 0
        
        # Thêm biến để theo dõi tỷ lệ giao dịch cùng shard
        self.same_shard_ratio = 0.8  # Mục tiêu tỷ lệ giao dịch nội shard
        
        # Xây dựng đồ thị shard level từ network
        self.shard_graph = self._build_shard_graph()
        
        # Thêm ma trận gần cận giữa các shard
        self.shard_affinity = np.ones((self.num_shards, self.num_shards)) - np.eye(self.num_shards)
        
        # Thêm lưu trữ cho giao dịch lịch sử
        self.transaction_history = []
        
        # Dynamic mesh connections
        self.dynamic_connections = set()
        
        # Thống kê lưu lượng giữa các cặp shard
        self.shard_pair_traffic = defaultdict(int)
        
        # Thời gian cuối cùng cập nhật dynamic mesh
        self.last_mesh_update = 0
        
        # Khoảng thời gian (bước) giữa các lần cập nhật dynamic mesh
        self.mesh_update_interval = 50
    
    def _build_shard_graph(self) -> nx.Graph:
        """
        Xây dựng đồ thị cấp shard từ đồ thị mạng với awareness về vị trí địa lý/logic.
        
        Returns:
            nx.Graph: Đồ thị cấp shard
        """
        shard_graph = nx.Graph()
        
        # Thêm các shard như là node
        for shard_id in range(self.num_shards):
            # Thêm thông tin vị trí cho mỗi shard (mô phỏng tọa độ logic hoặc địa lý)
            # Tạo tọa độ logic ngẫu nhiên cho các shard để mô phỏng vị trí địa lý/logic
            x_pos = random.uniform(0, 100)
            y_pos = random.uniform(0, 100)
            
            shard_graph.add_node(shard_id, 
                                congestion=0.0,
                                trust_score=0.0,
                                position=(x_pos, y_pos),
                                capacity=len(self.shards[shard_id]))
        
        # Tính toán khoảng cách địa lý giữa các shard (dựa trên tọa độ logic)
        geographical_distances = {}
        for i in range(self.num_shards):
            pos_i = shard_graph.nodes[i]['position']
            for j in range(i + 1, self.num_shards):
                pos_j = shard_graph.nodes[j]['position']
                # Tính khoảng cách Euclidean
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                geographical_distances[(i, j)] = distance
                geographical_distances[(j, i)] = distance
        
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
                            # Tính latency dựa trên khoảng cách địa lý
                            geo_dist = geographical_distances.get((i, j), 50)
                            base_latency = random.uniform(5, 20)  # Latency cơ bản
                            geo_factor = geo_dist / 100.0  # Normalized to [0,1]
                            # Latency tăng theo khoảng cách địa lý
                            self.network.edges[u, v]['latency'] = base_latency * (1 + geo_factor)
                        
                        if 'bandwidth' not in self.network.edges[u, v]:
                            # Bandwidth giảm theo khoảng cách địa lý
                            geo_dist = geographical_distances.get((i, j), 50)
                            base_bandwidth = random.uniform(5, 20)  # Bandwidth cơ bản (Mbps)
                            geo_factor = geo_dist / 100.0  # Normalized to [0,1]
                            # Bandwidth giảm theo khoảng cách địa lý
                            self.network.edges[u, v]['bandwidth'] = base_bandwidth * (1 - 0.5 * geo_factor)
                    
                    # Tính độ trễ và băng thông trung bình của các kết nối
                    avg_latency = np.mean([self.network.edges[u, v]['latency'] for u, v in cross_shard_edges])
                    avg_bandwidth = np.mean([self.network.edges[u, v]['bandwidth'] for u, v in cross_shard_edges])
                    
                    # Tính proximity factor dựa trên khoảng cách địa lý và số lượng kết nối
                    geo_dist = geographical_distances.get((i, j), 50)
                    num_connections = len(cross_shard_edges)
                    
                    # Proximity factor cao khi khoảng cách ngắn và nhiều kết nối
                    proximity_factor = (1 - geo_dist/100.0) * min(1.0, num_connections / 10.0)
                    
                    # Thêm cạnh giữa hai shard với các thuộc tính mở rộng
                    shard_graph.add_edge(i, j, 
                                        latency=avg_latency,
                                        bandwidth=avg_bandwidth,
                                        connection_count=len(cross_shard_edges),
                                        geographical_distance=geo_dist,
                                        proximity_factor=proximity_factor,
                                        historical_traffic=0,
                                        is_dynamic=False,
                                        stability=1.0,  # Độ ổn định của kết nối
                                        last_updated=time.time())
        
        return shard_graph
    
    def update_network_state(self, shard_congestion: np.ndarray, node_trust_scores: Dict[int, float], transaction_batch: List[Dict[str, Any]] = None):
        """
        Cập nhật trạng thái mạng với dữ liệu mới và cập nhật dynamic mesh nếu cần.
        
        Args:
            shard_congestion: Mảng chứa mức độ tắc nghẽn của từng shard
            node_trust_scores: Dictionary ánh xạ từ node ID đến điểm tin cậy
            transaction_batch: Batch giao dịch gần đây để cập nhật lịch sử
        """
        # Cập nhật bước hiện tại
        self.current_step += 1
        
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
        
        # Cập nhật lịch sử giao dịch nếu có
        if transaction_batch:
            # Giới hạn kích thước lịch sử
            max_history_size = self.predictive_window * 100
            if len(self.transaction_history) >= max_history_size:
                self.transaction_history = self.transaction_history[-max_history_size:]
            
            # Thêm giao dịch mới vào lịch sử
            self.transaction_history.extend(transaction_batch)
            
            # Cập nhật thống kê lưu lượng cho các cặp shard
            for tx in transaction_batch:
                if 'source_shard' in tx and 'destination_shard' in tx:
                    src = tx['source_shard']
                    dst = tx['destination_shard']
                    if src != dst:  # Chỉ quan tâm đến giao dịch xuyên shard
                        pair = tuple(sorted([src, dst]))  # Đảm bảo thứ tự nhất quán
                        self.shard_pair_traffic[pair] += 1
        
        # Xóa cache đường dẫn khi trạng thái mạng thay đổi
        # Chỉ xóa các cache quá cũ, không xóa toàn bộ cache
        current_time = self.current_step
        expired_keys = [k for k, (path, timestamp) in self.path_cache.items() 
                        if current_time - timestamp > self.cache_expire_time]
        for k in expired_keys:
            del self.path_cache[k]
        
        # Nếu cache quá lớn, xóa bớt các entry cũ nhất
        if len(self.path_cache) > self.max_cache_size:
            sorted_cache = sorted(self.path_cache.items(), key=lambda x: x[1][1])  # Sắp xếp theo timestamp
            entries_to_remove = len(self.path_cache) - self.max_cache_size
            for k, _ in sorted_cache[:entries_to_remove]:
                del self.path_cache[k]
        
        # Cập nhật dynamic mesh
        self.update_dynamic_mesh()
    
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
            # Giới hạn trong khoảng [0, 1]
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
        
        # Mô hình 4: Double Exponential Smoothing (Holt's method)
        if len(congestion_values) >= 4:
            alpha = 0.3  # Tham số làm mịn
            beta = 0.2   # Tham số xu hướng
            
            # Giá trị làm mịn ban đầu
            s_prev = congestion_values[-3]
            b_prev = congestion_values[-2] - congestion_values[-3]
            
            # Cập nhật cho bước cuối
            s_curr = alpha * congestion_values[-1] + (1 - alpha) * (s_prev + b_prev)
            b_curr = beta * (s_curr - s_prev) + (1 - beta) * b_prev
            
            # Dự đoán bước tiếp theo
            predicted_congestion_4 = s_curr + b_curr
            predicted_congestion_4 = np.clip(predicted_congestion_4, 0.0, 1.0)
        else:
            predicted_congestion_4 = congestion_values[-1]
        
        # Kết hợp các dự đoán với các trọng số dựa trên độ dài lịch sử
        # Lịch sử càng dài, càng tin cậy vào mô hình phức tạp hơn
        if len(congestion_values) >= 5:
            # Đủ dữ liệu cho mô hình phức tạp
            final_prediction = 0.2 * predicted_congestion_1 + 0.3 * predicted_congestion_2 + 0.2 * predicted_congestion_3 + 0.3 * predicted_congestion_4
        elif len(congestion_values) >= 3:
            # Dữ liệu trung bình
            final_prediction = 0.3 * predicted_congestion_1 + 0.3 * predicted_congestion_2 + 0.2 * predicted_congestion_3 + 0.2 * predicted_congestion_4
        else:
            # Dữ liệu ít
            final_prediction = 0.6 * predicted_congestion_1 + 0.4 * predicted_congestion_2
        
        # Điều chỉnh dựa trên mức độ biến động
        if len(congestion_values) >= 3:
            variance = np.var(congestion_values[-3:])
            # Nếu biến động cao, thêm hệ số an toàn
            if variance > 0.05:
                final_prediction += 0.1 * variance
        
        # Phân tích các yếu tố mạng
        # Kiểm tra số lượng kết nối đến shard này
        num_connections = sum(1 for u, v, data in self.shard_graph.edges(data=True) if u == shard_id or v == shard_id)
        
        # Yếu tố capacity - mỗi shard có số lượng node khác nhau
        shard_capacity = len(self.shards[shard_id])
        capacity_factor = max(0.0, 0.1 * (1.0 - shard_capacity / max(shard_capacity for shard in self.shards)))
        
        # Điều chỉnh dự đoán dựa trên số lượng kết nối và capacity
        connection_factor = max(0.0, 0.1 * (1.0 - num_connections / max(1, self.num_shards)))
        final_prediction += connection_factor + capacity_factor
        
        # Kiểm tra các shard kết nối trực tiếp có bị tắc nghẽn không
        connected_shards = [node for node in self.shard_graph.neighbors(shard_id)]
        if connected_shards:
            neighbor_congestion = np.mean([self.shard_graph.nodes[s]['congestion'] for s in connected_shards])
            # Nếu các shard lân cận bị tắc nghẽn, shard hiện tại cũng có khả năng bị ảnh hưởng
            if neighbor_congestion > 0.6:
                final_prediction += 0.1 * neighbor_congestion
        
        # Đảm bảo giá trị nằm trong khoảng [0, 1]
        return np.clip(final_prediction, 0.0, 1.0)
    
    def _calculate_path_cost(self, path: List[int], transaction: Dict[str, Any]) -> float:
        """
        Tính chi phí đường dẫn dựa trên các yếu tố hiệu suất và gần cận (proximity).
        
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
        total_proximity = 0.0
        
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
            bandwidth = edge_data['bandwidth']
            energy = (latency / 10.0) * (10.0 / bandwidth)
            total_energy += energy
            
            # Tính mức độ tắc nghẽn dự đoán cho mỗi shard
            congestion_from = self._predict_congestion(shard_from)
            congestion_to = self._predict_congestion(shard_to)
            
            # Tính congestion trung bình của cạnh
            edge_congestion = (congestion_from + congestion_to) / 2.0
            total_congestion += edge_congestion
            
            # Tính điểm tin cậy trung bình của hai shard
            trust_from = self.shard_graph.nodes[shard_from].get('trust_score', 0.5)
            trust_to = self.shard_graph.nodes[shard_to].get('trust_score', 0.5)
            
            # Điểm tin cậy cao sẽ giảm chi phí, nên ta lấy 1 - trust
            edge_risk = 1.0 - ((trust_from + trust_to) / 2.0)
            total_trust += edge_risk
            
            # Thêm chi phí về yếu tố gần cận (proximity)
            proximity_factor = edge_data.get('proximity_factor', 0.5)
            
            # Proximity cao sẽ giảm chi phí, nên ta lấy 1 - proximity_factor
            edge_proximity_cost = 1.0 - proximity_factor
            total_proximity += edge_proximity_cost
            
            # Xem xét tham số độ ổn định của kết nối
            stability = edge_data.get('stability', 1.0)
            
            # Ưu tiên các kết nối ổn định
            if stability < 0.8:
                total_latency *= (2.0 - stability)  # Tăng chi phí nếu kết nối không ổn định
            
            # Ưu tiên các kết nối dynamic đã được tạo để tối ưu cho lưu lượng cao
            if edge_data.get('is_dynamic', False):
                total_latency *= 0.8  # Giảm chi phí cho kết nối dynamic
        
        # Nếu có thông tin đặc biệt trong giao dịch, điều chỉnh chi phí theo
        # Ví dụ: giao dịch yêu cầu độ trễ thấp, bảo mật cao, hoặc tiết kiệm năng lượng
        tx_priority = transaction.get('priority', 'normal')
        if tx_priority == 'low_latency':
            total_latency *= 1.5  # Tăng trọng số cho độ trễ
        elif tx_priority == 'high_security':
            total_trust *= 1.5  # Tăng trọng số cho bảo mật
        elif tx_priority == 'energy_efficient':
            total_energy *= 1.5  # Tăng trọng số cho năng lượng
        
        # Tính chi phí tổng hợp
        total_cost = (
            self.latency_weight * total_latency +
            self.energy_weight * total_energy +
            self.congestion_weight * total_congestion +
            self.trust_weight * total_trust +
            self.proximity_weight * total_proximity +
            path_length_penalty
        )
        
        return total_cost
    
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
                         source_shard: int = None, 
                         destination_shard: int = None,
                         prioritize_security: bool = False) -> List[int]:
        """
        Tìm đường đi tối ưu cho giao dịch giữa hai shard.
        
        Args:
            transaction: Giao dịch cần định tuyến
            source_shard: Shard nguồn (nếu không cung cấp, sẽ sử dụng dự đoán)
            destination_shard: Shard đích (nếu không cung cấp, sẽ sử dụng dự đoán)
            prioritize_security: Ưu tiên an toàn hơn hiệu suất
            
        Returns:
            List[int]: Danh sách các shard ID tạo thành đường đi tối ưu
        """
        # Sử dụng predictive routing nếu source hoặc destination không được cung cấp
        if source_shard is None or destination_shard is None:
            src, dst, confidence = self.predictive_routing(transaction)
            source_shard = src if source_shard is None else source_shard
            destination_shard = dst if destination_shard is None else destination_shard
        
        # Xác nhận source và destination trong phạm vi hợp lệ
        source_shard = max(0, min(source_shard, self.num_shards - 1))
        destination_shard = max(0, min(destination_shard, self.num_shards - 1))
        
        # Trường hợp đặc biệt: source và destination là cùng một shard
        if source_shard == destination_shard:
            return [source_shard]
        
        # Kiểm tra cache trước
        cache_key = (source_shard, destination_shard, str(prioritize_security), 
                    transaction.get('priority', 'normal'))
        
        if cache_key in self.path_cache:
            path, timestamp = self.path_cache[cache_key]
            # Kiểm tra xem cache có còn hiệu lực không
            if self.current_step - timestamp <= self.cache_expire_time:
                return path
        
        # Chuẩn bị trọng số tạm thời nếu cần ưu tiên bảo mật
        tmp_weights = {
            'congestion': self.congestion_weight,
            'latency': self.latency_weight,
            'energy': self.energy_weight,
            'trust': self.trust_weight,
            'proximity': self.proximity_weight
        }
        
        if prioritize_security:
            # Tăng trọng số cho tin cậy, giảm cho hiệu suất
            tmp_weights['trust'] *= 2.0
            tmp_weights['latency'] *= 0.5
            tmp_weights['energy'] *= 0.5
        
        # Xem xét ưu tiên trong giao dịch
        tx_priority = transaction.get('priority', 'normal')
        if tx_priority == 'low_latency':
            tmp_weights['latency'] *= 1.5
        elif tx_priority == 'high_security':
            tmp_weights['trust'] *= 1.5
        elif tx_priority == 'energy_efficient':
            tmp_weights['energy'] *= 1.5
        
        # Lưu trọng số hiện tại
        old_weights = {
            'congestion': self.congestion_weight,
            'latency': self.latency_weight,
            'energy': self.energy_weight,
            'trust': self.trust_weight,
            'proximity': self.proximity_weight
        }
        
        # Đặt trọng số tạm thời
        self.congestion_weight = tmp_weights['congestion'] 
        self.latency_weight = tmp_weights['latency']
        self.energy_weight = tmp_weights['energy']
        self.trust_weight = tmp_weights['trust']
        self.proximity_weight = tmp_weights['proximity']
        
        # Tìm đường đi tối ưu
        path = self._dijkstra(source_shard, destination_shard, transaction)
        
        # Phục hồi trọng số
        self.congestion_weight = old_weights['congestion']
        self.latency_weight = old_weights['latency']
        self.energy_weight = old_weights['energy']
        self.trust_weight = old_weights['trust']
        self.proximity_weight = old_weights['proximity']
        
        # Lưu vào cache
        self.path_cache[cache_key] = (path, self.current_step)
        
        # Kiểm soát kích thước cache
        if len(self.path_cache) > self.max_cache_size:
            # Xóa entry cũ nhất
            oldest_entry = min(self.path_cache.items(), key=lambda x: x[1][1])
            del self.path_cache[oldest_entry[0]]
        
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
        Tìm đường đi tối ưu cho một tập hợp các giao dịch.
        
        Cải tiến đã thực hiện:
        1. Proximity-aware routing: Sử dụng thông tin vị trí địa lý/logic để tối ưu định tuyến
        2. Dynamic mesh connections: Tạo kết nối trực tiếp giữa các cặp shard có lưu lượng cao
        3. Predictive routing: Dự đoán tuyến đường tối ưu dựa trên lịch sử giao dịch
        4. Cải tiến dự đoán tắc nghẽn: Sử dụng nhiều mô hình dự đoán kết hợp
        5. Phân tích mẫu giao dịch: Tối ưu hóa định tuyến dựa trên mẫu lưu lượng
        
        Args:
            transaction_pool: Danh sách các giao dịch cần định tuyến
            
        Returns:
            Dict[int, List[int]]: Dictionary ánh xạ từ transaction ID đến đường đi tối ưu
        """
        # Cập nhật lịch sử giao dịch và phân tích mẫu
        self.transaction_history.extend(transaction_pool)
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]
        
        # Phân tích mẫu giao dịch để tối ưu hóa định tuyến
        pattern_analysis = self.analyze_transaction_patterns()
        
        # Nếu tìm thấy mẫu giao dịch, cập nhật dynamic mesh
        if pattern_analysis['patterns_found'] and pattern_analysis.get('high_traffic_pairs'):
            for pair in pattern_analysis['high_traffic_pairs']:
                self.shard_pair_traffic[pair] += 5  # Tăng ưu tiên cho cặp shard có lưu lượng cao
        
        # Cập nhật dynamic mesh nếu cần
        if self.use_dynamic_mesh and (self.current_step % 10 == 0):
            self.update_dynamic_mesh()
        
        # Tách các giao dịch nội shard và xuyên shard
        intra_shard_txs = []
        cross_shard_txs = []
        
        for tx in transaction_pool:
            if 'source_shard' in tx and 'destination_shard' in tx:
                if tx['source_shard'] == tx['destination_shard']:
                    intra_shard_txs.append(tx)
                else:
                    cross_shard_txs.append(tx)
            else:
                # Sử dụng predictive routing để dự đoán nguồn và đích
                src, dst, confidence = self.predictive_routing(tx)
                
                # Cập nhật giao dịch với thông tin dự đoán
                tx['source_shard'] = src
                tx['destination_shard'] = dst
                tx['predicted_route'] = True
                tx['prediction_confidence'] = confidence
                
                if src == dst:
                    intra_shard_txs.append(tx)
                else:
                    cross_shard_txs.append(tx)
        
        # Sắp xếp các giao dịch xuyên shard theo mức độ ưu tiên
        # và phân nhóm theo cặp shard nguồn-đích để tránh tính toán trùng lặp
        cross_shard_txs.sort(key=lambda tx: tx.get('priority_level', 0), reverse=True)
        
        # Nhóm các giao dịch theo cặp shard nguồn-đích
        route_groups = defaultdict(list)
        for tx in cross_shard_txs:
            route_key = (tx['source_shard'], tx['destination_shard'])
            route_groups[route_key].append(tx)
        
        # Từ điển lưu trữ đường đi tối ưu cho mỗi giao dịch
        tx_paths = {}
        
        # Xử lý các giao dịch nội shard (chỉ có 1 shard trong đường đi)
        for tx in intra_shard_txs:
            if 'id' in tx:
                tx_paths[tx['id']] = [tx['source_shard']]
        
        # Nhóm các giao dịch theo ưu tiên để cân bằng tải
        high_priority_txs = []
        normal_priority_txs = []
        low_priority_txs = []
        
        for txs in route_groups.values():
            for tx in txs:
                priority_level = tx.get('priority_level', 0)
                if priority_level > 8:
                    high_priority_txs.append(tx)
                elif priority_level > 4:
                    normal_priority_txs.append(tx)
                else:
                    low_priority_txs.append(tx)
        
        # Xử lý các giao dịch theo thứ tự ưu tiên
        # Giao dịch ưu tiên cao được xử lý trước với ưu tiên cho hiệu suất
        for tx in high_priority_txs:
            path = self.find_optimal_path(
                tx, 
                tx['source_shard'], 
                tx['destination_shard'],
                prioritize_security=False
            )
            if 'id' in tx:
                tx_paths[tx['id']] = path
        
        # Giao dịch ưu tiên trung bình được xử lý tiếp theo với cân bằng hiệu suất và bảo mật
        for tx in normal_priority_txs:
            path = self.find_optimal_path(
                tx, 
                tx['source_shard'], 
                tx['destination_shard'],
                prioritize_security=tx.get('prioritize_security', False)
            )
            if 'id' in tx:
                tx_paths[tx['id']] = path
        
        # Giao dịch ưu tiên thấp được xử lý sau cùng với ưu tiên cho bảo mật
        for tx in low_priority_txs:
            path = self.find_optimal_path(
                tx, 
                tx['source_shard'], 
                tx['destination_shard'],
                prioritize_security=True
            )
            if 'id' in tx:
                tx_paths[tx['id']] = path
        
        # Phát hiện các điểm tắc nghẽn để thông báo cho hệ thống
        congestion_hotspots = self.detect_congestion_hotspots()
        if congestion_hotspots:
            print(f"Cảnh báo: Phát hiện điểm tắc nghẽn tại các shard: {congestion_hotspots}")
        
        return tx_paths
    
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
    
    def update_dynamic_mesh(self):
        """
        Cập nhật dynamic mesh connections dựa trên lưu lượng giao dịch.
        Tạo kết nối trực tiếp giữa các cặp shard có lưu lượng cao.
        """
        current_time = time.time()
        
        # Chỉ cập nhật mesh sau một khoảng thời gian nhất định
        if not self.use_dynamic_mesh or (current_time - self.last_mesh_update < self.mesh_update_interval):
            return
        
        self.last_mesh_update = current_time
        
        # Xác định các cặp shard có lưu lượng cao nhất
        top_traffic_pairs = sorted(self.shard_pair_traffic.items(), key=lambda x: x[1], reverse=True)
        
        # Giới hạn số lượng kết nối động tối đa
        max_dynamic_connections = min(10, self.num_shards * (self.num_shards - 1) // 4)
        
        # Loại bỏ các kết nối dynamic cũ
        for i, j in list(self.dynamic_connections):
            if self.shard_graph.has_edge(i, j):
                # Nếu đã có kết nối trực tiếp giữa 2 shard, chỉ cập nhật thuộc tính
                self.shard_graph.edges[i, j]['is_dynamic'] = False
            self.dynamic_connections.remove((i, j))
        
        # Thêm kết nối mới cho các cặp shard có lưu lượng cao
        new_connections_count = 0
        for (i, j), traffic in top_traffic_pairs:
            # Nếu đã đạt số lượng kết nối tối đa, dừng lại
            if new_connections_count >= max_dynamic_connections:
                break
                
            # Bỏ qua nếu đã có kết nối trực tiếp giữa 2 shard
            if self.shard_graph.has_edge(i, j):
                self.shard_graph.edges[i, j]['is_dynamic'] = True
                self.shard_graph.edges[i, j]['historical_traffic'] = traffic
                self.dynamic_connections.add((i, j))
                new_connections_count += 1
                continue
                
            # Tính toán các thuộc tính cho kết nối mới
            pos_i = self.shard_graph.nodes[i]['position']
            pos_j = self.shard_graph.nodes[j]['position']
            geo_dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            
            # Latency và bandwidth tốt hơn cho kết nối dynamic do đã được tối ưu hóa
            latency = 5 + geo_dist * 0.3  # Latency thấp hơn cho kết nối động
            bandwidth = 15 - geo_dist * 0.1  # Bandwidth cao hơn cho kết nối động
            
            # Thêm cạnh mới vào đồ thị shard
            self.shard_graph.add_edge(i, j,
                                     latency=latency,
                                     bandwidth=bandwidth,
                                     connection_count=1,
                                     geographical_distance=geo_dist,
                                     proximity_factor=0.8,  # Cao hơn do đã được tối ưu hóa
                                     historical_traffic=traffic,
                                     is_dynamic=True,
                                     stability=0.9,  # Độ ổn định cao cho kết nối động
                                     last_updated=current_time)
            
            # Thêm vào tập kết nối động
            self.dynamic_connections.add((i, j))
            new_connections_count += 1
        
        # Reset lưu lượng sau khi đã cập nhật
        if new_connections_count > 0:
            self.shard_pair_traffic = defaultdict(int) 
    
    def predictive_routing(self, transaction: Dict[str, Any]) -> Tuple[int, int, bool]:
        """
        Sử dụng dữ liệu lịch sử để dự đoán tuyến đường tối ưu cho một giao dịch.
        
        Args:
            transaction: Giao dịch cần dự đoán tuyến đường
            
        Returns:
            Tuple[int, int, bool]: (source_shard, destination_shard, prediction_confidence)
            - source_shard: ID shard nguồn dự đoán
            - destination_shard: ID shard đích dự đoán
            - prediction_confidence: True nếu dự đoán có độ tin cậy cao
        """
        # Nếu có thông tin rõ ràng về nguồn và đích, không cần dự đoán
        if 'source_shard' in transaction and 'destination_shard' in transaction:
            return transaction['source_shard'], transaction['destination_shard'], True
        
        # Nếu không có đủ dữ liệu lịch sử, không thể dự đoán chính xác
        if len(self.transaction_history) < 50:
            # Trả về giá trị mặc định với độ tin cậy thấp
            default_source = transaction.get('source_shard', 0)
            default_dest = transaction.get('destination_shard', 0)
            if default_source == default_dest:
                default_dest = (default_source + 1) % self.num_shards
            return default_source, default_dest, False
        
        # Phân tích giao dịch để lấy các đặc trưng
        tx_type = transaction.get('type', 'default')
        tx_size = transaction.get('size', 1.0)
        tx_sender = transaction.get('sender', None)
        tx_receiver = transaction.get('receiver', None)
        
        # Tìm kiếm các giao dịch tương tự trong lịch sử
        similar_transactions = []
        
        for hist_tx in self.transaction_history[-100:]:  # Chỉ xem xét 100 giao dịch gần nhất
            # So khớp theo loại và kích thước
            type_match = hist_tx.get('type', 'default') == tx_type
            size_match = abs(hist_tx.get('size', 1.0) - tx_size) < 0.2  # Trong khoảng 20%
            
            # So khớp theo người gửi/nhận nếu có
            sender_match = True
            receiver_match = True
            
            if tx_sender and 'sender' in hist_tx:
                sender_match = hist_tx['sender'] == tx_sender
            
            if tx_receiver and 'receiver' in hist_tx:
                receiver_match = hist_tx['receiver'] == tx_receiver
            
            # Nếu match tất cả tiêu chí, thêm vào danh sách giao dịch tương tự
            if type_match and size_match and sender_match and receiver_match:
                if 'source_shard' in hist_tx and 'destination_shard' in hist_tx:
                    similar_transactions.append(hist_tx)
        
        # Nếu có đủ giao dịch tương tự, sử dụng thông tin để dự đoán
        if len(similar_transactions) >= 3:
            # Đếm tần suất của các cặp source-destination
            route_counts = {}
            for tx in similar_transactions:
                route = (tx['source_shard'], tx['destination_shard'])
                route_counts[route] = route_counts.get(route, 0) + 1
            
            # Tìm tuyến đường phổ biến nhất
            if route_counts:
                most_common_route = max(route_counts.items(), key=lambda x: x[1])
                route, count = most_common_route
                
                # Tính độ tin cậy dựa trên tỷ lệ xuất hiện
                confidence_threshold = 0.6  # Ngưỡng tỷ lệ để xem là đáng tin cậy
                confidence = count / len(similar_transactions)
                
                return route[0], route[1], confidence >= confidence_threshold
        
        # Nếu không tìm được tuyến đường đáng tin cậy, sử dụng giá trị mặc định
        default_source = transaction.get('source_shard', 0)
        default_dest = transaction.get('destination_shard', 0)
        if default_source == default_dest:
            default_dest = (default_source + 1) % self.num_shards
            
        return default_source, default_dest, False 

    def analyze_transaction_patterns(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Phân tích mẫu giao dịch để tối ưu hóa định tuyến.
        
        Args:
            window_size: Số lượng giao dịch gần nhất để phân tích
            
        Returns:
            Dict[str, Any]: Kết quả phân tích mẫu giao dịch
        """
        if len(self.transaction_history) < window_size:
            return {'patterns_found': False}
        
        # Lấy các giao dịch gần nhất để phân tích
        recent_txs = self.transaction_history[-window_size:]
        
        # Phân tích tỷ lệ giao dịch cùng shard và xuyên shard
        same_shard_count = 0
        cross_shard_count = 0
        
        # Đếm số lượng giao dịch theo từng cặp shard
        shard_pair_counts = defaultdict(int)
        
        # Phân tích theo loại giao dịch
        tx_type_counts = defaultdict(int)
        
        for tx in recent_txs:
            tx_type = tx.get('type', 'default')
            tx_type_counts[tx_type] += 1
            
            if 'source_shard' in tx and 'destination_shard' in tx:
                src = tx['source_shard']
                dst = tx['destination_shard']
                
                if src == dst:
                    same_shard_count += 1
                else:
                    cross_shard_count += 1
                    pair = tuple(sorted([src, dst]))
                    shard_pair_counts[pair] += 1
        
        # Tính tỷ lệ giao dịch cùng shard
        total_tx_with_route = same_shard_count + cross_shard_count
        same_shard_ratio = same_shard_count / max(1, total_tx_with_route)
        
        # Tìm các cặp shard có lưu lượng cao
        high_traffic_pairs = []
        if shard_pair_counts:
            avg_count = sum(shard_pair_counts.values()) / len(shard_pair_counts)
            high_traffic_pairs = [pair for pair, count in shard_pair_counts.items() 
                                if count > 2 * avg_count]
        
        # Cập nhật ma trận gần cận dựa trên lưu lượng giao dịch
        for (i, j), count in shard_pair_counts.items():
            # Tính hệ số dựa trên lưu lượng
            traffic_factor = min(1.0, count / max(1, window_size * 0.1))
            
            # Cập nhật ma trận gần cận - tăng affinity cho các cặp shard giao dịch nhiều
            current_affinity = self.shard_affinity[i, j]
            # Affinity tăng dần theo lưu lượng, nhưng không vượt quá 1.0
            new_affinity = min(1.0, current_affinity + 0.1 * traffic_factor)
            
            self.shard_affinity[i, j] = new_affinity
            self.shard_affinity[j, i] = new_affinity
        
        # Cập nhật tỷ lệ giao dịch cùng shard mục tiêu
        target_ratio = max(0.4, min(0.9, same_shard_ratio))
        self.same_shard_ratio = 0.8 * self.same_shard_ratio + 0.2 * target_ratio
        
        # Trả về kết quả phân tích
        return {
            'patterns_found': True,
            'same_shard_ratio': same_shard_ratio,
            'high_traffic_pairs': high_traffic_pairs,
            'tx_type_distribution': dict(tx_type_counts)
        } 