import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
import heapq
import random
import time
from collections import defaultdict, deque
import math

class MADRAPIDRouter:
    """
    Multi-Agent Dynamic Routing and Adaptive Path Intelligence Distribution (MAD-RAPID).
    Thuật toán định tuyến thông minh cho giao dịch xuyên shard trong mạng blockchain.
    Bao gồm proximity-aware routing, dynamic mesh connections và predictive routing.
    """
    
    def __init__(self, 
                 network: nx.Graph,
                 shards: List[List[int]],
                 congestion_weight: float = 0.4,
                 latency_weight: float = 0.3,
                 energy_weight: float = 0.1,
                 trust_weight: float = 0.1,
                 prediction_horizon: int = 5,
                 congestion_threshold: float = 0.7,
                 proximity_weight: float = 0.3,  # Tăng ảnh hưởng của proximity
                 use_dynamic_mesh: bool = True,
                 predictive_window: int = 15,  # Tăng cửa sổ dự đoán
                 max_cache_size: int = 2000,  # Tăng kích thước cache
                 geo_awareness: bool = True,  # Mới: Nhận thức vị trí địa lý
                 traffic_history_length: int = 100,  # Mới: Lưu lịch sử traffic
                 dynamic_connections_limit: int = 20,  # Mới: Giới hạn kết nối động
                 update_interval: int = 30):  # Mới: Cập nhật thường xuyên hơn
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
            geo_awareness: Tính năng nhận thức vị trí địa lý
            traffic_history_length: Số lượng bước lưu lịch sử traffic
            dynamic_connections_limit: Giới hạn số lượng kết nối động
            update_interval: Khoảng thời gian cập nhật (số bước)
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
        
        # Tham số mới
        self.geo_awareness = geo_awareness
        self.traffic_history_length = traffic_history_length
        self.dynamic_connections_limit = dynamic_connections_limit
        self.mesh_update_interval = update_interval
        
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
        
        # Thêm ma trận gần cận giữa các shard - cập nhật định kỳ
        self.shard_affinity = np.ones((self.num_shards, self.num_shards)) - np.eye(self.num_shards)
        
        # Ma trận vị trí địa lý các shard - mới
        self.geo_distance_matrix = np.zeros((self.num_shards, self.num_shards))
        self._calculate_geo_distance_matrix()
        
        # Thêm lưu trữ cho giao dịch lịch sử
        self.transaction_history = deque(maxlen=self.traffic_history_length)
        
        # Dynamic mesh connections
        self.dynamic_connections = set()
        
        # Thống kê lưu lượng giữa các cặp shard
        self.shard_pair_traffic = defaultdict(int)
        
        # Lịch sử lưu lượng giao dịch giữa các cặp shard - mới
        self.traffic_history = defaultdict(lambda: deque(maxlen=self.traffic_history_length))
        
        # Predictive routing model - mới
        self.shard_traffic_pattern = {}  # Lưu mẫu lưu lượng để dự đoán
        
        # Temporal localization - mới
        self.temporal_locality = defaultdict(lambda: defaultdict(float))  # (source, dest) -> time -> frequency
        
        # Thời gian cuối cùng cập nhật dynamic mesh
        self.last_mesh_update = 0
        
        # Lịch sử lưu lượng theo thời gian - mới  
        self.time_based_traffic = defaultdict(lambda: {})  # time_bucket -> (source, dest) -> count
        
        # Đường dẫn tối ưu trước đó
        self.last_optimal_paths = {}
    
    def _calculate_geo_distance_matrix(self):
        """Tính toán ma trận khoảng cách địa lý giữa các shard."""
        for i in range(self.num_shards):
            for j in range(self.num_shards):
                if i == j:
                    self.geo_distance_matrix[i, j] = 0
                else:
                    pos_i = self.shard_graph.nodes[i]['position']
                    pos_j = self.shard_graph.nodes[j]['position']
                    distance = math.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                    
                    # Chuẩn hóa khoảng cách về [0,1]
                    normalized_distance = min(1.0, distance / 100.0)
                    self.geo_distance_matrix[i, j] = normalized_distance
    
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
            
            # Tính toán zone dựa trên vị trí địa lý - mới
            zone_x = int(x_pos / 25)  # Chia không gian thành 4x4 zones
            zone_y = int(y_pos / 25)
            zone = zone_x + zone_y * 4  # 16 zones tổng cộng
            
            shard_graph.add_node(shard_id, 
                               congestion=0.0,
                               trust_score=0.0,
                               position=(x_pos, y_pos),
                               capacity=len(self.shards[shard_id]),
                               zone=zone,  # Thêm zone - mới
                               processing_power=random.uniform(0.7, 1.0),  # Thêm processing power - mới
                               stability=1.0)  # Độ ổn định của shard - mới
        
        # Tính toán khoảng cách địa lý giữa các shard (dựa trên tọa độ logic)
        geographical_distances = {}
        for i in range(self.num_shards):
            pos_i = shard_graph.nodes[i]['position']
            zone_i = shard_graph.nodes[i]['zone']
            
            for j in range(i + 1, self.num_shards):
                pos_j = shard_graph.nodes[j]['position']
                zone_j = shard_graph.nodes[j]['zone']
                
                # Tính khoảng cách Euclidean
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                geographical_distances[(i, j)] = distance
                geographical_distances[(j, i)] = distance
                
                # Nếu cùng zone, tăng proximity factor - mới
                same_zone_bonus = 0.3 if zone_i == zone_j else 0.0
        
        # Tìm các kết nối giữa các shard và tính toán độ trễ/băng thông trung bình
        for i in range(self.num_shards):
            for j in range(i + 1, self.num_shards):
                cross_shard_edges = []
                
                # Tìm tất cả các cạnh kết nối node giữa hai shard
                for node_i in self.shards[i]:
                    for node_j in self.shards[j]:
                        if self.network.has_edge(node_i, node_j):
                            cross_shard_edges.append((node_i, node_j))
                
                # Yếu tố vị trí địa lý - mới
                geo_dist = geographical_distances.get((i, j), 50)
                norm_geo_dist = min(1.0, geo_dist / 100.0)
                
                # Kiểm tra zone - mới
                zone_i = shard_graph.nodes[i]['zone']
                zone_j = shard_graph.nodes[j]['zone']
                same_zone = zone_i == zone_j
                
                if cross_shard_edges:
                    # Thêm thuộc tính latency và bandwidth nếu chưa có
                    for u, v in cross_shard_edges:
                        if 'latency' not in self.network.edges[u, v]:
                            # Tính latency dựa trên khoảng cách địa lý
                            # Giảm latency khi cùng zone - mới
                            zone_factor = 0.7 if same_zone else 1.0
                            base_latency = random.uniform(3, 18)  # Giảm latency cơ bản
                            geo_factor = norm_geo_dist  # Normalized to [0,1]
                            # Latency tăng theo khoảng cách địa lý
                            self.network.edges[u, v]['latency'] = base_latency * (1 + geo_factor) * zone_factor
                        
                        if 'bandwidth' not in self.network.edges[u, v]:
                            # Bandwidth giảm theo khoảng cách địa lý
                            # Tăng bandwidth khi cùng zone - mới
                            zone_factor = 1.3 if same_zone else 1.0
                            base_bandwidth = random.uniform(10, 30)  # Tăng bandwidth cơ bản
                            # Bandwidth giảm theo khoảng cách địa lý
                            self.network.edges[u, v]['bandwidth'] = base_bandwidth * (1 - 0.4 * norm_geo_dist) * zone_factor
                    
                    # Tính độ trễ và băng thông trung bình của các kết nối
                    avg_latency = np.mean([self.network.edges[u, v]['latency'] for u, v in cross_shard_edges])
                    avg_bandwidth = np.mean([self.network.edges[u, v]['bandwidth'] for u, v in cross_shard_edges])
                    
                    # Tính proximity factor dựa trên khoảng cách địa lý và số lượng kết nối
                    num_connections = len(cross_shard_edges)
                    
                    # Proximity factor cao khi khoảng cách ngắn và nhiều kết nối
                    # Tăng ảnh hưởng của proximity và same zone - mới
                    proximity_factor = (1 - norm_geo_dist * 0.8) * min(1.0, num_connections / 8.0)
                    if same_zone:
                        proximity_factor += 0.2  # Bonus cho cùng zone
                    proximity_factor = min(1.0, proximity_factor)
                    
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
                                       same_zone=same_zone,  # Mới: cờ đánh dấu cùng zone
                                       last_updated=time.time())
                
                # Tự động thêm kết nối ảo giữa các shard cùng zone - mới
                elif same_zone and self.geo_awareness:
                    virtual_latency = 25 + norm_geo_dist * 30  # Latency cao hơn nhưng vẫn hợp lý
                    virtual_bandwidth = 5 + (1 - norm_geo_dist) * 10
                    
                    shard_graph.add_edge(i, j,
                                       latency=virtual_latency,
                                       bandwidth=virtual_bandwidth,
                                       connection_count=0,  # Không có kết nối thật
                                       geographical_distance=geo_dist,
                                       proximity_factor=0.3,  # Proximity thấp hơn
                                       historical_traffic=0,
                                       is_dynamic=False,
                                       is_virtual=True,  # Đánh dấu là kết nối ảo
                                       stability=0.7,  # Độ ổn định thấp hơn
                                       same_zone=True,
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
        
        # Thực hiện phân tích mẫu giao dịch trước khi cập nhật mesh
        pattern_analysis = self.analyze_transaction_patterns(window_size=min(self.traffic_history_length, 50))
        
        # Xác định các cặp shard có lưu lượng cao nhất
        top_traffic_pairs = sorted(self.shard_pair_traffic.items(), key=lambda x: x[1], reverse=True)
        
        # Xem xét các mẫu thời gian để phát hiện các điểm nóng theo thời gian
        time_based_hotspots = []
        for time_bucket, traffic_map in self.time_based_traffic.items():
            if traffic_map:  # Kiểm tra không trống
                # Lấy top 3 cặp shard cho mỗi time bucket
                hotspots = sorted(traffic_map.items(), key=lambda x: x[1], reverse=True)[:3]
                time_based_hotspots.extend([pair for pair, _ in hotspots])
        
        # Giới hạn số lượng kết nối động tối đa
        max_dynamic_connections = min(self.dynamic_connections_limit, 
                                     self.num_shards * (self.num_shards - 1) // 3)
        
        # Loại bỏ các kết nối dynamic cũ
        old_connections = list(self.dynamic_connections)
        for i, j in old_connections:
            if self.shard_graph.has_edge(i, j):
                # Nếu đã có kết nối trực tiếp giữa 2 shard, cập nhật thuộc tính
                if 'is_dynamic' in self.shard_graph.edges[i, j]:
                    self.shard_graph.edges[i, j]['is_dynamic'] = False
            self.dynamic_connections.remove((i, j))
        
        # Thêm kết nối mới cho các cặp shard có lưu lượng cao
        new_connections_count = 0
        
        # Ưu tiên các cặp shard trong time_based_hotspots
        priority_pairs = []
        for pair in time_based_hotspots:
            if pair not in priority_pairs:
                priority_pairs.append(pair)
        
        # Thêm các cặp shard từ top_traffic_pairs
        for (i, j), traffic in top_traffic_pairs:
            if (i, j) not in priority_pairs and (j, i) not in priority_pairs:
                priority_pairs.append((i, j))
        
        # Đi qua danh sách ưu tiên và thêm các kết nối dynamik
        for i, j in priority_pairs:
            # Nếu đã đạt số lượng kết nối tối đa, dừng lại
            if new_connections_count >= max_dynamic_connections:
                break
            
            # Sắp xếp lại để đảm bảo i < j
            if i > j:
                i, j = j, i
                
            # Lấy lưu lượng giao dịch
            traffic = self.shard_pair_traffic.get((i, j), 0) + self.shard_pair_traffic.get((j, i), 0)
            
            # Bỏ qua nếu lưu lượng quá thấp
            if traffic < 5:  # Ngưỡng tối thiểu để tạo dynamic connection
                continue
                
            # Bỏ qua nếu đã có kết nối trực tiếp giữa 2 shard
            if self.shard_graph.has_edge(i, j):
                self.shard_graph.edges[i, j]['is_dynamic'] = True
                self.shard_graph.edges[i, j]['historical_traffic'] = traffic
                
                # Cập nhật thuộc tính của kết nối dựa trên dữ liệu traffic
                if traffic > 20:  # Traffic cao
                    # Giảm latency và tăng bandwidth
                    self.shard_graph.edges[i, j]['latency'] *= 0.8
                    self.shard_graph.edges[i, j]['bandwidth'] *= 1.2
                
                self.dynamic_connections.add((i, j))
                new_connections_count += 1
                continue
                
            # Tính toán các thuộc tính cho kết nối mới
            pos_i = self.shard_graph.nodes[i]['position']
            pos_j = self.shard_graph.nodes[j]['position']
            geo_dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            
            # Kiểm tra zone - mới
            zone_i = self.shard_graph.nodes[i].get('zone', -1)
            zone_j = self.shard_graph.nodes[j].get('zone', -1)
            same_zone = zone_i == zone_j and zone_i != -1
            
            # Tính proximity factor - mới
            normalized_dist = min(1.0, geo_dist / 100.0)
            proximity_factor = 0.5 + 0.5 * (1 - normalized_dist)
            if same_zone:
                proximity_factor += 0.2  # Bonus cho cùng zone
            proximity_factor = min(1.0, proximity_factor)
            
            # Latency và bandwidth tốt hơn cho kết nối dynamic do đã được tối ưu hóa
            zone_factor = 0.8 if same_zone else 1.0
            latency = (5 + normalized_dist * 15) * zone_factor  # Latency thấp hơn cho kết nối động
            bandwidth = (15 + (1 - normalized_dist) * 15) * (1.2 if same_zone else 1.0)  # Bandwidth cao hơn
            
            # Tính độ ổn định dựa trên mức độ ổn định của lưu lượng
            stability = 0.7
            if (i, j) in self.traffic_history or (j, i) in self.traffic_history:
                # Nếu có lịch sử lưu lượng, tính độ ổn định dựa trên biến thiên
                history = self.traffic_history.get((i, j), self.traffic_history.get((j, i), []))
                if len(history) > 5:
                    # Tính hệ số biến thiên (CV)
                    std_dev = np.std(history)
                    mean = np.mean(history)
                    if mean > 0:
                        cv = std_dev / mean
                        # Độ ổn định cao khi CV thấp
                        stability = max(0.5, min(0.95, 1.0 - cv))
            
            # Thêm cạnh mới vào đồ thị shard
            self.shard_graph.add_edge(i, j,
                                     latency=latency,
                                     bandwidth=bandwidth,
                                     connection_count=1,
                                     geographical_distance=geo_dist,
                                     proximity_factor=proximity_factor,
                                     historical_traffic=traffic,
                                     is_dynamic=True,
                                     same_zone=same_zone,  # Mới
                                     stability=stability,
                                     last_updated=current_time)
            
            # Thêm vào danh sách dynamic connections
            self.dynamic_connections.add((i, j))
            new_connections_count += 1
        
        # Nếu đã tạo ít nhất một kết nối mới, log thông tin
        if new_connections_count > 0:
            dynamic_edges = [(i, j, self.shard_graph.edges[i, j]['latency'], 
                             self.shard_graph.edges[i, j]['bandwidth']) 
                            for i, j in self.dynamic_connections]
            print(f"Đã tạo {new_connections_count} dynamic mesh connections: {dynamic_edges}")
        
        # Cập nhật dynamic weight cho routing algorithm dựa trên lưu lượng - mới
        self._update_routing_weights(pattern_analysis)
    
    def _update_routing_weights(self, pattern_analysis):
        """
        Cập nhật trọng số routing dựa trên phân tích mẫu lưu lượng.
        
        Args:
            pattern_analysis: Kết quả phân tích mẫu lưu lượng
        """
        if not pattern_analysis.get('patterns_found', False):
            return
            
        # Điều chỉnh trọng số dựa trên tỷ lệ giao dịch trong cùng một shard
        same_shard_ratio = pattern_analysis.get('same_shard_ratio', 0.8)
        
        # Nếu tỷ lệ giao dịch trong cùng shard thấp, tăng proximity_weight
        if same_shard_ratio < 0.5:
            new_proximity_weight = min(0.4, self.proximity_weight + 0.05)
            print(f"Điều chỉnh proximity_weight: {self.proximity_weight:.2f} -> {new_proximity_weight:.2f}")
            self.proximity_weight = new_proximity_weight
            
            # Giảm congestion_weight tương ứng
            self.congestion_weight = max(0.2, 1.0 - self.proximity_weight - self.latency_weight - self.energy_weight - self.trust_weight)
        
        # Nếu có nhiều giao dịch xuyên shard giữa các zone, tăng geo_awareness
        high_traffic_pairs = pattern_analysis.get('high_traffic_pairs', [])
        if high_traffic_pairs:
            # Kiểm tra xem các cặp có traffic cao có cùng zone không
            different_zone_pairs = 0
            for i, j in high_traffic_pairs:
                if i < len(self.shard_graph.nodes) and j < len(self.shard_graph.nodes):
                    zone_i = self.shard_graph.nodes[i].get('zone', -1)
                    zone_j = self.shard_graph.nodes[j].get('zone', -1)
                    if zone_i != zone_j or zone_i == -1:
                        different_zone_pairs += 1
            
            # Nếu nhiều cặp khác zone, tăng geo_awareness
            if different_zone_pairs > len(high_traffic_pairs) / 2:
                self.geo_awareness = True
    
    def predictive_routing(self, transaction: Dict[str, Any]) -> Tuple[int, int, bool]:
        """
        Dự đoán đường dẫn tối ưu dựa trên lịch sử giao dịch và mẫu giao dịch.
        
        Args:
            transaction: Giao dịch cần định tuyến
            
        Returns:
            Tuple[int, int, bool]: (next_shard, final_shard, is_direct_route)
            - next_shard: Shard tiếp theo để chuyển giao dịch
            - final_shard: Shard đích cuối cùng
            - is_direct_route: True nếu đường dẫn là trực tiếp (không qua shard trung gian)
        """
        source_shard = transaction['source_shard']
        dest_shard = transaction['destination_shard']
        tx_value = transaction.get('value', 0.0)
        tx_priority = transaction.get('priority', 0.5)
        tx_type = transaction.get('type', 'intra_shard')
        tx_category = transaction.get('category', 'simple_transfer')
        
        # Tạo ID cho cặp giao dịch
        tx_pair_id = f"{source_shard}_{dest_shard}"
        
        # Nếu là giao dịch nội shard, trả về source_shard là đích
        if source_shard == dest_shard:
            return source_shard, dest_shard, True
        
        # Kiểm tra trong cache trước
        cache_key = f"{source_shard}_{dest_shard}_{tx_category}_{round(tx_value)}"
        if cache_key in self.path_cache and self.current_step - self.path_cache[cache_key]['step'] < self.cache_expire_time:
            cached_path = self.path_cache[cache_key]['path']
            if len(cached_path) > 1:
                return cached_path[1], cached_path[-1], len(cached_path) == 2
        
        # Kiểm tra proximity trước tiên
        if self.geo_awareness:
            geo_distance = self.geo_distance_matrix[source_shard, dest_shard]
            
            # Nếu hai shard gần nhau địa lý và có kết nối trực tiếp
            if geo_distance < 0.4 and self.shard_graph.has_edge(source_shard, dest_shard):
                direct_edge = self.shard_graph.edges[source_shard, dest_shard]
                
                # Kiểm tra các điều kiện để sử dụng kết nối trực tiếp
                if direct_edge.get('bandwidth', 0) > 5 and direct_edge.get('stability', 0) > 0.5:
                    return dest_shard, dest_shard, True
        
        # Kiểm tra mẫu thời gian
        current_time_bucket = self.current_step % (24 * 60)  # Giả lập 24 giờ với mỗi phút là 1 bước
        time_slot = current_time_bucket // 60  # Chia thành 24 time slot
        
        # Nếu có mẫu thời gian cho cặp shard này
        if tx_pair_id in self.temporal_locality and time_slot in self.temporal_locality[tx_pair_id]:
            frequency = self.temporal_locality[tx_pair_id][time_slot]
            
            # Nếu tần suất cao, tìm đường đi nhanh nhất
            if frequency > 0.5:
                # Tìm đường đi nhanh nhất
                fast_path = self._find_fastest_path(source_shard, dest_shard)
                if fast_path and len(fast_path) > 1:
                    # Lưu vào cache
                    self.path_cache[cache_key] = {
                        'path': fast_path,
                        'step': self.current_step,
                        'type': 'temporal_pattern'
                    }
                    return fast_path[1], fast_path[-1], len(fast_path) == 2
        
        # Kiểm tra lịch sử giao dịch gần đây
        if len(self.transaction_history) > 0:
            # Lọc các giao dịch tương tự gần đây
            similar_txs = []
            for tx in self.transaction_history[-100:]:
                if (tx['source_shard'] == source_shard and 
                    tx['destination_shard'] == dest_shard and
                    tx.get('category', '') == tx_category and
                    abs(tx.get('value', 0) - tx_value) / max(1, tx_value) < 0.3):  # Value khác <30%
                    similar_txs.append(tx)
            
            if similar_txs:
                # Tìm giao dịch gần đây nhất có cùng nguồn và đích
                latest_similar_tx = max(similar_txs, key=lambda tx: tx.get('created_at', 0))
                
                # Nếu có routed_path và thành công
                if 'routed_path' in latest_similar_tx and latest_similar_tx.get('status', '') == 'completed':
                    historical_path = latest_similar_tx['routed_path']
                    
                    # Kiểm tra xem đường dẫn này còn hợp lệ không
                    is_valid_path = True
                    for i in range(len(historical_path) - 1):
                        if not self.shard_graph.has_edge(historical_path[i], historical_path[i+1]):
                            is_valid_path = False
                            break
                    
                    if is_valid_path and len(historical_path) > 1:
                        # Lưu vào cache
                        self.path_cache[cache_key] = {
                            'path': historical_path,
                            'step': self.current_step,
                            'type': 'historical'
                        }
                        return historical_path[1], historical_path[-1], len(historical_path) == 2
        
        # Xem xét dynamic mesh connections
        if self.use_dynamic_mesh and len(self.dynamic_connections) > 0:
            # Kiểm tra nếu có kết nối động trực tiếp
            if (source_shard, dest_shard) in self.dynamic_connections or (dest_shard, source_shard) in self.dynamic_connections:
                return dest_shard, dest_shard, True
            
            # Tìm đường đi qua dynamic mesh
            dynamic_path = self._find_dynamic_mesh_path(source_shard, dest_shard)
            if dynamic_path and len(dynamic_path) > 1:
                # Lưu vào cache
                self.path_cache[cache_key] = {
                    'path': dynamic_path,
                    'step': self.current_step,
                    'type': 'dynamic_mesh'
                }
                return dynamic_path[1], dynamic_path[-1], len(dynamic_path) == 2
        
        # Nếu tất cả các phương pháp trên thất bại, sử dụng Dijkstra truyền thống
        optimal_path = self._dijkstra(source_shard, dest_shard, transaction)
        
        # Lưu đường dẫn vào cache
        if optimal_path and len(optimal_path) > 1:
            self.path_cache[cache_key] = {
                'path': optimal_path,
                'step': self.current_step,
                'type': 'dijkstra'
            }
            return optimal_path[1], optimal_path[-1], len(optimal_path) == 2
        
        # Nếu không tìm được đường đi, trả về đích trực tiếp
        return dest_shard, dest_shard, True
    
    def _find_fastest_path(self, source_shard: int, dest_shard: int) -> List[int]:
        """
        Tìm đường đi nhanh nhất giữa hai shard.
        
        Args:
            source_shard: Shard nguồn
            dest_shard: Shard đích
            
        Returns:
            List[int]: Đường đi nhanh nhất (danh sách shard ID)
        """
        # Nếu đã có đường đi trực tiếp, sử dụng nó
        if self.shard_graph.has_edge(source_shard, dest_shard):
            return [source_shard, dest_shard]
        
        # Sử dụng thuật toán Dijkstra focus vào latency
        visited = set()
        distances = {source_shard: 0}
        previous = {}
        priority_queue = [(0, source_shard)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_node == dest_shard:
                # Đã tìm thấy đường đi tới đích
                return self._reconstruct_path(previous, dest_shard)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            for neighbor in self.shard_graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Lấy độ trễ của kết nối
                edge_latency = self.shard_graph.edges[current_node, neighbor].get('latency', 50)
                
                # Ưu tiên dynamic mesh connections
                if self.shard_graph.edges[current_node, neighbor].get('is_dynamic', False):
                    edge_latency *= 0.7  # Giảm latency cho dynamic connections
                
                # Tính khoảng cách mới
                new_distance = current_distance + edge_latency
                
                # Nếu tìm thấy đường đi ngắn hơn hoặc chưa có đường đi
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))
        
        # Nếu không tìm thấy đường đi
        return []
    
    def _find_dynamic_mesh_path(self, source_shard: int, dest_shard: int) -> List[int]:
        """
        Tìm đường đi thông qua các dynamic mesh connections.
        
        Args:
            source_shard: Shard nguồn
            dest_shard: Shard đích
            
        Returns:
            List[int]: Đường đi qua dynamic mesh (danh sách shard ID)
        """
        # Tạo đồ thị dynamic mesh
        dynamic_graph = nx.Graph()
        
        # Thêm tất cả các node
        for i in range(self.num_shards):
            dynamic_graph.add_node(i)
        
        # Thêm các kết nối dynamic
        for i, j in self.dynamic_connections:
            dynamic_graph.add_edge(i, j)
        
        # Thêm các kết nối trực tiếp giữa các shard
        for i, j in self.shard_graph.edges():
            dynamic_graph.add_edge(i, j)
        
        # Sử dụng thuật toán đường đi ngắn nhất của networkx
        try:
            # Kiểm tra nếu có đường đi
            if nx.has_path(dynamic_graph, source_shard, dest_shard):
                path = nx.shortest_path(dynamic_graph, source=source_shard, target=dest_shard)
                return path
        except Exception as e:
            print(f"Lỗi khi tìm dynamic mesh path: {e}")
        
        return []
    
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