import gym
import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Any, Optional
from gym import spaces
import time
import random
from collections import defaultdict

from qtrust.utils.logging import simulation_logger as logger

class BlockchainEnvironment(gym.Env):
    """
    Môi trường mô phỏng blockchain với sharding cho Deep Reinforcement Learning.
    Môi trường này mô phỏng một mạng blockchain với nhiều shard và giao dịch xuyên shard.
    Hỗ trợ resharding động và tối đa 32 shard.
    """
    
    def __init__(self, 
                 num_shards: int = 4, 
                 num_nodes_per_shard: int = 10,
                 max_transactions_per_step: int = 100,
                 transaction_value_range: Tuple[float, float] = (0.1, 100.0),
                 max_steps: int = 1000,
                 latency_penalty: float = 0.5,
                 energy_penalty: float = 0.3,
                 throughput_reward: float = 1.0,
                 security_reward: float = 0.8,
                 max_num_shards: int = 32,
                 min_num_shards: int = 2,
                 enable_dynamic_resharding: bool = True,
                 congestion_threshold_high: float = 0.85,
                 congestion_threshold_low: float = 0.15,
                 resharding_interval: int = 50):
        """
        Khởi tạo môi trường blockchain với sharding.
        
        Args:
            num_shards: Số lượng shard ban đầu trong mạng
            num_nodes_per_shard: Số lượng node trong mỗi shard
            max_transactions_per_step: Số lượng giao dịch tối đa mỗi bước
            transaction_value_range: Phạm vi giá trị giao dịch (min, max)
            max_steps: Số bước tối đa cho mỗi episode
            latency_penalty: Hệ số phạt cho độ trễ
            energy_penalty: Hệ số phạt cho tiêu thụ năng lượng
            throughput_reward: Hệ số thưởng cho throughput
            security_reward: Hệ số thưởng cho bảo mật
            max_num_shards: Số lượng shard tối đa có thể có trong hệ thống
            min_num_shards: Số lượng shard tối thiểu
            enable_dynamic_resharding: Cho phép resharding động hay không
            congestion_threshold_high: Ngưỡng tắc nghẽn để tăng số lượng shard
            congestion_threshold_low: Ngưỡng tắc nghẽn để giảm số lượng shard
            resharding_interval: Số bước giữa các lần kiểm tra resharding
        """
        super(BlockchainEnvironment, self).__init__()
        
        # Giới hạn số shard trong phạm vi cho phép
        self.num_shards = max(min_num_shards, min(num_shards, max_num_shards))
        self.num_nodes_per_shard = num_nodes_per_shard
        self.max_transactions_per_step = max_transactions_per_step
        self.transaction_value_range = transaction_value_range
        self.max_steps = max_steps
        
        # Tham số resharding động
        self.max_num_shards = max_num_shards
        self.min_num_shards = min_num_shards
        self.enable_dynamic_resharding = enable_dynamic_resharding
        self.congestion_threshold_high = congestion_threshold_high
        self.congestion_threshold_low = congestion_threshold_low
        self.resharding_interval = resharding_interval
        self.last_resharding_step = 0
        
        # Tính tổng số node trong hệ thống
        self.total_nodes = self.num_shards * self.num_nodes_per_shard
        
        # Các hệ số reward/penalty - tăng thưởng cho throughput và giảm phạt cho latency/energy
        self.latency_penalty = latency_penalty * 0.8  # Giảm 20% phạt độ trễ
        self.energy_penalty = energy_penalty * 0.8    # Giảm 20% phạt năng lượng
        self.throughput_reward = throughput_reward * 1.2  # Tăng 20% thưởng throughput
        self.security_reward = security_reward
        
        # Thêm hệ số để điều chỉnh performance
        self.network_efficiency = 1.0  # Hệ số điều chỉnh hiệu suất mạng
        self.cross_shard_penalty = 0.2  # Phạt cho giao dịch xuyên shard
        
        # Số bước hiện tại
        self.current_step = 0
        
        # Khởi tạo không gian trạng thái và hành động
        self._init_state_action_space()
        
        # Khởi tạo mạng blockchain
        self._init_blockchain_network()
        
        # Khởi tạo transaction pool
        self.transaction_pool = []
        
        # Lưu trữ lịch sử resharding
        self.resharding_history = []
        
        # Lưu trữ thông tin tắc nghẽn theo thời gian
        self.congestion_history = []
        
        # Thống kê hiệu suất
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy_consumption': [],
            'security_score': []
        }
        
        # Thêm performance metrics cho tests
        self.performance_metrics = {
            'transactions_processed': 0,
            'total_latency': 0,
            'total_energy': 0,
            'successful_transactions': 0
        }
        
        # Thêm thuộc tính blockchain_network cho tests
        self.blockchain_network = self.network
        
        logger.info(f"Khởi tạo môi trường blockchain với {self.num_shards} shard, mỗi shard có {num_nodes_per_shard} nodes")
        logger.info(f"Resharding động: {'Bật' if enable_dynamic_resharding else 'Tắt'}, Max shards: {max_num_shards}")
    
    def _init_state_action_space(self):
        """Khởi tạo không gian trạng thái và hành động."""
        # Không gian trạng thái:
        # - Mức độ tắc nghẽn mạng cho mỗi shard (0.0-1.0)
        # - Giá trị giao dịch trung bình trong mỗi shard
        # - Điểm tin cậy trung bình của các node trong mỗi shard (0.0-1.0)
        # - Tỷ lệ giao dịch thành công gần đây
        
        # Mỗi shard có 4 đặc trưng, cộng với 4 đặc trưng toàn cục
        # Thiết kế dynamic để hỗ trợ thay đổi số lượng shard
        max_features = self.max_num_shards * 4 + 4
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=float('inf'), 
            shape=(max_features,), 
            dtype=np.float32
        )
        
        # Không gian hành động:
        # - Lựa chọn shard đích cho một giao dịch (0 to max_num_shards-1)
        # - Lựa chọn giao thức đồng thuận (0: Fast BFT, 1: PBFT, 2: Robust BFT)
        self.action_space = spaces.MultiDiscrete([self.max_num_shards, 3])
        
        # Định nghĩa không gian trạng thái và hành động cho một cách nhìn dễ hiểu hơn
        self.state_space = {
            'network_congestion': [0.0, 1.0],  # Mức độ tắc nghẽn
            'transaction_value': [self.transaction_value_range[0], self.transaction_value_range[1]],
            'trust_scores': [0.0, 1.0],  # Điểm tin cậy
            'success_rate': [0.0, 1.0]   # Tỷ lệ thành công
        }
        
        self.action_space_dict = {
            'routing_decision': list(range(self.max_num_shards)),
            'consensus_selection': ['Fast_BFT', 'PBFT', 'Robust_BFT']
        }
    
    def _init_blockchain_network(self):
        """Khởi tạo mạng blockchain với shards và nodes."""
        # Sử dụng networkx để biểu diễn mạng blockchain
        self.network = nx.Graph()
        
        # Khởi tạo các node cho mỗi shard
        self.shards = []
        total_nodes = 0
        
        for shard_id in range(self.num_shards):
            shard_nodes = []
            for i in range(self.num_nodes_per_shard):
                node_id = total_nodes + i
                # Thêm node vào mạng với hiệu suất cao hơn
                self.network.add_node(
                    node_id, 
                    shard_id=shard_id,
                    trust_score=np.random.uniform(0.6, 1.0),  # Tăng điểm tin cậy ban đầu (từ 0.5-1.0 lên 0.6-1.0)
                    processing_power=np.random.uniform(0.8, 1.0),  # Tăng khả năng xử lý (từ 0.7-1.0 lên 0.8-1.0)
                    energy_efficiency=np.random.uniform(0.7, 0.95)  # Tăng hiệu suất năng lượng (từ 0.6-0.9 lên 0.7-0.95)
                )
                shard_nodes.append(node_id)
            
            self.shards.append(shard_nodes)
            total_nodes += self.num_nodes_per_shard
            logger.debug(f"Đã khởi tạo shard {shard_id} với {len(shard_nodes)} nodes")
        
        # Tạo kết nối giữa các node trong cùng một shard (đầy đủ kết nối)
        intra_shard_connections = 0
        for shard_nodes in self.shards:
            for i in range(len(shard_nodes)):
                for j in range(i + 1, len(shard_nodes)):
                    # Độ trễ từ 1ms đến 5ms cho các node trong cùng shard (giảm từ 1-10ms)
                    self.network.add_edge(
                        shard_nodes[i], 
                        shard_nodes[j], 
                        latency=np.random.uniform(1, 5),  # Giảm độ trễ tối đa
                        bandwidth=np.random.uniform(80, 150)  # Tăng băng thông (từ 50-100 lên 80-150 Mbps)
                    )
                    intra_shard_connections += 1
        
        # Tạo kết nối giữa các shard (một số kết nối ngẫu nhiên)
        inter_shard_connections = 0
        for i in range(self.num_shards):
            for j in range(i + 1, self.num_shards):
                # Chọn ngẫu nhiên 3 node từ mỗi shard để kết nối
                nodes_from_shard_i = np.random.choice(self.shards[i], 3, replace=False)
                nodes_from_shard_j = np.random.choice(self.shards[j], 3, replace=False)
                
                for node_i in nodes_from_shard_i:
                    for node_j in nodes_from_shard_j:
                        # Độ trễ từ 5ms đến 30ms cho các node giữa các shard (giảm từ 10-50ms)
                        self.network.add_edge(
                            node_i, 
                            node_j, 
                            latency=np.random.uniform(5, 30),  # Giảm độ trễ
                            bandwidth=np.random.uniform(20, 70)  # Tăng băng thông (từ 10-50 lên 20-70 Mbps)
                        )
                        inter_shard_connections += 1
        
        logger.info(f"Khởi tạo mạng blockchain thành công với {total_nodes} nodes, {intra_shard_connections} kết nối nội shard, {inter_shard_connections} kết nối liên shard")
        
        # Thiết lập trạng thái congestion ban đầu cho mỗi shard (giảm mức tắc nghẽn)
        self.shard_congestion = np.random.uniform(0.05, 0.2, self.num_shards)  # Giảm từ 0.1-0.3 xuống 0.05-0.2
        
        # Thiết lập trạng thái hiện tại cho consensus protocol của mỗi shard
        # 0: Fast BFT, 1: PBFT, 2: Robust BFT
        self.shard_consensus = np.zeros(self.num_shards, dtype=np.int32)
    
    def _generate_transactions(self) -> List[Dict[str, Any]]:
        """Tạo ngẫu nhiên các giao dịch cho bước hiện tại."""
        # Tạo số lượng giao dịch theo phân phối Poisson
        # Điều chỉnh lambda (tốc độ giao dịch) để phù hợp với quy mô mạng
        lambda_rate = 1.0 + 0.8 * self.num_shards  # Tăng tỷ lệ giao dịch
        # Số giao dịch thay đổi theo thời gian và tình trạng mạng
        # Giai đoạn đầu ít, sau đó tăng, rồi ổn định
        time_factor = min(1.0, self.current_step / (self.max_steps * 0.15))  # Tăng nhanh hơn
        congestion_factor = 1.0 - 0.4 * np.mean(self.shard_congestion)  # Giảm nhẹ ảnh hưởng của tắc nghẽn
        
        adjusted_lambda = lambda_rate * time_factor * congestion_factor
        num_transactions = np.random.poisson(adjusted_lambda)
        
        # Giới hạn số lượng giao dịch tối đa để tránh quá tải hệ thống
        num_transactions = min(num_transactions + 2, self.max_transactions_per_step)  # Tăng thêm ít nhất 2 giao dịch
        
        transactions = []
        for i in range(num_transactions):
            # Chọn shard nguồn và đích
            source_shard = np.random.randint(0, self.num_shards)
            
            # Xác định loại giao dịch: nội shard (60%), xuyên shard (40%)
            is_cross_shard = np.random.random() < 0.4
            if is_cross_shard and self.num_shards > 1:
                # Chọn một shard đích khác shard nguồn
                destination_shard = np.random.choice([s for s in range(self.num_shards) if s != source_shard])
                tx_type = "cross_shard"
            else:
                # Giao dịch nội shard
                destination_shard = source_shard
                tx_type = "intra_shard"
            
            # Chọn node nguồn và đích
            source_node = np.random.choice(self.shards[source_shard])
            destination_node = np.random.choice(self.shards[destination_shard])
            
            # Giá trị giao dịch
            value = np.random.uniform(self.transaction_value_range[0], self.transaction_value_range[1])
            
            # Mức độ ưu tiên và độ phức tạp
            priority = np.random.random()  # 0.0 - 1.0
            complexity = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
            
            # Loại giao dịch
            category = np.random.choice(['simple_transfer', 'smart_contract', 'token_exchange'], 
                                        p=[0.7, 0.2, 0.1])
            
            # Tạo giao dịch với các trường phù hợp với test
            transaction = {
                'id': f'tx_{source_shard}_{i}',
                'source': source_node,  # Thêm trường source cho test
                'destination': destination_node,  # Thêm trường destination cho test
                'source_shard': source_shard,
                'destination_shard': destination_shard,
                'source_node': source_node,
                'destination_node': destination_node,
                'value': value,
                'timestamp': self.current_step,  # Thêm trường timestamp cho test
                'priority': priority,
                'complexity': complexity,
                'category': category,
                'type': tx_type,
                'created_at': self.current_step,
                'status': 'pending',
                'routed_path': [source_shard],
                'consensus_protocol': None,
                'energy_consumed': 0.0
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def get_state(self) -> np.ndarray:
        """
        Lấy trạng thái hiện tại của môi trường.
        
        Returns:
            np.ndarray: Vector trạng thái
        """
        # Tạo vector trạng thái với đủ không gian cho max_num_shards
        # Sẽ padding 0 cho các shard không tồn tại
        
        # Tạo vector với các đặc trưng toàn cục
        global_features = np.array([
            np.mean(self.shard_congestion),  # Mức độ tắc nghẽn trung bình
            self.num_shards / self.max_num_shards,  # Tỷ lệ số lượng shard hiện tại
            self.current_step / self.max_steps,  # Tiến độ hiện tại
            self._get_network_stability()  # Độ ổn định của mạng
        ])
        
        # Tạo đặc trưng cho từng shard
        shard_features = np.zeros(self.max_num_shards * 4)
        
        for i in range(self.num_shards):
            # Vị trí trong vector trạng thái
            base_idx = i * 4
            
            # Lấy thông tin cho shard i
            shard_features[base_idx] = self.shard_congestion[i]  # Mức độ tắc nghẽn
            shard_features[base_idx + 1] = self._get_average_transaction_value(i)  # Giá trị giao dịch trung bình
            shard_features[base_idx + 2] = self._get_average_trust_score(i)  # Điểm tin cậy trung bình
            shard_features[base_idx + 3] = self._get_success_rate(i)  # Tỷ lệ thành công
        
        # Ghép các đặc trưng toàn cục và đặc trưng shard
        state = np.concatenate([global_features, shard_features]).astype(np.float32)
        
        # Đảm bảo state nằm trong không gian observation_space
        # Giới hạn giá trị trong phạm vi [0, inf)
        state = np.clip(state, 0.0, float('inf'))
        
        return state
    
    def _get_network_stability(self) -> float:
        """
        Tính toán độ ổn định của mạng dựa trên lịch sử resharding.
        
        Returns:
            float: Độ ổn định từ 0.0 đến 1.0
        """
        if not self.resharding_history:
            return 1.0
        
        # Số lần resharding gần đây (trong 100 bước gần nhất)
        recent_resharding_count = sum(1 for r in self.resharding_history 
                                     if self.current_step - r['step'] <= 100)
        
        # Độ ổn định giảm khi có nhiều lần resharding gần đây
        stability = max(0.0, 1.0 - recent_resharding_count * 0.2)
        
        return stability
    
    def _get_average_trust_score(self, shard_id: int) -> float:
        """
        Tính điểm tin cậy trung bình của các node trong một shard.
        
        Args:
            shard_id: ID của shard
            
        Returns:
            float: Điểm tin cậy trung bình
        """
        if shard_id >= self.num_shards:
            return 0.0
        
        trust_scores = []
        for node_id in self.shards[shard_id]:
            trust_score = self.network.nodes[node_id].get('trust_score', 0.7)
            trust_scores.append(trust_score)
        
        return np.mean(trust_scores) if trust_scores else 0.7
    
    def _get_success_rate(self, shard_id: int) -> float:
        """
        Tính tỷ lệ giao dịch thành công gần đây cho một shard.
        
        Args:
            shard_id: ID của shard
            
        Returns:
            float: Tỷ lệ thành công
        """
        if shard_id >= self.num_shards:
            return 0.0
        
        # Lấy các giao dịch gần đây liên quan đến shard này
        recent_txs = [tx for tx in self.transaction_pool 
                     if (tx['source_shard'] == shard_id or tx['destination_shard'] == shard_id) 
                     and self.current_step - tx['created_at'] <= 10]
        
        if not recent_txs:
            return 0.8  # Giá trị mặc định nếu không có giao dịch
        
        # Tính tỷ lệ thành công
        successful_txs = sum(1 for tx in recent_txs if tx['status'] == 'completed')
        return successful_txs / len(recent_txs)
    
    def _get_reward(self, action: np.ndarray, state: np.ndarray) -> float:
        """
        Tính toán phần thưởng dựa trên hành động và trạng thái.
        
        Args:
            action: Hành động đã chọn
            state: Trạng thái hiện tại
        
        Returns:
            float: Phần thưởng
        """
        # Thu thập các thành phần phần thưởng
        throughput_reward = self._get_throughput_reward()
        latency_penalty = self._get_latency_penalty()
        energy_penalty = self._get_energy_penalty()
        security_reward = self._get_security_reward()
        cross_shard_penalty = self._get_cross_shard_penalty()
        consensus_reward = self._get_consensus_reward(action)
        
        # Tổng hợp phần thưởng - Điều chỉnh trọng số:
        # - Tăng trọng số throughput_reward lên 1.5 (tăng 50%)
        # - Giảm trọng số latency_penalty xuống 0.6 (giảm 40%)
        # - Giảm trọng số energy_penalty xuống 0.6 (giảm 40%)
        # - Thêm ưu tiên cho innovation trong routing
        throughput_reward *= 1.5  # Tăng thêm 50%
        latency_penalty *= 0.6  # Giảm 40%
        energy_penalty *= 0.6  # Giảm 40%
        
        # Tổng hợp phần thưởng với trọng số mới
        reward = (throughput_reward - latency_penalty - energy_penalty + 
                security_reward - cross_shard_penalty + consensus_reward)
        
        # Thưởng thêm cho việc đạt hiệu suất cao
        if self._is_high_performance():
            reward += 2.5  # Tăng thưởng bổ sung cho hiệu suất xuất sắc (từ 2.0 lên 2.5)
        
        # Thêm thưởng cho innovation trong routing
        if self._is_innovative_routing(action):
            reward += 1.0  # Thưởng cho việc thử nghiệm các chiến lược routing mới
        
        return reward
    
    def _get_throughput_reward(self) -> float:
        """Tính phần thưởng cho throughput."""
        throughput = self.metrics['throughput'][-1] if self.metrics['throughput'] else 0
        return self.throughput_reward * throughput / 10.0  # Chuẩn hóa theo giá trị tối đa mong đợi
    
    def _get_latency_penalty(self) -> float:
        """Tính phạt cho độ trễ."""
        avg_latency = self.metrics['latency'][-1] if self.metrics['latency'] else 0
        return self.latency_penalty * min(1.0, avg_latency / 50.0)  # Chuẩn hóa (50ms là ngưỡng cao)
    
    def _get_energy_penalty(self) -> float:
        """Tính phạt cho tiêu thụ năng lượng."""
        energy_usage = self.metrics['energy_consumption'][-1] if self.metrics['energy_consumption'] else 0
        return self.energy_penalty * min(1.0, energy_usage / 400.0)  # Chuẩn hóa (400mJ/tx là ngưỡng cao)
    
    def _get_security_reward(self) -> float:
        """Tính phần thưởng cho độ bảo mật."""
        security_score = self.metrics['security_score'][-1] if self.metrics['security_score'] else 0
        return self.security_reward * security_score
    
    def _get_cross_shard_penalty(self) -> float:
        """Tính phạt cho tỷ lệ giao dịch xuyên shard."""
        cross_shard_txs = len([tx for tx in self.transaction_pool if tx.get('type') == 'cross_shard' and tx.get('status') == 'completed'])
        total_completed_txs = len([tx for tx in self.transaction_pool if tx.get('status') == 'completed'])
        cross_shard_ratio = cross_shard_txs / max(1, total_completed_txs)
        return self.cross_shard_penalty * max(0, cross_shard_ratio - 0.2)  # Chỉ phạt khi tỷ lệ > 20%
    
    def _get_consensus_reward(self, action) -> float:
        """
        Tính phần thưởng cho việc chọn đúng giao thức đồng thuận.
        
        Args:
            action: Hành động đã chọn
        
        Returns:
            float: Phần thưởng
        """
        # Kiểm tra kiểu dữ liệu của action và xử lý phù hợp
        if isinstance(action, (int, float, np.integer, np.floating)):
            # Nếu action là một số vô hướng, giả định đó là một hành động rời rạc
            # Cần phân tách thành destination_shard và consensus_protocol
            action_id = int(action)
            # Số lượng giao thức đồng thuận
            num_consensus_protocols = len(self.action_space_dict['consensus_selection'])
            # Tính destination_shard và consensus_protocol từ action_id
            destination_shard = action_id // num_consensus_protocols
            consensus_protocol = action_id % num_consensus_protocols
        elif hasattr(action, "__len__") and len(action) >= 2:
            # Nếu action là mảng với ít nhất 2 phần tử
            destination_shard = action[0]
            consensus_protocol = action[1]
        else:
            # Trường hợp không xác định, sử dụng giá trị mặc định an toàn
            logger.warning(f"Kiểu dữ liệu action không hợp lệ: {type(action)}. Sử dụng giá trị mặc định.")
            destination_shard = 0
            consensus_protocol = 0
        
        shard_congestion = self.shard_congestion[destination_shard]
        avg_tx_value = self._get_average_transaction_value(destination_shard)
        
        # Thưởng cho việc chọn đúng giao thức với trạng thái mạng
        if consensus_protocol == 0 and avg_tx_value < 30 and shard_congestion < 0.4:
            # Fast BFT phù hợp với giá trị thấp và tắc nghẽn thấp
            return 0.3
        elif consensus_protocol == 1 and 20 <= avg_tx_value <= 70:
            # PBFT phù hợp với giá trị trung bình
            return 0.2
        elif consensus_protocol == 2 and avg_tx_value > 60:
            # Robust BFT phù hợp với giá trị cao
            return 0.3
        
        return 0.0
    
    def _get_average_transaction_value(self, target_shard) -> float:
        """
        Tính giá trị trung bình của các giao dịch đang chờ xử lý trong một shard.
        
        Args:
            target_shard: ID của shard cần tính
            
        Returns:
            float: Giá trị trung bình của giao dịch
        """
        relevant_txs = [tx['value'] for tx in self.transaction_pool 
                      if tx.get('destination_shard') == target_shard and tx.get('status') == 'pending']
        
        if not relevant_txs:
            return 0.0
            
        return np.mean(relevant_txs)
    
    def _is_innovative_routing(self, action) -> bool:
        """
        Kiểm tra xem hành động routing có sáng tạo không.
        Sáng tạo được định nghĩa là chọn một shard ít được sử dụng gần đây
        hoặc sử dụng một giao thức đồng thuận phù hợp với tình huống hiện tại.
        
        Args:
            action: Hành động được thực hiện
            
        Returns:
            bool: True nếu hành động routing là sáng tạo
        """
        # Kiểm tra kiểu dữ liệu của action và xử lý phù hợp
        if isinstance(action, (int, float, np.integer, np.floating)):
            # Nếu action là một số vô hướng, giả định đó là một hành động rời rạc
            # Cần phân tách thành destination_shard và consensus_protocol
            action_id = int(action)
            # Số lượng giao thức đồng thuận
            num_consensus_protocols = len(self.action_space_dict['consensus_selection'])
            # Tính destination_shard và consensus_protocol từ action_id
            destination_shard = action_id // num_consensus_protocols
            consensus_protocol = action_id % num_consensus_protocols
        elif hasattr(action, "__len__") and len(action) >= 2:
            # Nếu action là mảng với ít nhất 2 phần tử
            destination_shard = action[0]
            consensus_protocol = action[1]
        else:
            # Trường hợp không xác định, sử dụng giá trị mặc định an toàn
            return False
        
        # Kiểm tra xem destination_shard có phải là shard ít được sử dụng
        shard_usage_history = getattr(self, 'shard_usage_history', None)
        if shard_usage_history is None:
            # Nếu chưa có lịch sử sử dụng, khởi tạo mảng với 0
            self.shard_usage_history = np.zeros(self.num_shards)
        
        # Kiểm tra xem shard đích có tắc nghẽn thấp hơn trung bình 
        # và ít được sử dụng trong lịch sử gần đây
        avg_congestion = np.mean(self.shard_congestion)
        is_less_congested = self.shard_congestion[destination_shard] < avg_congestion
        is_less_used = self.shard_usage_history[destination_shard] < np.mean(self.shard_usage_history)
        
        # Cập nhật lịch sử sử dụng
        self.shard_usage_history[destination_shard] += 1
        
        # Làm mờ lịch sử sử dụng để tránh tích lũy quá nhiều
        if self.current_step % 10 == 0:
            self.shard_usage_history *= 0.9
        
        # Kiểm tra xem giao thức đồng thuận có phù hợp với tình huống không
        # Lấy các giao dịch đang chờ xử lý trong shard
        pending_txs = [tx for tx in self.transaction_pool 
                     if tx.get('destination_shard') == destination_shard and tx.get('status') == 'pending']
        
        # Nếu không có giao dịch đang chờ, không thể đánh giá sự phù hợp
        if not pending_txs:
            return is_less_congested and is_less_used
        
        # Tính giá trị trung bình của giao dịch
        avg_value = np.mean([tx.get('value', 0.0) for tx in pending_txs])
        
        # Đánh giá sự phù hợp của giao thức đồng thuận với giá trị giao dịch
        consensus_appropriate = False
        if (consensus_protocol == 0 and avg_value < 30) or \
           (consensus_protocol == 1 and 20 <= avg_value <= 70) or \
           (consensus_protocol == 2 and avg_value > 60):
            consensus_appropriate = True
        
        return (is_less_congested and is_less_used) or consensus_appropriate
    
    def _is_high_performance(self) -> bool:
        """Kiểm tra xem hiệu suất hiện tại có cao không."""
        throughput = self.metrics['throughput'][-1] if self.metrics['throughput'] else 0
        avg_latency = self.metrics['latency'][-1] if self.metrics['latency'] else 0
        energy_usage = self.metrics['energy_consumption'][-1] if self.metrics['energy_consumption'] else 0
        
        # Điều chỉnh các ngưỡng để tăng cơ hội đạt được thưởng cho high performance
        # - Giảm ngưỡng throughput từ 20 xuống 18
        # - Tăng ngưỡng độ trễ từ 30 lên 35 ms (dễ đạt hơn)
        # - Tăng ngưỡng năng lượng từ 200 lên 220 (dễ đạt hơn)
        return throughput > 18 and avg_latency < 35 and energy_usage < 220
    
    def _process_transaction(self, transaction, action):
        """
        Xử lý một giao dịch với hành động được cung cấp.
        
        Args:
            transaction: Giao dịch cần xử lý
            action: Hành động được chọn [shard_index, consensus_protocol_index]
            
        Returns:
            Tuple gồm: (giao dịch đã xử lý, độ trễ)
        """
        # Lấy thông tin từ action
        destination_shard = min(action[0], self.num_shards - 1)  # Đảm bảo không vượt quá số shard hiện có
        consensus_protocol = action[1]  # Giao thức đồng thuận
        
        # Tính toán độ trễ dựa trên routing và consensus
        tx_latency = self._calculate_transaction_latency(transaction, destination_shard, consensus_protocol)
        
        # Cập nhật trạng thái giao dịch
        transaction['routed_path'].append(destination_shard)
        transaction['consensus_protocol'] = ['Fast_BFT', 'PBFT', 'Robust_BFT'][consensus_protocol]
        
        return transaction, tx_latency
    
    def _calculate_transaction_latency(self, transaction, destination_shard, consensus_protocol):
        """
        Tính toán độ trễ cho một giao dịch.
        
        Args:
            transaction: Giao dịch cần tính độ trễ
            destination_shard: Shard đích được chọn
            consensus_protocol: Giao thức đồng thuận được chọn
            
        Returns:
            float: Độ trễ của giao dịch (ms)
        """
        # Đảm bảo destination_shard hợp lệ
        destination_shard = min(destination_shard, self.num_shards - 1)
        
        # Kiểm tra nếu đây là giao dịch xuyên shard
        is_cross_shard = transaction['source_shard'] != destination_shard
        
        # Độ trễ cơ bản dựa trên loại giao dịch
        if is_cross_shard:
            base_latency = 15.0  # ms (cho giao dịch xuyên shard)
        else:
            base_latency = 5.0   # ms (cho giao dịch nội shard)
            
        # Hệ số độ trễ dựa trên giao thức đồng thuận
        consensus_factor = {
            0: 0.8,   # Fast BFT (nhanh nhất)
            1: 1.0,   # PBFT (trung bình)
            2: 1.3    # Robust BFT (chậm nhất, an toàn nhất)
        }.get(consensus_protocol, 1.0)
        
        # Hệ số độ trễ dựa trên mức độ tắc nghẽn của shard đích
        congestion_factor = 1.0 + self.shard_congestion[destination_shard]
        
        # Tính toán độ trễ cuối cùng với một ít ngẫu nhiên
        jitter = np.random.uniform(0.9, 1.1)  # +/- 10% biến động
        latency = base_latency * consensus_factor * congestion_factor * jitter
        
        return latency
    
    def _calculate_energy_consumption(self, transaction, destination_shard, consensus_protocol):
        """
        Tính năng lượng tiêu thụ khi xử lý giao dịch.
        
        Args:
            transaction: Giao dịch cần xử lý
            destination_shard: Shard đích được chọn
            consensus_protocol: Giao thức đồng thuận được chọn
            
        Returns:
            float: Năng lượng tiêu thụ
        """
        # Năng lượng cơ bản dựa trên loại giao dịch
        if transaction['type'] == 'cross_shard':
            base_energy = 200.0  # Tiêu thụ cao hơn cho giao dịch xuyên shard
        else:
            base_energy = 100.0  # Tiêu thụ thấp hơn cho giao dịch nội shard
            
        # Điều chỉnh năng lượng dựa trên độ phức tạp của giao dịch
        complexity_factor = {
            'low': 1.0,
            'medium': 1.5,
            'high': 2.5
        }.get(transaction['complexity'], 1.0)
        
        # Điều chỉnh năng lượng dựa trên giao thức đồng thuận
        consensus_factor = {
            0: 0.7,  # Fast BFT - tiêu thụ ít nhất
            1: 1.0,  # PBFT - tiêu thụ trung bình
            2: 1.8   # Robust BFT - tiêu thụ nhiều nhất
        }.get(consensus_protocol, 1.0)
        
        # Điều chỉnh năng lượng dựa trên loại giao dịch
        category_factor = {
            'simple_transfer': 1.0,
            'smart_contract': 1.5,
            'token_exchange': 1.3
        }.get(transaction['category'], 1.0)
        
        # Tính tổng năng lượng với yếu tố ngẫu nhiên
        energy = (base_energy * complexity_factor * consensus_factor * category_factor) * (0.9 + 0.2 * np.random.random())
        
        return energy
    
    def _determine_transaction_success(self, transaction, destination_shard, consensus_protocol, latency):
        """
        Xác định xem giao dịch có thành công hay không.
        
        Args:
            transaction: Giao dịch cần xử lý
            destination_shard: Shard đích được chọn
            consensus_protocol: Giao thức đồng thuận được chọn
            latency: Độ trễ của giao dịch
            
        Returns:
            bool: True nếu giao dịch thành công, False nếu thất bại
        """
        # Áp dụng xác thực chuyên biệt cho giao dịch xuyên shard
        if transaction['type'] == 'cross_shard':
            return self._verify_cross_shard_transaction(transaction, destination_shard, consensus_protocol, latency)
            
        # Tỷ lệ thành công cơ bản cho giao dịch nội shard
        base_success_rate = 0.95  # Tỷ lệ thành công cao hơn cho giao dịch nội shard
            
        # Điều chỉnh tỷ lệ thành công dựa trên độ phức tạp
        complexity_factor = {
            'low': 1.0,
            'medium': 0.95,
            'high': 0.9
        }.get(transaction['complexity'], 1.0)
        
        # Điều chỉnh tỷ lệ thành công dựa trên giao thức đồng thuận
        consensus_factor = {
            0: 0.92,   # Fast BFT - nâng cao từ 0.9
            1: 0.98,  # PBFT - giữ nguyên
            2: 0.995  # Robust BFT - giữ nguyên
        }.get(consensus_protocol, 0.95)
        
        # Đối với destination shard không phù hợp, giảm tỷ lệ thành công
        if transaction['destination_shard'] != destination_shard:
            routing_factor = 0.75  # Nâng cao từ 0.7, cải thiện khả năng rerouting
        else:
            routing_factor = 1.0  # Routing tối ưu
            
        # Nếu độ trễ quá cao, giảm tỷ lệ thành công
        if latency > 50.0:
            latency_factor = 0.8
        elif latency > 30.0:
            latency_factor = 0.9
        else:
            latency_factor = 1.0
            
        # Tính tỷ lệ thành công cuối cùng
        success_rate = base_success_rate * complexity_factor * consensus_factor * routing_factor * latency_factor
        
        # Thêm cache xác thực cho các giao dịch tương tự
        tx_key = f"{transaction['source_shard']}_{transaction['destination_shard']}_{transaction['category']}_{consensus_protocol}"
        
        # Cơ chế ổn định: nếu độ trễ thấp và routing tối ưu, cấp tăng tỷ lệ thành công
        if latency < 20.0 and transaction['destination_shard'] == destination_shard:
            success_rate = min(0.99, success_rate * 1.05)  # Tăng tối đa 5% nhưng không quá 0.99
            
        # Quyết định thành công dựa trên tỷ lệ
        return np.random.random() < success_rate
        
    def _verify_cross_shard_transaction(self, transaction, destination_shard, consensus_protocol, latency):
        """
        Xác thực chuyên biệt cho giao dịch xuyên shard với các tối ưu.
        
        Args:
            transaction: Giao dịch xuyên shard cần xác thực
            destination_shard: Shard đích được chọn
            consensus_protocol: Giao thức đồng thuận được chọn
            latency: Độ trễ của giao dịch
            
        Returns:
            bool: True nếu giao dịch thành công, False nếu thất bại
        """
        # Tỷ lệ thành công cơ bản cho giao dịch xuyên shard
        base_success_rate = 0.88  # Cải thiện từ 0.85 do tối ưu hóa
        
        # Kiểm tra nếu đường dẫn routing có tối ưu không
        routing_path = transaction.get('routed_path', [])
        path_length = len(routing_path)
        
        # Sử dụng witness mechanism để cải thiện xác thực
        if path_length > 2:  # Có ít nhất một shard trung gian
            # Kiểm tra nếu đường đi hợp lý (không quay lại shard đã đi qua)
            unique_shards = len(set(routing_path))
            if unique_shards < path_length:  # Đường đi có chứa loop
                base_success_rate *= 0.9  # Giảm tỷ lệ thành công
            elif path_length > 3:  # Đường đi dài
                base_success_rate *= 0.95  # Giảm nhẹ tỷ lệ thành công
        
        # Kiểm tra tính hợp lệ của shard nguồn và đích
        source_shard = transaction['source_shard']
        dest_shard = transaction['destination_shard']
        
        # Đánh giá quá trình routing
        if destination_shard == dest_shard:
            routing_factor = 1.0  # Routing chính xác
        elif destination_shard in routing_path:
            routing_factor = 0.9  # Đã đi qua shard này trong quá khứ
        else:
            routing_factor = 0.8  # Routing chưa tối ưu
            
        # Điều chỉnh dựa trên giao thức đồng thuận
        # Giao dịch xuyên shard yêu cầu giao thức mạnh hơn
        consensus_factor = {
            0: 0.85,   # Fast BFT - không đủ an toàn cho xuyên shard
            1: 0.95,   # PBFT - phù hợp cho xuyên shard
            2: 0.99    # Robust BFT - tối ưu cho xuyên shard
        }.get(consensus_protocol, 0.9)
        
        # Điều chỉnh dựa trên độ phức tạp
        complexity_factor = {
            'low': 1.0,
            'medium': 0.93,   # Cải thiện từ 0.9
            'high': 0.85      # Cải thiện từ 0.8
        }.get(transaction['complexity'], 1.0)
        
        # Điều chỉnh dựa trên độ trễ
        if latency > 70.0:  # Ngưỡng cao hơn cho xuyên shard
            latency_factor = 0.75
        elif latency > 40.0:
            latency_factor = 0.85
        else:
            latency_factor = 1.0
            
        # Witness mechanism: kiểm tra xem đã có giao dịch thành công tương tự gần đây không
        similar_tx_success = False
        for tx in self.transaction_pool[-100:]:  # Chỉ kiểm tra 100 giao dịch gần nhất
            if (tx.get('status') == 'completed' and 
                tx.get('type') == 'cross_shard' and
                tx.get('source_shard') == source_shard and 
                tx.get('destination_shard') == dest_shard):
                similar_tx_success = True
                break
                
        # Nếu có giao dịch tương tự thành công, tăng nhẹ tỷ lệ thành công
        witness_factor = 1.05 if similar_tx_success else 1.0
        
        # Tính toán tỷ lệ thành công cuối cùng
        success_rate = (base_success_rate * routing_factor * consensus_factor * 
                       complexity_factor * latency_factor * witness_factor)
                       
        # Cơ chế bảo vệ: bảo đảm tỷ lệ thành công nằm trong phạm vi hợp lý
        success_rate = min(0.98, max(0.5, success_rate))
        
        return np.random.random() < success_rate
    
    def _update_shard_congestion(self, transaction, destination_shard):
        """
        Cập nhật mức độ tắc nghẽn của shard sau khi xử lý giao dịch.
        
        Args:
            transaction: Giao dịch đã xử lý
            destination_shard: Shard đích được chọn
        """
        # Tăng mức độ tắc nghẽn dựa trên loại giao dịch
        if transaction['type'] == 'cross_shard':
            congestion_increase = 0.02  # Tắc nghẽn nhiều hơn cho giao dịch xuyên shard
        else:
            congestion_increase = 0.01  # Tắc nghẽn ít hơn cho giao dịch nội shard
            
        # Điều chỉnh dựa trên độ phức tạp
        complexity_factor = {
            'low': 1.0,
            'medium': 1.5,
            'high': 2.0
        }.get(transaction['complexity'], 1.0)
        
        # Tăng tắc nghẽn cho shard đích
        self.shard_congestion[destination_shard] += congestion_increase * complexity_factor
        
        # Giới hạn tắc nghẽn trong phạm vi [0.1, 1.0]
        self.shard_congestion[destination_shard] = min(1.0, max(0.1, self.shard_congestion[destination_shard]))
        
        # Giảm mức độ tắc nghẽn cho các shard khác (phục hồi tự nhiên)
        for shard_id in range(self.num_shards):
            if shard_id != destination_shard:
                self.shard_congestion[shard_id] = max(0.1, self.shard_congestion[shard_id] * 0.99)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Thực hiện một bước trong môi trường.
        
        Args:
            action: Hành động được chọn
                   [shard_index, consensus_protocol_index]
        
        Returns:
            Tuple gồm: (trạng thái mới, phần thưởng, tín hiệu kết thúc, thông tin bổ sung)
        """
        # Kiểm tra action có hợp lệ không
        assert self.action_space.contains(action), f"Action không hợp lệ: {action}"
        
        # Tăng step counter
        self.current_step += 1
        
        # Kiểm tra và thực hiện resharding nếu cần
        self._check_and_perform_resharding()
        
        # Tạo ngẫu nhiên các giao dịch mới cho bước hiện tại
        new_transactions = self._generate_transactions()
        self.transaction_pool.extend(new_transactions)
        
        # Lấy hành động (đã được rời rạc hóa) và đảm bảo shard index hợp lệ
        destination_shard = min(action[0], self.num_shards - 1)  # Đảm bảo không vượt quá số shard hiện có
        consensus_protocol = action[1]  # Giao thức đồng thuận
        
        # Tạo action đã được chuẩn hóa
        validated_action = np.array([destination_shard, consensus_protocol])
        
        # Xử lý các giao dịch và tính toán phần thưởng/phạt
        processed_transactions = 0
        successful_transactions = 0
        total_latency = 0
        total_energy = 0
        
        # Số lượng giao dịch để xử lý (tối đa 10)
        num_transactions_to_process = min(10, len(self.transaction_pool))
        
        if num_transactions_to_process > 0:
            # Chọn ngẫu nhiên các giao dịch để xử lý
            indices_to_process = np.random.choice(len(self.transaction_pool), 
                                                size=num_transactions_to_process, 
                                                replace=False)
            
            # Lấy danh sách giao dịch cần xử lý
            txs_to_process = []
            for idx in sorted(indices_to_process, reverse=True):
                txs_to_process.append(self.transaction_pool.pop(idx))
            
            # Quyết định xử lý hàng loạt hoặc riêng lẻ dựa trên số lượng giao dịch
            if len(txs_to_process) >= 3:  # Sử dụng xử lý hàng loạt nếu có từ 3 giao dịch trở lên
                # Xử lý hàng loạt các giao dịch
                processed_txs, batch_latency, batch_energy, batch_successful = self.batch_process_transactions(
                    txs_to_process, validated_action
                )
                
                # Cập nhật thống kê
                processed_transactions = len(processed_txs)
                successful_transactions = batch_successful
                total_latency = batch_latency
                total_energy = batch_energy
                
                # Thêm lại các giao dịch đã xử lý vào transaction pool
                self.transaction_pool.extend(processed_txs)
            else:
                # Xử lý từng giao dịch riêng lẻ
                for transaction in txs_to_process:
                    # Xử lý giao dịch với hành động được cung cấp
                    processed_tx, tx_latency = self._process_transaction(transaction, validated_action)
                    
                    # Tính toán năng lượng tiêu thụ
                    tx_energy = self._calculate_energy_consumption(
                        transaction, destination_shard, consensus_protocol
                    )
                    
                    # Xác định xem giao dịch có thành công không
                    success = self._determine_transaction_success(
                        transaction, destination_shard, consensus_protocol, tx_latency
                    )
                    
                    # Cập nhật các chỉ số
                    processed_transactions += 1
                    if success:
                        successful_transactions += 1
                        
                    total_latency += tx_latency
                    total_energy += tx_energy
                    
                    # Cập nhật transaction pool với giao dịch đã xử lý
                    transaction['status'] = 'completed' if success else 'failed'
                    transaction['completion_time'] = self.current_step
                    transaction['latency_ms'] = tx_latency
                    transaction['energy_consumed'] = tx_energy
                    self.transaction_pool.append(transaction)
                    
                    # Cập nhật tắc nghẽn mạng
                    self._update_shard_congestion(transaction, destination_shard)
        
        # Cập nhật thống kê hiệu suất
        # Cập nhật metrics cho test
        self.performance_metrics['transactions_processed'] += processed_transactions
        self.performance_metrics['successful_transactions'] += successful_transactions
        self.performance_metrics['total_latency'] += total_latency
        self.performance_metrics['total_energy'] += total_energy
        
        # Cập nhật metrics
        self._update_metrics()
        
        # Lấy trạng thái mới
        next_state = self.get_state()
        
        # Tính reward
        reward, reward_info = self._calculate_reward()
        
        # Kiểm tra xem episode đã kết thúc chưa
        done = self._is_done()
        
        # Tạo thông tin trả về
        info = {
            'successful_txs': successful_transactions,
            'total_txs': processed_transactions,
            'latency': total_latency,
            'energy_consumption': total_energy,
            'security_level': self._get_security_score(consensus_protocol),
            'cross_shard_txs': sum(1 for tx in new_transactions if tx['type'] == 'cross_shard'),
            'shard_congestion': self.shard_congestion.tolist(),
            # Thêm các trường cần thiết cho tests
            'transactions_processed': processed_transactions,
            'avg_latency': total_latency / max(1, processed_transactions),
            'avg_energy': total_energy / max(1, processed_transactions),
            'throughput': successful_transactions
        }
        
        return next_state, reward, done, info
    
    def _update_metrics(self):
        """Cập nhật các metrics hiệu suất từ transaction pool."""
        # Xác định hoàn thành trong step hiện tại (dựa trên completion_time)
        completed_txs = [tx for tx in self.transaction_pool if tx['status'] == 'completed']
        
        if completed_txs:
            # Tính throughput dựa trên số giao dịch hoàn thành trong bước hiện tại
            # và thời gian mô phỏng để quy đổi sang giao dịch/giây
            # 1 step = 100ms => 10 steps = 1 giây
            current_completed = [tx for tx in completed_txs 
                                if 'completion_time' in tx and int(tx['completion_time']) == self.current_step]
            tx_per_step = len(current_completed)
            tx_per_second = tx_per_step * 10  # Chuyển đổi sang tx/s
            self.metrics['throughput'].append(tx_per_second)
            logger.debug(f"Throughput: {tx_per_second} tx/s")
            
            # Độ trễ trung bình tính bằng ms thực tế
            latencies = [tx.get('latency_ms', 0) for tx in completed_txs if tx.get('latency_ms')]
            if latencies:
                avg_latency = np.mean(latencies)
                self.metrics['latency'].append(avg_latency)
                logger.debug(f"Độ trễ trung bình: {avg_latency:.2f} ms")
            
            # Tiêu thụ năng lượng trung bình tính bằng mJ/tx
            energies = [tx.get('energy_consumed', 0) for tx in completed_txs if tx.get('energy_consumed')]
            if energies:
                avg_energy = np.mean(energies)
                self.metrics['energy_consumption'].append(avg_energy)
                logger.debug(f"Tiêu thụ năng lượng trung bình: {avg_energy:.2f} mJ/tx")
            
            # Điểm bảo mật - tính dựa trên sự thành công của giao dịch và giao thức đồng thuận
            # Điểm cao hơn cho các giao thức mạnh hơn (Robust BFT > PBFT > Fast BFT)
            security_scores = []
            for tx in completed_txs:
                if 'consensus_protocol' in tx:
                    if tx['consensus_protocol'] == 'Fast_BFT':
                        security_scores.append(0.6)
                    elif tx['consensus_protocol'] == 'PBFT':
                        security_scores.append(0.8)
                    else:  # Robust_BFT
                        security_scores.append(1.0)
            
            if security_scores:
                avg_security = np.mean(security_scores)
                self.metrics['security_score'].append(avg_security)
                logger.debug(f"Điểm bảo mật trung bình: {avg_security:.2f}")
    
    def reset(self) -> np.ndarray:
        """Đặt lại môi trường về trạng thái ban đầu."""
        self.current_step = 0
        self.transaction_pool = []
        
        # Khởi tạo lại mạng blockchain
        self._init_blockchain_network()
        
        # Đặt lại metrics
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy_consumption': [],
            'security_score': []
        }
        
        return self.get_state()
    
    def render(self, mode='human'):
        """
        Hiển thị trạng thái hiện tại của môi trường.
        
        Args:
            mode: Chế độ render
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Number of transactions: {len(self.transaction_pool)}")
            print(f"Pending transactions: {len([tx for tx in self.transaction_pool if tx['status'] == 'pending'])}")
            print(f"Completed transactions: {len([tx for tx in self.transaction_pool if tx['status'] == 'completed'])}")
            print(f"Failed transactions: {len([tx for tx in self.transaction_pool if tx['status'] == 'failed'])}")
            
            if self.metrics['throughput']:
                print(f"Current throughput: {self.metrics['throughput'][-1]}")
            if self.metrics['latency']:
                print(f"Current average latency: {self.metrics['latency'][-1]:.2f}")
            if self.metrics['energy_consumption']:
                print(f"Current average energy consumption: {self.metrics['energy_consumption'][-1]:.2f}")
            if self.metrics['security_score']:
                print(f"Current security score: {self.metrics['security_score'][-1]:.2f}")
            
            print(f"Shard congestion: {', '.join([f'{c:.2f}' for c in self.shard_congestion])}")
            print("-" * 50)
    
    def close(self):
        """Đóng môi trường và giải phóng tài nguyên."""
        pass
    
    def get_congestion_level(self):
        """Lấy mức độ tắc nghẽn trung bình hiện tại của mạng."""
        return np.mean(self.shard_congestion)
    
    def get_congestion_data(self):
        """
        Lấy dữ liệu tắc nghẽn cho các shard.
        
        Returns:
            np.ndarray: Mảng chứa mức độ tắc nghẽn của mỗi shard
        """
        return self.shard_congestion.copy()

    def _get_state(self):
        """
        Lấy trạng thái hiện tại của môi trường.
        Đây là phiên bản đơn giản hơn của get_state() để sử dụng trong tests.
        
        Returns:
            np.ndarray: Vector trạng thái
        """
        return self.get_state()
    
    def _is_done(self):
        """
        Kiểm tra xem episode đã kết thúc chưa.
        
        Returns:
            bool: True nếu episode đã kết thúc, False nếu chưa
        """
        return self.current_step >= self.max_steps
    
    def _calculate_reward(self):
        """
        Tính toán phần thưởng dựa trên hiệu suất hiện tại.
        
        Returns:
            Tuple[float, Dict]: Phần thưởng và thông tin chi tiết
        """
        # Tính toán phần thưởng đơn giản dựa trên các metrics
        if self.performance_metrics['transactions_processed'] == 0:
            return 0.0, {'details': 'No transactions processed'}
        
        # Tính throughput reward với trọng số tăng 50%
        throughput_reward = self.throughput_reward * 1.5 * self.performance_metrics['successful_transactions']
        
        # Tính latency penalty với trọng số giảm 40%
        avg_latency = self.performance_metrics['total_latency'] / max(1, self.performance_metrics['transactions_processed'])
        latency_penalty = self.latency_penalty * 0.6 * min(1.0, avg_latency / 100.0)
        
        # Tính energy penalty với trọng số giảm 40%
        avg_energy = self.performance_metrics['total_energy'] / max(1, self.performance_metrics['transactions_processed'])
        energy_penalty = self.energy_penalty * 0.6 * min(1.0, avg_energy / 50.0)
        
        # Tổng hợp phần thưởng
        reward = throughput_reward - latency_penalty - energy_penalty
        
        # Thưởng thêm cho throughput cao
        if self.performance_metrics['successful_transactions'] > 15:
            bonus_factor = min(3.0, self.performance_metrics['successful_transactions'] / 15.0)
            throughput_bonus = bonus_factor * 0.5
            reward += throughput_bonus
        
        # Thông tin chi tiết
        info = {
            'throughput_reward': throughput_reward,
            'latency_penalty': latency_penalty,
            'energy_penalty': energy_penalty,
            'avg_latency': avg_latency,
            'avg_energy': avg_energy
        }
        
        return reward, info
    
    def _get_security_score(self, consensus_protocol):
        """
        Tính điểm bảo mật dựa trên giao thức đồng thuận.
        
        Args:
            consensus_protocol: Giao thức đồng thuận được sử dụng
            
        Returns:
            float: Điểm bảo mật (0.0-1.0)
        """
        # Điểm bảo mật dựa trên giao thức
        if consensus_protocol == 0:  # Fast BFT
            return 0.6  # Bảo mật thấp nhất
        elif consensus_protocol == 1:  # PBFT
            return 0.8  # Bảo mật trung bình
        else:  # Robust BFT
            return 0.95  # Bảo mật cao nhất

    def get_shard_congestion(self):
        """
        Trả về mức độ tắc nghẽn của các shard.
        
        Returns:
            np.ndarray: Mảng chứa mức độ tắc nghẽn của mỗi shard
        """
        return self.shard_congestion.copy()

    def _check_and_perform_resharding(self):
        """
        Kiểm tra và thực hiện resharding nếu cần thiết dựa trên tình trạng tắc nghẽn.
        """
        if not self.enable_dynamic_resharding:
            return
        
        # Chỉ resharding sau một khoảng thời gian nhất định
        if self.current_step - self.last_resharding_step < self.resharding_interval:
            return
        
        # Lưu lại mức độ tắc nghẽn hiện tại để phân tích
        self.congestion_history.append(self.shard_congestion.copy())
        
        # Tính toán mức độ tắc nghẽn trung bình và tối đa
        avg_congestion = np.mean(self.shard_congestion)
        max_congestion = np.max(self.shard_congestion)
        min_congestion = np.min(self.shard_congestion)
        
        # Các chỉ số ổn định của mạng
        need_split = False
        need_merge = False
        
        # Kiểm tra điều kiện để tăng số lượng shard (split)
        if max_congestion > self.congestion_threshold_high and self.num_shards < self.max_num_shards:
            need_split = True
            most_congested_shard = np.argmax(self.shard_congestion)
            logger.info(f"Phát hiện tắc nghẽn cao ở shard {most_congested_shard} ({max_congestion:.2f}), chuẩn bị tách shard")
        
        # Kiểm tra điều kiện để giảm số lượng shard (merge)
        elif avg_congestion < self.congestion_threshold_low and self.num_shards > self.min_num_shards:
            need_merge = True
            least_congested_shards = np.argsort(self.shard_congestion)[:2]
            logger.info(f"Phát hiện tắc nghẽn thấp ({avg_congestion:.2f}), chuẩn bị gộp các shard {least_congested_shards}")
        
        # Thực hiện resharding
        if need_split:
            self._split_shard(np.argmax(self.shard_congestion))
            self.last_resharding_step = self.current_step
            self.resharding_history.append({
                'type': 'split',
                'step': self.current_step,
                'shard_id': np.argmax(self.shard_congestion),
                'congestion_before': max_congestion,
                'num_shards_after': self.num_shards
            })
        elif need_merge:
            least_congested = np.argsort(self.shard_congestion)[:2]
            self._merge_shards(least_congested[0], least_congested[1])
            self.last_resharding_step = self.current_step
            self.resharding_history.append({
                'type': 'merge',
                'step': self.current_step,
                'shards_merged': [least_congested[0], least_congested[1]],
                'congestion_before': [self.shard_congestion[least_congested[0]], self.shard_congestion[least_congested[1]]],
                'num_shards_after': self.num_shards
            })
    
    def _split_shard(self, shard_id):
        """
        Tách một shard thành hai shard để giảm tắc nghẽn.
        
        Args:
            shard_id: ID của shard cần tách
        """
        if self.num_shards >= self.max_num_shards:
            logger.warning(f"Không thể tách shard {shard_id}: đã đạt giới hạn số lượng shard ({self.max_num_shards})")
            return
        
        logger.info(f"Tách shard {shard_id} thành hai shard")
        
        # Lấy danh sách các node trong shard cần tách
        nodes_to_split = self.shards[shard_id].copy()
        
        # Chia ngẫu nhiên các node thành hai nhóm
        np.random.shuffle(nodes_to_split)
        split_idx = len(nodes_to_split) // 2
        nodes_group1 = nodes_to_split[:split_idx]
        nodes_group2 = nodes_to_split[split_idx:]
        
        # Cập nhật shard hiện tại với nhóm node thứ nhất
        self.shards[shard_id] = nodes_group1
        
        # Tạo shard mới với nhóm node thứ hai
        new_shard_id = self.num_shards
        self.shards.append(nodes_group2)
        
        # Cập nhật thông tin shard_id cho các node trong mạng
        for node_id in nodes_group2:
            self.network.nodes[node_id]['shard_id'] = new_shard_id
        
        # Thêm kết nối giữa các node trong shard mới
        for i in range(len(nodes_group2)):
            for j in range(i + 1, len(nodes_group2)):
                # Nếu chưa có kết nối, thêm mới
                if not self.network.has_edge(nodes_group2[i], nodes_group2[j]):
                    self.network.add_edge(
                        nodes_group2[i], 
                        nodes_group2[j], 
                        latency=np.random.uniform(1, 5),
                        bandwidth=np.random.uniform(80, 150)
                    )
        
        # Tạo kết nối giữa shard mới và các shard khác
        for other_shard_id in range(self.num_shards):
            if other_shard_id != new_shard_id:
                # Chọn ngẫu nhiên 3 node từ mỗi shard để kết nối
                nodes_from_new_shard = np.random.choice(nodes_group2, min(3, len(nodes_group2)), replace=False)
                nodes_from_other_shard = np.random.choice(self.shards[other_shard_id], min(3, len(self.shards[other_shard_id])), replace=False)
                
                for node_i in nodes_from_new_shard:
                    for node_j in nodes_from_other_shard:
                        if not self.network.has_edge(node_i, node_j):
                            self.network.add_edge(
                                node_i, 
                                node_j, 
                                latency=np.random.uniform(5, 30),
                                bandwidth=np.random.uniform(20, 70)
                            )
        
        # Cập nhật số lượng shard và thông tin tắc nghẽn
        self.num_shards += 1
        
        # Cập nhật mảng congestion và consensus
        # Shard hiện tại giữ nguyên mức độ tắc nghẽn giảm 30%
        current_congestion = self.shard_congestion[shard_id] * 0.7
        
        # Mở rộng mảng congestion và consensus
        new_congestion = np.append(self.shard_congestion, current_congestion)
        new_congestion[shard_id] = current_congestion
        self.shard_congestion = new_congestion
        
        # Mở rộng mảng consensus protocol
        new_consensus = np.append(self.shard_consensus, self.shard_consensus[shard_id])
        self.shard_consensus = new_consensus
        
        logger.info(f"Tách shard thành công: {self.num_shards-1} -> {self.num_shards} shards")

    def _merge_shards(self, shard_id1, shard_id2):
        """
        Gộp hai shard thành một shard để tăng hiệu suất khi tắc nghẽn thấp.
        
        Args:
            shard_id1: ID của shard thứ nhất
            shard_id2: ID của shard thứ hai
        """
        if self.num_shards <= self.min_num_shards:
            logger.warning(f"Không thể gộp shard: số lượng shard đã ở mức tối thiểu ({self.min_num_shards})")
            return
        
        # Đảm bảo shard_id1 < shard_id2
        if shard_id1 > shard_id2:
            shard_id1, shard_id2 = shard_id2, shard_id1
        
        logger.info(f"Gộp shard {shard_id1} và {shard_id2}")
        
        # Lấy danh sách các node từ cả hai shard
        nodes_shard1 = self.shards[shard_id1].copy()
        nodes_shard2 = self.shards[shard_id2].copy()
        merged_nodes = nodes_shard1 + nodes_shard2
        
        # Cập nhật shard_id cho các node của shard2
        for node_id in nodes_shard2:
            self.network.nodes[node_id]['shard_id'] = shard_id1
        
        # Gộp hai shard: giữ shard1, xóa shard2
        self.shards[shard_id1] = merged_nodes
        
        # Thêm kết nối giữa các node của hai shard
        for node_i in nodes_shard1:
            for node_j in nodes_shard2:
                if not self.network.has_edge(node_i, node_j):
                    self.network.add_edge(
                        node_i, 
                        node_j, 
                        latency=np.random.uniform(1, 5),
                        bandwidth=np.random.uniform(80, 150)
                    )
        
        # Xóa shard thứ hai và cập nhật các shard còn lại
        self.shards.pop(shard_id2)
        
        # Cập nhật shard_id cho các node trong các shard có ID > shard_id2
        for shard_id in range(shard_id2 + 1, self.num_shards):
            for node_id in self.shards[shard_id - 1]:
                self.network.nodes[node_id]['shard_id'] = shard_id - 1
        
        # Cập nhật số lượng shard và thông tin tắc nghẽn
        self.num_shards -= 1
        
        # Tính toán mức độ tắc nghẽn mới cho shard sau khi gộp
        new_congestion_merged = (self.shard_congestion[shard_id1] * len(nodes_shard1) + 
                                 self.shard_congestion[shard_id2] * len(nodes_shard2)) / len(merged_nodes)
        
        # Cập nhật mảng congestion và consensus
        new_congestion = np.delete(self.shard_congestion, shard_id2)
        new_congestion[shard_id1] = new_congestion_merged
        self.shard_congestion = new_congestion
        
        # Cập nhật mảng consensus protocol
        new_consensus = np.delete(self.shard_consensus, shard_id2)
        self.shard_consensus = new_consensus
        
        logger.info(f"Gộp shard thành công: {self.num_shards+1} -> {self.num_shards} shards")

    def batch_process_transactions(self, transactions: List[Dict[str, Any]], action_array: np.ndarray) -> Tuple[List[Dict[str, Any]], float, float, int]:
        """
        Xử lý một loạt các giao dịch cùng một lúc để tăng hiệu suất.
        
        Args:
            transactions: Danh sách các giao dịch cần xử lý
            action_array: Mảng hành động [shard_index, consensus_protocol_index]
            
        Returns:
            Tuple chứa:
            - Danh sách các giao dịch đã xử lý
            - Tổng độ trễ
            - Tổng năng lượng tiêu thụ
            - Số giao dịch thành công
        """
        if not transactions:
            return [], 0.0, 0.0, 0
        
        destination_shard = action_array[0]
        consensus_protocol = action_array[1]
        
        # Phân loại giao dịch thành hai nhóm chính: nội shard và xuyên shard
        intra_shard_txs = []
        cross_shard_txs = []
        
        for tx in transactions:
            if tx['type'] == 'cross_shard':
                cross_shard_txs.append(tx)
            else:
                intra_shard_txs.append(tx)
        
        # Nhóm giao dịch theo shard đích
        tx_groups = defaultdict(list)
        for tx in intra_shard_txs:
            tx_groups[tx['destination_shard']].append(tx)
        
        # Nhóm giao dịch xuyên shard theo các tuyến đường
        cross_shard_groups = defaultdict(list)
        for tx in cross_shard_txs:
            route_key = f"{tx['source_shard']}_{tx['destination_shard']}"
            cross_shard_groups[route_key].append(tx)
        
        total_latency = 0.0
        total_energy = 0.0
        successful_txs = 0
        processed_txs = []
        
        # Xử lý từng nhóm giao dịch nội shard
        for dest_shard, group_txs in tx_groups.items():
            # Quyết định chi phí cơ bản cho mỗi loại giao dịch
            if dest_shard == destination_shard:
                # Đúng routing - chi phí thấp
                base_latency = 5.0  # ms
                base_energy = 2.0   # mJ
                success_prob = 0.95
            else:
                # Sai routing - chi phí cao hơn
                base_latency = 15.0  # ms
                base_energy = 5.0    # mJ
                success_prob = 0.75
            
            # Điều chỉnh chi phí và tỷ lệ thành công dựa trên giao thức đồng thuận
            # 0: Fast BFT, 1: PBFT, 2: Robust BFT
            if consensus_protocol == 0:  # Fast BFT
                latency_factor = 0.8    # Nhanh nhất
                energy_factor = 0.7     # Ít năng lượng nhất
                consensus_success = 0.92  # Nâng cao từ 0.9
            elif consensus_protocol == 1:  # PBFT
                latency_factor = 1.0    # Trung bình
                energy_factor = 1.0     # Trung bình
                consensus_success = 0.95  # Giữ nguyên
            else:  # Robust BFT
                latency_factor = 1.3    # Chậm nhất
                energy_factor = 1.5     # Nhiều năng lượng nhất
                consensus_success = 0.99  # Giữ nguyên
            
            # Số lượng giao dịch trong nhóm ảnh hưởng đến hiệu quả xử lý hàng loạt
            batch_size = len(group_txs)
            if batch_size > 1:
                # Giảm chi phí trung bình khi xử lý hàng loạt
                batch_efficiency = 0.8 + 0.2 / batch_size  # Giảm từ 20% đến gần 0% khi batch_size lớn
            else:
                batch_efficiency = 1.0  # Không có lợi ích từ xử lý hàng loạt
            
            # Tính tổng chi phí cho nhóm
            batch_latency = base_latency * latency_factor * batch_efficiency * batch_size
            batch_energy = base_energy * energy_factor * batch_efficiency * batch_size
            
            # Tính số lượng giao dịch thành công dựa trên xác suất
            batch_success_prob = success_prob * consensus_success
            successful_in_batch = 0
            
            # Cập nhật trạng thái của từng giao dịch trong nhóm
            for tx in group_txs:
                # Xác định thành công hay thất bại
                is_success = np.random.random() < batch_success_prob
                
                # Cập nhật trạng thái giao dịch
                tx['status'] = 'completed' if is_success else 'failed'
                tx['completion_time'] = self.current_step
                tx['consensus_protocol'] = ['Fast_BFT', 'PBFT', 'Robust_BFT'][consensus_protocol]
                
                # Tính toán độ trễ cho giao dịch cụ thể với một chút biến động
                individual_latency = batch_latency / batch_size * (0.9 + 0.2 * np.random.random())
                tx['latency_ms'] = individual_latency
                
                # Tính toán năng lượng tiêu thụ cho giao dịch cụ thể với một chút biến động
                individual_energy = batch_energy / batch_size * (0.9 + 0.2 * np.random.random())
                tx['energy_consumed'] = individual_energy
                
                # Thêm shard đích vào đường dẫn routing
                if destination_shard not in tx['routed_path']:
                    tx['routed_path'].append(destination_shard)
                
                # Cập nhật thống kê
                if is_success:
                    successful_in_batch += 1
                
                processed_txs.append(tx)
            
            # Cập nhật tổng thống kê
            total_latency += batch_latency
            total_energy += batch_energy
            successful_txs += successful_in_batch
            
            # Cập nhật mức độ tắc nghẽn cho shard đích
            congestion_increase = 0.01 * batch_size  # Tăng tắc nghẽn theo số lượng giao dịch
            self.shard_congestion[dest_shard] = min(1.0, self.shard_congestion[dest_shard] + congestion_increase)
        
        # Xử lý từng nhóm giao dịch xuyên shard
        for route_key, group_txs in cross_shard_groups.items():
            source_shard, dest_shard = map(int, route_key.split('_'))
            
            # Xác thực giao dịch xuyên shard theo lô
            batch_size = len(group_txs)
            
            # Tính chi phí xác thực xuyên shard
            # Xuyên shard có chi phí cao hơn
            base_latency = 12.0  # ms (giảm từ 15.0 nhờ tối ưu hóa)
            base_energy = 4.5    # mJ (giảm từ 5.0 nhờ tối ưu hóa)
            
            # Điều chỉnh chi phí và tỷ lệ thành công dựa trên giao thức đồng thuận
            if consensus_protocol == 0:  # Fast BFT - không tối ưu cho xuyên shard
                latency_factor = 0.9     # Nhanh nhưng không hiệu quả cho xuyên shard
                energy_factor = 0.8      # Ít năng lượng
                base_success_prob = 0.80  # Tỷ lệ thành công thấp cho xuyên shard
            elif consensus_protocol == 1:  # PBFT - tốt cho xuyên shard
                latency_factor = 1.0     # Trung bình
                energy_factor = 1.0      # Trung bình
                base_success_prob = 0.90  # Tỷ lệ thành công khá tốt
            else:  # Robust BFT - tối ưu cho xuyên shard
                latency_factor = 1.2     # Chậm hơn nhưng vẫn được tối ưu (giảm từ 1.3)
                energy_factor = 1.3      # Tiêu thụ nhiều (giảm từ 1.5)
                base_success_prob = 0.95  # Tỷ lệ thành công cao
            
            # Tối ưu hiệu quả xử lý theo lô cho giao dịch xuyên shard
            if batch_size > 1:
                # Hiệu quả cao hơn cho xử lý hàng loạt xuyên shard
                batch_efficiency = 0.75 + 0.2 / batch_size  # Cải thiện từ 0.8
            else:
                batch_efficiency = 1.0
            
            # Tính toán chi phí cho cả lô
            batch_latency = base_latency * latency_factor * batch_efficiency * batch_size
            batch_energy = base_energy * energy_factor * batch_efficiency * batch_size
            
            # Điều chỉnh tỷ lệ thành công dựa trên đường đi
            # Nếu đi qua đúng shard đích, tăng tỷ lệ thành công
            if destination_shard == dest_shard:
                path_factor = 1.05  # Thưởng cho routing đúng
            elif destination_shard in [source_shard, dest_shard]:
                path_factor = 1.0   # Chấp nhận được
            else:
                path_factor = 0.85  # Phạt cho routing không tối ưu
                
            # Tính tỷ lệ thành công cuối cùng cho lô
            batch_success_prob = min(0.98, base_success_prob * path_factor)
            
            # Witness mechanism: nếu gần đây có giao dịch tương tự thành công
            has_successful_witness = False
            for tx in self.transaction_pool[-50:]:  # Chỉ kiểm tra 50 giao dịch gần nhất
                if (tx.get('status') == 'completed' and 
                    tx.get('source_shard') == source_shard and 
                    tx.get('destination_shard') == dest_shard):
                    has_successful_witness = True
                    break
                    
            # Tăng tỷ lệ thành công nếu có witness
            if has_successful_witness:
                batch_success_prob = min(0.98, batch_success_prob * 1.05)
                
            # Áp dụng khả năng thành công cho mỗi giao dịch, với cơ chế phụ thuộc
            # Trong một nhóm, nếu giao dịch trước thành công, giao dịch sau có xác suất cao hơn
            prev_success = False
            successful_in_batch = 0
            
            # Sắp xếp giao dịch để các giao dịch giá trị thấp hơn xác thực trước
            sorted_txs = sorted(group_txs, key=lambda tx: tx.get('value', 0))
            
            for tx in sorted_txs:
                # Điều chỉnh xác suất thành công dựa trên kết quả giao dịch trước
                adj_success_prob = batch_success_prob
                if prev_success:
                    adj_success_prob = min(0.98, adj_success_prob * 1.03)  # Tăng 3% nếu giao dịch trước thành công
                
                # Điều chỉnh theo độ phức tạp của giao dịch
                complexity_factor = {
                    'low': 1.0,
                    'medium': 0.95,
                    'high': 0.9
                }.get(tx.get('complexity', 'medium'), 1.0)
                
                # Đặc thù cho từng giao dịch dựa trên giá trị
                value = tx.get('value', 0)
                if value > 40.0:  # Giá trị cao yêu cầu bảo mật cao hơn
                    value_factor = 0.98 if consensus_protocol == 2 else 0.95  # Chỉ Robust BFT tối ưu cho giá trị cao
                else:
                    value_factor = 1.0
                
                # Tính xác suất thành công cuối cùng cho giao dịch
                tx_success_prob = adj_success_prob * complexity_factor * value_factor
                
                # Xác định thành công hay thất bại
                is_success = np.random.random() < tx_success_prob
                prev_success = is_success  # Cập nhật kết quả cho giao dịch tiếp theo
                
                # Cập nhật trạng thái giao dịch
                tx['status'] = 'completed' if is_success else 'failed'
                tx['completion_time'] = self.current_step
                tx['consensus_protocol'] = ['Fast_BFT', 'PBFT', 'Robust_BFT'][consensus_protocol]
                
                # Tính toán chi phí cá nhân với một chút biến động
                tx_complexity_weight = {'low': 0.9, 'medium': 1.0, 'high': 1.1}.get(tx.get('complexity', 'medium'), 1.0)
                individual_latency = (batch_latency / batch_size) * tx_complexity_weight * (0.9 + 0.2 * np.random.random())
                individual_energy = (batch_energy / batch_size) * tx_complexity_weight * (0.9 + 0.2 * np.random.random())
                
                tx['latency_ms'] = individual_latency
                tx['energy_consumed'] = individual_energy
                
                # Thêm shard đích vào đường dẫn routing
                if destination_shard not in tx['routed_path']:
                    tx['routed_path'].append(destination_shard)
                
                # Cập nhật thống kê
                if is_success:
                    successful_in_batch += 1
                
                processed_txs.append(tx)
            
            # Cập nhật tổng thống kê
            total_latency += batch_latency
            total_energy += batch_energy
            successful_txs += successful_in_batch
            
            # Cập nhật mức độ tắc nghẽn cho các shard liên quan
            # Xuyên shard tạo tắc nghẽn ở cả nguồn và đích
            source_congestion = 0.005 * batch_size  # Tắc nghẽn ít hơn ở nguồn
            dest_congestion = 0.015 * batch_size    # Tắc nghẽn nhiều hơn ở đích
            
            # Cập nhật mức độ tắc nghẽn
            if source_shard < len(self.shard_congestion):
                self.shard_congestion[source_shard] = min(1.0, self.shard_congestion[source_shard] + source_congestion)
            if dest_shard < len(self.shard_congestion):
                self.shard_congestion[dest_shard] = min(1.0, self.shard_congestion[dest_shard] + dest_congestion)
        
        return processed_txs, total_latency, total_energy, successful_txs