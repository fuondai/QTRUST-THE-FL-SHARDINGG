import gym
import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Any
from gym import spaces

from qtrust.utils.logging import simulation_logger as logger

class BlockchainEnvironment(gym.Env):
    """
    Môi trường mô phỏng blockchain với sharding cho Deep Reinforcement Learning.
    Môi trường này mô phỏng một mạng blockchain với nhiều shard và giao dịch xuyên shard.
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
                 security_reward: float = 0.8):
        """
        Khởi tạo môi trường blockchain với sharding.
        
        Args:
            num_shards: Số lượng shard trong mạng
            num_nodes_per_shard: Số lượng node trong mỗi shard
            max_transactions_per_step: Số lượng giao dịch tối đa mỗi bước
            transaction_value_range: Phạm vi giá trị giao dịch (min, max)
            max_steps: Số bước tối đa cho mỗi episode
            latency_penalty: Hệ số phạt cho độ trễ
            energy_penalty: Hệ số phạt cho tiêu thụ năng lượng
            throughput_reward: Hệ số thưởng cho throughput
            security_reward: Hệ số thưởng cho bảo mật
        """
        super(BlockchainEnvironment, self).__init__()
        
        self.num_shards = num_shards
        self.num_nodes_per_shard = num_nodes_per_shard
        self.max_transactions_per_step = max_transactions_per_step
        self.transaction_value_range = transaction_value_range
        self.max_steps = max_steps
        
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
        
        logger.info(f"Khởi tạo môi trường blockchain với {num_shards} shard, mỗi shard có {num_nodes_per_shard} nodes")
    
    def _init_state_action_space(self):
        """Khởi tạo không gian trạng thái và hành động."""
        # Không gian trạng thái:
        # - Mức độ tắc nghẽn mạng cho mỗi shard (0.0-1.0)
        # - Giá trị giao dịch trung bình trong mỗi shard
        # - Điểm tin cậy trung bình của các node trong mỗi shard (0.0-1.0)
        # - Tỷ lệ giao dịch thành công gần đây
        
        # Mỗi shard có 4 đặc trưng, cộng với 4 đặc trưng toàn cục
        num_features = self.num_shards * 4 + 4
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=float('inf'), 
            shape=(num_features,), 
            dtype=np.float32
        )
        
        # Không gian hành động:
        # - Lựa chọn shard đích cho một giao dịch (0 to num_shards-1)
        # - Lựa chọn giao thức đồng thuận (0: Fast BFT, 1: PBFT, 2: Robust BFT)
        self.action_space = spaces.MultiDiscrete([self.num_shards, 3])
        
        # Định nghĩa không gian trạng thái và hành động cho một cách nhìn dễ hiểu hơn
        self.state_space = {
            'network_congestion': [0.0, 1.0],  # Mức độ tắc nghẽn
            'transaction_value': [self.transaction_value_range[0], self.transaction_value_range[1]],
            'trust_scores': [0.0, 1.0],  # Điểm tin cậy
            'success_rate': [0.0, 1.0]   # Tỷ lệ thành công
        }
        
        self.action_space_dict = {
            'routing_decision': list(range(self.num_shards)),
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
        # Tính toán các đặc trưng trạng thái
        features = []
        
        # 1. Mức độ tắc nghẽn mạng cho mỗi shard (0.0-1.0)
        features.extend(self.shard_congestion)
        
        # 2. Mức độ tín nhiệm trung bình của các node trong mỗi shard
        shard_trust_scores = []
        for shard_nodes in self.shards:
            avg_trust = np.mean([self.network.nodes[node]['trust_score'] for node in shard_nodes])
            shard_trust_scores.append(avg_trust)
        features.extend(shard_trust_scores)
        
        # 3. Tỷ lệ giao dịch xuyên shard
        cross_shard_txs = [tx for tx in self.transaction_pool if tx['source_shard'] != tx['destination_shard']]
        cross_shard_ratio = len(cross_shard_txs) / max(1, len(self.transaction_pool))
        features.append(cross_shard_ratio)
        
        # 4. Giá trị giao dịch trung bình trong mỗi shard
        shard_avg_values = []
        for s in range(self.num_shards):
            shard_txs = [tx['value'] for tx in self.transaction_pool if tx['destination_shard'] == s]
            avg_value = np.mean(shard_txs) if shard_txs else 0.0
            # Chuẩn hóa giá trị để fit vào observation space
            normalized_value = avg_value / self.transaction_value_range[1]
            shard_avg_values.append(normalized_value)
        features.extend(shard_avg_values)
        
        # 5. Tỷ lệ giao dịch thành công gần đây
        completed_txs = [tx for tx in self.transaction_pool if tx['status'] == 'completed']
        success_rate = len(completed_txs) / max(1, len(self.transaction_pool))
        features.append(success_rate)
        
        # 6. Đặc trưng hiệu suất (throughput, latency, energy)
        avg_throughput = np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0.0
        # Chuẩn hóa
        normalized_throughput = min(1.0, avg_throughput / 100.0)  # Giả sử throughput tối đa là 100
        features.append(normalized_throughput)
        
        avg_latency = np.mean(self.metrics['latency']) if self.metrics['latency'] else 0.0
        # Chuẩn hóa
        normalized_latency = min(1.0, avg_latency / 100.0)  # Giả sử latency tối đa là 100ms
        features.append(normalized_latency)
        
        avg_energy = np.mean(self.metrics['energy_consumption']) if self.metrics['energy_consumption'] else 0.0
        # Chuẩn hóa
        normalized_energy = min(1.0, avg_energy / 500.0)  # Giả sử energy tối đa là 500 đơn vị
        features.append(normalized_energy)
        
        # Đảm bảo state có kích thước đúng với observation_space
        while len(features) < self.observation_space.shape[0]:
            features.append(0.0)  # Thêm padding nếu cần
        
        # Chỉ lấy số lượng feature cần thiết
        features = features[:self.observation_space.shape[0]]
        
        # Convert to numpy array với dtype float32
        state = np.array(features, dtype=np.float32)
        
        return state
    
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
        
        # Tổng hợp phần thưởng
        reward = (throughput_reward - latency_penalty - energy_penalty + 
                security_reward - cross_shard_penalty + consensus_reward)
        
        # Thưởng thêm cho việc đạt hiệu suất cao
        if self._is_high_performance():
            reward += 2.0  # Thưởng bổ sung cho hiệu suất xuất sắc
        
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
    
    def _is_high_performance(self) -> bool:
        """Kiểm tra xem hiệu suất hiện tại có cao không."""
        throughput = self.metrics['throughput'][-1] if self.metrics['throughput'] else 0
        avg_latency = self.metrics['latency'][-1] if self.metrics['latency'] else 0
        energy_usage = self.metrics['energy_consumption'][-1] if self.metrics['energy_consumption'] else 0
        
        return throughput > 20 and avg_latency < 30 and energy_usage < 200
    
    def _process_transaction(self, transaction, action):
        """
        Xử lý một giao dịch cụ thể với hành động đã chọn.
        
        Args:
            transaction: Giao dịch cần xử lý
            action: Hành động được chọn (shard đích, giao thức đồng thuận)
        
        Returns:
            Tuple[Dict, float]: Giao dịch đã xử lý và độ trễ
        """
        destination_shard = action[0]
        consensus_protocol = action[1]
        
        # Tính độ trễ dựa trên loại giao dịch và giao thức đồng thuận
        tx_latency = self._calculate_transaction_latency(transaction, destination_shard, consensus_protocol)
        
        # Cập nhật trạng thái giao dịch
        transaction['status'] = 'completed'
        transaction['consensus_protocol'] = consensus_protocol
        
        # Thêm shard đích vào đường dẫn
        if destination_shard not in transaction['routed_path']:
            transaction['routed_path'].append(destination_shard)
            
        return transaction, tx_latency
        
    def _calculate_transaction_latency(self, transaction, destination_shard, consensus_protocol):
        """
        Tính độ trễ khi xử lý giao dịch.
        
        Args:
            transaction: Giao dịch cần xử lý
            destination_shard: Shard đích được chọn
            consensus_protocol: Giao thức đồng thuận được chọn
            
        Returns:
            float: Độ trễ xử lý giao dịch
        """
        # Độ trễ cơ bản dựa trên loại giao dịch
        if transaction['type'] == 'cross_shard':
            base_latency = 20.0  # Độ trễ cao hơn cho giao dịch xuyên shard
        else:
            base_latency = 10.0  # Độ trễ thấp hơn cho giao dịch nội shard
            
        # Điều chỉnh độ trễ dựa trên độ phức tạp của giao dịch
        complexity_factor = {
            'low': 1.0,
            'medium': 1.5,
            'high': 2.0
        }.get(transaction['complexity'], 1.0)
        
        # Điều chỉnh độ trễ dựa trên giao thức đồng thuận
        consensus_factor = {
            0: 0.8,  # Fast BFT - nhanh nhất
            1: 1.0,  # PBFT - trung bình
            2: 1.5   # Robust BFT - chậm nhất
        }.get(consensus_protocol, 1.0)
        
        # Điều chỉnh độ trễ dựa trên tắc nghẽn shard
        congestion_factor = 1.0 + self.shard_congestion[destination_shard]
        
        # Tính tổng độ trễ với yếu tố ngẫu nhiên
        latency = (base_latency * complexity_factor * consensus_factor * congestion_factor) * (0.9 + 0.2 * np.random.random())
        
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
        # Tỷ lệ thành công cơ bản dựa trên loại giao dịch
        if transaction['type'] == 'cross_shard':
            base_success_rate = 0.85  # Tỷ lệ thành công thấp hơn cho giao dịch xuyên shard
        else:
            base_success_rate = 0.95  # Tỷ lệ thành công cao hơn cho giao dịch nội shard
            
        # Điều chỉnh tỷ lệ thành công dựa trên độ phức tạp
        complexity_factor = {
            'low': 1.0,
            'medium': 0.95,
            'high': 0.9
        }.get(transaction['complexity'], 1.0)
        
        # Điều chỉnh tỷ lệ thành công dựa trên giao thức đồng thuận
        consensus_factor = {
            0: 0.9,   # Fast BFT - ít an toàn nhất
            1: 0.98,  # PBFT - an toàn trung bình
            2: 0.995  # Robust BFT - an toàn nhất
        }.get(consensus_protocol, 0.95)
        
        # Đối với destination shard không phù hợp, giảm tỷ lệ thành công
        if transaction['destination_shard'] != destination_shard:
            routing_factor = 0.7  # Routing không tối ưu
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
        
        # Quyết định thành công dựa trên tỷ lệ
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
            action: Hành động của agent (shard_id, consensus_protocol)
        
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: (state, reward, done, info)
        """
        # Kiểm tra action có hợp lệ không
        assert self.action_space.contains(action), f"Action không hợp lệ: {action}"
        
        # Tăng step counter
        self.current_step += 1
        
        # Tạo ngẫu nhiên các giao dịch mới cho bước hiện tại
        new_transactions = self._generate_transactions()
        self.transaction_pool.extend(new_transactions)
        
        # Lấy hành động (đã được rời rạc hóa)
        destination_shard = action[0]  # Shard đích
        consensus_protocol = action[1]  # Giao thức đồng thuận
        
        # Xử lý các giao dịch và tính toán phần thưởng/phạt
        processed_transactions = 0
        successful_transactions = 0
        total_latency = 0
        total_energy = 0
        
        # Chọn một số lượng giao dịch ngẫu nhiên để xử lý (tối đa 10)
        num_transactions_to_process = min(10, len(self.transaction_pool))
        
        if num_transactions_to_process > 0:
            # Chọn ngẫu nhiên các giao dịch để xử lý
            indices_to_process = np.random.choice(len(self.transaction_pool), 
                                                size=num_transactions_to_process, 
                                                replace=False)
            
            for idx in sorted(indices_to_process, reverse=True):
                transaction = self.transaction_pool.pop(idx)
                
                # Xử lý giao dịch với hành động được cung cấp
                processed_tx, tx_latency = self._process_transaction(transaction, action)
                
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
        
        # Tính throughput reward
        throughput_reward = self.throughput_reward * self.performance_metrics['successful_transactions']
        
        # Tính latency penalty
        avg_latency = self.performance_metrics['total_latency'] / max(1, self.performance_metrics['transactions_processed'])
        latency_penalty = self.latency_penalty * min(1.0, avg_latency / 100.0)
        
        # Tính energy penalty
        avg_energy = self.performance_metrics['total_energy'] / max(1, self.performance_metrics['transactions_processed'])
        energy_penalty = self.energy_penalty * min(1.0, avg_energy / 50.0)
        
        # Tổng hợp phần thưởng
        reward = throughput_reward - latency_penalty - energy_penalty
        
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