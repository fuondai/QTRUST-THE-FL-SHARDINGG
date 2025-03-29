import os
import sys
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Thêm thư mục hiện tại vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class Node:
    def __init__(self, node_id, shard_id, is_malicious=False, attack_type=None):
        self.node_id = node_id
        self.shard_id = shard_id
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.trust_score = 1.0
        self.processing_power = random.uniform(0.8, 1.2)
        self.connections = []
        self.transactions_processed = 0
        self.uptime = 100.0  # Tỷ lệ thời gian hoạt động (%)
        self.energy_efficiency = random.uniform(0.7, 1.0)  # Hiệu suất năng lượng
        self.last_active = time.time()
        self.reputation_history = []
        
    def __str__(self):
        return f"Node {self.node_id} (Shard {self.shard_id})"
    
    def update_trust_score(self, success_rate):
        """Cập nhật điểm tin cậy dựa trên tỷ lệ thành công."""
        self.trust_score = 0.9 * self.trust_score + 0.1 * success_rate
        self.reputation_history.append(self.trust_score)
        return self.trust_score

class Transaction:
    def __init__(self, tx_id, source_shard, target_shard, size=1.0):
        self.tx_id = tx_id
        self.source_shard = source_shard
        self.target_shard = target_shard
        self.size = size
        self.is_cross_shard = source_shard != target_shard
        self.route = []
        self.hops = 0
        self.latency = 0
        self.energy = 0
        self.is_processed = False
        self.timestamp = time.time()
        self.priority = random.uniform(0, 1)  # Độ ưu tiên của giao dịch
        self.data_integrity = 1.0  # Tính toàn vẹn dữ liệu
        self.processing_attempts = 0  # Số lần thử xử lý
        self.completion_time = None
        self.resource_cost = 0.0  # Chi phí tài nguyên cho việc xử lý
        
    def is_cross_shard_tx(self):
        return self.is_cross_shard
    
    def calculate_resource_cost(self):
        """Tính toán chi phí tài nguyên dựa trên kích thước và số hop."""
        base_cost = self.size * 0.5
        hop_factor = 1.0 + (self.hops * 0.2)
        self.resource_cost = base_cost * hop_factor
        return self.resource_cost
    
    def mark_completed(self):
        """Đánh dấu giao dịch đã hoàn thành và ghi nhận thời gian."""
        self.is_processed = True
        self.completion_time = time.time()
        self.calculate_resource_cost()

class Shard:
    def __init__(self, shard_id, num_nodes, malicious_percentage=0, attack_types=None):
        self.shard_id = shard_id
        self.nodes = []
        self.congestion_level = 0.0
        self.transactions_queue = []
        self.processed_transactions = []
        self.blocked_transactions = []  # Giao dịch bị chặn
        self.network_stability = 1.0  # Độ ổn định mạng
        self.resource_utilization = 0.0  # Mức độ sử dụng tài nguyên
        self.consensus_difficulty = random.uniform(0.5, 1.5)  # Độ khó đạt đồng thuận
        self.last_metrics_update = time.time()
        self.historical_congestion = []  # Lịch sử mức độ tắc nghẽn
        
        # Tính số lượng nút độc hại
        num_malicious = int(num_nodes * malicious_percentage / 100)
        
        # Tạo danh sách các loại tấn công nếu được chỉ định
        if attack_types is None:
            attack_types = []
        
        # Tạo nút
        for i in range(num_nodes):
            is_malicious = i < num_malicious
            attack_type = None
            if is_malicious and attack_types:
                attack_type = random.choice(attack_types)
            
            node = Node(
                node_id=f"{shard_id}_{i}", 
                shard_id=shard_id,
                is_malicious=is_malicious,
                attack_type=attack_type
            )
            self.nodes.append(node)
    
    def get_non_malicious_nodes(self):
        return [node for node in self.nodes if not node.is_malicious]
    
    def get_malicious_nodes(self):
        return [node for node in self.nodes if node.is_malicious]
    
    def compute_power_distribution(self):
        total_power = sum(node.processing_power for node in self.nodes)
        return [(node.node_id, node.processing_power / total_power) for node in self.nodes]

    def update_congestion(self):
        # Cập nhật mức độ tắc nghẽn dựa trên số lượng giao dịch trong hàng đợi
        queue_size = len(self.transactions_queue)
        prev_congestion = self.congestion_level
        self.congestion_level = min(1.0, queue_size / 100)  # Tối đa là 1.0
        
        # Thêm vào lịch sử
        self.historical_congestion.append(self.congestion_level)
        
        # Chỉ giữ 100 giá trị gần nhất
        if len(self.historical_congestion) > 100:
            self.historical_congestion.pop(0)
        
        # Cập nhật độ ổn định mạng dựa trên mức độ biến động tắc nghẽn
        if prev_congestion > 0:
            stability_factor = 1.0 - abs(self.congestion_level - prev_congestion) / prev_congestion
            self.network_stability = 0.9 * self.network_stability + 0.1 * stability_factor
        
        # Cập nhật mức sử dụng tài nguyên
        self.resource_utilization = 0.8 * self.resource_utilization + 0.2 * self.congestion_level
        
    def get_shard_health(self):
        """Tính toán chỉ số sức khỏe tổng thể của shard."""
        # Kết hợp các chỉ số
        non_malicious_ratio = len(self.get_non_malicious_nodes()) / max(1, len(self.nodes))
        congestion_factor = 1.0 - self.congestion_level
        stability_factor = self.network_stability
        
        # Tính điểm sức khỏe (0-1)
        health_score = (non_malicious_ratio * 0.4 + 
                      congestion_factor * 0.3 + 
                      stability_factor * 0.3)
        
        return health_score
        
    def __str__(self):
        return f"Shard {self.shard_id} with {len(self.nodes)} nodes"

# Hàm xử lý giao dịch đơn lẻ cho đa luồng
def _process_single_transaction(tx, sim_context):
    """Xử lý một giao dịch đơn lẻ trong luồng riêng"""
    shards = sim_context['shards']
    
    # Kiểm tra xem đây có phải là giao dịch xuyên shard không
    if tx.is_cross_shard:
        # Mô phỏng việc định tuyến thông qua các shard
        tx.hops = max(1, abs(tx.target_shard - tx.source_shard))
        
        # Tính độ trễ và tiêu thụ năng lượng
        base_latency = 10 + (5 * tx.hops)  # Cơ sở + 5ms cho mỗi hop
        congestion_factor = max(shards[tx.source_shard].congestion_level, 
                             shards[tx.target_shard].congestion_level)
        
        # Điều chỉnh độ trễ dựa trên mức độ tắc nghẽn
        tx.latency = base_latency * (1 + congestion_factor * 2)
        
        # Tính toán năng lượng tiêu thụ
        base_energy = 2.0 * tx.size  # Cơ sở 2 đơn vị năng lượng cho mỗi đơn vị kích thước
        hop_energy = 0.5 * tx.hops    # 0.5 đơn vị cho mỗi hop
        tx.energy = base_energy + hop_energy
    else:
        # Giao dịch nội shard
        tx.hops = 1
        tx.latency = 5 * (1 + shards[tx.source_shard].congestion_level)  # 5ms cơ sở
        tx.energy = 1.0 * tx.size  # 1 đơn vị năng lượng cho mỗi đơn vị kích thước
    
    # Kiểm tra xem giao dịch có bị ảnh hưởng bởi nút độc hại không
    source_shard = shards[tx.source_shard]
    target_shard = shards[tx.target_shard]
    malicious_node_ratio = (
        len(source_shard.get_malicious_nodes()) / len(source_shard.nodes) +
        len(target_shard.get_malicious_nodes()) / len(target_shard.nodes)
    ) / 2
    
    # Xác suất thành công dựa trên tỷ lệ nút độc hại
    success_probability = 1.0 - (malicious_node_ratio * 0.8)  # 80% ảnh hưởng từ nút độc hại
    
    # Đánh dấu giao dịch là đã xử lý
    tx.processing_attempts += 1
    if random.random() < success_probability:
        tx.mark_completed()
        return (tx, True)
    else:
        return (tx, False)

class LargeScaleBlockchainSimulation:
    def __init__(self, 
                 num_shards=10, 
                 nodes_per_shard=20,
                 malicious_percentage=0,
                 attack_scenario=None,
                 max_workers=None):
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.malicious_percentage = malicious_percentage
        self.attack_scenario = attack_scenario
        self.shards = []
        self.transactions = []
        self.processed_transactions = []
        self.current_step = 0
        self.tx_counter = 0
        self.simulation_start_time = time.time()
        
        # Số lượng worker tối đa cho đa luồng, mặc định là số lõi CPU * 2
        self.max_workers = max_workers or (multiprocessing.cpu_count() * 2)
        
        # Mở rộng các chỉ số
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy': [],
            'security': [],
            'cross_shard_ratio': [],
            'transaction_success_rate': [],  # Tỷ lệ thành công
            'network_stability': [],         # Độ ổn định mạng
            'resource_utilization': [],      # Mức sử dụng tài nguyên
            'consensus_efficiency': [],      # Hiệu quả đồng thuận
            'shard_health': [],              # Sức khỏe của các shard
            'avg_hops': [],                  # Số bước trung bình
            'network_resilience': [],        # Khả năng phục hồi
            'avg_block_size': [],            # Kích thước khối trung bình
            'network_partition_events': []   # Sự kiện phân mảnh mạng
        }
        
        # Thêm thống kê theo thời gian thực
        self.real_time_stats = {
            'start_time': time.time(),
            'elapsed_time': 0,
            'tx_per_second': 0,
            'avg_processing_time': 0,
            'peak_throughput': 0,
            'peak_latency': 0,
        }
        
        self.attack_types = []
        
        # Cấu hình loại tấn công dựa trên kịch bản
        if attack_scenario:
            if attack_scenario == '51_percent':
                self.attack_types = ['51_percent']
                # Đảm bảo tỷ lệ độc hại ít nhất 51% cho kịch bản này
                self.malicious_percentage = max(51, malicious_percentage)
            elif attack_scenario == 'sybil':
                self.attack_types = ['sybil']
            elif attack_scenario == 'eclipse':
                self.attack_types = ['eclipse']
            elif attack_scenario == 'selfish_mining':
                self.attack_types = ['selfish_mining']
            elif attack_scenario == 'bribery':
                self.attack_types = ['bribery']
            elif attack_scenario == 'ddos':
                self.attack_types = ['ddos']
            elif attack_scenario == 'finney':
                self.attack_types = ['finney']
            elif attack_scenario == 'mixed':
                self.attack_types = ['51_percent', 'sybil', 'eclipse', 'selfish_mining']
        
        self._initialize_blockchain()
        
    def _initialize_blockchain(self):
        # Tạo các shard
        for i in range(self.num_shards):
            shard = Shard(
                shard_id=i,
                num_nodes=self.nodes_per_shard,
                malicious_percentage=self.malicious_percentage,
                attack_types=self.attack_types
            )
            self.shards.append(shard)
        
        # Tạo kết nối giữa các nút
        self._create_network_connections()
        
        print(f"Khởi tạo blockchain với {self.num_shards} shard, mỗi shard có {self.nodes_per_shard} nút")
        print(f"Tỷ lệ nút độc hại: {self.malicious_percentage}%")
        if self.attack_scenario:
            print(f"Kịch bản tấn công: {self.attack_scenario}")
    
    def _create_network_connections(self):
        # Tạo kết nối liên shard
        for source_shard in self.shards:
            for target_shard in self.shards:
                if source_shard.shard_id != target_shard.shard_id:
                    # Chọn ngẫu nhiên nút từ mỗi shard để kết nối
                    source_nodes = random.sample(source_shard.nodes, min(5, len(source_shard.nodes)))
                    target_nodes = random.sample(target_shard.nodes, min(5, len(target_shard.nodes)))
                    
                    for s_node in source_nodes:
                        for t_node in target_nodes:
                            s_node.connections.append(t_node)
                            t_node.connections.append(s_node)
        
        # Thêm kết nối nội shard
        for shard in self.shards:
            for i, node in enumerate(shard.nodes):
                # Mỗi nút kết nối với 80% nút khác trong shard
                potential_connections = [n for n in shard.nodes if n != node]
                num_connections = int(len(potential_connections) * 0.8)
                connections = random.sample(potential_connections, num_connections)
                
                for conn in connections:
                    if conn not in node.connections:
                        node.connections.append(conn)
                    if node not in conn.connections:
                        conn.connections.append(node)
                        
        # Nếu có kịch bản tấn công Eclipse, thay đổi kết nối
        if 'eclipse' in self.attack_types:
            self._setup_eclipse_attack()
    
    def _setup_eclipse_attack(self):
        # Chọn ngẫu nhiên một shard để thực hiện tấn công
        target_shard = random.choice(self.shards)
        malicious_nodes = target_shard.get_malicious_nodes()
        
        if malicious_nodes:
            # Chọn ngẫu nhiên một nút để bị cô lập
            victim_nodes = random.sample(target_shard.get_non_malicious_nodes(), 
                                         min(3, len(target_shard.get_non_malicious_nodes())))
            
            for victim in victim_nodes:
                print(f"Thiết lập tấn công Eclipse cho nút {victim.node_id}")
                
                # Xóa tất cả các kết nối hiện tại
                for conn in victim.connections[:]:
                    if conn in victim.connections:
                        victim.connections.remove(conn)
                    if victim in conn.connections:
                        conn.connections.remove(victim)
                
                # Chỉ kết nối với các nút độc hại
                for attacker in malicious_nodes:
                    victim.connections.append(attacker)
                    attacker.connections.append(victim)
    
    def _generate_transactions(self, num_transactions):
        new_transactions = []
        
        for _ in range(num_transactions):
            source_shard = random.randint(0, self.num_shards - 1)
            
            # 30% là giao dịch xuyên shard
            if random.random() < 0.3:
                target_shard = random.randint(0, self.num_shards - 1)
                while target_shard == source_shard:
                    target_shard = random.randint(0, self.num_shards - 1)
            else:
                target_shard = source_shard
            
            # Tạo giao dịch với size ngẫu nhiên
            tx = Transaction(
                tx_id=f"tx_{self.tx_counter}",
                source_shard=source_shard,
                target_shard=target_shard,
                size=random.uniform(0.5, 2.0)
            )
            self.tx_counter += 1
            new_transactions.append(tx)
            
            # Thêm vào hàng đợi của shard nguồn
            self.shards[source_shard].transactions_queue.append(tx)
        
        self.transactions.extend(new_transactions)
        return new_transactions
    
    def _process_transactions(self):
        processed_count = 0
        blocked_count = 0
        total_hops = 0
        
        for shard in self.shards:
            # Cập nhật mức độ tắc nghẽn
            shard.update_congestion()
        
        # Chuẩn bị context cho xử lý đa luồng
        sim_context = {
            'shards': self.shards,
            'attack_types': [self.attack_scenario] if self.attack_scenario else []
        }
        
        # Thu thập tất cả các giao dịch cần xử lý từ tất cả các shard
        all_transactions = []
        for shard in self.shards:
            # Lấy các nút không độc hại để xử lý
            validators = shard.get_non_malicious_nodes()
            
            # Nếu không có đủ nút xác thực, bỏ qua shard này
            if len(validators) < self.nodes_per_shard / 2:
                continue
                
            # Sắp xếp giao dịch theo ưu tiên
            shard.transactions_queue.sort(key=lambda tx: tx.priority, reverse=True)
            
            # Chỉ xử lý tối đa 15 giao dịch mỗi shard
            all_transactions.extend(shard.transactions_queue[:15])
            
        # Sử dụng ThreadPoolExecutor để xử lý song song
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tạo một partial function với context cố định
            process_tx_with_context = partial(_process_single_transaction, sim_context=sim_context)
            
            # Submit tất cả các giao dịch cho executor
            future_to_tx = {executor.submit(process_tx_with_context, tx): tx for tx in all_transactions}
            
            # Xử lý kết quả khi hoàn thành
            for future in as_completed(future_to_tx):
                tx = future_to_tx[future]
                try:
                    result_tx, success = future.result()
                    
                    # Xóa khỏi hàng đợi của shard
                    source_shard = self.shards[tx.source_shard]
                    if tx in source_shard.transactions_queue:
                        source_shard.transactions_queue.remove(tx)
                    
                    if success:
                        # Giao dịch thành công
                        self.processed_transactions.append(result_tx)
                        source_shard.processed_transactions.append(result_tx)
                        processed_count += 1
                        
                        # Cập nhật thống kê
                        if result_tx.is_cross_shard:
                            total_hops += result_tx.hops
                    else:
                        # Giao dịch thất bại
                        blocked_count += 1
                        source_shard.blocked_transactions.append(result_tx)
                except Exception as e:
                    print(f"Lỗi khi xử lý giao dịch: {e}")
        
        # Cập nhật thống kê
        if processed_count > 0:
            avg_hops = total_hops / max(1, processed_count)
        else:
            avg_hops = 0
            
        # Cập nhật tin cậy của các nút
        for shard in self.shards:
            for node in shard.nodes:
                # Cập nhật điểm tin cậy dựa trên số giao dịch đã xử lý
                success_rate = min(1.0, node.transactions_processed / max(1, processed_count + blocked_count))
                node.update_trust_score(success_rate)
        
        return processed_count, blocked_count, avg_hops
    
    def _calculate_metrics(self):
        # Chỉ tính toán các chỉ số nếu có giao dịch đã xử lý
        if not self.processed_transactions:
            return {
                'throughput': 0,
                'latency': 0,
                'energy': 0,
                'security': 0,
                'cross_shard_ratio': 0,
                'transaction_success_rate': 0,
                'network_stability': 0,
                'resource_utilization': 0,
                'consensus_efficiency': 0,
                'shard_health': 0,
                'avg_hops': 0,
                'network_resilience': 0,
                'avg_block_size': 0,
                'network_partition_events': 0
            }
        
        # Tính throughput
        throughput = len(self.processed_transactions) / max(1, self.current_step)
        
        # Tính độ trễ trung bình
        avg_latency = sum(tx.latency for tx in self.processed_transactions) / len(self.processed_transactions)
        
        # Tính năng lượng trung bình
        avg_energy = sum(tx.energy for tx in self.processed_transactions) / len(self.processed_transactions)
        
        # Tính tỷ lệ giao dịch xuyên shard
        cross_shard_txs = [tx for tx in self.processed_transactions if tx.is_cross_shard_tx()]
        cross_shard_ratio = len(cross_shard_txs) / len(self.processed_transactions) if self.processed_transactions else 0
        
        # Tính tỷ lệ thành công giao dịch
        success_rate = len(self.processed_transactions) / max(1, len(self.transactions))
        
        # Tính độ ổn định mạng
        network_stability = sum(shard.network_stability for shard in self.shards) / len(self.shards)
        
        # Tính mức sử dụng tài nguyên
        resource_utilization = sum(shard.resource_utilization for shard in self.shards) / len(self.shards)
        
        # Tính hiệu quả đồng thuận
        consensus_efficiency = 1.0 - (avg_latency / 1000.0)  # Đơn vị là ms
        consensus_efficiency = max(0, min(1, consensus_efficiency))
        
        # Tính điểm sức khỏe trung bình của các shard
        shard_health = sum(shard.get_shard_health() for shard in self.shards) / len(self.shards)
        
        # Tính số hop trung bình
        avg_hops = sum(tx.hops for tx in self.processed_transactions) / len(self.processed_transactions)
        
        # Tính an toàn (security score) dựa trên phân bố điểm tin cậy
        total_nodes = sum(len(shard.nodes) for shard in self.shards)
        malicious_nodes = sum(len(shard.get_malicious_nodes()) for shard in self.shards)
        malicious_ratio = malicious_nodes / total_nodes if total_nodes > 0 else 0
        
        # Cách tính cải tiến, xem xét cả điểm tin cậy của các nút
        trust_scores = []
        for shard in self.shards:
            for node in shard.nodes:
                trust_scores.append(node.trust_score)
        
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0
        
        # Tính security score - cải thiện công thức
        max_malicious_threshold = 0.45  # Ngưỡng tỷ lệ độc hại tối đa có thể chịu được
        if malicious_ratio >= 0.51:  # Tấn công 51%
            security = max(0, 0.2 - (malicious_ratio - 0.51) * 2) * avg_trust
        elif malicious_ratio > max_malicious_threshold:  # Gần với ngưỡng nguy hiểm
            security = max(0, 1 - ((malicious_ratio - max_malicious_threshold) / (0.51 - max_malicious_threshold))) * avg_trust
        else:
            security = (1 - (malicious_ratio / max_malicious_threshold)) * avg_trust
        
        # Tính khả năng phục hồi mạng
        # Cải thiện khả năng phục hồi đối với tấn công 51%
        if '51_percent' in self.attack_types and malicious_ratio >= 0.51:
            # Tỷ lệ nút độc hại có điểm tin cậy thấp (đã bị phát hiện)
            detected_malicious = sum(1 for shard in self.shards 
                                   for node in shard.get_malicious_nodes() 
                                   if node.trust_score < 0.5)
            detection_rate = detected_malicious / max(1, malicious_nodes)
            
            # Khả năng phục hồi sẽ phụ thuộc vào tỷ lệ phát hiện và số lượng nút độc hại
            network_resilience = detection_rate * (1 - malicious_ratio/0.7) * security
            network_resilience = max(0, min(0.6, network_resilience))  # Giới hạn tối đa 0.6 cho tấn công 51%
        elif malicious_ratio > 0.3:
            # Đối với tỷ lệ độc hại cao nhưng chưa đạt 51%
            network_resilience = (1 - malicious_ratio) * security * 0.7
        else:
            # Đối với tỷ lệ độc hại thấp
            network_resilience = (1 - malicious_ratio) * security
        
        network_resilience = max(0, min(1, network_resilience))  # Giới hạn 0-1
        
        # Tính kích thước khối trung bình (giả lập)
        avg_block_size = sum(tx.size for tx in self.processed_transactions[-100:]) / min(100, len(self.processed_transactions))
        
        # Số sự kiện phân mảnh mạng (giả lập)
        network_partition_events = int(10 * (1 - network_stability))
        
        return {
            'throughput': throughput,
            'latency': avg_latency,
            'energy': avg_energy,
            'security': security,
            'cross_shard_ratio': cross_shard_ratio,
            'transaction_success_rate': success_rate,
            'network_stability': network_stability,
            'resource_utilization': resource_utilization,
            'consensus_efficiency': consensus_efficiency,
            'shard_health': shard_health,
            'avg_hops': avg_hops,
            'network_resilience': network_resilience,
            'avg_block_size': avg_block_size,
            'network_partition_events': network_partition_events
        }
    
    def _update_metrics(self):
        metrics = self._calculate_metrics()
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def run_simulation(self, num_steps=1000, transactions_per_step=50):
        print(f"Bắt đầu mô phỏng với {num_steps} bước, {transactions_per_step} giao dịch/bước")
        
        for step in tqdm(range(num_steps)):
            self.current_step = step + 1
            
            # Tạo giao dịch mới
            self._generate_transactions(transactions_per_step)
            
            # Xử lý giao dịch
            processed, blocked, avg_hops = self._process_transactions()
            
            # Cập nhật các chỉ số
            self._update_metrics()
            
            # In thông tin mỗi 100 bước
            if (step + 1) % 100 == 0:
                metrics = self._calculate_metrics()
                print(f"\nBước {step + 1}/{num_steps}:")
                print(f"  Throughput: {metrics['throughput']:.2f} tx/s")
                print(f"  Độ trễ trung bình: {metrics['latency']:.2f} ms")
                print(f"  Tỷ lệ thành công: {metrics['transaction_success_rate']:.2f}")
                print(f"  Độ ổn định mạng: {metrics['network_stability']:.2f}")
                print(f"  Sức khỏe shard: {metrics['shard_health']:.2f}")
                print(f"  Giao dịch đã xử lý: {processed}, bị chặn: {blocked}")
        
        print("\nMô phỏng hoàn tất!")
        return self.metrics
    
    def plot_metrics(self, save_dir=None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Đặt style cho biểu đồ
        plt.style.use('dark_background')
        sns.set(style="darkgrid")
        
        # Tạo bảng màu tùy chỉnh
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(colors))
        
        # Tạo figure với nhiều subplot
        fig = plt.figure(figsize=(20, 16))
        
        # Thiết lập GridSpec
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Throughput
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.metrics['throughput'], color=colors[0], linewidth=2)
        ax1.set_title('Throughput (tx/s)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Bước')
        ax1.set_ylabel('tx/s')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(range(len(self.metrics['throughput'])), 
                         self.metrics['throughput'], 
                         alpha=0.3, 
                         color=colors[0])
        
        # 2. Latency
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.metrics['latency'], color=colors[1], linewidth=2)
        ax2.set_title('Độ trễ (ms)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bước')
        ax2.set_ylabel('ms')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(range(len(self.metrics['latency'])), 
                         self.metrics['latency'], 
                         alpha=0.3, 
                         color=colors[1])
        
        # 3. Energy
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.metrics['energy'], color=colors[2], linewidth=2)
        ax3.set_title('Tiêu thụ năng lượng', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Bước')
        ax3.set_ylabel('Đơn vị năng lượng')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(range(len(self.metrics['energy'])), 
                         self.metrics['energy'], 
                         alpha=0.3, 
                         color=colors[2])
        
        # 4. Security
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.metrics['security'], color=colors[3], linewidth=2)
        ax4.set_title('Điểm bảo mật', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Bước')
        ax4.set_ylabel('Điểm (0-1)')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(range(len(self.metrics['security'])), 
                         self.metrics['security'], 
                         alpha=0.3, 
                         color=colors[3])
        
        # 5. Cross-shard ratio
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.metrics['cross_shard_ratio'], color=colors[4], linewidth=2)
        ax5.set_title('Tỷ lệ giao dịch xuyên shard', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Bước')
        ax5.set_ylabel('Tỷ lệ')
        ax5.grid(True, alpha=0.3)
        ax5.fill_between(range(len(self.metrics['cross_shard_ratio'])), 
                         self.metrics['cross_shard_ratio'], 
                         alpha=0.3, 
                         color=colors[4])
        
        # 6. Transaction success rate
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.metrics['transaction_success_rate'], color=colors[5], linewidth=2)
        ax6.set_title('Tỷ lệ giao dịch thành công', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Bước')
        ax6.set_ylabel('Tỷ lệ')
        ax6.grid(True, alpha=0.3)
        ax6.fill_between(range(len(self.metrics['transaction_success_rate'])), 
                         self.metrics['transaction_success_rate'], 
                         alpha=0.3, 
                         color=colors[5])
        
        # 7. Network stability
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(self.metrics['network_stability'], color=colors[0], linewidth=2)
        ax7.set_title('Độ ổn định mạng', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Bước')
        ax7.set_ylabel('Điểm (0-1)')
        ax7.grid(True, alpha=0.3)
        ax7.fill_between(range(len(self.metrics['network_stability'])), 
                         self.metrics['network_stability'], 
                         alpha=0.3, 
                         color=colors[0])
        
        # 8. Resource utilization
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(self.metrics['resource_utilization'], color=colors[1], linewidth=2)
        ax8.set_title('Mức sử dụng tài nguyên', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Bước')
        ax8.set_ylabel('Tỷ lệ')
        ax8.grid(True, alpha=0.3)
        ax8.fill_between(range(len(self.metrics['resource_utilization'])), 
                         self.metrics['resource_utilization'], 
                         alpha=0.3, 
                         color=colors[1])
        
        # 9. Consensus efficiency
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(self.metrics['consensus_efficiency'], color=colors[2], linewidth=2)
        ax9.set_title('Hiệu quả đồng thuận', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Bước')
        ax9.set_ylabel('Điểm (0-1)')
        ax9.grid(True, alpha=0.3)
        ax9.fill_between(range(len(self.metrics['consensus_efficiency'])), 
                         self.metrics['consensus_efficiency'], 
                         alpha=0.3, 
                         color=colors[2])
        
        # 10. Shard health
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.plot(self.metrics['shard_health'], color=colors[3], linewidth=2)
        ax10.set_title('Sức khỏe shard', fontsize=14, fontweight='bold')
        ax10.set_xlabel('Bước')
        ax10.set_ylabel('Điểm (0-1)')
        ax10.grid(True, alpha=0.3)
        ax10.fill_between(range(len(self.metrics['shard_health'])), 
                         self.metrics['shard_health'], 
                         alpha=0.3, 
                         color=colors[3])
        
        # 11. Network resilience
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.plot(self.metrics['network_resilience'], color=colors[4], linewidth=2)
        ax11.set_title('Khả năng phục hồi', fontsize=14, fontweight='bold')
        ax11.set_xlabel('Bước')
        ax11.set_ylabel('Điểm (0-1)')
        ax11.grid(True, alpha=0.3)
        ax11.fill_between(range(len(self.metrics['network_resilience'])), 
                         self.metrics['network_resilience'], 
                         alpha=0.3, 
                         color=colors[4])
        
        # 12. Average block size
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.plot(self.metrics['avg_block_size'], color=colors[5], linewidth=2)
        ax12.set_title('Kích thước khối trung bình', fontsize=14, fontweight='bold')
        ax12.set_xlabel('Bước')
        ax12.set_ylabel('Kích thước')
        ax12.grid(True, alpha=0.3)
        ax12.fill_between(range(len(self.metrics['avg_block_size'])), 
                         self.metrics['avg_block_size'], 
                         alpha=0.3, 
                         color=colors[5])
        
        # Tiêu đề chính
        fig.suptitle(f'QTrust Blockchain Metrics - {self.num_shards} Shards, {self.nodes_per_shard} Nodes/Shard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Thêm chú thích về tấn công
        attack_text = f"Attack Scenario: {self.attack_scenario}" if self.attack_scenario else "No Attack"
        plt.figtext(0.5, 0.01, attack_text, ha="center", fontsize=16, bbox={"facecolor":"red", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Lưu biểu đồ
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attack_suffix = f"_{self.attack_scenario}" if self.attack_scenario else ""
            filename = f"{save_dir}/detailed_metrics_{self.num_shards}shards_{self.nodes_per_shard}nodes{attack_suffix}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ chi tiết tại: {filename}")
        
        # Đóng biểu đồ để giải phóng bộ nhớ
        plt.close(fig)
        
        # Tạo biểu đồ radar
        self._plot_radar_chart(save_dir)
        
        # Tạo biểu đồ heatmap cho mức độ tắc nghẽn
        self._plot_congestion_heatmap(save_dir)
    
    def _plot_radar_chart(self, save_dir=None):
        # Lấy chỉ số từ 100 bước gần nhất
        recent_metrics = {}
        for key in self.metrics:
            if self.metrics[key]:
                recent_metrics[key] = np.mean(self.metrics[key][-100:])
            else:
                recent_metrics[key] = 0
        
        # Các nhãn cho biểu đồ radar
        categories = [
            'Throughput', 'Độ bảo mật', 'Tỷ lệ thành công',
            'Độ ổn định mạng', 'Hiệu quả đồng thuận', 'Khả năng phục hồi'
        ]
        
        # Chuẩn hóa giá trị về thang 0-1
        values = [
            min(1, recent_metrics['throughput'] / max(1, recent_metrics['throughput'])),
            recent_metrics['security'],
            recent_metrics['transaction_success_rate'],
            recent_metrics['network_stability'],
            recent_metrics['consensus_efficiency'],
            recent_metrics['network_resilience']
        ]
        
        # Số lượng biến
        N = len(categories)
        
        # Góc cho mỗi trục
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Đóng vòng
        
        # Thêm giá trị đầu tiên vào cuối để đóng vòng
        values += values[:1]
        
        # Thiết lập biểu đồ
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # Vẽ đường và điền màu
        ax.plot(angles, values, 'o-', linewidth=2, color='cyan')
        ax.fill(angles, values, alpha=0.25, color='cyan')
        
        # Thiết lập nhãn
        plt.xticks(angles[:-1], categories, size=14)
        
        # Thiết lập tỷ lệ trục y
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                  color="grey", size=12)
        plt.ylim(0, 1)
        
        # Tiêu đề
        attack_text = f"- {self.attack_scenario}" if self.attack_scenario else ""
        plt.title(f"QTrust Performance Radar - {self.num_shards} Shards {attack_text}", 
                 size=16, color='white', pad=20)
        
        # Lưu biểu đồ
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attack_suffix = f"_{self.attack_scenario}" if self.attack_scenario else ""
            filename = f"{save_dir}/radar_chart_{self.num_shards}shards_{self.nodes_per_shard}nodes{attack_suffix}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ radar tại: {filename}")
        
        # Đóng biểu đồ để giải phóng bộ nhớ
        plt.close()
    
    def _plot_congestion_heatmap(self, save_dir=None):
        # Kiểm tra xem có dữ liệu tắc nghẽn không
        has_congestion_data = all(hasattr(shard, 'historical_congestion') and len(shard.historical_congestion) > 0 
                             for shard in self.shards)
        
        if not has_congestion_data:
            print("Không có dữ liệu tắc nghẽn để tạo heatmap")
            return
            
        # Tạo ma trận congestion cho các shard
        history_length = min(100, min(len(shard.historical_congestion) for shard in self.shards))
        
        if history_length == 0:
            print("Không đủ dữ liệu tắc nghẽn lịch sử để tạo heatmap")
            return
            
        congestion_data = np.zeros((self.num_shards, history_length))
        
        for i, shard in enumerate(self.shards):
            congestion_data[i, :history_length] = shard.historical_congestion[-history_length:]
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 8))
        
        # Tạo heatmap
        heatmap = sns.heatmap(congestion_data, cmap='inferno', 
                   xticklabels=20, yticklabels=[f"Shard {i}" for i in range(self.num_shards)])
        
        plt.title("Congestion Levels Across Shards", fontsize=16)
        plt.xlabel("Time Steps (most recent)", fontsize=12)
        plt.ylabel("Shards", fontsize=12)
        
        # Thêm colorbar
        # colorbar đã được tạo tự động bởi sns.heatmap
        
        # Lưu biểu đồ
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attack_suffix = f"_{self.attack_scenario}" if self.attack_scenario else ""
            filename = f"{save_dir}/congestion_heatmap_{self.num_shards}shards_{self.nodes_per_shard}nodes{attack_suffix}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ mức độ tắc nghẽn tại: {filename}")
        
        # Đóng biểu đồ để giải phóng bộ nhớ
        plt.close()
    
    def generate_report(self, save_dir=None):
        # Tính toán metrics cuối cùng
        metrics = self._calculate_metrics()
        
        # Chuẩn bị báo cáo
        report = f"""
QTrust Large-Scale Blockchain Simulation Report
==============================================

Cấu hình:
- Số lượng shard: {self.num_shards}
- Số nút trên mỗi shard: {self.nodes_per_shard}
- Tổng số nút: {self.num_shards * self.nodes_per_shard}
- Tỷ lệ nút độc hại: {self.malicious_percentage}%
- Kịch bản tấn công: {self.attack_scenario if self.attack_scenario else "Không có"}

Hiệu suất cơ bản:
- Throughput: {metrics['throughput']:.2f} tx/s
- Độ trễ trung bình: {metrics['latency']:.2f} ms
- Tiêu thụ năng lượng trung bình: {metrics['energy']:.2f}
- Điểm bảo mật: {metrics['security']:.2f}
- Tỷ lệ giao dịch xuyên shard: {metrics['cross_shard_ratio']:.2f}

Chỉ số hiệu suất nâng cao:
- Tỷ lệ giao dịch thành công: {metrics['transaction_success_rate']:.2f}
- Độ ổn định mạng: {metrics['network_stability']:.2f}
- Mức sử dụng tài nguyên: {metrics['resource_utilization']:.2f}
- Hiệu quả đồng thuận: {metrics['consensus_efficiency']:.2f}
- Sức khỏe shard: {metrics['shard_health']:.2f}
- Số hop trung bình: {metrics['avg_hops']:.2f}
- Khả năng phục hồi mạng: {metrics['network_resilience']:.2f}
- Kích thước khối trung bình: {metrics['avg_block_size']:.2f}
- Số sự kiện phân mảnh mạng: {metrics['network_partition_events']}

Thống kê thời gian thực:
- Thời gian chạy: {self.real_time_stats['elapsed_time']:.2f} giây
- Giao dịch/giây: {self.real_time_stats['tx_per_second']:.2f}
- Thời gian xử lý trung bình: {self.real_time_stats['avg_processing_time']:.4f} giây
- Throughput đỉnh: {self.real_time_stats['peak_throughput']:.2f} tx/s

Số liệu thống kê:
- Tổng số giao dịch đã tạo: {len(self.transactions)}
- Tổng số giao dịch đã xử lý: {len(self.processed_transactions)}
- Tỷ lệ giao dịch thành công: {len(self.processed_transactions) / max(1, len(self.transactions)):.2f}
- Tổng số giao dịch bị chặn: {sum(len(shard.blocked_transactions) for shard in self.shards)}

Phân tích mạng:
- Số lượng kết nối trung bình mỗi nút: {sum(len(node.connections) for shard in self.shards for node in shard.nodes) / (self.num_shards * self.nodes_per_shard):.2f}
- Tỷ lệ nút độc hại có trust score thấp: {sum(1 for shard in self.shards for node in shard.nodes if node.is_malicious and node.trust_score < 0.5) / max(1, sum(1 for shard in self.shards for node in shard.nodes if node.is_malicious)):.2f}
"""
        
        # In báo cáo
        print(report)
        
        # Lưu báo cáo
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attack_suffix = f"_{self.attack_scenario}" if self.attack_scenario else ""
            filename = f"{save_dir}/detailed_report_{self.num_shards}shards_{self.nodes_per_shard}nodes{attack_suffix}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"Đã lưu báo cáo chi tiết tại: {filename}")
            
            # Tạo báo cáo JSON
            json_filename = f"{save_dir}/metrics_{self.num_shards}shards_{self.nodes_per_shard}nodes{attack_suffix}_{timestamp}.json"
            self._save_metrics_json(json_filename)
        
        return report
    
    def _save_metrics_json(self, filename):
        """Lưu metrics dưới dạng JSON để có thể tái sử dụng."""
        import json
        
        # Chuẩn bị dữ liệu
        data = {
            "config": {
                "num_shards": self.num_shards,
                "nodes_per_shard": self.nodes_per_shard,
                "total_nodes": self.num_shards * self.nodes_per_shard,
                "malicious_percentage": self.malicious_percentage,
                "attack_scenario": self.attack_scenario
            },
            "final_metrics": self._calculate_metrics(),
            "real_time_stats": self.real_time_stats,
            "transaction_stats": {
                "total_created": len(self.transactions),
                "total_processed": len(self.processed_transactions),
                "success_rate": len(self.processed_transactions) / max(1, len(self.transactions))
            }
        }
        
        # Lưu file JSON
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Đã lưu metrics JSON tại: {filename}")
        
def main():
    parser = argparse.ArgumentParser(description='QTrust Large-Scale Blockchain Simulation')
    parser.add_argument('--num-shards', type=int, default=10, help='Số lượng shard')
    parser.add_argument('--nodes-per-shard', type=int, default=20, help='Số nút trên mỗi shard')
    parser.add_argument('--steps', type=int, default=1000, help='Số bước mô phỏng')
    parser.add_argument('--tx-per-step', type=int, default=50, help='Số giao dịch mỗi bước')
    parser.add_argument('--malicious', type=float, default=0, help='Tỷ lệ % nút độc hại')
    parser.add_argument('--attack', type=str, 
                        choices=['51_percent', 'sybil', 'eclipse', 'selfish_mining', 'bribery', 'ddos', 'finney', 'mixed', None], 
                        default=None, help='Kịch bản tấn công')
    parser.add_argument('--save-dir', type=str, default='results_attack', help='Thư mục lưu kết quả')
    parser.add_argument('--no-display', action='store_true', help='Không hiển thị biểu đồ trên màn hình')
    
    args = parser.parse_args()
    
    # Tạo mô phỏng
    simulation = LargeScaleBlockchainSimulation(
        num_shards=args.num_shards,
        nodes_per_shard=args.nodes_per_shard,
        malicious_percentage=args.malicious,
        attack_scenario=args.attack
    )
    
    # Chạy mô phỏng
    metrics = simulation.run_simulation(
        num_steps=args.steps,
        transactions_per_step=args.tx_per_step
    )
    
    # Tạo biểu đồ và báo cáo
    simulation.plot_metrics(save_dir=args.save_dir)
    simulation.generate_report(save_dir=args.save_dir)

if __name__ == "__main__":
    main() 