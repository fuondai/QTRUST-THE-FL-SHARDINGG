import os
import sys
import time
import random
import argparse
import multiprocessing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Cấu hình encoding cho output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục hiện tại vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class Node:
    __slots__ = ('node_id', 'shard_id', 'is_malicious', 'attack_type', 'trust_score', 
                'processing_power', 'connections', 'transactions_processed', 'uptime',
                'energy_efficiency', 'last_active', 'reputation_history')
    
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
    __slots__ = ('tx_id', 'source_shard', 'target_shard', 'size', 'is_cross_shard', 
                'route', 'hops', 'latency', 'energy', 'is_processed', 'timestamp',
                'priority', 'data_integrity', 'processing_attempts', 'completion_time',
                'resource_cost')
    
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

class TransactionPipeline:
    """Lớp xử lý giao dịch theo pipeline, tối ưu hóa quá trình xử lý giao dịch."""
    
    __slots__ = ('shards', 'max_workers', 'optimal_workers', 'pipeline_metrics',
                'processing_cache', 'validation_result_cache', 'routing_result_cache')
    
    def __init__(self, shards, max_workers=None):
        self.shards = shards
        self.max_workers = max_workers
        # Xác định số lượng worker tối ưu
        self.optimal_workers = min(
            multiprocessing.cpu_count() * 2,  # 2x số lượng CPU core
            32  # Giới hạn tối đa
        ) if max_workers is None else max_workers
        
        # Đếm số lượng giao dịch được xử lý qua mỗi giai đoạn
        self.pipeline_metrics = {
            'validation': 0,
            'routing': 0,
            'consensus': 0,
            'execution': 0,
            'commit': 0
        }
        
        # Cache để tăng tốc xử lý
        self.processing_cache = {}  # Cache kết quả xử lý
        self.validation_result_cache = {}  # Cache kết quả kiểm tra
        self.routing_result_cache = {}  # Cache kết quả định tuyến
    
    def validate_transaction(self, tx):
        """Giai đoạn 1: Kiểm tra tính hợp lệ của giao dịch."""
        # Kiểm tra cache trước
        if tx.tx_id in self.validation_result_cache:
            return self.validation_result_cache[tx.tx_id]
            
        # Giả lập việc kiểm tra tính hợp lệ
        self.pipeline_metrics['validation'] += 1
        
        # Kiểm tra xem source và target shard có hợp lệ không
        if tx.source_shard < 0 or tx.source_shard >= len(self.shards) or \
           tx.target_shard < 0 or tx.target_shard >= len(self.shards):
            self.validation_result_cache[tx.tx_id] = False
            return False
        
        # Kiểm tra kích thước giao dịch
        if tx.size <= 0 or tx.size > 10:  # Giới hạn kích thước từ 0-10
            self.validation_result_cache[tx.tx_id] = False
            return False
        
        # Lưu kết quả vào cache
        self.validation_result_cache[tx.tx_id] = True
        return True
    
    def route_transaction(self, tx):
        """Giai đoạn 2: Tính toán đường đi của giao dịch."""
        # Kiểm tra cache
        cache_key = f"{tx.tx_id}_{tx.source_shard}_{tx.target_shard}"
        if cache_key in self.routing_result_cache:
            cached_result = self.routing_result_cache[cache_key]
            tx.route = cached_result['route']
            tx.hops = cached_result['hops']
            tx.latency = cached_result['latency']
            return True
            
        self.pipeline_metrics['routing'] += 1
        
        if tx.is_cross_shard:
            # Tính toán số lượng hop
            tx.hops = max(1, abs(tx.target_shard - tx.source_shard))
            
            # Tạo đường đi
            tx.route = self._generate_route(tx.source_shard, tx.target_shard)
            
            # Tính toán độ trễ dựa trên đường đi và tình trạng tắc nghẽn
            route_congestion = [self.shards[shard].congestion_level for shard in tx.route]
            avg_congestion = sum(route_congestion) / len(route_congestion) if route_congestion else 0
            tx.latency = (10 + (5 * tx.hops)) * (1 + avg_congestion * 2)
        else:
            # Giao dịch trong cùng shard
            tx.hops = 1
            tx.route = [tx.source_shard]
            tx.latency = 5 * (1 + self.shards[tx.source_shard].congestion_level)
        
        # Lưu kết quả vào cache
        self.routing_result_cache[cache_key] = {
            'route': tx.route,
            'hops': tx.hops,
            'latency': tx.latency
        }
        
        return True
    
    def _generate_route(self, source, target):
        """Tạo đường đi tối ưu giữa các shard."""
        # Cache key cho route
        cache_key = f"route_{source}_{target}"
        if cache_key in self.processing_cache:
            return self.processing_cache[cache_key]
            
        if source == target:
            return [source]
        
        route = [source]
        current = source
        
        # Tìm đường đi ngắn nhất (đơn giản hóa)
        while current != target:
            if current < target:
                current += 1
            else:
                current -= 1
            route.append(current)
        
        # Lưu vào cache
        self.processing_cache[cache_key] = route
        return route
    
    def reach_consensus(self, tx):
        """Giai đoạn 3: Đạt đồng thuận giữa các nút trong shard."""
        self.pipeline_metrics['consensus'] += 1
        
        source_shard = self.shards[tx.source_shard]
        target_shard = self.shards[tx.target_shard]
        
        # Tính tỷ lệ nút độc hại
        malicious_node_ratio = (
            len(source_shard.get_malicious_nodes()) / max(1, len(source_shard.nodes)) +
            len(target_shard.get_malicious_nodes()) / max(1, len(target_shard.nodes))
        ) / 2
        
        # Xác suất đạt đồng thuận dựa trên tỷ lệ nút độc hại
        consensus_probability = 1.0 - (malicious_node_ratio * 0.8)
        
        # Kiểm tra có đạt đồng thuận không
        return random.random() < consensus_probability
    
    def execute_transaction(self, tx):
        """Giai đoạn 4: Thực thi giao dịch."""
        self.pipeline_metrics['execution'] += 1
        
        # Cache key cho energy
        cache_key = f"energy_{tx.tx_id}_{tx.is_cross_shard}_{tx.size}_{tx.hops}"
        if cache_key in self.processing_cache:
            tx.energy = self.processing_cache[cache_key]
            return True
        
        # Tính toán chi phí năng lượng
        if tx.is_cross_shard:
            base_energy = 2.0 * tx.size
            hop_energy = 0.5 * tx.hops
            tx.energy = base_energy + hop_energy
        else:
            tx.energy = 1.0 * tx.size
        
        # Lưu vào cache
        self.processing_cache[cache_key] = tx.energy
        
        # Kiểm tra xem giao dịch có thành công hay không
        success_probability = 0.98  # Xác suất thành công cao ở giai đoạn này
        return random.random() < success_probability
    
    def commit_transaction(self, tx):
        """Giai đoạn 5: Hoàn tất giao dịch."""
        self.pipeline_metrics['commit'] += 1
        
        # Đánh dấu giao dịch đã hoàn thành
        tx.mark_completed()
        
        # Tính toàn vẹn dữ liệu
        source_shard = self.shards[tx.source_shard]
        target_shard = self.shards[tx.target_shard]
        
        malicious_node_ratio = (
            len(source_shard.get_malicious_nodes()) / max(1, len(source_shard.nodes)) +
            len(target_shard.get_malicious_nodes()) / max(1, len(target_shard.nodes))
        ) / 2
        
        if malicious_node_ratio > 0:
            tx.data_integrity = max(0.7, 1.0 - (malicious_node_ratio * 0.5))
        
        return True
    
    def process_transaction(self, tx, sim_context=None):
        """Xử lý giao dịch qua toàn bộ quy trình pipeline."""
        # Cache key cho kết quả xử lý
        cache_key = f"process_{tx.tx_id}_{tx.processing_attempts}"
        if cache_key in self.processing_cache:
            return self.processing_cache[cache_key]
            
        # Giai đoạn 1: Kiểm tra tính hợp lệ
        if not self.validate_transaction(tx):
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Giai đoạn 2: Tính toán đường đi
        if not self.route_transaction(tx):
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Giai đoạn 3: Đạt đồng thuận
        if not self.reach_consensus(tx):
            tx.processing_attempts += 1
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Giai đoạn 4: Thực thi giao dịch
        if not self.execute_transaction(tx):
            tx.processing_attempts += 1
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Giai đoạn 5: Hoàn tất giao dịch
        self.commit_transaction(tx)
        
        result = (tx, True)
        self.processing_cache[cache_key] = result
        return result
    
    def process_transactions_batch(self, transactions, sim_context=None):
        """Xử lý song song một nhóm giao dịch qua pipeline."""
        results = []
        
        # Phân nhóm giao dịch theo độ phức tạp để xử lý hiệu quả hơn
        simple_transactions = [tx for tx in transactions if not tx.is_cross_shard]
        complex_transactions = [tx for tx in transactions if tx.is_cross_shard]
        
        with ThreadPoolExecutor(max_workers=self.optimal_workers) as executor:
            # Xử lý các giao dịch đơn giản với ưu tiên cao hơn
            simple_futures = [executor.submit(self.process_transaction, tx, sim_context) for tx in simple_transactions]
            # Xử lý các giao dịch phức tạp
            complex_futures = [executor.submit(self.process_transaction, tx, sim_context) for tx in complex_transactions]
            
            # Thu thập kết quả từ các giao dịch đơn giản
            for future in as_completed(simple_futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Lỗi trong quá trình xử lý pipeline (giao dịch đơn giản): {e}")
            
            # Thu thập kết quả từ các giao dịch phức tạp
            for future in as_completed(complex_futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Lỗi trong quá trình xử lý pipeline (giao dịch phức tạp): {e}")
        
        # Dọn dẹp cache để tránh tràn bộ nhớ
        if len(self.processing_cache) > 10000:
            self.processing_cache = {}
        if len(self.validation_result_cache) > 10000:
            self.validation_result_cache = {}
        if len(self.routing_result_cache) > 10000:
            self.routing_result_cache = {}
            
        return results
    
    def process_shard_transactions(self, shard_transactions, sim_context=None):
        """Xử lý các giao dịch được nhóm theo shard để tối ưu hóa locality."""
        all_results = []
        
        # Tạo danh sách tất cả các giao dịch để xử lý
        all_transactions = []
        for shard_id, transactions in shard_transactions.items():
            if transactions:
                all_transactions.extend(transactions)
        
        # Nếu số lượng giao dịch nhỏ, xử lý một lần
        if len(all_transactions) <= self.optimal_workers * 2:
            return self.process_transactions_batch(all_transactions, sim_context)
        
        # Nếu có nhiều giao dịch, xử lý theo từng shard để tối ưu locality
        with ThreadPoolExecutor(max_workers=min(len(shard_transactions), 8)) as executor:
            # Submit các công việc xử lý từng shard
            future_to_shard = {}
            for shard_id, transactions in shard_transactions.items():
                if transactions:
                    future = executor.submit(self.process_transactions_batch, transactions, sim_context)
                    future_to_shard[future] = shard_id
            
            # Thu thập kết quả
            for future in as_completed(future_to_shard):
                try:
                    shard_results = future.result()
                    all_results.extend(shard_results)
                except Exception as e:
                    shard_id = future_to_shard[future]
                    print(f"Lỗi khi xử lý giao dịch cho shard {shard_id}: {e}")
        
        return all_results
    
    def get_pipeline_metrics(self):
        """Trả về số liệu thống kê về pipeline."""
        return self.pipeline_metrics.copy()
    
    def clear_caches(self):
        """Xóa tất cả cache để giải phóng bộ nhớ."""
        self.processing_cache.clear()
        self.validation_result_cache.clear()
        self.routing_result_cache.clear()
        return True

class Shard:
    __slots__ = ('shard_id', 'nodes', 'congestion_level', 'transactions_queue', 'processed_transactions',
                'blocked_transactions', 'network_stability', 'resource_utilization', 'consensus_difficulty',
                'last_metrics_update', 'historical_congestion', '_non_malicious_nodes_cache', '_malicious_nodes_cache',
                '_last_cache_update')
    
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
        
        # Cache để tối ưu hóa truy vấn
        self._non_malicious_nodes_cache = None
        self._malicious_nodes_cache = None
        self._last_cache_update = 0
        
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
        """Trả về danh sách các nút không độc hại, sử dụng cache để tối ưu."""
        # Kiểm tra xem cache có hợp lệ không
        current_time = time.time()
        if self._non_malicious_nodes_cache is None or current_time - self._last_cache_update > 1.0:
            # Cập nhật cache nếu đã quá hạn hoặc chưa được tạo
            self._non_malicious_nodes_cache = [node for node in self.nodes if not node.is_malicious]
            self._last_cache_update = current_time
        return self._non_malicious_nodes_cache
    
    def get_malicious_nodes(self):
        """Trả về danh sách các nút độc hại, sử dụng cache để tối ưu."""
        # Kiểm tra xem cache có hợp lệ không
        current_time = time.time()
        if self._malicious_nodes_cache is None or current_time - self._last_cache_update > 1.0:
            # Cập nhật cache nếu đã quá hạn hoặc chưa được tạo
            self._malicious_nodes_cache = [node for node in self.nodes if node.is_malicious]
            self._last_cache_update = current_time
        return self._malicious_nodes_cache
    
    def compute_power_distribution(self):
        """Tính toán phân phối sức mạnh xử lý giữa các nút."""
        total_power = sum(node.processing_power for node in self.nodes)
        return [(node.node_id, node.processing_power / total_power) for node in self.nodes]

    def update_congestion(self):
        """Cập nhật mức độ tắc nghẽn của shard dựa trên số lượng giao dịch đang chờ xử lý."""
        # Cập nhật mức độ tắc nghẽn dựa trên số lượng giao dịch trong hàng đợi
        queue_size = len(self.transactions_queue)
        prev_congestion = self.congestion_level
        self.congestion_level = min(1.0, queue_size / 100)  # Tối đa là 1.0
        
        # Thêm vào lịch sử
        self.historical_congestion.append(self.congestion_level)
        
        # Giới hạn kích thước lịch sử để tránh sử dụng quá nhiều bộ nhớ
        if len(self.historical_congestion) > 100:
            self.historical_congestion.pop(0)
        
        # Cập nhật độ ổn định mạng dựa trên mức độ biến động tắc nghẽn
        if prev_congestion > 0:
            stability_factor = 1.0 - abs(self.congestion_level - prev_congestion) / prev_congestion
            self.network_stability = 0.9 * self.network_stability + 0.1 * stability_factor
        
        # Cập nhật mức sử dụng tài nguyên
        self.resource_utilization = 0.8 * self.resource_utilization + 0.2 * self.congestion_level
        
        # Đánh dấu cache đã hết hạn khi cập nhật trạng thái shard
        self._last_cache_update = 0
        
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
        
    def add_transaction_to_queue(self, tx):
        """Thêm giao dịch vào hàng đợi theo thứ tự ưu tiên."""
        self.transactions_queue.append(tx)
        # Sắp xếp hàng đợi theo độ ưu tiên để các giao dịch quan trọng hơn được xử lý trước
        self.transactions_queue.sort(key=lambda tx: tx.priority, reverse=True)
        return True
    
    def clear_old_data(self, max_processed=1000, max_blocked=500):
        """Xóa dữ liệu cũ để tránh tiêu tốn quá nhiều bộ nhớ."""
        # Giữ lại tối đa số lượng giao dịch đã xử lý
        if len(self.processed_transactions) > max_processed:
            self.processed_transactions = self.processed_transactions[-max_processed:]
        
        # Giữ lại tối đa số lượng giao dịch bị chặn
        if len(self.blocked_transactions) > max_blocked:
            self.blocked_transactions = self.blocked_transactions[-max_blocked:]
        
        return True
        
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
    
    # Kiểm tra thành công/thất bại
    success = random.random() < success_probability
    
    if success:
        # Cập nhật thông tin giao dịch thành công
        tx.mark_completed()
        
        # Cập nhật tính toàn vẹn dữ liệu nếu có nút độc hại
        if malicious_node_ratio > 0:
            tx.data_integrity = max(0.7, 1.0 - (malicious_node_ratio * 0.5))
    else:
        # Giao dịch thất bại, tăng số lần thử
        tx.processing_attempts += 1
        
    # Trả về tuple chứa transaction và trạng thái thành công
    return (tx, success)

class LargeScaleBlockchainSimulation:
    __slots__ = ('num_shards', 'nodes_per_shard', 'malicious_percentage', 'attack_scenario',
                'attack_types', 'max_workers', 'shards', 'transactions', 'processed_transactions',
                'current_step', 'metrics_history', 'transaction_pipeline', 'total_processed_tx',
                'total_cross_shard_tx', 'avg_latency', 'avg_energy', 'pipeline_stats',
                'tx_counter')
    
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
        self.attack_types = []
        self.max_workers = max_workers
        
        if attack_scenario:
            if attack_scenario == 'random':
                self.attack_types = ['eclipse', 'sybil', '51_percent', 'selfish_mining']
            elif attack_scenario == 'eclipse':
                self.attack_types = ['eclipse']
            elif attack_scenario == 'sybil':
                self.attack_types = ['sybil']
            elif attack_scenario == '51_percent':
                self.attack_types = ['51_percent']
            elif attack_scenario == 'selfish_mining':
                self.attack_types = ['selfish_mining']
        
        # Khởi tạo blockchain
        self.shards = []
        self._initialize_blockchain()
        
        # Tạo kết nối mạng
        self._create_network_connections()
        
        # Thiết lập tấn công nếu được chỉ định
        if 'eclipse' in self.attack_types:
            self._setup_eclipse_attack()
        
        # Khởi tạo các biến theo dõi
        self.transactions = []
        self.processed_transactions = []
        self.current_step = 0
        self.tx_counter = 0  # Bộ đếm ID cho giao dịch
        
        # Lưu trữ các chỉ số hiệu suất
        self.metrics_history = {
            'throughput': [],
            'latency': [],
            'energy': [],
            'security': [],
            'cross_shard_ratio': [],
            'transaction_success_rate': [],
            'network_stability': [],
            'resource_utilization': [],
            'consensus_efficiency': [],
            'shard_health': [],
            'avg_hops': [],
            'network_resilience': [],
            'avg_block_size': [],
            'network_partition_events': []
        }
        
        # Khởi tạo pipeline xử lý giao dịch
        self.transaction_pipeline = TransactionPipeline(self.shards, self.max_workers)
        
        # Các biến theo dõi bổ sung
        self.total_processed_tx = 0
        self.total_cross_shard_tx = 0
        self.avg_latency = 0
        self.avg_energy = 0
        self.pipeline_stats = {}
    
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
        
        # Tạo trước danh sách cặp nguồn/đích để giảm tải xử lý
        pairs = []
        cross_shard_count = int(num_transactions * 0.3)  # 30% là giao dịch xuyên shard
        same_shard_count = num_transactions - cross_shard_count
        
        # Tạo các cặp trong cùng shard
        for _ in range(same_shard_count):
            shard_id = random.randint(0, self.num_shards - 1)
            pairs.append((shard_id, shard_id))
        
        # Tạo các cặp xuyên shard
        for _ in range(cross_shard_count):
            source_shard = random.randint(0, self.num_shards - 1)
            target_shard = random.randint(0, self.num_shards - 1)
            while target_shard == source_shard:
                target_shard = random.randint(0, self.num_shards - 1)
            pairs.append((source_shard, target_shard))
        
        # Trộn các cặp để tránh xử lý theo đợt
        random.shuffle(pairs)
        
        # Tạo các giao dịch từ các cặp đã tạo
        for source_shard, target_shard in pairs:
            # Tạo giao dịch với size ngẫu nhiên
            tx = Transaction(
                tx_id=f"tx_{self.tx_counter}",
                source_shard=source_shard,
                target_shard=target_shard,
                size=random.uniform(0.5, 2.0)
            )
            self.tx_counter += 1
            new_transactions.append(tx)
            
            # Thêm vào hàng đợi của shard nguồn - sử dụng phương thức mới đã tối ưu
            self.shards[source_shard].add_transaction_to_queue(tx)
        
        # Thêm vào danh sách giao dịch tổng thể
        self.transactions.extend(new_transactions)
        
        # Định kỳ dọn dẹp cache của pipeline để tránh sử dụng quá nhiều bộ nhớ
        if self.tx_counter % 1000 == 0:
            self.transaction_pipeline.clear_caches()
        
        return new_transactions
    
    def _process_transactions(self):
        processed_count = 0
        blocked_count = 0
        total_hops = 0
        
        # Cập nhật mức độ tắc nghẽn cho tất cả các shard
        for shard in self.shards:
            shard.update_congestion()
        
        # Chuẩn bị context cho xử lý đa luồng
        sim_context = {
            'shards': self.shards,
            'attack_types': [self.attack_scenario] if self.attack_scenario else []
        }
        
        # Thu thập các giao dịch cần xử lý từ tất cả các shard và phân nhóm theo shard
        shard_transactions = {i: [] for i in range(len(self.shards))}
        
        # Xử lý hiệu quả hơn bằng cách giới hạn số lượng giao dịch được xử lý mỗi shard
        max_tx_per_shard = 20
        
        for idx, shard in enumerate(self.shards):
            # Lấy số lượng nút không độc hại
            non_malicious_nodes = shard.get_non_malicious_nodes()
            
            # Nếu không có đủ nút xác thực, bỏ qua shard này
            if len(non_malicious_nodes) < self.nodes_per_shard / 2:
                continue
                
            # Lấy các giao dịch ưu tiên cao nhất (đã được sắp xếp trong add_transaction_to_queue)
            transactions_to_process = shard.transactions_queue[:max_tx_per_shard]
            
            # Thêm vào danh sách giao dịch cần xử lý
            shard_transactions[idx].extend(transactions_to_process)
        
        # Sử dụng pipeline để xử lý song song theo từng giai đoạn
        results = self.transaction_pipeline.process_shard_transactions(shard_transactions, sim_context)
        
        # Lưu trữ pipeline stats
        self.pipeline_stats = self.transaction_pipeline.get_pipeline_metrics()
        
        # Theo dõi nút được phân bổ công việc
        node_work_allocation = {}
        
        # Xử lý kết quả
        for tx, success in results:
            # Xóa khỏi hàng đợi của shard nguồn - tối ưu hóa bằng cách tìm kiếm hiệu quả
            source_shard = self.shards[tx.source_shard]
            try:
                source_shard.transactions_queue.remove(tx)
            except ValueError:
                # Giao dịch có thể đã bị xóa bởi một xử lý khác
                pass
            
            if success:
                # Giao dịch thành công
                self.processed_transactions.append(tx)
                source_shard.processed_transactions.append(tx)
                processed_count += 1
                
                # Cập nhật thống kê
                if tx.is_cross_shard:
                    self.total_cross_shard_tx += 1
                    total_hops += tx.hops
                
                self.total_processed_tx += 1
                
                # Cập nhật số lượng giao dịch đã xử lý cho các nút
                valid_nodes = source_shard.get_non_malicious_nodes()
                if valid_nodes:
                    # Chọn nút ít bận rộn nhất để phân phối công việc đồng đều
                    selected_node = min(valid_nodes, key=lambda node: node_work_allocation.get(node.node_id, 0))
                    selected_node.transactions_processed += 1
                    
                    # Cập nhật phân bổ công việc
                    node_work_allocation[selected_node.node_id] = node_work_allocation.get(selected_node.node_id, 0) + 1
            else:
                # Giao dịch thất bại
                blocked_count += 1
                source_shard.blocked_transactions.append(tx)
        
        # Tính trung bình số hop
        if processed_count > 0:
            avg_hops = total_hops / max(1, self.total_cross_shard_tx)
        else:
            avg_hops = 0
            
        # Định kỳ xóa dữ liệu cũ để tiết kiệm bộ nhớ
        if self.current_step % 10 == 0:
            for shard in self.shards:
                shard.clear_old_data()
        
        # Cập nhật độ tin cậy của các nút
        for shard in self.shards:
            for node in shard.nodes:
                # Tính tỷ lệ thành công dựa trên số giao dịch đã xử lý
                success_rate = min(1.0, node.transactions_processed / max(1, self.total_processed_tx / len(self.shards)))
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
            self.metrics_history[key].append(value)
    
    def run_simulation(self, num_steps=1000, transactions_per_step=50):
        print(f"Bat dau mo phong voi {num_steps} buoc, {transactions_per_step} giao dich/buoc")
        
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
                print(f"\nBuoc {step + 1}/{num_steps}:")
                print(f"  Throughput: {metrics['throughput']:.2f} tx/s")
                print(f"  Do tre trung binh: {metrics['latency']:.2f} ms")
                print(f"  Ty le thanh cong: {metrics['transaction_success_rate']:.2f}")
                print(f"  Do on dinh mang: {metrics['network_stability']:.2f}")
                print(f"  Suc khoe shard: {metrics['shard_health']:.2f}")
                print(f"  Giao dich da xu ly: {processed}, bi chan: {blocked}")
        
        print("\nMo phong hoan tat!")
        return self.metrics_history
    
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
            ax1.plot(self.metrics_history['throughput'], color=colors[0], linewidth=2)
            ax1.set_title('Throughput (tx/s)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Buoc')
            ax1.set_ylabel('tx/s')
            ax1.grid(True, alpha=0.3)
            ax1.fill_between(range(len(self.metrics_history['throughput'])), 
                             self.metrics_history['throughput'], 
                             alpha=0.3, 
                             color=colors[0])
            
            # 2. Latency
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.metrics_history['latency'], color=colors[1], linewidth=2)
            ax2.set_title('Do tre (ms)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Buoc')
            ax2.set_ylabel('ms')
            ax2.grid(True, alpha=0.3)
            ax2.fill_between(range(len(self.metrics_history['latency'])), 
                             self.metrics_history['latency'], 
                             alpha=0.3, 
                             color=colors[1])
            
            # 3. Energy
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(self.metrics_history['energy'], color=colors[2], linewidth=2)
            ax3.set_title('Tieu thu nang luong', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Buoc')
            ax3.set_ylabel('Don vi nang luong')
            ax3.grid(True, alpha=0.3)
            ax3.fill_between(range(len(self.metrics_history['energy'])), 
                             self.metrics_history['energy'], 
                             alpha=0.3, 
                             color=colors[2])
            
            # 4. Security
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.plot(self.metrics_history['security'], color=colors[3], linewidth=2)
            ax4.set_title('Diem bao mat', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Buoc')
            ax4.set_ylabel('Diem (0-1)')
            ax4.grid(True, alpha=0.3)
            ax4.fill_between(range(len(self.metrics_history['security'])), 
                             self.metrics_history['security'], 
                             alpha=0.3, 
                             color=colors[3])
            
            # 5. Cross-shard ratio
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.plot(self.metrics_history['cross_shard_ratio'], color=colors[4], linewidth=2)
            ax5.set_title('Ty le giao dich xuyen shard', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Buoc')
            ax5.set_ylabel('Ty le')
            ax5.grid(True, alpha=0.3)
            ax5.fill_between(range(len(self.metrics_history['cross_shard_ratio'])), 
                             self.metrics_history['cross_shard_ratio'], 
                             alpha=0.3, 
                             color=colors[4])
            
            # 6. Transaction success rate
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(self.metrics_history['transaction_success_rate'], color=colors[5], linewidth=2)
            ax6.set_title('Ty le giao dich thanh cong', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Buoc')
            ax6.set_ylabel('Ty le')
            ax6.grid(True, alpha=0.3)
            ax6.fill_between(range(len(self.metrics_history['transaction_success_rate'])), 
                             self.metrics_history['transaction_success_rate'], 
                             alpha=0.3, 
                             color=colors[5])
            
            # 7. Network stability
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(self.metrics_history['network_stability'], color=colors[0], linewidth=2)
            ax7.set_title('Do on dinh mang', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Buoc')
            ax7.set_ylabel('Diem (0-1)')
            ax7.grid(True, alpha=0.3)
            ax7.fill_between(range(len(self.metrics_history['network_stability'])), 
                             self.metrics_history['network_stability'], 
                             alpha=0.3, 
                             color=colors[0])
            
            # 8. Resource utilization
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(self.metrics_history['resource_utilization'], color=colors[1], linewidth=2)
            ax8.set_title('Muc su dung tai nguyen', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Buoc')
            ax8.set_ylabel('Ty le')
            ax8.grid(True, alpha=0.3)
            ax8.fill_between(range(len(self.metrics_history['resource_utilization'])), 
                             self.metrics_history['resource_utilization'], 
                             alpha=0.3, 
                             color=colors[1])
            
            # 9. Consensus efficiency
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.plot(self.metrics_history['consensus_efficiency'], color=colors[2], linewidth=2)
            ax9.set_title('Hieu qua dong thuan', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Buoc')
            ax9.set_ylabel('Diem (0-1)')
            ax9.grid(True, alpha=0.3)
            ax9.fill_between(range(len(self.metrics_history['consensus_efficiency'])), 
                             self.metrics_history['consensus_efficiency'], 
                             alpha=0.3, 
                             color=colors[2])
            
            # 10. Shard health
            ax10 = fig.add_subplot(gs[3, 0])
            ax10.plot(self.metrics_history['shard_health'], color=colors[3], linewidth=2)
            ax10.set_title('Suc khoe shard', fontsize=14, fontweight='bold')
            ax10.set_xlabel('Buoc')
            ax10.set_ylabel('Diem (0-1)')
            ax10.grid(True, alpha=0.3)
            ax10.fill_between(range(len(self.metrics_history['shard_health'])), 
                             self.metrics_history['shard_health'], 
                             alpha=0.3, 
                             color=colors[3])
            
            # 11. Network resilience
            ax11 = fig.add_subplot(gs[3, 1])
            ax11.plot(self.metrics_history['network_resilience'], color=colors[4], linewidth=2)
            ax11.set_title('Kha nang phuc hoi', fontsize=14, fontweight='bold')
            ax11.set_xlabel('Buoc')
            ax11.set_ylabel('Diem (0-1)')
            ax11.grid(True, alpha=0.3)
            ax11.fill_between(range(len(self.metrics_history['network_resilience'])), 
                             self.metrics_history['network_resilience'], 
                             alpha=0.3, 
                             color=colors[4])
            
            # 12. Average block size
            ax12 = fig.add_subplot(gs[3, 2])
            ax12.plot(self.metrics_history['avg_block_size'], color=colors[5], linewidth=2)
            ax12.set_title('Kich thuoc khoi trung binh', fontsize=14, fontweight='bold')
            ax12.set_xlabel('Buoc')
            ax12.set_ylabel('Kich thuoc')
            ax12.grid(True, alpha=0.3)
            ax12.fill_between(range(len(self.metrics_history['avg_block_size'])), 
                             self.metrics_history['avg_block_size'], 
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(save_dir, f"detailed_metrics_{self.num_shards}shards_{self.nodes_per_shard}nodes_{timestamp}.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Da luu bieu do chi tiet tai: {save_file}")
            
            plt.close(fig)
            
            # Tạo biểu đồ radar
            self._plot_radar_chart(save_dir)
            
            # Tạo biểu đồ heatmap cho mức độ tắc nghẽn
            self._plot_congestion_heatmap(save_dir)
    
    def _plot_congestion_heatmap(self, save_dir=None):
        # Kiểm tra xem có dữ liệu tắc nghẽn không
        has_congestion_data = True
        for shard in self.shards:
            if not hasattr(shard, 'historical_congestion') or not shard.historical_congestion:
                has_congestion_data = False
                break
        
        if not has_congestion_data:
            return
        
        # Chuẩn bị dữ liệu cho heatmap
        # Lấy lịch sử tắc nghẽn từ mỗi shard
        congestion_data = []
        for shard in self.shards:
            # Đảm bảo rằng tất cả các shard có cùng số lượng điểm dữ liệu
            # bằng cách lấy 100 điểm cuối cùng
            if len(shard.historical_congestion) > 100:
                congestion_data.append(shard.historical_congestion[-100:])
            else:
                # Nếu ít hơn 100 điểm, thêm 0 vào đầu để đủ 100 điểm
                padding = [0] * (100 - len(shard.historical_congestion))
                congestion_data.append(padding + shard.historical_congestion)
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Tạo heatmap
        congestion_array = np.array(congestion_data)
        
        # Đổi thứ tự để shard 0 ở dưới cùng
        congestion_array = np.flip(congestion_array, axis=0)
        
        # Tùy chỉnh bảng màu
        cmap = LinearSegmentedColormap.from_list("custom", ["#1a9850", "#ffffbf", "#d73027"], N=256)
        
        # Vẽ heatmap
        im = ax.imshow(congestion_array, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        
        # Thêm colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Muc do tac nghen', rotation=270, labelpad=15)
        
        # Đặt nhãn
        ax.set_xlabel('Buoc thoi gian')
        ax.set_ylabel('Shard ID')
        
        # Đặt các giá trị trục y để hiển thị ID shard
        shard_ids = [f'Shard {self.num_shards - i - 1}' for i in range(self.num_shards)]
        ax.set_yticks(np.arange(len(shard_ids)))
        ax.set_yticklabels(shard_ids)
        
        # Giảm số lượng nhãn trục x để tránh chồng chéo
        step = max(1, len(congestion_array[0]) // 10)
        ax.set_xticks(np.arange(0, len(congestion_array[0]), step))
        ax.set_xticklabels(np.arange(0, len(congestion_array[0]), step))
        
        # Tiêu đề
        plt.title(f'Phan tich tac nghen shard {self.num_shards} Shards, {self.nodes_per_shard} Nodes/Shard', fontsize=14)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        if save_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(save_dir, f"congestion_heatmap_{self.num_shards}shards_{self.nodes_per_shard}nodes_{timestamp}.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Da luu bieu do muc do tac nghen tai: {save_file}")
        
        plt.close(fig)
    
    def generate_report(self, save_dir=None):
        # Tính toán metrics cuối cùng
        metrics = self._calculate_metrics()
        
        # Tạo thư mục nếu không tồn tại
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Tạo báo cáo
        report = {
            'configuration': {
                'num_shards': self.num_shards,
                'nodes_per_shard': self.nodes_per_shard,
                'malicious_percentage': self.malicious_percentage,
                'attack_scenario': self.attack_scenario,
                'attack_types': self.attack_types
            },
            'performance': {
                'throughput': metrics['throughput'],
                'latency': metrics['latency'],
                'energy': metrics['energy'],
                'security': metrics['security'],
                'cross_shard_ratio': metrics['cross_shard_ratio'],
                'transaction_success_rate': metrics['transaction_success_rate'],
                'network_stability': metrics['network_stability'],
                'resource_utilization': metrics['resource_utilization'],
                'consensus_efficiency': metrics['consensus_efficiency'],
                'shard_health': metrics['shard_health'],
                'network_resilience': metrics['network_resilience']
            },
            'shard_statistics': {},
            'pipeline_stats': self.pipeline_stats
        }
        
        # Thu thập thống kê theo shard
        for idx, shard in enumerate(self.shards):
            report['shard_statistics'][f'shard_{idx}'] = {
                'num_nodes': len(shard.nodes),
                'malicious_nodes': len(shard.get_malicious_nodes()),
                'congestion_level': shard.congestion_level,
                'network_stability': shard.network_stability,
                'resource_utilization': shard.resource_utilization,
                'transactions_processed': len(shard.processed_transactions),
                'transactions_blocked': len(shard.blocked_transactions),
                'health_score': shard.get_shard_health()
            }
        
        # Thu thập thống kê về pipeline
        pipeline_efficiency = self._calculate_pipeline_efficiency()
        report['pipeline_performance'] = pipeline_efficiency
            
        # In báo cáo tóm tắt
        self._print_summary_report(report)
        
        # Lưu kết quả dưới dạng JSON
        if save_dir:
            report_file = os.path.join(save_dir, 'simulation_report.json')
            with open(report_file, 'w') as f:
                import json
                json.dump(report, f, indent=4)
            
            # Lưu tất cả các chỉ số theo thời gian
            self._save_metrics_json(os.path.join(save_dir, 'metrics_history.json'))
            
            # Vẽ biểu đồ và lưu
            self.plot_metrics(save_dir)
            self._plot_radar_chart(save_dir)
            self._plot_congestion_heatmap(save_dir)
            
            print(f"\nBáo cáo đã được lưu vào: {save_dir}")
        
        return report
    
    def _calculate_pipeline_efficiency(self):
        """Tính toán hiệu suất xử lý pipeline."""
        if not self.pipeline_stats:
            return {
                'pipeline_throughput': 0,
                'stage_efficiency': {},
                'bottleneck_stage': 'N/A',
                'parallel_efficiency': 0
            }
        
        # Tính tổng số giao dịch đã xử lý qua mỗi giai đoạn
        total_validation = self.pipeline_stats.get('validation', 0)
        total_routing = self.pipeline_stats.get('routing', 0)
        total_consensus = self.pipeline_stats.get('consensus', 0)
        total_execution = self.pipeline_stats.get('execution', 0)
        total_commit = self.pipeline_stats.get('commit', 0)
        
        # Tính hiệu suất từng giai đoạn (tỷ lệ giao dịch đi qua mỗi giai đoạn)
        stage_efficiency = {}
        if total_validation > 0:
            stage_efficiency['validation'] = 1.0
            stage_efficiency['routing'] = total_routing / total_validation if total_validation else 0
            stage_efficiency['consensus'] = total_consensus / total_validation if total_validation else 0
            stage_efficiency['execution'] = total_execution / total_validation if total_validation else 0
            stage_efficiency['commit'] = total_commit / total_validation if total_validation else 0
        
        # Tìm giai đoạn bottleneck
        if stage_efficiency:
            bottleneck_stage = min(stage_efficiency, key=stage_efficiency.get)
        else:
            bottleneck_stage = 'N/A'
        
        # Tính hiệu suất song song
        if bottleneck_stage != 'N/A' and stage_efficiency[bottleneck_stage] > 0:
            # Hiệu quả song song = Tổng thời gian tuần tự / (Số giai đoạn * Thời gian dài nhất)
            # Giả định mỗi giai đoạn mất thời gian bằng nhau
            parallel_efficiency = 1.0 / (5 * (1.0 - stage_efficiency[bottleneck_stage] + 0.2))
            parallel_efficiency = min(1.0, parallel_efficiency)  # Không vượt quá 1.0
        else:
            parallel_efficiency = 0
        
        # Tính throughput pipeline
        pipeline_throughput = total_commit / max(1, self.current_step)
        
        return {
            'pipeline_throughput': pipeline_throughput,
            'stage_efficiency': stage_efficiency,
            'bottleneck_stage': bottleneck_stage,
            'parallel_efficiency': parallel_efficiency,
            'total_transactions': {
                'validation': total_validation,
                'routing': total_routing,
                'consensus': total_consensus,
                'execution': total_execution,
                'commit': total_commit
            }
        }
    
    def _print_summary_report(self, report):
        """In báo cáo tóm tắt."""
        print("\n" + "="*80)
        print(" "*30 + "BAO CAO MO PHONG")
        print("="*80)
        
        print("\nCAU HINH:")
        print(f"- So luong shard: {report['configuration']['num_shards']}")
        print(f"- So nut moi shard: {report['configuration']['nodes_per_shard']}")
        print(f"- Ty le nut doc hai: {report['configuration']['malicious_percentage']}%")
        print(f"- Kich ban tan cong: {report['configuration']['attack_scenario'] or 'None'}")
        
        print("\nHIEU SUAT:")
        print(f"- Throughput: {report['performance']['throughput']:.2f} tx/s")
        print(f"- Do tre: {report['performance']['latency']:.2f} ms")
        print(f"- Nang luong: {report['performance']['energy']:.2f} don vi")
        print(f"- Diem bao mat: {report['performance']['security']:.2f}")
        print(f"- Ty le giao dich xuyen shard: {report['performance']['cross_shard_ratio']:.2f}")
        print(f"- Ty le giao dich thanh cong: {report['performance']['transaction_success_rate']:.2f}")
        
        print("\nHIEU SUAT PIPELINE:")
        if 'pipeline_performance' in report:
            pp = report['pipeline_performance']
            print(f"- Pipeline throughput: {pp['pipeline_throughput']:.2f} tx/s")
            print(f"- Bottleneck stage: {pp['bottleneck_stage']}")
            print(f"- Hieu suat xu ly song song: {pp['parallel_efficiency']:.2f}")
            
            if 'stage_efficiency' in pp and pp['stage_efficiency']:
                print("\n  Hieu suat theo giai doan:")
                for stage, eff in pp['stage_efficiency'].items():
                    print(f"  - {stage}: {eff:.2f}")
            
            if 'total_transactions' in pp:
                print("\n  So giao dich xu ly theo giai doan:")
                for stage, count in pp['total_transactions'].items():
                    print(f"  - {stage}: {count}")
        
        print("\nTHONG KE THEO SHARD:")
        # Chỉ hiển thị 3 shard đầu tiên để giữ báo cáo ngắn gọn
        displayed_shards = list(report['shard_statistics'].keys())[:3]
        for shard_id in displayed_shards:
            shard_stats = report['shard_statistics'][shard_id]
            print(f"\n{shard_id.upper()}:")
            print(f"- So nut: {shard_stats['num_nodes']}")
            print(f"- Nut doc hai: {shard_stats['malicious_nodes']}")
            print(f"- Muc do tac nghen: {shard_stats['congestion_level']:.2f}")
            print(f"- Giao dich da xu ly: {shard_stats['transactions_processed']}")
            print(f"- Giao dich bi chan: {shard_stats['transactions_blocked']}")
            print(f"- Diem suc khoe: {shard_stats['health_score']:.2f}")
        
        if len(report['shard_statistics']) > 3:
            print(f"\n... va {len(report['shard_statistics']) - 3} shard khac")
        
        print("\n" + "="*80)
    
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
            "metrics": self.metrics_history,
            "processed_transactions": len(self.processed_transactions)
        }
        
        # Lưu file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Da luu metrics vao file: {filename}")
    
    def parallel_save_results(self, save_dir):
        """Lưu tất cả kết quả song song sử dụng đa luồng."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_suffix = f"_{self.attack_scenario}" if self.attack_scenario else ""
        base_filename = f"{self.num_shards}shards_{self.nodes_per_shard}nodes{attack_suffix}_{timestamp}"
        
        # Danh sách các công việc cần thực hiện
        tasks = [
            {
                "name": "metrics_json",
                "file": os.path.join(save_dir, f"metrics_{base_filename}.json"),
                "func": self._save_metrics_json
            },
            {
                "name": "report_json",
                "file": os.path.join(save_dir, f"report_{base_filename}.json"),
                "func": self._save_report_json
            },
            {
                "name": "transaction_stats",
                "file": os.path.join(save_dir, f"transactions_{base_filename}.json"),
                "func": self._save_transaction_stats
            },
            {
                "name": "node_stats",
                "file": os.path.join(save_dir, f"nodes_{base_filename}.json"),
                "func": self._save_node_stats
            }
        ]
        
        # Thực hiện song song các tác vụ lưu
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for task in tasks:
                futures.append(executor.submit(task["func"], task["file"]))
            
            # Chờ hoàn thành
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Loi khi luu ket qua: {e}")
        
        # Tạo biểu đồ (không chạy song song vì matplotlib không thread-safe)
        self.plot_metrics(save_dir)
        self._plot_radar_chart(save_dir)
        self._plot_congestion_heatmap(save_dir)
        
        print(f"\nDa luu day du ket qua vao: {save_dir}")
        
        return os.path.join(save_dir, f"report_{base_filename}.json")
    
    def _save_report_json(self, filename):
        """Lưu report dạng JSON."""
        import json
        
        # Tạo report
        report = self.generate_report(None)  # Tạo report nhưng không lưu và vẽ biểu đồ
        
        # Lưu file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Da luu report vao file: {filename}")
    
    def _save_transaction_stats(self, filename):
        """Lưu thống kê về giao dịch."""
        import json
        
        # Tính toán thống kê
        processed_count = len(self.processed_transactions)
        cross_shard_count = sum(1 for tx in self.processed_transactions if tx.is_cross_shard)
        avg_latency = sum(tx.latency for tx in self.processed_transactions) / max(1, processed_count)
        avg_energy = sum(tx.energy for tx in self.processed_transactions) / max(1, processed_count)
        avg_hops = sum(tx.hops for tx in self.processed_transactions) / max(1, processed_count)
        
        # Phân phối độ trễ
        latency_distribution = {}
        for tx in self.processed_transactions:
            latency_range = int(tx.latency / 10) * 10  # Làm tròn đến 10ms gần nhất
            latency_distribution[latency_range] = latency_distribution.get(latency_range, 0) + 1
        
        # Dữ liệu thống kê
        data = {
            "total_transactions": len(self.transactions),
            "processed_transactions": processed_count,
            "cross_shard_transactions": cross_shard_count,
            "cross_shard_ratio": cross_shard_count / max(1, processed_count),
            "avg_latency": avg_latency,
            "avg_energy": avg_energy,
            "avg_hops": avg_hops,
            "latency_distribution": {str(k): v for k, v in sorted(latency_distribution.items())}
        }
        
        # Lưu file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Da luu thong ke giao dich vao file: {filename}")
    
    def _save_node_stats(self, filename):
        """Lưu thống kê về các nút."""
        import json
        
        # Tính toán thống kê
        node_stats = []
        for shard in self.shards:
            for node in shard.nodes:
                node_stats.append({
                    "node_id": node.node_id,
                    "shard_id": node.shard_id,
                    "is_malicious": node.is_malicious,
                    "attack_type": node.attack_type,
                    "trust_score": node.trust_score,
                    "processing_power": node.processing_power,
                    "transactions_processed": node.transactions_processed,
                    "uptime": node.uptime,
                    "energy_efficiency": node.energy_efficiency
                })
        
        # Lưu file
        with open(filename, 'w') as f:
            json.dump(node_stats, f, indent=2)
            
        print(f"Da luu thong ke nut vao file: {filename}")

    def _plot_radar_chart(self, save_dir=None):
        """Vẽ biểu đồ radar cho các chỉ số hiệu suất."""
        if not save_dir:
            return
            
        # Lấy chỉ số từ phiên mô phỏng
        metrics = self.metrics_history
        
        if not metrics or not all(key in metrics for key in ['throughput', 'latency', 'energy', 'security', 'cross_shard_ratio', 'transaction_success_rate']):
            return
            
        # Lấy giá trị cuối cùng từ mỗi metric
        categories = ['Throughput', 'Do tre', 'Nang luong',
                 'Bao mat', 'Giao dich xuyen shard', 
                 'Ty le thanh cong', 'On dinh mang',
                 'Su dung tai nguyen', 'Dong thuan', 
                 'Suc khoe shard']
        
        # Chuẩn hóa giá trị
        values = []
        if metrics['throughput']:
            values.append(min(1.0, metrics['throughput'][-1] / 50))  # Giả sử tối đa 50 tx/s là 1.0
        else:
            values.append(0)
            
        if metrics['latency']:
            values.append(max(0, 1.0 - (metrics['latency'][-1] / 50)))  # Càng thấp càng tốt
        else:
            values.append(0)
            
        if metrics['energy']:
            values.append(max(0, 1.0 - (metrics['energy'][-1] / 5.0)))  # Càng thấp càng tốt
        else:
            values.append(0)
            
        for key in ['security', 'cross_shard_ratio', 'transaction_success_rate', 'network_stability']:
            if metrics[key]:
                values.append(metrics[key][-1])
            else:
                values.append(0)
        
        if metrics['resource_utilization']:
            values.append(max(0, 1.0 - metrics['resource_utilization'][-1]))  # Càng thấp càng tốt
        else:
            values.append(0)
            
        if metrics['consensus_efficiency']:
            values.append(metrics['consensus_efficiency'][-1])
        else:
            values.append(0)
            
        if metrics['shard_health']:
            values.append(metrics['shard_health'][-1])
        else:
            values.append(0)
        
        # Tạo biểu đồ
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Số lượng các danh mục
        N = len(categories)
        
        # Góc cho mỗi danh mục (chia đều 360 độ)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Khép kín 
        
        # Thêm giá trị cuối cùng để khép kín biểu đồ
        values += values[:1]
        
        # Vẽ biểu đồ
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
        ax.fill(angles, values, color='#1f77b4', alpha=0.4)
        
        # Đặt các nhãn
        plt.xticks(angles[:-1], categories, size=12)
        
        # Tạo nhãn cấp độ (0.2, 0.4, 0.6, 0.8, 1.0)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                  color="grey", size=10)
        plt.ylim(0, 1)
        
        # Tiêu đề
        plt.title(f'Blockchain Performance Radar - {self.num_shards} Shards', 
                 size=15, y=1.05)
        
        # Lưu biểu đồ
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(save_dir, f"radar_chart_{self.num_shards}shards_{self.nodes_per_shard}nodes_{timestamp}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Da luu bieu do radar tai: {save_file}")
        
        plt.close(fig)
    
    def save_final_metrics(self, result_subdir):
        """Lưu và trả về performance metrics cuối cùng"""
        print(f"\nHoan tat! Ket qua duoc luu tai: {result_subdir}")
        
        # Trả về performance metrics cuối cùng
        final_metrics = self._calculate_metrics()
        return final_metrics

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="QTrust Large Scale Blockchain Simulation")
    
    # Tham số mô phỏng
    parser.add_argument("--num-shards", type=int, default=10, help="So luong shards (mac dinh: 10)")
    parser.add_argument("--nodes-per-shard", type=int, default=20, help="So nut moi shard (mac dinh: 20)")
    parser.add_argument("--malicious-percentage", type=float, default=10, help="Ty le phan tram nut doc hai (mac dinh: 10%%)")
    parser.add_argument("--attack-scenario", type=str, choices=["51_percent", "sybil", "eclipse", "selfish_mining", "random", None], 
                       default=None, help="Kich ban tan cong")
    parser.add_argument("--steps", type=int, default=1000, help="So buoc mo phong (mac dinh: 1000)")
    parser.add_argument("--transactions-per-step", type=int, default=50, help="So giao dich moi buoc (mac dinh: 50)")
    parser.add_argument("--max-workers", type=int, default=None, help="So worker toi da cho xu ly da luong")
    parser.add_argument("--save-dir", type=str, default="results", help="Thu muc luu ket qua (mac dinh: 'results')")
    parser.add_argument("--skip-plots", action="store_true", help="Bo qua ve bieu do (chi luu du lieu JSON)")
    parser.add_argument("--parallel-save", action="store_true", help="Luu ket qua song song su dung da luong")
    
    args = parser.parse_args()
    
    # In cấu hình
    print("\nQTrust Large Scale Blockchain Simulation")
    print("=======================================")
    print(f"So luong shards: {args.num_shards}")
    print(f"So nut moi shard: {args.nodes_per_shard}")
    print(f"Ty le nut doc hai: {args.malicious_percentage}%")
    print(f"Kich ban tan cong: {args.attack_scenario or 'Khong co tan cong'}")
    print(f"So buoc mo phong: {args.steps}")
    print(f"So giao dich moi buoc: {args.transactions_per_step}")
    if args.max_workers:
        print(f"So worker toi da: {args.max_workers}")
    print("=======================================\n")
    
    # Tạo thư mục lưu kết quả nếu chưa tồn tại
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Khởi tạo và chạy mô phỏng
    start_time = time.time()
    
    simulation = LargeScaleBlockchainSimulation(
        num_shards=args.num_shards,
        nodes_per_shard=args.nodes_per_shard,
        malicious_percentage=args.malicious_percentage,
        attack_scenario=args.attack_scenario,
        max_workers=args.max_workers
    )
    
    print(f"Khoi tao blockchain voi {args.num_shards} shard, moi shard co {args.nodes_per_shard} nut")
    print(f"Ty le nut doc hai: {args.malicious_percentage}%")
    
    metrics = simulation.run_simulation(num_steps=args.steps, 
                                     transactions_per_step=args.transactions_per_step)
    
    # Tính thời gian chạy
    elapsed_time = time.time() - start_time
    print(f"\nTong thoi gian mo phong: {elapsed_time:.2f} giay")
    
    # Tạo thư mục con cho kết quả theo timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(args.save_dir, f"sim_{timestamp}")
    os.makedirs(result_subdir, exist_ok=True)
    
    # Lưu kết quả và tạo báo cáo
    if args.parallel_save:
        # Lưu kết quả song song
        simulation.parallel_save_results(result_subdir)
    else:
        # Tạo báo cáo và lưu kết quả tuần tự
        simulation.generate_report(result_subdir)
        
        # Chỉ vẽ biểu đồ nếu không bỏ qua
        if not args.skip_plots:
            simulation.plot_metrics(result_subdir)
    
    print(f"\nHoan tat! Ket qua duoc luu tai: {result_subdir}")
    
    # Trả về performance metrics cuối cùng
    final_metrics = simulation.save_final_metrics(result_subdir)
    return final_metrics

if __name__ == "__main__":
    main() 