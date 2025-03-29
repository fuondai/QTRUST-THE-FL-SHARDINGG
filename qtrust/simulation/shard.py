import random
from typing import List, Optional, Dict, Any

class BlockchainNode:
    """
    Đại diện một node trong mạng blockchain.
    """
    def __init__(self, node_id: int, shard_id: int, is_malicious: bool = False):
        """
        Khởi tạo node blockchain.
        
        Args:
            node_id: ID của node
            shard_id: ID của shard mà node thuộc về
            is_malicious: True nếu node là độc hại
        """
        self.node_id = node_id
        self.shard_id = shard_id
        self.is_malicious = is_malicious
        self.connections = []  # Danh sách các node kết nối
        self.transactions_processed = 0
        self.blocks_created = 0
        self.attack_behaviors = []  # Danh sách các hành vi tấn công
        self.resource_usage = 0.0  # Sử dụng tài nguyên (0.0-1.0)
        
    def process_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Xử lý một giao dịch.
        
        Args:
            transaction: Thông tin giao dịch
            
        Returns:
            bool: True nếu giao dịch được xử lý thành công
        """
        # Nếu là node độc hại, có thể thực hiện tấn công
        if self.is_malicious and self.attack_behaviors:
            for behavior in self.attack_behaviors:
                # Kiểm tra xác suất thực hiện tấn công
                if random.random() < behavior['probability']:
                    # Thực hiện hành vi tấn công
                    if 'reject_valid_tx' in behavior['actions'] and transaction.get('valid', True):
                        return False
                    elif 'validate_invalid_tx' in behavior['actions'] and not transaction.get('valid', True):
                        return True
                    elif 'double_spend' in behavior['actions']:
                        # Mô phỏng double-spending
                        pass
                    # Mô phỏng các hành vi khác dựa trên loại tấn công
                    
        # Xử lý bình thường cho node trung thực hoặc node độc hại không thực hiện tấn công
        self.transactions_processed += 1
        
        return transaction.get('valid', True)
    
    def create_block(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tạo một block mới.
        
        Args:
            transactions: Danh sách các giao dịch để đưa vào block
            
        Returns:
            Dict[str, Any]: Thông tin block đã tạo
        """
        self.blocks_created += 1
        
        # Kiểm tra xem node có đang thực hiện tấn công selfish mining không
        is_withholding = False
        if self.is_malicious:
            for behavior in self.attack_behaviors:
                if behavior['type'] == 'selfish_mining' and 'withhold_blocks' in behavior['actions']:
                    if random.random() < behavior['probability']:
                        is_withholding = True
        
        return {
            'creator': self.node_id,
            'shard_id': self.shard_id,
            'transactions': transactions,
            'is_withheld': is_withholding,
            'timestamp': random.randint(100000, 999999)  # Giả lập thời gian tạo
        }
    
    def execute_ddos_attack(self) -> float:
        """
        Thực hiện tấn công DDoS nếu node được cấu hình để làm điều đó.
        
        Returns:
            float: Mức độ tấn công (0.0-1.0)
        """
        if not self.is_malicious:
            return 0.0
            
        for behavior in self.attack_behaviors:
            if behavior['type'] == 'ddos' and random.random() < behavior['probability']:
                # Mô phỏng tấn công DDoS
                attack_intensity = random.uniform(0.5, 1.0)
                return attack_intensity
                
        return 0.0
        
    def attempt_bribery(self, target_nodes: List['BlockchainNode']) -> List[int]:
        """
        Thực hiện tấn công hối lộ nếu node được cấu hình để làm điều đó.
        
        Args:
            target_nodes: Danh sách các node có thể bị hối lộ
            
        Returns:
            List[int]: IDs của các node đã bị hối lộ
        """
        bribed_nodes = []
        
        if not self.is_malicious:
            return bribed_nodes
            
        for behavior in self.attack_behaviors:
            if behavior['type'] == 'bribery' and random.random() < behavior['probability']:
                # Mô phỏng tấn công hối lộ
                for target in target_nodes:
                    # Chỉ node trung thực mới có thể bị hối lộ
                    if not target.is_malicious and random.random() < 0.3:  # 30% cơ hội hối lộ thành công
                        bribed_nodes.append(target.node_id)
                
        return bribed_nodes

class Shard:
    """
    Đại diện một shard trong mạng blockchain.
    """
    def __init__(self, shard_id: int, num_nodes: int, 
                 malicious_percentage: float = 0.0,
                 attack_types: List[str] = None):
        """
        Khởi tạo một shard.
        
        Args:
            shard_id: ID của shard
            num_nodes: Số lượng node trong shard
            malicious_percentage: Tỷ lệ % node độc hại
            attack_types: Danh sách các loại tấn công
        """
        self.shard_id = shard_id
        self.num_nodes = num_nodes
        self.malicious_percentage = malicious_percentage
        self.attack_types = attack_types or []
        
        # Khởi tạo các node
        self.nodes = []
        self.malicious_nodes = []
        self._initialize_nodes()
        
        # Thống kê shard
        self.transactions_pool = []
        self.confirmed_transactions = []
        self.blocks = []
        self.network_stability = 1.0
        self.resource_utilization = 0.0
        
    def _initialize_nodes(self):
        """Khởi tạo các node trong shard."""
        # Số lượng node độc hại
        num_malicious = int(self.num_nodes * self.malicious_percentage / 100.0)
        
        # Tạo node trung thực
        for i in range(self.num_nodes - num_malicious):
            node_id = self.shard_id * self.num_nodes + i
            node = BlockchainNode(node_id, self.shard_id, is_malicious=False)
            self.nodes.append(node)
        
        # Tạo node độc hại
        for i in range(num_malicious):
            node_id = self.shard_id * self.num_nodes + (self.num_nodes - num_malicious) + i
            node = BlockchainNode(node_id, self.shard_id, is_malicious=True)
            self.nodes.append(node)
            self.malicious_nodes.append(node)
        
        # Thiết lập hành vi tấn công
        if self.malicious_nodes and self.attack_types:
            self._setup_attack_behavior()
    
    def _setup_attack_behavior(self):
        """Thiết lập hành vi tấn công cho các node độc hại."""
        for node in self.malicious_nodes:
            # Các hành vi tấn công dựa trên loại tấn công
            if '51_percent' in self.attack_types:
                node.attack_behaviors.append({
                    'type': '51_percent',
                    'probability': 0.8,
                    'actions': ['reject_valid_tx', 'validate_invalid_tx', 'double_spend']
                })
            
            if 'sybil' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'sybil',
                    'probability': 0.7,
                    'actions': ['create_fake_identities', 'vote_manipulation']
                })
            
            if 'eclipse' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'eclipse',
                    'probability': 0.75,
                    'actions': ['isolate_node', 'filter_transactions']
                })
                
            if 'selfish_mining' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'selfish_mining',
                    'probability': 0.6,
                    'actions': ['withhold_blocks', 'release_selectively', 'fork_chain']
                })
                
            if 'bribery' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'bribery',
                    'probability': 0.5,
                    'actions': ['bribe_validators', 'incentivize_forks', 'corrupt_consensus']
                })
                
            if 'ddos' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'ddos',
                    'probability': 0.9,
                    'actions': ['flood_requests', 'resource_exhaustion', 'connection_overload']
                })
                
            if 'finney' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'finney',
                    'probability': 0.65,
                    'actions': ['prepare_hidden_chain', 'double_spend_attack', 'revert_transactions']
                })
    
    def process_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Xử lý một loạt các giao dịch trong shard.
        
        Args:
            transactions: Danh sách các giao dịch cần xử lý
            
        Returns:
            Dict[str, Any]: Kết quả xử lý
        """
        # Giả lập tình trạng tấn công DDoS
        ddos_intensity = 0.0
        for node in self.malicious_nodes:
            ddos_intensity = max(ddos_intensity, node.execute_ddos_attack())
        
        # Nếu có tấn công DDoS, giảm hiệu suất xử lý
        performance_factor = max(0.1, 1.0 - ddos_intensity * 0.8)
        
        # Chọn ngẫu nhiên các node xác thực
        available_nodes = [node for node in self.nodes 
                          if not (node.is_malicious and any(b['type'] == 'ddos' for b in node.attack_behaviors))]
        
        validator_count = max(3, int(len(available_nodes) * 0.3))  # Ít nhất 3 node xác thực
        validators = random.sample(available_nodes, min(validator_count, len(available_nodes)))
        
        # Xử lý từng giao dịch
        processed_count = 0
        successful_count = 0
        rejected_count = 0
        
        for tx in transactions:
            # Xử lý lần lượt qua các validators
            votes = []
            for validator in validators:
                # Mô phỏng bribery attack
                if any(node.is_malicious and any(b['type'] == 'bribery' for b in node.attack_behaviors) 
                      for node in self.malicious_nodes):
                    # Kiểm tra xem validator có bị hối lộ không
                    for attacker in [n for n in self.malicious_nodes 
                                  if any(b['type'] == 'bribery' for b in n.attack_behaviors)]:
                        bribed_nodes = attacker.attempt_bribery([validator])
                        if validator.node_id in bribed_nodes:
                            # Node bị hối lộ sẽ bỏ phiếu theo ý muốn của node tấn công
                            votes.append(not tx.get('valid', True))
                            break
                    else:
                        # Không bị hối lộ, bỏ phiếu bình thường
                        votes.append(validator.process_transaction(tx))
                else:
                    # Không có tấn công hối lộ, xử lý bình thường
                    votes.append(validator.process_transaction(tx))
            
            # Kiểm tra kết quả
            if sum(votes) > len(votes) / 2:  # Đa số đồng ý
                successful_count += 1
                tx['status'] = 'completed'
                self.confirmed_transactions.append(tx)
            else:
                rejected_count += 1
                tx['status'] = 'rejected'
            
            processed_count += 1
        
        # Xử lý tấn công Finney nếu có
        if any(node.is_malicious and any(b['type'] == 'finney' for b in node.attack_behaviors) 
              for node in self.malicious_nodes):
            # Có cơ hội đảo ngược một số giao dịch đã hoàn thành
            finney_attacker = next((node for node in self.malicious_nodes 
                                if any(b['type'] == 'finney' for b in node.attack_behaviors)), None)
            
            if finney_attacker and self.confirmed_transactions:
                # Xác suất thành công của tấn công Finney
                finney_probability = 0.0
                for behavior in finney_attacker.attack_behaviors:
                    if behavior['type'] == 'finney':
                        finney_probability = behavior['probability']
                
                if random.random() < finney_probability:
                    # Lựa chọn ngẫu nhiên một số giao dịch để đảo ngược
                    revert_count = min(3, len(self.confirmed_transactions))
                    transactions_to_revert = random.sample(self.confirmed_transactions, revert_count)
                    
                    for tx in transactions_to_revert:
                        tx['status'] = 'reverted'
                        self.confirmed_transactions.remove(tx)
                        successful_count -= 1
        
        # Cập nhật thống kê shard
        self.network_stability = max(0.2, self.network_stability - ddos_intensity * 0.3)
        self.resource_utilization = min(1.0, self.resource_utilization + ddos_intensity * 0.4)
        
        return {
            'processed': processed_count,
            'successful': successful_count,
            'rejected': rejected_count,
            'network_stability': self.network_stability,
            'resource_utilization': self.resource_utilization
        }
    
    def get_shard_health(self) -> float:
        """
        Tính toán mức độ sức khỏe của shard dựa trên nhiều yếu tố.
        
        Returns:
            float: Điểm sức khỏe (0.0-1.0)
        """
        # Tỷ lệ node trung thực
        honest_ratio = 1.0 - len(self.malicious_nodes) / max(1, len(self.nodes))
        
        # Kết hợp với độ ổn định mạng
        health = 0.6 * honest_ratio + 0.3 * self.network_stability + 0.1 * (1.0 - self.resource_utilization)
        
        return health
    
    def get_malicious_nodes(self) -> List[BlockchainNode]:
        """
        Lấy danh sách các node độc hại trong shard.
        
        Returns:
            List[BlockchainNode]: Danh sách các node độc hại
        """
        return self.malicious_nodes 