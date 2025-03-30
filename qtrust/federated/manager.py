"""
Quản lý quá trình Federated Learning trong hệ thống QTrust.

Module này cung cấp các công cụ để điều phối quá trình học liên bang
giữa nhiều nút mạng phân tán.
"""

import logging
import random
import time
from typing import Dict, List, Any, Callable, Optional, Tuple, Union

import numpy as np
import torch

from qtrust.utils.cache import lru_cache, ttl_cache, compute_hash
from qtrust.federated.model_aggregation import ModelAggregator 
from qtrust.federated.protocol import FederatedProtocol
from qtrust.federated.client import FederatedClient
from qtrust.federated.privacy import PrivacyManager

logger = logging.getLogger("qtrust.federated")

class FederatedLearningManager:
    """
    Quản lý quá trình huấn luyện Federated Learning.
    
    Lớp này điều phối việc phân phối mô hình, tổng hợp cập nhật,
    và triển khai các chiến lược bảo mật và hiệu quả.
    """
    
    def __init__(self, 
                 initial_model: Dict[str, torch.Tensor],
                 clients: List[FederatedClient],
                 aggregation_method: str = 'weighted_average',
                 client_fraction: float = 1.0,
                 rounds: int = 10,
                 local_epochs: int = 5,
                 protocol: Optional[FederatedProtocol] = None,
                 privacy_manager: Optional[PrivacyManager] = None,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Khởi tạo FederatedLearningManager.
        
        Args:
            initial_model: Mô hình khởi tạo ban đầu (từ điển state_dict)
            clients: Danh sách các client tham gia
            aggregation_method: Phương pháp tổng hợp cập nhật mô hình
            client_fraction: Tỷ lệ client được chọn mỗi vòng
            rounds: Số vòng huấn luyện
            local_epochs: Số epoch huấn luyện cục bộ trên mỗi client
            protocol: Giao thức giao tiếp giữa server và client
            privacy_manager: Quản lý các kỹ thuật bảo mật
            device: Thiết bị tính toán (cpu/cuda)
            seed: Seed để đảm bảo reproducibility
        """
        self.global_model = initial_model
        self.clients = clients
        self.client_fraction = client_fraction
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.protocol = protocol or FederatedProtocol()
        self.privacy_manager = privacy_manager or PrivacyManager()
        self.device = device
        self.seed = seed
        
        # Khởi tạo bộ tổng hợp mô hình
        self.aggregator = ModelAggregator()
        self.aggregation_method = aggregation_method
        
        # Thiết lập seed để đảm bảo reproducibility
        self._set_seed(seed)
        
        # Lưu trữ lịch sử huấn luyện
        self.history = {
            'global_performance': [],
            'client_performances': {},
            'selected_clients': [],
            'aggregation_time': [],
            'communication_volume': []
        }
        
        # Cache cho các phép tính toán lặp lại
        self.cache = {}
        
        # Khởi tạo lịch sử hiệu suất cho mỗi client
        for client in self.clients:
            self.history['client_performances'][client.id] = []
    
    def _set_seed(self, seed: int):
        """Thiết lập các seed để đảm bảo reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @lru_cache(maxsize=100)
    def _select_clients(self, round_idx: int, client_ids: Tuple[str, ...], fraction: float) -> List[FederatedClient]:
        """
        Chọn một tập hợp các client tham gia vào vòng huấn luyện hiện tại.
        
        Args:
            round_idx: Chỉ số vòng huấn luyện hiện tại
            client_ids: Tuple các ID client (dùng cho caching)
            fraction: Tỷ lệ client được chọn
        
        Returns:
            List[FederatedClient]: Danh sách các client được chọn
        """
        num_clients = max(1, int(fraction * len(self.clients)))
        
        # Sử dụng seed cố định cho mỗi vòng
        local_random = random.Random(self.seed + round_idx)
        
        # Chọn client dựa trên một số yếu tố
        if hasattr(self, 'client_trust_scores'):
            # Ưu tiên client có độ tin cậy cao
            trust_weights = [max(0.1, score) for score in self.client_trust_scores]
            chosen_clients = local_random.choices(
                self.clients, weights=trust_weights, k=num_clients
            )
        else:
            # Nếu không có điểm tin cậy, chọn ngẫu nhiên
            chosen_clients = local_random.sample(self.clients, num_clients)
            
        # Lưu lại danh sách client được chọn
        self.history['selected_clients'].append([client.id for client in chosen_clients])
        
        return chosen_clients
    
    @ttl_cache(ttl=3600)  # Cache trong 1 giờ
    def _evaluate_global_model(self, model_hash: str, round_idx: int) -> Dict[str, float]:
        """
        Đánh giá hiệu suất của mô hình toàn cục trên tập dữ liệu đánh giá.
        
        Args:
            model_hash: Hash của mô hình để cache kết quả
            round_idx: Chỉ số vòng huấn luyện hiện tại
            
        Returns:
            Dict[str, float]: Chỉ số hiệu suất của mô hình
        """
        logger.info(f"Đánh giá mô hình toàn cục sau vòng {round_idx}...")
        
        # Đánh giá trên một tập dữ liệu giữ lại
        if hasattr(self, 'test_data'):
            # Placeholder for actual evaluation
            performance = {'accuracy': 0.85 + round_idx * 0.01, 'loss': 0.4 - round_idx * 0.02}
            
            # TODO: Thực hiện đánh giá thực tế
            # performance = evaluate_model(self.global_model, self.test_data)
            
            self.history['global_performance'].append(performance)
            return performance
        
        # Nếu không có tập test, đánh giá trên dữ liệu của các client
        performance_scores = []
        
        for client in self.clients:
            # Gửi mô hình tới client để đánh giá
            client_perf = client.evaluate(self.global_model)
            performance_scores.append(client_perf)
        
        # Tính trung bình các chỉ số hiệu suất
        avg_performance = {}
        for metric in performance_scores[0].keys():
            avg_performance[metric] = sum(p[metric] for p in performance_scores) / len(performance_scores)
        
        self.history['global_performance'].append(avg_performance)
        return avg_performance
    
    @lru_cache(maxsize=10)
    def _compute_client_trust_scores(self, performance_cache_key: str) -> List[float]:
        """
        Tính toán điểm tin cậy cho mỗi client dựa trên hiệu suất quá khứ.
        
        Args:
            performance_cache_key: Khóa cache cho dữ liệu hiệu suất
            
        Returns:
            List[float]: Điểm tin cậy cho mỗi client
        """
        # Mô phỏng tính toán điểm tin cậy
        trust_scores = []
        
        for client in self.clients:
            # Lấy lịch sử hiệu suất của client
            performances = self.history['client_performances'][client.id]
            
            if not performances:
                # Nếu không có dữ liệu, sử dụng giá trị mặc định
                trust_scores.append(0.5)
                continue
            
            # Tính điểm tin cậy dựa trên độ chính xác đóng góp
            accuracy_contribution = []
            last_n = min(5, len(performances))  # Xem xét 5 vòng gần nhất
            
            for perf in performances[-last_n:]:
                if 'accuracy' in perf:
                    accuracy_contribution.append(perf['accuracy'])
                elif 'loss' in perf:
                    # Chuyển đổi loss thành độ chính xác tương đối
                    accuracy_contribution.append(1.0 / (1.0 + perf['loss']))
            
            # Tính điểm tin cậy dựa trên độ chính xác trung bình
            if accuracy_contribution:
                avg_accuracy = sum(accuracy_contribution) / len(accuracy_contribution)
                trust_score = min(1.0, avg_accuracy)  # Giới hạn tối đa là 1.0
            else:
                trust_score = 0.5  # Giá trị mặc định
            
            trust_scores.append(trust_score)
        
        return trust_scores
    
    def train(self) -> Dict[str, torch.Tensor]:
        """
        Huấn luyện mô hình sử dụng Federated Learning.
        
        Returns:
            Dict[str, torch.Tensor]: Mô hình toàn cục cuối cùng
        """
        logger.info(f"Bắt đầu quá trình Federated Learning với {len(self.clients)} clients...")
        
        for round_idx in range(self.rounds):
            start_time = time.time()
            logger.info(f"------ Vòng {round_idx+1}/{self.rounds} ------")
            
            # Tính điểm tin cậy cho mỗi client
            client_ids = tuple(client.id for client in self.clients)
            perf_cache_key = compute_hash(self.history['client_performances'])
            self.client_trust_scores = self._compute_client_trust_scores(perf_cache_key)
            
            # Chọn client tham gia vòng huấn luyện hiện tại
            selected_clients = self._select_clients(round_idx, client_ids, self.client_fraction)
            logger.info(f"Đã chọn {len(selected_clients)}/{len(self.clients)} clients")
            
            # Phân phối mô hình cho các client
            client_updates = []
            client_weights = []
            client_performance = []
            trust_scores = []
            
            for client in selected_clients:
                client_idx = self.clients.index(client)
                trust_score = self.client_trust_scores[client_idx]
                trust_scores.append(trust_score)
                
                # Áp dụng các kỹ thuật bảo mật nếu cần
                secure_model = self.privacy_manager.secure_model_for_client(
                    self.global_model, client.id, trust_score
                )
                
                # Gửi mô hình tới client và nhận cập nhật
                client_model, client_stats = client.train(
                    secure_model, 
                    epochs=self.local_epochs
                )
                
                # Áp dụng các kỹ thuật bảo mật cho cập nhật
                verified_model = self.privacy_manager.verify_client_update(
                    client_model, client.id, trust_score
                )
                
                # Thu thập cập nhật
                client_updates.append(verified_model)
                client_weights.append(client_stats['sample_size'])
                client_performance.append(client_stats['performance'])
                
                # Cập nhật lịch sử hiệu suất
                self.history['client_performances'][client.id].append(client_stats['performance'])
            
            # Tổng hợp các cập nhật từ client
            aggregation_start = time.time()
            
            # Chuẩn bị tham số cho bộ tổng hợp
            agg_params = {
                'params_list': client_updates,
                'weights': client_weights
            }
            
            # Thêm tham số cho các phương pháp đặc biệt
            if self.aggregation_method == 'fedprox':
                agg_params['global_params'] = self.global_model
            elif self.aggregation_method == 'fedadam':
                agg_params['global_params'] = self.global_model
            elif self.aggregation_method == 'fedtrust':
                agg_params['trust_scores'] = trust_scores
                agg_params['performance_scores'] = [p.get('accuracy', 0.5) for p in client_performance]
            
            # Gọi phương thức tổng hợp tương ứng
            if self.aggregation_method in self.aggregator.methods:
                self.global_model = self.aggregator.methods[self.aggregation_method](**agg_params)
            else:
                # Fallback to weighted_average
                self.global_model = self.aggregator.weighted_average(**agg_params)
            
            aggregation_time = time.time() - aggregation_start
            self.history['aggregation_time'].append(aggregation_time)
            
            # Xóa cache quá cũ để giảm bộ nhớ
            if round_idx % 5 == 0:
                self.aggregator.clear_cache()
            
            # Đánh giá mô hình toàn cục
            model_hash = compute_hash(self.global_model)
            performance = self._evaluate_global_model(model_hash, round_idx)
            
            # Ghi lại thời gian hoàn thành vòng
            round_time = time.time() - start_time
            logger.info(f"Vòng {round_idx+1} hoàn thành trong {round_time:.2f}s")
            logger.info(f"Hiệu suất: {performance}")
        
        logger.info("Quá trình Federated Learning hoàn thành!")
        return self.global_model
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Lấy lịch sử huấn luyện.
        
        Returns:
            Dict[str, Any]: Lịch sử huấn luyện
        """
        return self.history 