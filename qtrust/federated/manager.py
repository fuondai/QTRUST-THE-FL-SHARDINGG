import numpy as np
import random
from typing import Dict, List, Any, Optional

class FederatedLearningManager:
    """
    Quản lý quá trình học liên bang (Federated Learning) giữa các node trong mạng blockchain.
    """
    
    def __init__(self, num_shards: int, nodes_per_shard: int, aggregation_method: str = 'weighted_average'):
        """
        Khởi tạo trình quản lý học liên bang.
        
        Args:
            num_shards: Số lượng shard trong mạng
            nodes_per_shard: Số lượng node trong mỗi shard
            aggregation_method: Phương pháp tổng hợp mô hình ('weighted_average', 'median', 'trimmed_mean')
        """
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.aggregation_method = aggregation_method
        
        # Lưu trữ mô hình toàn cục và cục bộ
        self.global_model = None
        self.local_models = {}
        
        # Thông tin về dữ liệu và độ tin cậy của node
        self.node_weights = {}  # Trọng số cho mỗi node khi tổng hợp
        self.contribution_history = {}  # Lịch sử đóng góp của mỗi node
        
        # Các tham số cho học liên bang
        self.round = 0
        self.min_clients_required = max(2, int(num_shards * nodes_per_shard * 0.1))  # Tối thiểu 10% node tham gia
        
        # Khởi tạo trọng số mặc định cho mỗi node
        total_nodes = num_shards * nodes_per_shard
        for node_id in range(total_nodes):
            self.node_weights[node_id] = 1.0 / total_nodes
            self.contribution_history[node_id] = []
    
    def reset(self):
        """Khởi tạo lại trạng thái của trình quản lý học liên bang."""
        self.global_model = None
        self.local_models = {}
        self.round = 0
        
        # Khôi phục trọng số về giá trị mặc định
        total_nodes = self.num_shards * self.nodes_per_shard
        for node_id in range(total_nodes):
            self.node_weights[node_id] = 1.0 / total_nodes
            self.contribution_history[node_id] = []
    
    def aggregate_models(self, local_data: Dict[int, Any]) -> Optional[Any]:
        """
        Tổng hợp các mô hình cục bộ từ các node thành mô hình toàn cục.
        
        Args:
            local_data: Từ điển ánh xạ từ node ID đến dữ liệu cục bộ (mô hình + metrics)
            
        Returns:
            Optional[Any]: Mô hình toàn cục mới hoặc None nếu không đủ dữ liệu
        """
        if not local_data or len(local_data) < self.min_clients_required:
            return None
        
        # Lưu trữ các mô hình cục bộ đã nhận
        for node_id, data in local_data.items():
            if 'model' in data:
                self.local_models[node_id] = data['model']
                
                # Cập nhật lịch sử đóng góp
                if 'quality' in data:
                    self.contribution_history[node_id].append(data['quality'])
                    # Giới hạn kích thước lịch sử
                    if len(self.contribution_history[node_id]) > 10:
                        self.contribution_history[node_id].pop(0)
        
        # Cập nhật trọng số của các node dựa trên lịch sử đóng góp
        self._update_node_weights()
        
        # Tổng hợp các mô hình cục bộ thành mô hình toàn cục
        if len(self.local_models) >= self.min_clients_required:
            if self.aggregation_method == 'weighted_average':
                self.global_model = self._weighted_average_aggregation()
            elif self.aggregation_method == 'median':
                self.global_model = self._median_aggregation()
            elif self.aggregation_method == 'trimmed_mean':
                self.global_model = self._trimmed_mean_aggregation()
            else:
                # Mặc định là weighted_average
                self.global_model = self._weighted_average_aggregation()
            
            self.round += 1
            return self.global_model
        
        return None
    
    def get_global_model(self) -> Optional[Any]:
        """
        Trả về mô hình toàn cục hiện tại.
        
        Returns:
            Optional[Any]: Mô hình toàn cục hoặc None nếu chưa có
        """
        return self.global_model
    
    def _update_node_weights(self):
        """Cập nhật trọng số của các node dựa trên lịch sử đóng góp."""
        # Tính trọng số mới dựa trên chất lượng đóng góp gần đây
        new_weights = {}
        
        for node_id, history in self.contribution_history.items():
            if history:
                # Tính trung bình chất lượng đóng góp gần đây, ưu tiên các đóng góp mới hơn
                if len(history) > 3:
                    # Sử dụng trung bình có trọng số nếu có đủ lịch sử
                    weights = np.exp(np.linspace(0, 1, len(history)))
                    weights = weights / np.sum(weights)
                    avg_quality = np.sum(np.array(history) * weights)
                else:
                    # Nếu không đủ lịch sử, sử dụng trung bình đơn giản
                    avg_quality = np.mean(history)
                
                # Trọng số là hàm của chất lượng đóng góp
                # Sử dụng hàm sigmoid để ánh xạ chất lượng vào khoảng (0, 1)
                # và tăng độ dốc để làm nổi bật sự khác biệt
                new_weights[node_id] = 1.0 / (1.0 + np.exp(-5 * (avg_quality - 0.5)))
            else:
                # Nếu không có lịch sử, sử dụng trọng số mặc định thấp
                new_weights[node_id] = 0.1
        
        # Chuẩn hóa trọng số
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for node_id in new_weights:
                new_weights[node_id] /= total_weight
        
        # Cập nhật trọng số
        self.node_weights.update(new_weights)
    
    def _weighted_average_aggregation(self) -> Any:
        """
        Tổng hợp mô hình bằng phương pháp trung bình có trọng số.
        
        Returns:
            Any: Mô hình toàn cục mới
        """
        # Trong mô hình mô phỏng, chúng ta có thể giả định mô hình đơn giản là một số hoặc một danh sách thông số
        # Trong thực tế, các mô hình có thể là tensor hoặc mảng đa chiều
        
        if not self.local_models:
            return None
        
        # Tạo một mô hình mẫu (giả sử tất cả các mô hình có cùng cấu trúc)
        if self.global_model is None:
            # Khởi tạo mô hình toàn cục lần đầu
            first_model = next(iter(self.local_models.values()))
            if isinstance(first_model, dict):
                # Nếu mô hình là từ điển (ví dụ: {layer1: weights1, layer2: weights2, ...})
                self.global_model = {key: 0.0 for key in first_model}
            elif isinstance(first_model, list):
                # Nếu mô hình là danh sách (ví dụ: [weight1, weight2, ...])
                self.global_model = [0.0] * len(first_model)
            else:
                # Nếu mô hình là số (đơn giản hóa cho mô phỏng)
                self.global_model = 0.0
        
        # Tổng hợp mô hình bằng trung bình có trọng số
        if isinstance(self.global_model, dict):
            # Khởi tạo lại mô hình toàn cục
            for key in self.global_model:
                self.global_model[key] = 0.0
            
            # Tính trung bình có trọng số cho từng tham số
            for node_id, model in self.local_models.items():
                weight = self.node_weights.get(node_id, 0.0)
                for key, value in model.items():
                    if key in self.global_model:
                        self.global_model[key] += weight * value
            
        elif isinstance(self.global_model, list):
            # Khởi tạo lại mô hình toàn cục
            self.global_model = [0.0] * len(self.global_model)
            
            # Tính trung bình có trọng số cho từng tham số
            for node_id, model in self.local_models.items():
                weight = self.node_weights.get(node_id, 0.0)
                for i, value in enumerate(model):
                    if i < len(self.global_model):
                        self.global_model[i] += weight * value
            
        else:
            # Đơn giản hóa cho mô phỏng với một số duy nhất
            self.global_model = 0.0
            for node_id, model in self.local_models.items():
                weight = self.node_weights.get(node_id, 0.0)
                self.global_model += weight * model
        
        return self.global_model
    
    def _median_aggregation(self) -> Any:
        """
        Tổng hợp mô hình bằng phương pháp median để giảm ảnh hưởng của outlier.
        
        Returns:
            Any: Mô hình toàn cục mới
        """
        if not self.local_models:
            return None
        
        # Tạo một mô hình mẫu (giả sử tất cả các mô hình có cùng cấu trúc)
        if self.global_model is None:
            # Khởi tạo mô hình toàn cục lần đầu
            first_model = next(iter(self.local_models.values()))
            if isinstance(first_model, dict):
                self.global_model = {key: 0.0 for key in first_model}
            elif isinstance(first_model, list):
                self.global_model = [0.0] * len(first_model)
            else:
                self.global_model = 0.0
        
        # Tổng hợp mô hình bằng median
        if isinstance(self.global_model, dict):
            for key in self.global_model:
                values = [model[key] for model in self.local_models.values() if key in model]
                if values:
                    self.global_model[key] = np.median(values)
                    
        elif isinstance(self.global_model, list):
            for i in range(len(self.global_model)):
                values = [model[i] for model in self.local_models.values() if i < len(model)]
                if values:
                    self.global_model[i] = np.median(values)
                    
        else:
            # Đơn giản hóa cho mô phỏng với một số duy nhất
            self.global_model = np.median(list(self.local_models.values()))
        
        return self.global_model
    
    def _trimmed_mean_aggregation(self, trim_ratio: float = 0.2) -> Any:
        """
        Tổng hợp mô hình bằng phương pháp trimmed mean.
        
        Args:
            trim_ratio: Tỷ lệ phần trăm các giá trị nhỏ nhất và lớn nhất để loại bỏ (mặc định: 0.2)
            
        Returns:
            Any: Mô hình toàn cục mới
        """
        if not self.local_models:
            return None
        
        # Tạo một mô hình mẫu (giả sử tất cả các mô hình có cùng cấu trúc)
        if self.global_model is None:
            # Khởi tạo mô hình toàn cục lần đầu
            first_model = next(iter(self.local_models.values()))
            if isinstance(first_model, dict):
                self.global_model = {key: 0.0 for key in first_model}
            elif isinstance(first_model, list):
                self.global_model = [0.0] * len(first_model)
            else:
                self.global_model = 0.0
        
        # Tổng hợp mô hình bằng trimmed mean
        if isinstance(self.global_model, dict):
            for key in self.global_model:
                values = [model[key] for model in self.local_models.values() if key in model]
                if values:
                    # Loại bỏ trim_ratio các giá trị ở mỗi đầu
                    k = int(len(values) * trim_ratio)
                    if k > 0:
                        sorted_values = sorted(values)
                        trimmed_values = sorted_values[k:-k] if len(sorted_values) > 2*k else sorted_values
                        self.global_model[key] = np.mean(trimmed_values)
                    else:
                        self.global_model[key] = np.mean(values)
                    
        elif isinstance(self.global_model, list):
            for i in range(len(self.global_model)):
                values = [model[i] for model in self.local_models.values() if i < len(model)]
                if values:
                    # Loại bỏ trim_ratio các giá trị ở mỗi đầu
                    k = int(len(values) * trim_ratio)
                    if k > 0:
                        sorted_values = sorted(values)
                        trimmed_values = sorted_values[k:-k] if len(sorted_values) > 2*k else sorted_values
                        self.global_model[i] = np.mean(trimmed_values)
                    else:
                        self.global_model[i] = np.mean(values)
                    
        else:
            # Đơn giản hóa cho mô phỏng với một số duy nhất
            values = list(self.local_models.values())
            k = int(len(values) * trim_ratio)
            if k > 0:
                sorted_values = sorted(values)
                trimmed_values = sorted_values[k:-k] if len(sorted_values) > 2*k else sorted_values
                self.global_model = np.mean(trimmed_values)
            else:
                self.global_model = np.mean(values)
        
        return self.global_model 