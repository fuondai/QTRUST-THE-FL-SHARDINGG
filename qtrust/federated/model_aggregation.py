import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import copy

class OptimizedAggregator:
    """
    Lớp thực hiện các phương pháp tổng hợp mô hình tối ưu hóa cho Federated Learning.
    """
    def __init__(self):
        """
        Khởi tạo lớp tổng hợp mô hình tối ưu.
        """
        # Dữ liệu lịch sử
        self.aggregation_history = []
        
    @staticmethod
    def weighted_average(params_list: List[Dict[str, torch.Tensor]], 
                        weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình bằng trung bình có trọng số.
        
        Args:
            params_list: Danh sách các từ điển tham số mô hình từ các client
            weights: Danh sách trọng số cho mỗi client
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if not params_list:
            raise ValueError("Không có tham số để tổng hợp")
            
        if len(params_list) != len(weights):
            raise ValueError("Số lượng tham số và trọng số không khớp")
        
        # Chuẩn hóa trọng số
        weights_sum = sum(weights)
        if weights_sum == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / weights_sum for w in weights]
        
        # Khởi tạo tham số kết quả từ client đầu tiên
        result_params = copy.deepcopy(params_list[0])
        
        # Thực hiện tổng hợp với trọng số đã chuẩn hóa
        for key in result_params.keys():
            # Khởi tạo tensor kết quả với giá trị 0
            result_params[key] = torch.zeros_like(result_params[key])
            
            # Cộng tham số từ mỗi client với trọng số tương ứng
            for client_idx, client_params in enumerate(params_list):
                result_params[key] += normalized_weights[client_idx] * client_params[key]
                
        return result_params
    
    @staticmethod
    def median(params_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình bằng giá trị trung vị.
        Phương pháp này giúp chống lại Byzantine attacks.
        
        Args:
            params_list: Danh sách các từ điển tham số mô hình từ các client
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if not params_list:
            raise ValueError("Không có tham số để tổng hợp")
            
        # Khởi tạo tham số kết quả từ client đầu tiên
        result_params = copy.deepcopy(params_list[0])
        
        # Thực hiện tổng hợp median cho từng tham số
        for key in result_params.keys():
            # Lấy tất cả giá trị từ client cho tham số hiện tại
            param_tensors = [params[key].cpu().numpy() for params in params_list]
            
            # Tính giá trị trung vị trên trục client
            median_values = np.median(param_tensors, axis=0)
            
            # Chuyển lại thành tensor
            result_params[key] = torch.tensor(median_values, dtype=result_params[key].dtype, 
                                            device=result_params[key].device)
                
        return result_params
    
    @staticmethod
    def trimmed_mean(params_list: List[Dict[str, torch.Tensor]], 
                    trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình bằng trung bình cắt ngọn.
        Phương pháp này loại bỏ % giá trị lớn nhất và nhỏ nhất trước khi tính trung bình.
        
        Args:
            params_list: Danh sách các từ điển tham số mô hình từ các client
            trim_ratio: Tỷ lệ phần trăm cắt ngọn (0.1 = cắt 10% cao nhất và 10% thấp nhất)
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if not params_list:
            raise ValueError("Không có tham số để tổng hợp")
            
        if trim_ratio < 0 or trim_ratio > 0.5:
            raise ValueError("Tỷ lệ cắt ngọn phải nằm trong khoảng [0, 0.5]")
            
        # Khởi tạo tham số kết quả từ client đầu tiên
        result_params = copy.deepcopy(params_list[0])
        
        # Thực hiện tổng hợp trimmed mean cho từng tham số
        for key in result_params.keys():
            # Lấy tất cả giá trị từ client cho tham số hiện tại
            param_tensors = [params[key].cpu().numpy() for params in params_list]
            
            # Số lượng client cần cắt ở mỗi đầu
            num_clients = len(param_tensors)
            k = int(np.ceil(num_clients * trim_ratio))
            
            if 2*k >= num_clients:
                # Nếu cắt quá nhiều, sử dụng phương pháp trung bình thông thường
                mean_values = np.mean(param_tensors, axis=0)
            else:
                # Sắp xếp các giá trị theo thứ tự tăng dần trên trục client
                sorted_params = np.sort(param_tensors, axis=0)
                
                # Lấy giá trị trung bình của các giá trị còn lại sau khi cắt
                mean_values = np.mean(sorted_params[k:num_clients-k], axis=0)
            
            # Chuyển lại thành tensor
            result_params[key] = torch.tensor(mean_values, dtype=result_params[key].dtype, 
                                            device=result_params[key].device)
                
        return result_params
    
    @staticmethod
    def krum(params_list: List[Dict[str, torch.Tensor]], 
            num_byzantine: int = 0) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình bằng thuật toán Krum.
        Krum chọn client có tổng khoảng cách Euclidean đến n-f-2 client gần nhất là nhỏ nhất.
        
        Args:
            params_list: Danh sách các từ điển tham số mô hình từ các client
            num_byzantine: Số lượng client Byzantine tối đa có thể có
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if not params_list:
            raise ValueError("Không có tham số để tổng hợp")
            
        num_clients = len(params_list)
        if num_byzantine >= num_clients / 2:
            raise ValueError("Số lượng client Byzantine quá lớn")
            
        # Chuyển các tham số thành vector phẳng
        flat_params = []
        for params in params_list:
            flat_param = torch.cat([param.flatten() for param in params.values()])
            flat_params.append(flat_param)
        
        # Tính khoảng cách giữa các cặp clients
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = torch.norm(flat_params[i] - flat_params[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Krum score là tổng khoảng cách đến n-f-2 client gần nhất
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            # Lấy n-f-2 client gần nhất
            closest_distances = torch.topk(distances[i], k=num_clients-num_byzantine-2, 
                                         largest=False).values
            scores[i] = torch.sum(closest_distances)
        
        # Chọn client có score nhỏ nhất
        best_client_idx = torch.argmin(scores).item()
        
        return copy.deepcopy(params_list[best_client_idx])
    
    @staticmethod
    def adaptive_federated_averaging(params_list: List[Dict[str, torch.Tensor]],
                                   trust_scores: List[float],
                                   performance_scores: List[float],
                                   adaptive_alpha: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình bằng trung bình thích nghi dựa trên điểm tin cậy và hiệu suất.
        
        Args:
            params_list: Danh sách các từ điển tham số mô hình từ các client
            trust_scores: Điểm tin cậy của mỗi client
            performance_scores: Điểm hiệu suất của mỗi client
            adaptive_alpha: Tỷ lệ kết hợp giữa điểm tin cậy và hiệu suất
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if not params_list:
            raise ValueError("Không có tham số để tổng hợp")
            
        if len(params_list) != len(trust_scores) or len(params_list) != len(performance_scores):
            raise ValueError("Số lượng tham số, điểm tin cậy và điểm hiệu suất không khớp")
        
        # Tính trọng số dựa trên kết hợp điểm tin cậy và hiệu suất
        combined_scores = [adaptive_alpha * trust + (1 - adaptive_alpha) * perf 
                          for trust, perf in zip(trust_scores, performance_scores)]
        
        # Chuẩn hóa điểm thành trọng số
        total_score = sum(combined_scores)
        if total_score == 0:
            weights = [1.0 / len(combined_scores)] * len(combined_scores)
        else:
            weights = [score / total_score for score in combined_scores]
        
        # Tổng hợp bằng trung bình có trọng số
        return OptimizedAggregator.weighted_average(params_list, weights)
    
    @staticmethod
    def fedprox(params_list: List[Dict[str, torch.Tensor]],
               global_params: Dict[str, torch.Tensor],
               weights: List[float],
               mu: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Thực hiện tổng hợp mô hình bằng FedProx, thêm điều hạn chế gần với mô hình toàn cục.
        
        Args:
            params_list: Danh sách các từ điển tham số mô hình từ các client
            global_params: Tham số mô hình toàn cục hiện tại
            weights: Danh sách trọng số cho mỗi client
            mu: Hệ số điều chuẩn (regularization)
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if not params_list:
            raise ValueError("Không có tham số để tổng hợp")
            
        if len(params_list) != len(weights):
            raise ValueError("Số lượng tham số và trọng số không khớp")
        
        # Tổng hợp cơ bản bằng trung bình có trọng số
        aggregated_params = OptimizedAggregator.weighted_average(params_list, weights)
        
        # Thêm điều chuẩn để giữ cho mô hình mới không quá khác biệt so với mô hình toàn cục
        for key in aggregated_params:
            aggregated_params[key] = (1 - mu) * aggregated_params[key] + mu * global_params[key]
                
        return aggregated_params

class ModelAggregationManager:
    """
    Quản lý việc tổng hợp mô hình với nhiều phương pháp khác nhau.
    """
    def __init__(self, default_method: str = 'weighted_average'):
        """
        Khởi tạo trình quản lý tổng hợp mô hình.
        
        Args:
            default_method: Phương pháp tổng hợp mặc định
        """
        self.aggregator = OptimizedAggregator()
        self.default_method = default_method
        self.supported_methods = {
            'weighted_average': self.aggregator.weighted_average,
            'median': self.aggregator.median,
            'trimmed_mean': self.aggregator.trimmed_mean,
            'krum': self.aggregator.krum,
            'adaptive_fedavg': self.aggregator.adaptive_federated_averaging,
            'fedprox': self.aggregator.fedprox
        }
        
    def aggregate(self, method: str, params_list: List[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp mô hình sử dụng phương pháp đã chọn.
        
        Args:
            method: Tên phương pháp tổng hợp
            params_list: Danh sách các từ điển tham số mô hình từ các client
            **kwargs: Các tham số bổ sung cho phương pháp tổng hợp
            
        Returns:
            Dict: Tham số mô hình đã tổng hợp
        """
        if method not in self.supported_methods:
            print(f"Phương pháp {method} không được hỗ trợ, sử dụng {self.default_method}")
            method = self.default_method
            
        aggregation_func = self.supported_methods[method]
        
        try:
            result = aggregation_func(params_list, **kwargs)
            return result
        except Exception as e:
            print(f"Lỗi khi tổng hợp mô hình với phương pháp {method}: {e}")
            # Fallback về phương pháp mặc định nếu có lỗi
            if method != self.default_method:
                print(f"Sử dụng phương pháp {self.default_method} thay thế")
                kwargs_for_default = {}
                if self.default_method == 'weighted_average' and 'weights' in kwargs:
                    kwargs_for_default = {'weights': kwargs['weights']}
                return self.supported_methods[self.default_method](params_list, **kwargs_for_default)
            raise
    
    def recommend_method(self, 
                       num_clients: int, 
                       has_trust_scores: bool = False, 
                       suspected_byzantine: bool = False) -> str:
        """
        Đề xuất phương pháp tổng hợp tốt nhất dựa trên số lượng client và dữ liệu có sẵn.
        
        Args:
            num_clients: Số lượng client tham gia tổng hợp
            has_trust_scores: Có dữ liệu về điểm tin cậy hay không
            suspected_byzantine: Có nghi ngờ client Byzantine hay không
            
        Returns:
            str: Tên phương pháp tổng hợp được đề xuất
        """
        if suspected_byzantine:
            if num_clients >= 7:  # Krum cần ít nhất 2f+3 client, với f là số client Byzantine
                return 'krum'
            else:
                return 'trimmed_mean'
        
        if has_trust_scores:
            return 'adaptive_fedavg'
        
        if num_clients < 5:
            return 'weighted_average'
        
        return 'fedprox' 