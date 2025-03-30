import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Any, Union
from collections import deque
import random
import os
import time

class AnomalyDetector:
    """
    Lớp phát hiện dị thường sử dụng Machine Learning cho hệ thống HTDCM.
    """
    def __init__(self, input_features: int = 10, hidden_size: int = 64, 
                 anomaly_threshold: float = 0.85, memory_size: int = 1000,
                 learning_rate: float = 0.001, device: str = None):
        """
        Khởi tạo Anomaly Detector.
        
        Args:
            input_features: Số lượng đặc trưng đầu vào từ hành vi node
            hidden_size: Kích thước lớp ẩn của model
            anomaly_threshold: Ngưỡng để xác định dị thường
            memory_size: Kích thước bộ nhớ để lưu trữ mẫu
            learning_rate: Tốc độ học của model
            device: Thiết bị tính toán (None để tự phát hiện)
        """
        # Thiết lập thiết bị tính toán
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mô hình
        self.encoder = AutoEncoder(input_features, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Tham số phát hiện dị thường
        self.input_features = input_features
        self.anomaly_threshold = anomaly_threshold
        
        # Bộ nhớ cho việc huấn luyện
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Thống kê
        self.anomaly_history = []
        self.normal_history = []
        self.detected_anomalies = 0
        self.false_positives = 0
        self.training_loss_history = []
        
        # Trạng thái
        self.is_trained = False
        self.min_samples_for_training = 64
        self.reconstruction_errors = []
    
    def extract_features(self, node_data: Dict[str, Any]) -> np.ndarray:
        """
        Trích xuất các đặc trưng từ dữ liệu node để phát hiện dị thường.
        
        Args:
            node_data: Dữ liệu về hoạt động của node
            
        Returns:
            np.ndarray: Vector đặc trưng đầu vào cho mô hình
        """
        features = []
        
        # 1. Tỷ lệ thành công giao dịch
        total_txs = node_data.get('successful_txs', 0) + node_data.get('failed_txs', 0)
        success_rate = node_data.get('successful_txs', 0) / max(1, total_txs)
        features.append(success_rate)
        
        # 2. Thống kê thời gian phản hồi
        response_times = node_data.get('response_times', [])
        if response_times:
            features.append(np.mean(response_times))
            features.append(np.std(response_times))
            features.append(np.max(response_times))
            features.append(np.min(response_times))
        else:
            features.extend([0, 0, 0, 0])  # Giá trị mặc định
        
        # 3. Đánh giá từ peers
        peer_ratings = list(node_data.get('peer_ratings', {}).values())
        if peer_ratings:
            features.append(np.mean(peer_ratings))
            features.append(np.std(peer_ratings))
        else:
            features.extend([0.5, 0])  # Giá trị mặc định
        
        # 4. Số lượng hoạt động độc hại
        features.append(min(1.0, node_data.get('malicious_activities', 0) / 10.0))  # Chuẩn hóa
        
        # 5. Mẫu hoạt động gần đây
        recent_activities = node_data.get('activity_history', [])
        success_history = [1.0 if act[0] == 'success' else 0.0 for act in recent_activities[-10:]]
        # Đảm bảo có đủ 10 giá trị
        success_history = success_history + [0.5] * (10 - len(success_history))
        features.extend(success_history)
        
        # Chuẩn hóa và đảm bảo đủ kích thước
        features = np.array(features, dtype=np.float32)
        if len(features) < self.input_features:
            features = np.pad(features, (0, self.input_features - len(features)), 'constant')
        elif len(features) > self.input_features:
            features = features[:self.input_features]
            
        return features
    
    def add_sample(self, node_data: Dict[str, Any], is_anomaly: bool = False):
        """
        Thêm một mẫu vào bộ nhớ để huấn luyện.
        
        Args:
            node_data: Dữ liệu về hoạt động của node
            is_anomaly: Có phải là dị thường đã biết hay không
        """
        features = self.extract_features(node_data)
        self.memory.append((features, is_anomaly))
    
    def train(self, epochs: int = 5) -> float:
        """
        Huấn luyện mô hình phát hiện dị thường.
        
        Args:
            epochs: Số lượng epochs để huấn luyện
            
        Returns:
            float: Mất mát trung bình trong lần huấn luyện cuối cùng
        """
        if len(self.memory) < self.min_samples_for_training:
            return 0.0  # Không đủ dữ liệu
            
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            
            # Lấy chủ yếu các mẫu bình thường (không phải dị thường) để huấn luyện
            normal_samples = [(features, label) for features, label in self.memory if not label]
            if len(normal_samples) < self.batch_size // 2:
                samples = random.sample(self.memory, min(self.batch_size, len(self.memory)))
            else:
                samples = random.sample(normal_samples, min(self.batch_size, len(normal_samples)))
            
            # Chia batch
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i+self.batch_size]
                if len(batch) < 2:  # Quá ít mẫu
                    continue
                    
                features_batch = np.array([sample[0] for sample in batch])
                features_tensor = torch.FloatTensor(features_batch).to(self.device)
                
                # Huấn luyện autoencoder
                self.optimizer.zero_grad()
                reconstructed = self.encoder(features_tensor)
                loss = self.criterion(reconstructed, features_tensor)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            # Lưu loss trung bình của epoch
            if batches > 0:
                avg_epoch_loss = epoch_loss / batches
                self.training_loss_history.append(avg_epoch_loss)
                losses.append(avg_epoch_loss)
        
        # Tính các ngưỡng phát hiện dị thường dựa trên lỗi tái tạo
        self._calculate_reconstruction_thresholds()
        
        self.is_trained = True
        return np.mean(losses) if losses else 0.0
    
    def _calculate_reconstruction_thresholds(self):
        """
        Tính toán các ngưỡng lỗi tái tạo dựa trên dữ liệu trong bộ nhớ.
        """
        if len(self.memory) < self.min_samples_for_training:
            return
            
        # Lấy mẫu bình thường
        normal_samples = [(features, label) for features, label in self.memory if not label]
        if not normal_samples:
            return
            
        features_batch = np.array([sample[0] for sample in normal_samples])
        features_tensor = torch.FloatTensor(features_batch).to(self.device)
        
        self.encoder.eval()
        with torch.no_grad():
            reconstructed = self.encoder(features_tensor)
            # Tính lỗi tái tạo
            reconstruction_errors = F.mse_loss(reconstructed, features_tensor, reduction='none').mean(dim=1).cpu().numpy()
        self.encoder.train()
        
        self.reconstruction_errors = reconstruction_errors
        # Ngưỡng là giá trị trung bình cộng với x lần độ lệch chuẩn
        self.anomaly_threshold = np.mean(reconstruction_errors) + 2.0 * np.std(reconstruction_errors)
    
    def detect_anomaly(self, node_data: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Phát hiện dị thường từ dữ liệu node.
        
        Args:
            node_data: Dữ liệu về hoạt động của node
            
        Returns:
            Tuple[bool, float, Dict[str, Any]]: 
                - Có phải dị thường hay không
                - Điểm dị thường (càng cao càng chắc chắn)
                - Thông tin chi tiết về phát hiện
        """
        if not self.is_trained:
            # Huấn luyện mô hình nếu chưa được huấn luyện
            if len(self.memory) >= self.min_samples_for_training:
                self.train()
            else:
                # Nếu không đủ dữ liệu để huấn luyện, thêm mẫu và trả về False
                self.add_sample(node_data)
                return False, 0.0, {"message": "Not enough data for anomaly detection"}
        
        # Trích xuất đặc trưng
        features = self.extract_features(node_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Tính lỗi tái tạo
        self.encoder.eval()
        with torch.no_grad():
            reconstructed = self.encoder(features_tensor)
            reconstruction_error = F.mse_loss(reconstructed, features_tensor).item()
        self.encoder.train()
        
        # Phát hiện dị thường nếu lỗi tái tạo lớn hơn ngưỡng
        is_anomaly = reconstruction_error > self.anomaly_threshold
        anomaly_score = reconstruction_error / self.anomaly_threshold if self.anomaly_threshold > 0 else 0.0
        
        # Thêm kết quả vào lịch sử
        if is_anomaly:
            self.anomaly_history.append((features, anomaly_score))
            self.detected_anomalies += 1
        else:
            self.normal_history.append(features)
            
        # Thêm mẫu vào bộ nhớ
        self.add_sample(node_data, is_anomaly)
        
        # Tính toán thêm thông tin
        details = {
            "reconstruction_error": reconstruction_error,
            "anomaly_threshold": self.anomaly_threshold,
            "anomaly_score": anomaly_score,
            "is_trained": self.is_trained,
            "memory_size": len(self.memory)
        }
        
        return is_anomaly, anomaly_score, details
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về phát hiện dị thường.
        
        Returns:
            Dict[str, Any]: Thống kê của bộ phát hiện dị thường
        """
        return {
            "detected_anomalies": self.detected_anomalies,
            "false_positives": self.false_positives,
            "is_trained": self.is_trained,
            "training_loss": self.training_loss_history[-1] if self.training_loss_history else 0.0,
            "memory_size": len(self.memory),
            "anomaly_threshold": self.anomaly_threshold,
            "recent_normal_samples": len(self.normal_history),
            "recent_anomaly_samples": len(self.anomaly_history)
        }
    
    def save_model(self, path: str = "models/anomaly_detector.pt"):
        """
        Lưu mô hình vào file.
        
        Args:
            path: Đường dẫn để lưu mô hình
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'anomaly_threshold': self.anomaly_threshold,
            'input_features': self.input_features,
            'detected_anomalies': self.detected_anomalies,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str = "models/anomaly_detector.pt"):
        """
        Tải mô hình từ file.
        
        Args:
            path: Đường dẫn để tải mô hình
        """
        if not os.path.exists(path):
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Tạo lại mô hình với đúng số lượng đặc trưng
        self.input_features = checkpoint['input_features']
        hidden_size = self.encoder.hidden_size
        self.encoder = AutoEncoder(self.input_features, hidden_size).to(self.device)
        
        # Tải trạng thái
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.encoder.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.anomaly_threshold = checkpoint['anomaly_threshold']
        self.detected_anomalies = checkpoint['detected_anomalies']
        self.is_trained = checkpoint['is_trained']
        
        return True


class AutoEncoder(nn.Module):
    """
    Mô hình AutoEncoder cho phát hiện dị thường.
    """
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Khởi tạo AutoEncoder.
        
        Args:
            input_size: Kích thước đầu vào
            hidden_size: Kích thước layer ẩn
        """
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, input_size),
            nn.Sigmoid()  # Giá trị đầu ra trong khoảng [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Truyền dữ liệu qua autoencoder.
        
        Args:
            x: Dữ liệu đầu vào
            
        Returns:
            torch.Tensor: Dữ liệu tái tạo
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LSTMAnomalyDetector(nn.Module):
    """
    Mô hình LSTM cho phát hiện dị thường dựa trên chuỗi thời gian.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """
        Khởi tạo LSTM Anomaly Detector.
        
        Args:
            input_size: Kích thước đầu vào
            hidden_size: Kích thước hidden state của LSTM
            num_layers: Số lớp LSTM chồng lên nhau
        """
        super(LSTMAnomalyDetector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Prediction layer
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Truyền dữ liệu qua LSTM.
        
        Args:
            x: Chuỗi thời gian đầu vào (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Dự đoán chuỗi thời gian tiếp theo
        """
        batch_size = x.size(0)
        
        # Khởi tạo hidden state và cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Lấy output từ time step cuối cùng
        last_time_step = lstm_out[:, -1, :]
        
        # Dự đoán giá trị tiếp theo
        predictions = self.linear(last_time_step)
        
        return predictions


class MLBasedAnomalyDetectionSystem:
    """
    Hệ thống phát hiện dị thường dựa trên ML tích hợp nhiều phương pháp.
    """
    def __init__(self, input_features: int = 20, time_series_length: int = 10):
        """
        Khởi tạo hệ thống phát hiện dị thường.
        
        Args:
            input_features: Số lượng đặc trưng cho mô hình AutoEncoder
            time_series_length: Độ dài chuỗi thời gian cho mô hình LSTM
        """
        # Phát hiện dị thường dựa trên AutoEncoder
        self.anomaly_detector = AnomalyDetector(input_features=input_features)
        
        # Phát hiện dị thường dựa trên chuỗi thời gian (sẽ triển khai sau)
        self.time_series_length = time_series_length
        
        # Lưu trữ lịch sử dữ liệu
        self.node_history = {}  # node_id -> lịch sử dữ liệu
        
        # Thống kê
        self.total_detections = 0
        self.detection_per_node = {}
        self.last_training_time = 0
        
    def process_node_data(self, node_id: int, node_data: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Xử lý dữ liệu từ một node và phát hiện dị thường.
        
        Args:
            node_id: ID của node
            node_data: Dữ liệu về hoạt động của node
            
        Returns:
            Tuple[bool, float, Dict[str, Any]]: 
                - Có phải dị thường hay không
                - Điểm dị thường (càng cao càng chắc chắn)
                - Thông tin chi tiết về phát hiện
        """
        # Lưu dữ liệu node vào lịch sử
        if node_id not in self.node_history:
            self.node_history[node_id] = []
        self.node_history[node_id].append(node_data)
        
        # Giới hạn lịch sử
        if len(self.node_history[node_id]) > 100:
            self.node_history[node_id] = self.node_history[node_id][-100:]
            
        # Huấn luyện định kỳ nếu cần
        current_time = time.time()
        if current_time - self.last_training_time > 300:  # 5 phút
            self._periodic_training()
            self.last_training_time = current_time
            
        # Phát hiện dị thường
        is_anomaly, score, details = self.anomaly_detector.detect_anomaly(node_data)
        
        # Cập nhật thống kê
        if is_anomaly:
            self.total_detections += 1
            if node_id not in self.detection_per_node:
                self.detection_per_node[node_id] = 0
            self.detection_per_node[node_id] += 1
            
        # Thêm thông tin chi tiết
        details["node_id"] = node_id
        details["detection_count"] = self.detection_per_node.get(node_id, 0)
        
        return is_anomaly, score, details
        
    def _periodic_training(self):
        """
        Huấn luyện định kỳ các mô hình phát hiện dị thường.
        """
        # Huấn luyện autoencoder anomaly detector
        if len(self.anomaly_detector.memory) >= self.anomaly_detector.min_samples_for_training:
            self.anomaly_detector.train(epochs=3)
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về hệ thống phát hiện dị thường.
        
        Returns:
            Dict[str, Any]: Thống kê của hệ thống phát hiện dị thường
        """
        detector_stats = self.anomaly_detector.get_statistics()
        
        # Thống kê tổng hợp
        return {
            "total_detections": self.total_detections,
            "nodes_with_anomalies": len(self.detection_per_node),
            "detector_stats": detector_stats,
            "top_anomalous_nodes": sorted(self.detection_per_node.items(), key=lambda x: x[1], reverse=True)[:5]
        } 