"""
Module chứa các hàm đánh giá hiệu suất cho hệ thống QTrust.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional

def calculate_transaction_throughput(successful_txs: int, total_time: float) -> float:
    """
    Tính toán thông lượng giao dịch.
    
    Args:
        successful_txs: Số lượng giao dịch thành công
        total_time: Tổng thời gian (ms)
        
    Returns:
        float: Thông lượng giao dịch (giao dịch/ms)
    """
    if total_time == 0:
        return 0.0
    return successful_txs / total_time

def calculate_latency_metrics(latencies: List[float]) -> Dict[str, float]:
    """
    Tính toán các chỉ số về độ trễ.
    
    Args:
        latencies: Danh sách độ trễ của các giao dịch (ms)
        
    Returns:
        Dict: Các thông số về độ trễ (trung bình, trung vị, cao nhất, v.v.)
    """
    if not latencies:
        return {
            'avg_latency': 0.0,
            'median_latency': 0.0,
            'min_latency': 0.0,
            'max_latency': 0.0,
            'p95_latency': 0.0,
            'p99_latency': 0.0
        }
    
    return {
        'avg_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99)
    }

def calculate_energy_efficiency(energy_consumption: float, successful_txs: int) -> float:
    """
    Tính toán hiệu suất năng lượng.
    
    Args:
        energy_consumption: Tổng năng lượng tiêu thụ
        successful_txs: Số lượng giao dịch thành công
        
    Returns:
        float: Năng lượng tiêu thụ trên mỗi giao dịch thành công
    """
    if successful_txs == 0:
        return float('inf')
    return energy_consumption / successful_txs

def calculate_security_metrics(
    trust_scores: Dict[int, float], 
    malicious_nodes: List[int]
) -> Dict[str, float]:
    """
    Tính toán các chỉ số về bảo mật.
    
    Args:
        trust_scores: Điểm tin cậy của các node
        malicious_nodes: Danh sách ID của các node độc hại
        
    Returns:
        Dict: Các thông số về bảo mật
    """
    total_nodes = len(trust_scores)
    if total_nodes == 0:
        return {
            'avg_trust': 0.0,
            'malicious_ratio': 0.0,
            'trust_variance': 0.0
        }
    
    avg_trust = np.mean(list(trust_scores.values()))
    trust_variance = np.var(list(trust_scores.values()))
    malicious_ratio = len(malicious_nodes) / total_nodes if total_nodes > 0 else 0
    
    return {
        'avg_trust': avg_trust,
        'malicious_ratio': malicious_ratio,
        'trust_variance': trust_variance
    }

def calculate_cross_shard_metrics(
    cross_shard_txs: int, 
    total_txs: int, 
    cross_shard_latencies: List[float],
    intra_shard_latencies: List[float]
) -> Dict[str, float]:
    """
    Tính toán các chỉ số về giao dịch xuyên shard.
    
    Args:
        cross_shard_txs: Số lượng giao dịch xuyên shard
        total_txs: Tổng số giao dịch
        cross_shard_latencies: Độ trễ của các giao dịch xuyên shard
        intra_shard_latencies: Độ trễ của các giao dịch trong shard
        
    Returns:
        Dict: Các thông số về giao dịch xuyên shard
    """
    cross_shard_ratio = cross_shard_txs / total_txs if total_txs > 0 else 0
    
    cross_shard_avg_latency = np.mean(cross_shard_latencies) if cross_shard_latencies else 0
    intra_shard_avg_latency = np.mean(intra_shard_latencies) if intra_shard_latencies else 0
    
    latency_overhead = (cross_shard_avg_latency / intra_shard_avg_latency) if intra_shard_avg_latency > 0 else 0
    
    return {
        'cross_shard_ratio': cross_shard_ratio,
        'cross_shard_avg_latency': cross_shard_avg_latency,
        'intra_shard_avg_latency': intra_shard_avg_latency,
        'latency_overhead': latency_overhead
    }

def generate_performance_report(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Tạo báo cáo hiệu suất từ các chỉ số.
    
    Args:
        metrics: Dictionary chứa các chỉ số hiệu suất
        
    Returns:
        pd.DataFrame: Báo cáo hiệu suất dạng bảng
    """
    report = pd.DataFrame({
        'Metric': [
            'Throughput (tx/s)',
            'Average Latency (ms)',
            'Median Latency (ms)',
            'P95 Latency (ms)',
            'Energy per Transaction',
            'Average Trust Score',
            'Malicious Node Ratio',
            'Cross-Shard Transaction Ratio',
            'Cross-Shard Latency Overhead'
        ],
        'Value': [
            metrics.get('throughput', 0) * 1000,  # Chuyển đổi từ tx/ms sang tx/s
            metrics.get('latency', {}).get('avg_latency', 0),
            metrics.get('latency', {}).get('median_latency', 0),
            metrics.get('latency', {}).get('p95_latency', 0),
            metrics.get('energy_per_tx', 0),
            metrics.get('security', {}).get('avg_trust', 0),
            metrics.get('security', {}).get('malicious_ratio', 0),
            metrics.get('cross_shard', {}).get('cross_shard_ratio', 0),
            metrics.get('cross_shard', {}).get('latency_overhead', 0)
        ]
    })
    
    return report

def plot_trust_distribution(trust_scores: Dict[int, float], 
                           malicious_nodes: List[int], 
                           title: str = "Trust Score Distribution",
                           save_path: Optional[str] = None):
    """
    Vẽ đồ thị phân phối điểm tin cậy của các node.
    
    Args:
        trust_scores: Điểm tin cậy của các node
        malicious_nodes: Danh sách ID của các node độc hại
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(10, 6))
    
    # Tạo danh sách điểm tin cậy cho mỗi nhóm node
    normal_nodes = [node_id for node_id in trust_scores if node_id not in malicious_nodes]
    
    normal_scores = [trust_scores[node_id] for node_id in normal_nodes]
    malicious_scores = [trust_scores[node_id] for node_id in malicious_nodes if node_id in trust_scores]
    
    # Vẽ histogram
    sns.histplot(normal_scores, color='green', alpha=0.5, label='Normal Nodes', bins=20)
    if malicious_scores:
        sns.histplot(malicious_scores, color='red', alpha=0.5, label='Malicious Nodes', bins=20)
    
    plt.title(title)
    plt.xlabel('Trust Score')
    plt.ylabel('Number of Nodes')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_performance_comparison(results: Dict[str, Dict[str, List[float]]], 
                               metric_name: str,
                               title: str,
                               ylabel: str,
                               save_path: Optional[str] = None):
    """
    Vẽ đồ thị so sánh hiệu suất giữa các phương pháp.
    
    Args:
        results: Dictionary chứa kết quả của các phương pháp
        metric_name: Tên của chỉ số cần so sánh
        title: Tiêu đề của đồ thị
        ylabel: Nhãn trục y
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(12, 8))
    
    # Tạo dữ liệu cho boxplot
    data = []
    labels = []
    
    for method_name, method_results in results.items():
        if metric_name in method_results:
            data.append(method_results[metric_name])
            labels.append(method_name)
    
    # Vẽ boxplot
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Đặt màu cho các hộp
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for patch, color in zip(box['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_time_series(data: List[float], 
                    title: str, 
                    xlabel: str, 
                    ylabel: str,
                    window: int = 10,
                    save_path: Optional[str] = None):
    """
    Vẽ đồ thị chuỗi thời gian.
    
    Args:
        data: Dữ liệu cần vẽ
        title: Tiêu đề của đồ thị
        xlabel: Nhãn trục x
        ylabel: Nhãn trục y
        window: Kích thước cửa sổ cho đường trung bình động
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(12, 6))
    
    # Vẽ dữ liệu gốc
    plt.plot(data, alpha=0.5, label='Raw Data')
    
    # Vẽ đường trung bình động
    if len(data) >= window:
        moving_avg = pd.Series(data).rolling(window=window).mean().values
        plt.plot(range(window-1, len(data)), moving_avg[window-1:], 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_heatmap(data: np.ndarray, 
                x_labels: List[str], 
                y_labels: List[str],
                title: str,
                save_path: Optional[str] = None):
    """
    Vẽ heatmap.
    
    Args:
        data: Dữ liệu dạng ma trận
        x_labels: Nhãn cho trục x
        y_labels: Nhãn cho trục y
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, cmap='viridis', xticklabels=x_labels, yticklabels=y_labels)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def calculate_throughput(successful_txs: int, total_time: float) -> float:
    """
    Tính toán thông lượng giao dịch.
    
    Args:
        successful_txs: Số lượng giao dịch thành công
        total_time: Tổng thời gian (s)
        
    Returns:
        float: Thông lượng giao dịch (giao dịch/giây)
    """
    if total_time == 0:
        return 0.0
    return successful_txs / total_time

def calculate_cross_shard_transaction_ratio(cross_shard_txs: int, total_txs: int) -> float:
    """
    Tính toán tỷ lệ giao dịch xuyên shard.
    
    Args:
        cross_shard_txs: Số lượng giao dịch xuyên shard
        total_txs: Tổng số giao dịch
        
    Returns:
        float: Tỷ lệ giao dịch xuyên shard
    """
    if total_txs == 0:
        return 0.0
    return cross_shard_txs / total_txs

def plot_performance_metrics(metrics: Dict[str, List[float]], 
                            title: str = "Performance Metrics Over Time",
                            save_path: Optional[str] = None):
    """
    Vẽ đồ thị các chỉ số hiệu suất theo thời gian.
    
    Args:
        metrics: Dictionary chứa dữ liệu của các chỉ số theo thời gian
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(15, 10))
    
    metric_names = list(metrics.keys())
    num_metrics = len(metric_names)
    
    rows = (num_metrics + 1) // 2  # Số hàng của lưới đồ thị
    cols = min(2, num_metrics)      # Số cột của lưới đồ thị
    
    for i, metric_name in enumerate(metric_names):
        plt.subplot(rows, cols, i+1)
        
        values = metrics[metric_name]
        x = range(len(values))
        
        plt.plot(x, values)
        plt.title(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Thêm đường trung bình động
        window = min(10, len(values))
        if len(values) >= window:
            moving_avg = pd.Series(values).rolling(window=window).mean().values
            plt.plot(range(window-1, len(values)), moving_avg[window-1:], 'r-', linewidth=2, 
                    label=f'Moving Avg (w={window})')
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_comparison_charts(comparison_data: Dict[str, Dict[str, float]], 
                          metrics: List[str],
                          title: str = "Performance Comparison",
                          save_path: Optional[str] = None):
    """
    Vẽ đồ thị so sánh hiệu suất giữa các phương pháp.
    
    Args:
        comparison_data: Dictionary chứa dữ liệu so sánh giữa các phương pháp
        metrics: Danh sách các chỉ số cần so sánh
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(15, 10))
    
    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2
    cols = min(2, num_metrics)
    
    method_names = list(comparison_data.keys())
    
    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i+1)
        
        # Chuẩn bị dữ liệu
        values = [comparison_data[method][metric] for method in method_names]
        
        # Vẽ biểu đồ cột
        bars = plt.bar(method_names, values)
        
        # Thêm nhãn giá trị
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(metric)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_metrics_over_time(metrics_over_time: Dict[str, List[float]],
                           labels: List[str],
                           title: str = "Metrics Over Time",
                           xlabel: str = "Step",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None):
    """
    Vẽ đồ thị các chỉ số theo thời gian.
    
    Args:
        metrics_over_time: Dictionary chứa dữ liệu của các chỉ số theo thời gian
        labels: Nhãn cho từng chỉ số
        title: Tiêu đề của đồ thị
        xlabel: Nhãn trục x
        figsize: Kích thước của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    for i, (metric_name, values) in enumerate(metrics_over_time.items()):
        plt.subplot(len(metrics_over_time), 1, i+1)
        plt.plot(values)
        plt.ylabel(labels[i] if i < len(labels) else metric_name)
        if i == len(metrics_over_time) - 1:
            plt.xlabel(xlabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

class SecurityMetrics:
    """
    Lớp theo dõi và phân tích các chỉ số bảo mật trong hệ thống QTrust.
    """
    def __init__(self, window_size: int = 20):
        """
        Khởi tạo đối tượng SecurityMetrics.
        
        Args:
            window_size: Kích thước cửa sổ phân tích cho các chỉ số bảo mật
        """
        self.analysis_window = window_size
        self.thresholds = {
            "51_percent": 0.7,
            "ddos": 0.65,
            "mixed": 0.6,
            "selfish_mining": 0.75,
            "bribery": 0.7
        }
        self.weights = {
            "51_percent": {
                "failed_tx_ratio": 0.3,
                "high_value_tx_failure": 0.5,
                "node_trust_variance": 0.2
            },
            "ddos": {
                "latency_deviation": 0.5,
                "failed_tx_ratio": 0.3,
                "network_congestion": 0.2
            },
            "mixed": {
                "failed_tx_ratio": 0.25,
                "latency_deviation": 0.25,
                "node_trust_variance": 0.25,
                "high_value_tx_failure": 0.25
            },
            "selfish_mining": {
                "block_withholding": 0.6,
                "fork_rate": 0.4
            },
            "bribery": {
                "voting_deviation": 0.5,
                "trust_inconsistency": 0.5
            }
        }
        
        # Lịch sử các chỉ số tấn công
        self.history = {
            "attack_indicators": [],  # Danh sách các chỉ số tấn công theo thời gian
            "detected_attacks": [],   # Danh sách các tấn công đã phát hiện
            "node_trust_variance": [],  # Lịch sử phương sai tin cậy
            "latency_deviation": [],  # Lịch sử biến động độ trễ
            "failed_tx_ratio": [],    # Lịch sử tỷ lệ giao dịch thất bại
            "security_metrics": []    # Lịch sử các chỉ số bảo mật
        }
        
        # Chỉ số tấn công hiện tại
        self.attack_indicators = {
            "51_percent": 0.0,
            "ddos": 0.0,
            "mixed": 0.0,
            "selfish_mining": 0.0,
            "bribery": 0.0
        }
        
        # Trạng thái tấn công hiện tại
        self.current_attack = None
        self.attack_confidence = 0.0
        
    def calculate_attack_indicators(self, 
                                  failed_tx_ratio: float,
                                  node_trust_variance: float,
                                  latency_deviation: float,
                                  network_metrics: Dict[str, Any],
                                  transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Tính toán các chỉ số cho từng loại tấn công.
        
        Args:
            failed_tx_ratio: Tỷ lệ giao dịch thất bại
            node_trust_variance: Phương sai của điểm tin cậy giữa các node
            latency_deviation: Độ lệch của độ trễ
            network_metrics: Các chỉ số mạng
            transactions: Danh sách giao dịch gần đây
            
        Returns:
            Dict[str, float]: Các chỉ số tấn công
        """
        indicators = {}
        
        # Tính chỉ số tấn công 51%
        indicators['51_percent'] = self._calculate_51_percent_indicator(
            failed_tx_ratio, node_trust_variance, transactions)
        
        # Tính chỉ số tấn công DDoS
        indicators['ddos'] = self._calculate_ddos_indicator(
            failed_tx_ratio, latency_deviation, network_metrics)
        
        # Tính chỉ số tấn công hỗn hợp
        indicators['mixed'] = self._calculate_mixed_indicator(
            failed_tx_ratio, node_trust_variance, latency_deviation, network_metrics, transactions)
        
        # Tính chỉ số tấn công selfish mining
        indicators['selfish_mining'] = self._calculate_selfish_mining_indicator(
            network_metrics, transactions)
        
        # Tính chỉ số tấn công bribery
        indicators['bribery'] = self._calculate_bribery_indicator(
            node_trust_variance, network_metrics, transactions)
        
        # Cập nhật lịch sử
        self.history["attack_indicators"].append(indicators.copy())
        self.history["node_trust_variance"].append(node_trust_variance)
        self.history["latency_deviation"].append(latency_deviation)
        self.history["failed_tx_ratio"].append(failed_tx_ratio)
        
        # Giới hạn kích thước lịch sử
        if len(self.history["attack_indicators"]) > self.analysis_window * 2:
            self.history["attack_indicators"].pop(0)
            self.history["node_trust_variance"].pop(0)
            self.history["latency_deviation"].pop(0)
            self.history["failed_tx_ratio"].pop(0)
        
        # Cập nhật chỉ số tấn công hiện tại
        self.attack_indicators = indicators
        
        return indicators
    
    def detect_attack(self) -> Tuple[Optional[str], float]:
        """
        Phát hiện loại tấn công dựa trên các chỉ số hiện tại.
        
        Returns:
            Tuple[Optional[str], float]: (Loại tấn công, độ tin cậy)
        """
        # Tìm loại tấn công có chỉ số cao nhất
        highest_indicator = 0.0
        detected_attack = None
        
        for attack_type, indicator in self.attack_indicators.items():
            if indicator > highest_indicator:
                highest_indicator = indicator
                detected_attack = attack_type
        
        # Kiểm tra có vượt quá ngưỡng không
        if detected_attack and highest_indicator >= self.thresholds.get(detected_attack, 0.7):
            attack_confidence = highest_indicator
        else:
            detected_attack = None
            attack_confidence = 0.0
        
        # Cập nhật trạng thái tấn công hiện tại
        self.current_attack = detected_attack
        self.attack_confidence = attack_confidence
        
        # Lưu vào lịch sử
        self.history["detected_attacks"].append((detected_attack, attack_confidence))
        
        return detected_attack, attack_confidence
    
    def _calculate_51_percent_indicator(self, 
                                      failed_tx_ratio: float, 
                                      node_trust_variance: float,
                                      transactions: List[Dict[str, Any]]) -> float:
        """
        Tính chỉ số tấn công 51%.
        
        Args:
            failed_tx_ratio: Tỷ lệ giao dịch thất bại
            node_trust_variance: Phương sai của điểm tin cậy giữa các node
            transactions: Danh sách giao dịch gần đây
            
        Returns:
            float: Chỉ số tấn công 51% (0.0-1.0)
        """
        # Tính tỷ lệ thất bại của các giao dịch có giá trị cao
        high_value_txs = [tx for tx in transactions if tx.get('value', 0) > 50]
        high_value_failure_ratio = 0.0
        if high_value_txs:
            high_value_failure_ratio = sum(1 for tx in high_value_txs if tx.get('status') != 'completed') / len(high_value_txs)
        
        # Tính chỉ số tấn công 51% dựa trên trọng số
        weights = self.weights["51_percent"]
        indicator = (
            weights["failed_tx_ratio"] * failed_tx_ratio +
            weights["node_trust_variance"] * min(1.0, node_trust_variance * 10) +
            weights["high_value_tx_failure"] * high_value_failure_ratio
        )
        
        return min(1.0, indicator)
    
    def _calculate_ddos_indicator(self, 
                                failed_tx_ratio: float, 
                                latency_deviation: float,
                                network_metrics: Dict[str, Any]) -> float:
        """
        Tính chỉ số tấn công DDoS.
        
        Args:
            failed_tx_ratio: Tỷ lệ giao dịch thất bại
            latency_deviation: Độ lệch của độ trễ
            network_metrics: Các chỉ số mạng
            
        Returns:
            float: Chỉ số tấn công DDoS (0.0-1.0)
        """
        # Tính chỉ số tắc nghẽn mạng
        network_congestion = network_metrics.get('congestion', latency_deviation)
        
        # Tính chỉ số tấn công DDoS dựa trên trọng số
        weights = self.weights["ddos"]
        indicator = (
            weights["failed_tx_ratio"] * failed_tx_ratio +
            weights["latency_deviation"] * min(1.0, latency_deviation * 2) +
            weights["network_congestion"] * network_congestion
        )
        
        return min(1.0, indicator)
    
    def _calculate_mixed_indicator(self, 
                                 failed_tx_ratio: float, 
                                 node_trust_variance: float,
                                 latency_deviation: float,
                                 network_metrics: Dict[str, Any],
                                 transactions: List[Dict[str, Any]]) -> float:
        """
        Tính chỉ số tấn công hỗn hợp.
        
        Args:
            failed_tx_ratio: Tỷ lệ giao dịch thất bại
            node_trust_variance: Phương sai của điểm tin cậy giữa các node
            latency_deviation: Độ lệch của độ trễ
            network_metrics: Các chỉ số mạng
            transactions: Danh sách giao dịch gần đây
            
        Returns:
            float: Chỉ số tấn công hỗn hợp (0.0-1.0)
        """
        # Tính tỷ lệ thất bại của các giao dịch có giá trị cao
        high_value_txs = [tx for tx in transactions if tx.get('value', 0) > 50]
        high_value_failure_ratio = 0.0
        if high_value_txs:
            high_value_failure_ratio = sum(1 for tx in high_value_txs if tx.get('status') != 'completed') / len(high_value_txs)
        
        # Tính chỉ số tấn công hỗn hợp dựa trên trọng số
        weights = self.weights["mixed"]
        indicator = (
            weights["failed_tx_ratio"] * failed_tx_ratio +
            weights["latency_deviation"] * latency_deviation +
            weights["node_trust_variance"] * min(1.0, node_trust_variance * 5) +
            weights["high_value_tx_failure"] * high_value_failure_ratio
        )
        
        # Tính entropy của sự biến đổi tin cậy
        trust_entropy = self._calculate_trust_entropy(network_metrics)
        
        # Tấn công hỗn hợp thường có các chỉ số không nhất quán và sự biến đổi lớn
        # Phát hiện các mẫu đa dạng
        mixed_patterns = 0
        if failed_tx_ratio > 0.4:
            mixed_patterns += 1
        if latency_deviation > 0.5:
            mixed_patterns += 1
        if node_trust_variance > 0.1:  # Ngưỡng thấp hơn để tăng độ nhạy
            mixed_patterns += 1
        if high_value_failure_ratio > 0.6:
            mixed_patterns += 1
        if trust_entropy > 0.6:
            mixed_patterns += 1
            
        # Tăng chỉ số nếu phát hiện nhiều mẫu đặc trưng của tấn công hỗn hợp
        if mixed_patterns >= 3:
            indicator *= 1.0 + (mixed_patterns - 2) * 0.1
            
        return min(1.0, indicator)
    
    def _calculate_selfish_mining_indicator(self, 
                                          network_metrics: Dict[str, Any],
                                          transactions: List[Dict[str, Any]]) -> float:
        """
        Tính chỉ số tấn công selfish mining.
        
        Args:
            network_metrics: Các chỉ số mạng
            transactions: Danh sách giao dịch gần đây
            
        Returns:
            float: Chỉ số tấn công selfish mining (0.0-1.0)
        """
        # Selfish mining thường gây ra tỷ lệ fork cao và withholding block
        fork_rate = network_metrics.get('fork_rate', 0.0)
        block_withholding = network_metrics.get('block_withholding', 0.0)
        
        # Tính chỉ số tấn công selfish mining dựa trên trọng số
        weights = self.weights["selfish_mining"]
        indicator = (
            weights["fork_rate"] * fork_rate +
            weights["block_withholding"] * block_withholding
        )
        
        return min(1.0, indicator)
    
    def _calculate_bribery_indicator(self, 
                                   node_trust_variance: float,
                                   network_metrics: Dict[str, Any],
                                   transactions: List[Dict[str, Any]]) -> float:
        """
        Tính chỉ số tấn công bribery.
        
        Args:
            node_trust_variance: Phương sai của điểm tin cậy giữa các node
            network_metrics: Các chỉ số mạng
            transactions: Danh sách giao dịch gần đây
            
        Returns:
            float: Chỉ số tấn công bribery (0.0-1.0)
        """
        # Bribery thường gây ra sự không nhất quán trong biểu quyết và tin cậy
        voting_deviation = network_metrics.get('voting_deviation', 0.0)
        trust_inconsistency = min(1.0, node_trust_variance * 5)
        
        # Tính chỉ số tấn công bribery dựa trên trọng số
        weights = self.weights["bribery"]
        indicator = (
            weights["voting_deviation"] * voting_deviation +
            weights["trust_inconsistency"] * trust_inconsistency
        )
        
        return min(1.0, indicator)

    def _calculate_trust_entropy(self, network_metrics: Dict[str, Any]) -> float:
        """
        Tính entropy của sự biến đổi tin cậy.
        
        Args:
            network_metrics: Các chỉ số mạng
            
        Returns:
            float: Entropy của sự biến đổi tin cậy (0.0-1.0)
        """
        # Lấy lịch sử phương sai tin cậy
        if len(self.history["node_trust_variance"]) < 2:
            return 0.0
            
        # Tính độ biến đổi
        trust_variances = self.history["node_trust_variance"][-self.analysis_window:]
        trust_variance_changes = [abs(trust_variances[i] - trust_variances[i-1]) 
                                  for i in range(1, len(trust_variances))]
        
        if not trust_variance_changes:
            return 0.0
            
        # Tính entropy đơn giản dựa trên mức độ biến động
        avg_change = np.mean(trust_variance_changes)
        max_change = max(trust_variance_changes)
        entropy = avg_change / max(0.001, max_change)
        
        return min(1.0, entropy * 2)  # Nhân 2 để tăng độ nhạy

    def update_security_metrics(self, 
                              detected_attack: str,
                              attack_confidence: float,
                              network_metrics: Dict[str, Any],
                              previous_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Cập nhật và trả về các chỉ số bảo mật dựa trên tấn công được phát hiện.
        
        Args:
            detected_attack: Loại tấn công được phát hiện
            attack_confidence: Độ tin cậy của phát hiện tấn công
            network_metrics: Các chỉ số mạng hiện tại
            previous_state: Trạng thái bảo mật trước đó
            
        Returns:
            Dict[str, float]: Các chỉ số bảo mật được cập nhật
        """
        security_metrics = {}
        
        # Cập nhật chỉ số tổng quát
        security_metrics['overall_security'] = 1.0 - (attack_confidence if detected_attack else 0.0)
        
        # Cập nhật chỉ số phục hồi
        prev_security = previous_state.get('overall_security', 1.0)
        security_metrics['recovery_rate'] = max(0.0, (security_metrics['overall_security'] - prev_security) / max(0.01, 1.0 - prev_security))
        
        # Cập nhật chỉ số phát hiện
        security_metrics['detection_level'] = attack_confidence if detected_attack else 0.0
        
        # Cập nhật điểm bảo vệ cho từng loại tấn công
        security_metrics['51_percent_protection'] = 1.0 - self.attack_indicators.get('51_percent', 0.0)
        security_metrics['ddos_protection'] = 1.0 - self.attack_indicators.get('ddos', 0.0)
        security_metrics['mixed_protection'] = 1.0 - self.attack_indicators.get('mixed', 0.0)
        
        # Chỉ số ổn định mạng
        security_metrics['network_stability'] = max(0.0, 1.0 - network_metrics.get('latency_deviation', 0.0))
        
        # Chỉ số tin cậy giao dịch
        security_metrics['transaction_reliability'] = max(0.0, 1.0 - network_metrics.get('failed_tx_ratio', 0.0))
        
        return security_metrics

def plot_attack_detection_results(security_metrics, output_dir=None):
    """
    Vẽ biểu đồ kết quả phát hiện tấn công.
    
    Args:
        security_metrics: Đối tượng SecurityMetrics chứa lịch sử phát hiện tấn công
        output_dir: Thư mục đầu ra cho biểu đồ
    """
    # Tạo dữ liệu cho biểu đồ
    history = security_metrics.history
    
    if len(history["attack_indicators"]) < 2:
        return
    
    # Chuẩn bị dữ liệu
    time_points = list(range(len(history["attack_indicators"])))
    attack_types = list(history["attack_indicators"][0].keys())
    
    # Tạo dataframe cho dễ vẽ
    attack_data = {attack_type: [indicators[attack_type] for indicators in history["attack_indicators"]] 
                  for attack_type in attack_types}
    attack_data['time'] = time_points
    df = pd.DataFrame(attack_data)
    
    # Chuẩn bị dữ liệu cho tấn công được phát hiện
    detected_data = []
    for i, (attack, confidence) in enumerate(history["detected_attacks"]):
        if attack:
            detected_data.append((i, attack, confidence))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    
    # Vẽ chỉ số các loại tấn công
    for attack_type in attack_types:
        plt.plot(df['time'], df[attack_type], label=f'{attack_type} Indicator')
    
    # Vẽ ngưỡng phát hiện
    for attack_type, threshold in security_metrics.thresholds.items():
        if attack_type in attack_types:
            plt.axhline(y=threshold, linestyle='--', alpha=0.5, color='gray', 
                       label=f'{attack_type} Threshold')
    
    # Đánh dấu các điểm phát hiện tấn công
    if detected_data:
        for time_point, attack, confidence in detected_data:
            plt.scatter(time_point, confidence, marker='*', s=150, 
                       color='red', label=f'Detected {attack}' if attack != detected_data[0][1] else None)
            plt.annotate(f'{attack}', (time_point, confidence), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('Attack Detection Results Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Attack Indicator Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu biểu đồ nếu cần
    if output_dir:
        plt.savefig(f"{output_dir}/attack_detection_results.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_security_metrics_comparison(security_metrics, output_dir=None):
    """
    So sánh các chỉ số bảo mật giữa các thời điểm khác nhau.
    
    Args:
        security_metrics: Danh sách các chỉ số bảo mật theo thời gian
        output_dir: Thư mục đầu ra cho biểu đồ
    """
    # Kiểm tra dữ liệu đầu vào
    if not isinstance(security_metrics, list) or len(security_metrics) < 2:
        return
    
    # Chuẩn bị dữ liệu
    metrics_keys = ['overall_security', 'recovery_rate', 'detection_level', 
                   '51_percent_protection', 'ddos_protection', 'mixed_protection',
                   'network_stability', 'transaction_reliability']
    
    # Tạo dataframe chứa dữ liệu từ các thời điểm
    data = []
    for i, metrics in enumerate(security_metrics):
        metrics_values = {key: metrics.get(key, 0.0) for key in metrics_keys if key in metrics}
        metrics_values['time_point'] = i
        data.append(metrics_values)
    
    df = pd.DataFrame(data)
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(14, 10))
    
    # Vẽ từng chỉ số theo thời gian
    for key in metrics_keys:
        if key in df.columns:
            plt.plot(df['time_point'], df[key], marker='o', label=key)
    
    plt.title('Security Metrics Comparison Over Time')
    plt.xlabel('Time Points')
    plt.ylabel('Metric Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu biểu đồ nếu cần
    if output_dir:
        plt.savefig(f"{output_dir}/security_metrics_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    # Biểu đồ heatmap cho sự thay đổi của các chỉ số theo thời gian
    if len(df) > 5:
        plt.figure(figsize=(12, 8))
        
        # Chuẩn bị dữ liệu cho heatmap
        heatmap_data = df[metrics_keys].T if all(key in df.columns for key in metrics_keys) else df.drop('time_point', axis=1).T
        
        # Vẽ heatmap
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
        
        plt.title('Security Metrics Change Over Time')
        plt.xlabel('Time Points')
        plt.ylabel('Security Metrics')
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if output_dir:
            plt.savefig(f"{output_dir}/security_metrics_heatmap.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def plot_attack_impact_radar(attack_metrics, output_dir=None):
    """
    Vẽ biểu đồ radar để so sánh tác động của các loại tấn công khác nhau.
    
    Args:
        attack_metrics: Dict chứa các chỉ số ảnh hưởng của các loại tấn công
        output_dir: Thư mục đầu ra cho biểu đồ
    """
    # Kiểm tra dữ liệu đầu vào
    if not isinstance(attack_metrics, dict) or not attack_metrics:
        return
    
    # Chuẩn bị dữ liệu
    attack_types = list(attack_metrics.keys())
    metrics = list(attack_metrics[attack_types[0]].keys()) if attack_types else []
    
    if not metrics:
        return
    
    # Tạo dataframe
    data = {metric: [attack_metrics[attack][metric] for attack in attack_types] for metric in metrics}
    df = pd.DataFrame(data, index=attack_types)
    
    # Chuẩn bị dữ liệu cho biểu đồ radar
    categories = metrics
    N = len(categories)
    
    # Tạo góc cho mỗi chỉ số
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Đóng vòng
    
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Thêm dữ liệu cho từng loại tấn công
    for attack in attack_types:
        values = df.loc[attack].tolist()
        values += values[:1]  # Đóng vòng
        ax.plot(angles, values, linewidth=2, label=attack)
        ax.fill(angles, values, alpha=0.25)
    
    # Tùy chỉnh biểu đồ
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Attack Impact Comparison')
    
    # Lưu biểu đồ nếu cần
    if output_dir:
        plt.savefig(f"{output_dir}/attack_impact_radar.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close() 