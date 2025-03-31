import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Thêm thư mục hiện tại vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from large_scale_simulation import LargeScaleBlockchainSimulation

def run_attack_comparison(base_args, output_dir='results_attack_comparison'):
    """Chạy so sánh các loại tấn công khác nhau."""
    attack_types = [None, '51_percent', 'sybil', 'eclipse', 'selfish_mining', 'bribery', 'ddos', 'finney', 'mixed']
    all_metrics = {}
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Bắt đầu so sánh các loại tấn công ===")
    
    # Chạy mô phỏng cho mỗi loại tấn công
    for attack_type in attack_types:
        attack_name = attack_type if attack_type else "no_attack"
        print(f"\n== Đang chạy mô phỏng với tấn công: {attack_name} ==")
        
        # Cài đặt tỷ lệ nút độc hại phù hợp với loại tấn công
        malicious_percentage = base_args.malicious
        if attack_type == '51_percent':
            malicious_percentage = 51
        elif attack_type == 'sybil':
            malicious_percentage = 30
        elif attack_type == 'selfish_mining':
            malicious_percentage = 20
        elif attack_type == 'bribery':
            malicious_percentage = 25
        elif attack_type == 'ddos':
            malicious_percentage = 15
        elif attack_type == 'finney':
            malicious_percentage = 20
        elif attack_type == 'mixed':
            malicious_percentage = 40
        
        # Tạo mô phỏng
        simulation = LargeScaleBlockchainSimulation(
            num_shards=base_args.num_shards,
            nodes_per_shard=base_args.nodes_per_shard,
            malicious_percentage=malicious_percentage,
            attack_scenario=attack_type
        )
        
        # Chạy mô phỏng với ít bước hơn để tiết kiệm thời gian
        steps = base_args.steps // 2 if attack_type else base_args.steps
        metrics = simulation.run_simulation(
            num_steps=steps,
            transactions_per_step=base_args.tx_per_step
        )
        
        # Lưu metrics
        all_metrics[attack_name] = {
            'throughput': np.mean(metrics['throughput'][-100:]),
            'latency': np.mean(metrics['latency'][-100:]),
            'energy': np.mean(metrics['energy'][-100:]),
            'security': np.mean(metrics['security'][-100:]),
            'cross_shard_ratio': np.mean(metrics['cross_shard_ratio'][-100:])
        }
        
        # Tạo biểu đồ và báo cáo
        print(f"Đang tạo biểu đồ chi tiết cho tấn công {attack_name}...")
        simulation.plot_metrics(save_dir=output_dir)
        simulation.generate_report(save_dir=output_dir)
        print(f"Đã hoàn thành mô phỏng với tấn công {attack_name}")
    
    # Tạo báo cáo so sánh
    print("\nĐang tạo báo cáo so sánh giữa các loại tấn công...")
    generate_comparison_report(all_metrics, output_dir)
    
    # Tạo biểu đồ so sánh
    print("Đang tạo biểu đồ so sánh giữa các loại tấn công...")
    plot_comparison_charts(all_metrics, output_dir)
    
    print(f"Đã hoàn thành so sánh các loại tấn công. Kết quả được lưu trong {output_dir}")
    
    return all_metrics

def run_scale_comparison(base_args, output_dir='results_scale_comparison'):
    """Chạy so sánh các quy mô mạng khác nhau."""
    # Các cấu hình quy mô khác nhau
    scale_configs = [
        {"name": "small", "num_shards": 4, "nodes_per_shard": 10},
        {"name": "medium", "num_shards": 8, "nodes_per_shard": 20},
        {"name": "large", "num_shards": 16, "nodes_per_shard": 30},
        {"name": "xlarge", "num_shards": 32, "nodes_per_shard": 40}
    ]
    
    all_metrics = {}
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Bắt đầu so sánh các quy mô mạng ===")
    
    # Chạy mô phỏng cho mỗi quy mô
    for config in scale_configs:
        print(f"\n== Đang chạy mô phỏng với quy mô: {config['name']} ==")
        print(f"Cấu hình: {config['num_shards']} shard, {config['nodes_per_shard']} nút/shard")
        
        # Tạo mô phỏng
        simulation = LargeScaleBlockchainSimulation(
            num_shards=config['num_shards'],
            nodes_per_shard=config['nodes_per_shard'],
            malicious_percentage=base_args.malicious,
            attack_scenario=base_args.attack
        )
        
        # Điều chỉnh số bước dựa trên quy mô
        scale_factor = config['num_shards'] / base_args.num_shards
        steps = int(base_args.steps / scale_factor)
        steps = max(steps, 200)  # Ít nhất 200 bước
        
        # Chạy mô phỏng
        metrics = simulation.run_simulation(
            num_steps=steps,
            transactions_per_step=base_args.tx_per_step
        )
        
        # Lưu metrics
        all_metrics[config['name']] = {
            'throughput': np.mean(metrics['throughput'][-100:]),
            'latency': np.mean(metrics['latency'][-100:]),
            'energy': np.mean(metrics['energy'][-100:]),
            'security': np.mean(metrics['security'][-100:]),
            'cross_shard_ratio': np.mean(metrics['cross_shard_ratio'][-100:]),
            'num_nodes': config['num_shards'] * config['nodes_per_shard']
        }
        
        # Tạo biểu đồ và báo cáo
        print(f"Đang tạo biểu đồ chi tiết cho quy mô {config['name']}...")
        simulation.plot_metrics(save_dir=output_dir)
        simulation.generate_report(save_dir=output_dir)
        print(f"Đã hoàn thành mô phỏng với quy mô {config['name']}")
    
    # Tạo báo cáo so sánh
    print("\nĐang tạo báo cáo so sánh giữa các quy mô mạng...")
    generate_scale_comparison_report(all_metrics, output_dir)
    
    # Tạo biểu đồ so sánh
    print("Đang tạo biểu đồ so sánh giữa các quy mô mạng...")
    plot_scale_comparison_charts(all_metrics, output_dir)
    
    print(f"Đã hoàn thành so sánh các quy mô mạng. Kết quả được lưu trong {output_dir}")
    
    return all_metrics

def generate_comparison_report(metrics, output_dir):
    """Tạo báo cáo so sánh các loại tấn công."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/attack_comparison_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("So sánh các loại tấn công trong QTrust\n")
        f.write("=====================================\n\n")
        
        # Tiêu đề cột
        f.write(f"{'Loại tấn công':<20} {'Throughput':<12} {'Độ trễ':<12} {'Năng lượng':<12} {'Bảo mật':<12} {'Xuyên shard':<12}\n")
        f.write(f"{'-'*76}\n")
        
        # Dữ liệu từng dòng
        for attack_name, attack_metrics in metrics.items():
            display_name = "Không tấn công" if attack_name == "no_attack" else attack_name
            f.write(f"{display_name:<20} {attack_metrics['throughput']:<12.2f} {attack_metrics['latency']:<12.2f} ")
            f.write(f"{attack_metrics['energy']:<12.2f} {attack_metrics['security']:<12.2f} ")
            f.write(f"{attack_metrics['cross_shard_ratio']:<12.2f}\n")
        
        f.write("\nPhân tích:\n\n")
        
        # Tìm kiếm giá trị tốt/xấu nhất
        best_throughput = max(metrics.items(), key=lambda x: x[1]['throughput'])
        worst_latency = max(metrics.items(), key=lambda x: x[1]['latency'])
        best_security = max(metrics.items(), key=lambda x: x[1]['security'])
        
        f.write(f"1. Throughput tốt nhất: {best_throughput[0]} ({best_throughput[1]['throughput']:.2f} tx/s)\n")
        f.write(f"2. Độ trễ cao nhất: {worst_latency[0]} ({worst_latency[1]['latency']:.2f} ms)\n")
        f.write(f"3. Bảo mật tốt nhất: {best_security[0]} ({best_security[1]['security']:.2f})\n\n")
        
        f.write("Đánh giá tác động của các loại tấn công:\n\n")
        
        # So sánh với trường hợp không tấn công
        baseline = metrics["no_attack"]
        for attack_name, attack_metrics in metrics.items():
            if attack_name == "no_attack":
                continue
                
            throughput_change = ((attack_metrics['throughput'] - baseline['throughput']) / baseline['throughput']) * 100
            latency_change = ((attack_metrics['latency'] - baseline['latency']) / baseline['latency']) * 100
            security_change = ((attack_metrics['security'] - baseline['security']) / baseline['security']) * 100
            
            f.write(f"Tấn công {attack_name}:\n")
            f.write(f"  - Throughput: {throughput_change:.2f}%\n")
            f.write(f"  - Độ trễ: {latency_change:.2f}%\n")
            f.write(f"  - Bảo mật: {security_change:.2f}%\n\n")
    
    print(f"Đã lưu báo cáo so sánh tại: {filename}")

def generate_scale_comparison_report(metrics, output_dir):
    """Tạo báo cáo so sánh các quy mô mạng."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/scale_comparison_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("So sánh hiệu suất QTrust trên các quy mô mạng\n")
        f.write("===========================================\n\n")
        
        # Tiêu đề cột
        f.write(f"{'Quy mô':<12} {'Số nút':<12} {'Throughput':<12} {'Độ trễ':<12} {'Năng lượng':<12} {'Bảo mật':<12} {'Xuyên shard':<12}\n")
        f.write(f"{'-'*84}\n")
        
        # Dữ liệu từng dòng
        for scale_name, scale_metrics in metrics.items():
            f.write(f"{scale_name:<12} {scale_metrics['num_nodes']:<12d} {scale_metrics['throughput']:<12.2f} ")
            f.write(f"{scale_metrics['latency']:<12.2f} {scale_metrics['energy']:<12.2f} ")
            f.write(f"{scale_metrics['security']:<12.2f} {scale_metrics['cross_shard_ratio']:<12.2f}\n")
        
        f.write("\nPhân tích mở rộng quy mô:\n\n")
        
        # Tính toán mối quan hệ giữa quy mô và hiệu suất
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['num_nodes'])
        
        smallest = sorted_metrics[0][1]
        largest = sorted_metrics[-1][1]
        
        node_scale = largest['num_nodes'] / smallest['num_nodes']
        throughput_scale = largest['throughput'] / smallest['throughput']
        latency_scale = largest['latency'] / smallest['latency']
        
        f.write(f"1. Tỷ lệ mở rộng nút: {node_scale:.2f}x\n")
        f.write(f"2. Tỷ lệ mở rộng throughput: {throughput_scale:.2f}x\n")
        f.write(f"3. Tỷ lệ tăng độ trễ: {latency_scale:.2f}x\n\n")
        
        # Tính toán hiệu quả mở rộng
        scaling_efficiency = throughput_scale / node_scale * 100
        f.write(f"Hiệu quả mở rộng: {scaling_efficiency:.2f}%\n")
        f.write(f"(100% là tuyến tính hoàn hảo, >100% là siêu tuyến tính, <100% là dưới tuyến tính)\n\n")
        
        f.write("Mối quan hệ giữa quy mô và độ trễ:\n")
        for i in range(1, len(sorted_metrics)):
            prev = sorted_metrics[i-1][1]
            curr = sorted_metrics[i][1]
            
            node_increase = (curr['num_nodes'] - prev['num_nodes']) / prev['num_nodes'] * 100
            latency_increase = (curr['latency'] - prev['latency']) / prev['latency'] * 100
            
            f.write(f"  - Khi tăng số nút {node_increase:.2f}%, độ trễ tăng {latency_increase:.2f}%\n")
    
    print(f"Đã lưu báo cáo so sánh quy mô tại: {filename}")

def plot_comparison_charts(metrics, output_dir):
    """Tạo biểu đồ so sánh các loại tấn công."""
    # Chuẩn bị dữ liệu
    attack_names = list(metrics.keys())
    display_names = ["No Attack" if name == "no_attack" else name for name in attack_names]
    
    throughputs = [metrics[name]['throughput'] for name in attack_names]
    latencies = [metrics[name]['latency'] for name in attack_names]
    securities = [metrics[name]['security'] for name in attack_names]
    energies = [metrics[name]['energy'] for name in attack_names]
    
    # Thiết lập style
    plt.style.use('dark_background')
    sns.set(style="darkgrid")
    
    # Tạo bảng màu đẹp
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    # Tạo biểu đồ cột
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Throughput
    bars1 = axes[0, 0].bar(display_names, throughputs, color=colors[0], alpha=0.7)
    axes[0, 0].set_title('Throughput (tx/s)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('tx/s', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    # Thêm giá trị trên đầu cột
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Latency
    bars2 = axes[0, 1].bar(display_names, latencies, color=colors[1], alpha=0.7)
    axes[0, 1].set_title('Độ trễ (ms)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('ms', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    # Thêm giá trị trên đầu cột
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Security
    bars3 = axes[1, 0].bar(display_names, securities, color=colors[2], alpha=0.7)
    axes[1, 0].set_title('Điểm bảo mật', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Điểm (0-1)', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    # Thêm giá trị trên đầu cột
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Energy
    bars4 = axes[1, 1].bar(display_names, energies, color=colors[3], alpha=0.7)
    axes[1, 1].set_title('Tiêu thụ năng lượng', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Đơn vị năng lượng', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)
    # Thêm giá trị trên đầu cột
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Tiêu đề chính
    fig.suptitle('So sánh hiệu suất dưới các loại tấn công khác nhau', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/attack_comparison_chart_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ so sánh tại: {filename}")
    
    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close(fig)
    
    # Tạo biểu đồ radar để so sánh tất cả các chỉ số
    plot_radar_comparison(metrics, output_dir)

def plot_scale_comparison_charts(metrics, output_dir):
    """Tạo biểu đồ so sánh các quy mô mạng."""
    # Chuẩn bị dữ liệu
    scale_names = list(metrics.keys())
    num_nodes = [metrics[name]['num_nodes'] for name in scale_names]
    throughputs = [metrics[name]['throughput'] for name in scale_names]
    latencies = [metrics[name]['latency'] for name in scale_names]
    securities = [metrics[name]['security'] for name in scale_names]
    
    # Sắp xếp dữ liệu theo số nút
    sorted_data = sorted(zip(num_nodes, scale_names, throughputs, latencies, securities))
    num_nodes = [d[0] for d in sorted_data]
    scale_names = [d[1] for d in sorted_data]
    throughputs = [d[2] for d in sorted_data]
    latencies = [d[3] for d in sorted_data]
    securities = [d[4] for d in sorted_data]
    
    # Thiết lập style
    plt.style.use('dark_background')
    sns.set(style="darkgrid")
    
    # Bảng màu tốt hơn
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    # Tạo biểu đồ
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Throughput vs Number of Nodes
    axes[0].plot(num_nodes, throughputs, 'o-', color=colors[0], linewidth=3, markersize=10)
    axes[0].set_title('Throughput vs Số nút', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Số nút', fontsize=14)
    axes[0].set_ylabel('Throughput (tx/s)', fontsize=14, color=colors[0])
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(num_nodes, throughputs, alpha=0.2, color=colors[0])
    
    # Đánh dấu điểm với nhãn
    for i, (x, y) in enumerate(zip(num_nodes, throughputs)):
        axes[0].annotate(f"{scale_names[i]}: {y:.2f}",
                        (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12)
    
    # Latency vs Number of Nodes
    axes[1].plot(num_nodes, latencies, 'o-', color=colors[1], linewidth=3, markersize=10)
    axes[1].set_title('Độ trễ vs Số nút', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Số nút', fontsize=14)
    axes[1].set_ylabel('Độ trễ (ms)', fontsize=14, color=colors[1])
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(num_nodes, latencies, alpha=0.2, color=colors[1])
    
    # Đánh dấu điểm với nhãn
    for i, (x, y) in enumerate(zip(num_nodes, latencies)):
        axes[1].annotate(f"{scale_names[i]}: {y:.2f}",
                        (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12)
    
    # Security vs Number of Nodes
    axes[2].plot(num_nodes, securities, 'o-', color=colors[2], linewidth=3, markersize=10)
    axes[2].set_title('Bảo mật vs Số nút', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Số nút', fontsize=14)
    axes[2].set_ylabel('Điểm bảo mật', fontsize=14, color=colors[2])
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(num_nodes, securities, alpha=0.2, color=colors[2])
    
    # Đánh dấu điểm với nhãn
    for i, (x, y) in enumerate(zip(num_nodes, securities)):
        axes[2].annotate(f"{scale_names[i]}: {y:.2f}",
                        (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12)
    
    plt.tight_layout()
    
    # Tiêu đề chính
    fig.suptitle('Đánh giá khả năng mở rộng của QTrust Blockchain', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/scale_comparison_chart_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ so sánh quy mô tại: {filename}")
    
    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close(fig)
    
    # Tạo biểu đồ tăng trưởng
    plot_scaling_efficiency(metrics, output_dir)

def plot_radar_comparison(metrics, output_dir):
    """Tạo biểu đồ radar để so sánh tất cả các loại tấn công."""
    # Chuẩn bị dữ liệu
    attack_names = list(metrics.keys())
    display_names = ["No Attack" if name == "no_attack" else name for name in attack_names]
    
    # Chuẩn hóa các chỉ số
    max_throughput = max(metrics[name]['throughput'] for name in attack_names)
    max_latency = max(metrics[name]['latency'] for name in attack_names)
    max_energy = max(metrics[name]['energy'] for name in attack_names)
    
    normalized_metrics = {}
    for name in attack_names:
        normalized_metrics[name] = {
            'throughput': metrics[name]['throughput'] / max_throughput,
            # Đảo ngược latency và energy vì giá trị thấp hơn là tốt hơn
            'latency': 1 - (metrics[name]['latency'] / max_latency),
            'energy': 1 - (metrics[name]['energy'] / max_energy),
            'security': metrics[name]['security']
        }
    
    # Các nhãn kết quả
    categories = ['Throughput', 'Độ trễ', 'Năng lượng', 'Bảo mật']
    
    # Số lượng biến
    N = len(categories)
    
    # Góc cho mỗi trục
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Thiết lập style
    plt.style.use('dark_background')
    
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Bảng màu tốt hơn
    colors = plt.cm.rainbow(np.linspace(0, 1, len(attack_names)))
    
    # Thêm dữ liệu cho mỗi loại tấn công
    for i, name in enumerate(attack_names):
        values = [
            normalized_metrics[name]['throughput'],
            normalized_metrics[name]['latency'],
            normalized_metrics[name]['energy'],
            normalized_metrics[name]['security']
        ]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=display_names[i], color=colors[i])
        ax.fill(angles, values, alpha=0.2, color=colors[i])
    
    # Cài đặt nhãn
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=14)
    
    # Cài đặt y-ticks
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rlabel_position(0)
    ax.tick_params(axis='y', labelsize=10, colors='gray')
    
    # Thêm tiêu đề và chú thích
    ax.set_title("So sánh hiệu suất dưới các loại tấn công khác nhau", size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/attack_radar_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ radar so sánh tại: {filename}")
    
    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close(fig)

def plot_scaling_efficiency(metrics, output_dir):
    """Tạo biểu đồ hiệu quả mở rộng."""
    # Chuẩn bị dữ liệu
    scale_names = list(metrics.keys())
    sorted_data = sorted([(metrics[name]['num_nodes'], name) for name in scale_names])
    
    num_nodes = [d[0] for d in sorted_data]
    scale_names = [d[1] for d in sorted_data]
    
    # Tính hiệu quả mở rộng
    base_nodes = num_nodes[0]
    base_throughput = metrics[scale_names[0]]['throughput']
    
    ideal_scaling = [base_throughput * (n / base_nodes) for n in num_nodes]
    actual_throughput = [metrics[name]['throughput'] for name in scale_names]
    
    scaling_efficiency = [(actual / ideal) * 100 for actual, ideal in zip(actual_throughput, ideal_scaling)]
    
    # Thiết lập style
    plt.style.use('dark_background')
    
    # Tạo biểu đồ
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Trục chính: Throughput
    line1 = ax1.plot(num_nodes, actual_throughput, 'o-', color='#3498db', linewidth=3, markersize=10, label='Actual Throughput')
    line2 = ax1.plot(num_nodes, ideal_scaling, '--', color='#95a5a6', linewidth=2, label='Ideal Linear Scaling')
    ax1.set_xlabel('Số nút', fontsize=14)
    ax1.set_ylabel('Throughput (tx/s)', color='#3498db', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.grid(True, alpha=0.3)
    
    # Đánh dấu điểm với nhãn
    for i, (x, y) in enumerate(zip(num_nodes, actual_throughput)):
        ax1.annotate(f"{scale_names[i]}",
                    (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=12)
    
    # Trục phụ: Hiệu quả mở rộng
    ax2 = ax1.twinx()
    line3 = ax2.plot(num_nodes, scaling_efficiency, 'o-', color='#e74c3c', linewidth=3, markersize=10, label='Scaling Efficiency')
    ax2.set_ylabel('Hiệu quả mở rộng (%)', color='#e74c3c', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Thêm đường ngang ở 100%
    ax2.axhline(y=100, linestyle='--', color='#2ecc71', alpha=0.7, linewidth=2)
    
    # Đánh dấu hiệu quả mở rộng
    for i, (x, y) in enumerate(zip(num_nodes, scaling_efficiency)):
        ax2.annotate(f"{y:.1f}%",
                    (x, y), xytext=(0, -20),
                    textcoords='offset points', fontsize=12, color='#e74c3c')
    
    # Kết hợp các đường cho chú thích
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)
    
    # Tiêu đề
    plt.title('Hiệu quả mở rộng của QTrust Blockchain', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/scaling_efficiency_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ hiệu quả mở rộng tại: {filename}")
    
    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='QTrust Attack and Scale Simulation Runner')
    parser.add_argument('--mode', type=str, choices=['attack', 'scale', 'both'], default='both',
                        help='Chế độ mô phỏng: attack (so sánh tấn công), scale (so sánh quy mô), both (cả hai)')
    parser.add_argument('--num-shards', type=int, default=10, help='Số lượng shard cơ bản')
    parser.add_argument('--nodes-per-shard', type=int, default=20, help='Số nút trên mỗi shard cơ bản')
    parser.add_argument('--steps', type=int, default=500, help='Số bước mô phỏng')
    parser.add_argument('--tx-per-step', type=int, default=50, help='Số giao dịch mỗi bước')
    parser.add_argument('--malicious', type=float, default=10, help='Tỷ lệ %% nút độc hại')
    parser.add_argument('--attack', type=str, choices=['51_percent', 'sybil', 'eclipse', 'mixed', None], 
                        default=None, help='Kịch bản tấn công cơ bản')
    parser.add_argument('--output-dir', type=str, default='results_comparison', help='Thư mục lưu kết quả')
    parser.add_argument('--no-display', action='store_true', help='Không hiển thị kết quả chi tiết trên màn hình')
    parser.add_argument('--high-quality', action='store_true', help='Tạo biểu đồ chất lượng cao (dpi cao hơn)')
    
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Thiết lập dpi cho biểu đồ chất lượng cao
    if args.high_quality:
        plt.rcParams['figure.dpi'] = 300
    
    # Thiết lập để không hiển thị biểu đồ trên màn hình
    if args.no_display:
        plt.switch_backend('Agg')
    
    if args.mode == 'attack' or args.mode == 'both':
        attack_output_dir = os.path.join(args.output_dir, 'attack_comparison')
        attack_metrics = run_attack_comparison(args, attack_output_dir)
    
    if args.mode == 'scale' or args.mode == 'both':
        scale_output_dir = os.path.join(args.output_dir, 'scale_comparison')
        scale_metrics = run_scale_comparison(args, scale_output_dir)
    
    print("\nMô phỏng hoàn tất! Kết quả đã được lưu trong thư mục:", args.output_dir)

if __name__ == "__main__":
    main() 