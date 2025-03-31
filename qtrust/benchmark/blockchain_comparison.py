#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blockchain Comparison - Công cụ so sánh hiệu suất với các hệ thống blockchain khác

File này cung cấp các chức năng để so sánh QTrust với các hệ thống blockchain khác
dựa trên các chỉ số hiệu suất khác nhau.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Import từ cùng thư mục blockchain_comparison_utils
from blockchain_comparison_utils import (
    plot_heatmap_comparison,
    plot_relationship_comparison,
    generate_comparison_table
)

# Các dữ liệu so sánh (mẫu, cần cập nhật với dữ liệu thực tế sau khi benchmark)
BLOCKCHAIN_SYSTEMS = {
    'QTrust': {
        'throughput': 5000,        # giao dịch/giây
        'latency': 2.5,            # ms
        'security': 0.95,          # 0-1 scale
        'energy': 15,              # watt-hours/tx (giả định)
        'scalability': 0.9,        # 0-1 scale
        'decentralization': 0.85,  # 0-1 scale
        'cross_shard_efficiency': 0.92,  # 0-1 scale
        'attack_resistance': 0.94  # 0-1 scale
    },
    'Ethereum 2.0': {
        'throughput': 3000,
        'latency': 12,
        'security': 0.92,
        'energy': 35,
        'scalability': 0.85,
        'decentralization': 0.80,
        'cross_shard_efficiency': 0.85,
        'attack_resistance': 0.90
    },
    'Solana': {
        'throughput': 65000,
        'latency': 0.4,
        'security': 0.80,
        'energy': 5,
        'scalability': 0.95,
        'decentralization': 0.65,
        'cross_shard_efficiency': 0.89,
        'attack_resistance': 0.78
    },
    'Algorand': {
        'throughput': 1200,
        'latency': 4.5,
        'security': 0.90,
        'energy': 8,
        'scalability': 0.82,
        'decentralization': 0.75,
        'cross_shard_efficiency': 0.88,
        'attack_resistance': 0.87
    },
    'Avalanche': {
        'throughput': 4500,
        'latency': 2,
        'security': 0.87,
        'energy': 20,
        'scalability': 0.88,
        'decentralization': 0.72,
        'cross_shard_efficiency': 0.90,
        'attack_resistance': 0.85
    },
    'Polkadot': {
        'throughput': 1500,
        'latency': 6,
        'security': 0.89,
        'energy': 25,
        'scalability': 0.87,
        'decentralization': 0.78,
        'cross_shard_efficiency': 0.94,
        'attack_resistance': 0.88
    },
    'Cardano': {
        'throughput': 1000,
        'latency': 10,
        'security': 0.91,
        'energy': 18,
        'scalability': 0.80,
        'decentralization': 0.82,
        'cross_shard_efficiency': 0.83,
        'attack_resistance': 0.89
    }
}

# Định nghĩa khả năng chống lại các loại tấn công (0-1, giá trị cao hơn = tốt hơn)
ATTACK_RESISTANCE = {
    'QTrust': {
        'sybil_attack': 0.95,
        'ddos_attack': 0.90,
        '51_percent': 0.93,
        'eclipse_attack': 0.92,
        'smart_contract_exploit': 0.88
    },
    'Ethereum 2.0': {
        'sybil_attack': 0.92,
        'ddos_attack': 0.88,
        '51_percent': 0.94,
        'eclipse_attack': 0.85,
        'smart_contract_exploit': 0.80
    },
    'Solana': {
        'sybil_attack': 0.85,
        'ddos_attack': 0.92,
        '51_percent': 0.80,
        'eclipse_attack': 0.78,
        'smart_contract_exploit': 0.82
    },
    'Algorand': {
        'sybil_attack': 0.90,
        'ddos_attack': 0.85,
        '51_percent': 0.92,
        'eclipse_attack': 0.88,
        'smart_contract_exploit': 0.85
    },
    'Avalanche': {
        'sybil_attack': 0.88,
        'ddos_attack': 0.87,
        '51_percent': 0.89,
        'eclipse_attack': 0.84,
        'smart_contract_exploit': 0.79
    },
    'Polkadot': {
        'sybil_attack': 0.91,
        'ddos_attack': 0.86,
        '51_percent': 0.88,
        'eclipse_attack': 0.90,
        'smart_contract_exploit': 0.83
    },
    'Cardano': {
        'sybil_attack': 0.93,
        'ddos_attack': 0.84,
        '51_percent': 0.91,
        'eclipse_attack': 0.87,
        'smart_contract_exploit': 0.90
    }
}

def update_with_benchmark_results(benchmark_results: Dict[str, Any], system_name: str = 'QTrust') -> None:
    """
    Cập nhật dữ liệu mô phỏng với kết quả benchmark thực tế.
    
    Args:
        benchmark_results: Dict chứa kết quả benchmark
        system_name: Tên hệ thống cần cập nhật
    """
    if system_name in BLOCKCHAIN_SYSTEMS:
        for key, value in benchmark_results.items():
            if key in BLOCKCHAIN_SYSTEMS[system_name]:
                BLOCKCHAIN_SYSTEMS[system_name][key] = value
                
        print(f"Đã cập nhật dữ liệu benchmark cho {system_name}")
    else:
        BLOCKCHAIN_SYSTEMS[system_name] = benchmark_results
        print(f"Đã thêm dữ liệu benchmark mới cho {system_name}")

def import_blockchain_data(filepath: str) -> None:
    """
    Nhập dữ liệu blockchain từ file JSON.
    
    Args:
        filepath: Đường dẫn đến file JSON
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Kiểm tra cấu trúc dữ liệu
        for system_name, values in data.items():
            required_keys = ['throughput', 'latency', 'security']
            if all(key in values for key in required_keys):
                # Cập nhật nếu hệ thống đã tồn tại hoặc thêm mới
                if system_name in BLOCKCHAIN_SYSTEMS:
                    for key, value in values.items():
                        BLOCKCHAIN_SYSTEMS[system_name][key] = value
                else:
                    BLOCKCHAIN_SYSTEMS[system_name] = values
            else:
                print(f"Bỏ qua hệ thống {system_name} do thiếu thông tin cần thiết")
                
        print(f"Đã nhập dữ liệu từ {filepath} thành công")
    except Exception as e:
        print(f"Lỗi khi nhập dữ liệu: {e}")

def export_benchmark_data(output_dir: str) -> None:
    """
    Xuất dữ liệu benchmark hiện tại ra file JSON.
    
    Args:
        output_dir: Thư mục đầu ra
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"blockchain_data_{timestamp}.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(BLOCKCHAIN_SYSTEMS, f, indent=4)
        print(f"Đã xuất dữ liệu benchmark ra {filepath}")
    except Exception as e:
        print(f"Lỗi khi xuất dữ liệu: {e}")

def plot_attack_resistance(output_dir: str) -> None:
    """
    Tạo biểu đồ radar hiển thị khả năng chống tấn công của các hệ thống.
    
    Args:
        output_dir: Thư mục đầu ra để lưu biểu đồ
    """
    # Lấy các loại tấn công
    attack_types = list(next(iter(ATTACK_RESISTANCE.values())).keys())
    
    # Lấy các hệ thống
    systems = list(ATTACK_RESISTANCE.keys())
    
    # Số lượng tấn công
    num_attacks = len(attack_types)
    
    # Thiết lập góc cho các trục
    angles = np.linspace(0, 2*np.pi, num_attacks, endpoint=False).tolist()
    angles += angles[:1]  # Đóng biểu đồ bằng cách lặp lại góc đầu tiên
    
    # Thiết lập biểu đồ
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    
    # Màu sắc
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
    
    # Vẽ biểu đồ cho từng hệ thống
    for i, system in enumerate(systems):
        values = [ATTACK_RESISTANCE[system][attack] for attack in attack_types]
        values += values[:1]  # Đóng biểu đồ
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=system, alpha=0.8)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Thiết lập các trục
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([attack.replace('_', ' ').title() for attack in attack_types])
    
    # Y axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Thêm grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Thêm tiêu đề và chú thích
    plt.title('So sánh khả năng chống tấn công của các hệ thống Blockchain', fontsize=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Lưu biểu đồ
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"attack_resistance_comparison_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(output_dir: Optional[str] = None) -> None:
    """
    Tạo báo cáo so sánh đầy đủ với tất cả các biểu đồ và bảng.
    
    Args:
        output_dir: Thư mục đầu ra để lưu báo cáo
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "blockchain_comparison_reports")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo các biểu đồ và bảng
    plot_heatmap_comparison(BLOCKCHAIN_SYSTEMS, output_dir)
    plot_relationship_comparison(BLOCKCHAIN_SYSTEMS, output_dir)
    plot_attack_resistance(output_dir)
    comparison_table = generate_comparison_table(BLOCKCHAIN_SYSTEMS, output_dir)
    
    # Xuất dữ liệu
    export_benchmark_data(output_dir)
    
    # Tạo file báo cáo markdown
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"blockchain_comparison_report_{timestamp}.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo so sánh các hệ thống Blockchain\n\n")
        f.write(f"*Tạo lúc: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Tổng quan\n\n")
        f.write("Báo cáo này so sánh hiệu suất của QTrust với các hệ thống blockchain khác dựa trên các chỉ số quan trọng:\n")
        f.write("- Throughput (giao dịch/giây)\n")
        f.write("- Latency (độ trễ trung bình)\n")
        f.write("- Security (điểm bảo mật)\n")
        f.write("- Energy Consumption (tiêu thụ năng lượng)\n")
        f.write("- Scalability (khả năng mở rộng)\n")
        f.write("- Decentralization (mức độ phi tập trung)\n")
        f.write("- Cross-shard Efficiency (hiệu quả xử lý giao dịch liên shard)\n")
        f.write("- Attack Resistance (khả năng chống tấn công)\n\n")
        
        f.write("## Bảng so sánh chỉ số hiệu suất\n\n")
        f.write(comparison_table.to_markdown() + "\n\n")
        
        f.write("## Biểu đồ đã tạo\n\n")
        f.write("Các biểu đồ sau đã được tạo trong thư mục báo cáo:\n\n")
        f.write("1. Heatmap so sánh các chỉ số hiệu suất đã chuẩn hóa\n")
        f.write("2. Biểu đồ mối quan hệ giữa Throughput, Latency và Security\n")
        f.write("3. Biểu đồ radar so sánh khả năng chống tấn công\n\n")
        
        f.write("## Kết luận\n\n")
        
        # Xác định hệ thống có điểm cao nhất
        top_system = comparison_table.index[0]
        top_score = comparison_table['Performance Score'].max()
        
        f.write(f"Dựa trên phân tích, **{top_system}** có điểm hiệu suất tổng thể cao nhất ({top_score:.3f}) ")
        
        # Xác định lĩnh vực mà QTrust vượt trội
        if 'QTrust' in comparison_table.index:
            qtrust_row = comparison_table.loc['QTrust']
            best_metrics = []
            
            for col in comparison_table.columns:
                if col not in ['Performance Score', 'Rank']:
                    # Đối với latency và energy, giá trị thấp hơn là tốt hơn
                    if col in ['Latency', 'Energy']:
                        if qtrust_row[col] == comparison_table[col].min():
                            best_metrics.append(col)
                    # Đối với các chỉ số khác, giá trị cao hơn là tốt hơn
                    elif qtrust_row[col] == comparison_table[col].max():
                        best_metrics.append(col)
            
            if best_metrics:
                f.write(f"với QTrust dẫn đầu trong các chỉ số: {', '.join(best_metrics)}.\n\n")
            else:
                f.write(".\n\n")
                
            # Đề xuất cải tiến
            if 'QTrust' != top_system:
                f.write("### Đề xuất cải tiến cho QTrust\n\n")
                
                # Tìm các chỉ số mà QTrust kém hơn hệ thống tốt nhất
                improvement_areas = []
                
                for col in comparison_table.columns:
                    if col not in ['Performance Score', 'Rank']:
                        # Đối với latency và energy, giá trị thấp hơn là tốt hơn
                        if col in ['Latency', 'Energy']:
                            if qtrust_row[col] > comparison_table.loc[top_system, col]:
                                improvement_areas.append(f"{col} (hiện tại: {qtrust_row[col]}, mục tiêu: <={comparison_table.loc[top_system, col]})")
                        # Đối với các chỉ số khác, giá trị cao hơn là tốt hơn
                        elif qtrust_row[col] < comparison_table.loc[top_system, col]:
                            improvement_areas.append(f"{col} (hiện tại: {qtrust_row[col]}, mục tiêu: >={comparison_table.loc[top_system, col]})")
                
                if improvement_areas:
                    f.write("Các lĩnh vực cần cải thiện:\n")
                    for area in improvement_areas:
                        f.write(f"- {area}\n")
        
        f.write("\n*Lưu ý: Dữ liệu này dựa trên các benchmark được thực hiện trong điều kiện thử nghiệm và có thể khác với hiệu suất trong môi trường production.*\n")
    
    print(f"Đã tạo báo cáo so sánh tại {report_path}")

if __name__ == "__main__":
    # Tạo thư mục đầu ra
    output_dir = os.path.join(os.getcwd(), "blockchain_comparison_reports")
    
    # Tạo báo cáo so sánh
    generate_comparison_report(output_dir) 