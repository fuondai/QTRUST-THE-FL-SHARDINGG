#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example Blockchain Comparison - Ví dụ sử dụng công cụ so sánh blockchain

File này minh họa cách sử dụng các chức năng của blockchain_comparison
để tạo báo cáo và biểu đồ so sánh các hệ thống blockchain.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thêm đường dẫn project
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import các chức năng so sánh
from blockchain_comparison import (
    BLOCKCHAIN_SYSTEMS,
    ATTACK_RESISTANCE,
    update_with_benchmark_results,
    import_blockchain_data,
    generate_comparison_report,
    plot_attack_resistance
)

def main():
    """Chạy ví dụ minh họa công cụ so sánh blockchain."""
    # Đường dẫn lưu báo cáo
    output_dir = os.path.join(project_root, "blockchain_comparison_examples")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Bắt đầu tạo báo cáo so sánh blockchain...\n")
    
    # 1. In thông tin về các hệ thống đang so sánh
    print("Các hệ thống blockchain đang so sánh:")
    for system, metrics in BLOCKCHAIN_SYSTEMS.items():
        print(f"- {system}: Throughput={metrics['throughput']} tx/s, Latency={metrics['latency']} ms")
    print()
    
    # 2. Cập nhật dữ liệu QTrust từ benchmark giả định
    print("Cập nhật thông số từ benchmark mới nhất của QTrust...")
    benchmark_results = {
        'throughput': 7500,        # tx/s (giả định từ benchmark)
        'latency': 1.8,            # ms
        'security': 0.96,          # 0-1 scale
        'energy': 12,              # watt-hours/tx
        'scalability': 0.92,       # 0-1 scale
        'cross_shard_efficiency': 0.94
    }
    update_with_benchmark_results(benchmark_results)
    print("Đã cập nhật dữ liệu benchmark cho QTrust\n")
    
    # 3. Tạo báo cáo so sánh
    print("Tạo báo cáo so sánh đầy đủ...")
    generate_comparison_report(output_dir)
    print()
    
    # 4. Tạo biểu đồ bổ sung
    print("Tạo biểu đồ bổ sung cho phân tích QTrust...")
    
    # Biểu đồ cột đơn giản về throughput
    plt.figure(figsize=(12, 6))
    systems = list(BLOCKCHAIN_SYSTEMS.keys())
    throughputs = [BLOCKCHAIN_SYSTEMS[s]['throughput'] for s in systems]
    
    # Đánh dấu QTrust
    colors = ['green' if s == 'QTrust' else 'skyblue' for s in systems]
    
    plt.bar(systems, throughputs, color=colors)
    plt.title('So sánh Throughput giữa các Blockchain', fontsize=14)
    plt.ylabel('Throughput (tx/s)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Thêm nhãn giá trị
    for i, v in enumerate(throughputs):
        plt.text(i, v + 0.01*max(throughputs), f"{v:,}", 
                 ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Hoàn thành! Xem kết quả tại: {output_dir}")

if __name__ == "__main__":
    main() 