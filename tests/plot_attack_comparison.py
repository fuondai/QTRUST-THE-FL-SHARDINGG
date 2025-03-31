#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot Attack Comparison - Script để so sánh hiệu suất của QTrust dưới các tấn công khác nhau
và vẽ các biểu đồ so sánh cho báo cáo khoa học.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime

# Thêm thư mục hiện tại vào PYTHONPATH để có thể import các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from attack_simulation_runner import run_attack_comparison
from large_scale_simulation import LargeScaleBlockchainSimulation

def plot_comparison_radar(metrics_data, output_dir='results_comparison'):
    """Vẽ biểu đồ radar so sánh hiệu suất giữa các loại tấn công."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Các loại metrics cần so sánh
    metrics = ['throughput', 'latency', 'energy', 'security', 'cross_shard_ratio']
    
    # Chuẩn hóa dữ liệu để so sánh trên radar chart
    normalized_data = {}
    
    # Tìm giá trị min/max cho mỗi metrics
    min_values = {m: float('inf') for m in metrics}
    max_values = {m: float('-inf') for m in metrics}
    
    for attack, values in metrics_data.items():
        for m in metrics:
            if values[m] < min_values[m]:
                min_values[m] = values[m]
            if values[m] > max_values[m]:
                max_values[m] = values[m]
    
    # Chuẩn hóa
    for attack, values in metrics_data.items():
        normalized_data[attack] = {}
        for m in metrics:
            # Đối với latency và energy, giá trị thấp hơn là tốt hơn, nên đảo ngược
            if m in ['latency', 'energy']:
                if max_values[m] == min_values[m]:
                    normalized_data[attack][m] = 1.0
                else:
                    normalized_data[attack][m] = 1 - (values[m] - min_values[m]) / (max_values[m] - min_values[m])
            else:
                if max_values[m] == min_values[m]:
                    normalized_data[attack][m] = 1.0
                else:
                    normalized_data[attack][m] = (values[m] - min_values[m]) / (max_values[m] - min_values[m])
    
    # Số lượng biến
    N = len(metrics)
    
    # Góc cho mỗi trục
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    # Tên metrics
    labels = [m.replace('_', ' ').title() for m in metrics]
    labels += labels[:1]  # Đóng vòng tròn
    
    # Màu sắc và style cho mỗi loại tấn công
    colors = sns.color_palette("Set1", len(normalized_data))
    
    # Tạo một figure lớn hơn
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # Vẽ cho từng loại tấn công
    for i, (attack, values) in enumerate(normalized_data.items()):
        # Chuẩn bị dữ liệu để vẽ
        attack_values = [values[m] for m in metrics]
        attack_values += attack_values[:1]  # Đóng vòng tròn
        
        # Vẽ đường và điền màu
        ax.plot(angles, attack_values, linewidth=2, linestyle='solid', 
                label=attack.replace('_', ' ').title(), color=colors[i])
        ax.fill(angles, attack_values, alpha=0.1, color=colors[i])
    
    # Tùy chỉnh radar chart
    ax.set_theta_offset(np.pi / 2)  # Bắt đầu từ trên cùng
    ax.set_theta_direction(-1)  # Đi theo chiều kim đồng hồ
    
    # Đặt labels cho mỗi trục
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=12)
    
    # Thêm mức đô mờ
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_ylim(0, 1)
    
    # Thêm lưới
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm tiêu đề và legend
    plt.title('QTrust Performance Comparison Across Attack Scenarios', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attack_comparison_radar_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_bars(metrics_data, output_dir='results_comparison'):
    """Vẽ biểu đồ cột so sánh hiệu suất giữa các loại tấn công."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuyển đổi dữ liệu để dễ vẽ với seaborn
    df_data = []
    for attack, values in metrics_data.items():
        for metric, value in values.items():
            df_data.append({
                'Attack': attack.replace('_', ' ').title(),
                'Metric': metric.replace('_', ' ').title(),
                'Value': value
            })
    
    df = pd.DataFrame(df_data)
    
    # Tạo figure lớn
    plt.figure(figsize=(15, 12))
    
    # Vẽ cho từng metric
    metrics = ['Throughput', 'Latency', 'Energy', 'Security', 'Cross Shard Ratio']
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        metric_data = df[df['Metric'] == metric]
        
        # Sắp xếp các tấn công theo giá trị
        if metric in ['Latency', 'Energy']:  # Giá trị thấp hơn là tốt hơn
            metric_data = metric_data.sort_values('Value')
        else:  # Giá trị cao hơn là tốt hơn
            metric_data = metric_data.sort_values('Value', ascending=False)
        
        # Vẽ biểu đồ cột
        sns.barplot(data=metric_data, x='Attack', y='Value', 
                    palette='viridis', alpha=0.8)
        
        # Thêm chú thích
        for j, row in enumerate(metric_data.itertuples()):
            plt.text(j, row.Value + (max(metric_data['Value']) * 0.02), 
                     f"{row.Value:.2f}", ha='center', fontsize=9)
        
        # Tùy chỉnh trục và tiêu đề
        plt.title(f"{metric} Comparison", fontsize=13, fontweight='bold')
        plt.xlabel('')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/attack_comparison_bars_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_security_vs_performance(metrics_data, output_dir='results_comparison'):
    """Vẽ biểu đồ so sánh an ninh và hiệu suất."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Chuẩn bị dữ liệu cho scatter plot
    attacks = []
    throughputs = []
    latencies = []
    securities = []
    
    for attack, values in metrics_data.items():
        attacks.append(attack.replace('_', ' ').title())
        throughputs.append(values['throughput'])
        latencies.append(values['latency'])
        securities.append(values['security'])
    
    # Màu sắc dựa trên latency (thấp hơn = tốt hơn)
    norm_latencies = [l/max(latencies) for l in latencies]
    colors = plt.cm.cool(norm_latencies)
    
    # Kích thước dựa trên throughput (cao hơn = tốt hơn)
    norm_throughputs = [t/max(throughputs) * 500 for t in throughputs]
    
    # Vẽ scatter plot
    plt.scatter(securities, latencies, s=norm_throughputs, c=colors, alpha=0.6)
    
    # Thêm annotations
    for i, attack in enumerate(attacks):
        plt.annotate(attack, (securities[i], latencies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Tùy chỉnh biểu đồ
    plt.title('Security vs. Latency Trade-off Across Attack Scenarios', fontsize=14, fontweight='bold')
    plt.xlabel('Security Score', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm chú thích về kích thước
    plt.annotate('Bubble size represents throughput', xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                                fc="white", ec="gray", alpha=0.8))
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/security_vs_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(metrics_data, output_dir='results_comparison'):
    """Tạo bảng tóm tắt so sánh các loại tấn công."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo DataFrame từ dữ liệu metrics
    df = pd.DataFrame(metrics_data).T
    
    # Đổi tên các cột để dễ đọc
    df.columns = [c.replace('_', ' ').title() for c in df.columns]
    
    # Đổi tên index (tên các loại tấn công)
    df.index = [idx.replace('_', ' ').title() for idx in df.index]
    
    # Tạo cột Performance Score bằng cách tính điểm tổng hợp
    # Công thức: tăng throughput + security, giảm latency + energy
    df['Performance Score'] = (
        df['Throughput'] / df['Throughput'].max() * 0.3 + 
        (1 - df['Latency'] / df['Latency'].max()) * 0.3 + 
        (1 - df['Energy'] / df['Energy'].max()) * 0.1 + 
        df['Security'] / df['Security'].max() * 0.3
    )
    
    # Sắp xếp theo Performance Score
    df = df.sort_values('Performance Score', ascending=False)
    
    # Làm tròn các giá trị
    df = df.round(3)
    
    # Lưu bảng thành HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tạo style cho HTML table
    html = df.to_html()
    
    # Thêm CSS
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attack Comparison Results</title>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
            }}
            th, td {{
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #ddd;
            }}
            caption {{
                font-size: 1.5em;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>QTrust Performance Under Different Attack Scenarios</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        {html}
    </body>
    </html>
    """
    
    # Lưu HTML
    with open(f"{output_dir}/attack_comparison_table_{timestamp}.html", "w") as f:
        f.write(styled_html)
    
    # Lưu CSV
    df.to_csv(f"{output_dir}/attack_comparison_table_{timestamp}.csv")
    
    print(f"Bảng tóm tắt đã được lưu tại {output_dir}")
    
    return df

def run_comprehensive_analysis(args):
    """Chạy phân tích toàn diện với so sánh giữa các loại tấn công."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Bắt đầu phân tích toàn diện QTrust dưới các loại tấn công ===")
    
    # Chạy so sánh tấn công
    all_metrics = run_attack_comparison(args, output_dir)
    
    # Lưu dữ liệu metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{output_dir}/attack_metrics_{timestamp}.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    # Vẽ các biểu đồ so sánh
    plot_comparison_radar(all_metrics, output_dir)
    plot_comparison_bars(all_metrics, output_dir)
    plot_security_vs_performance(all_metrics, output_dir)
    
    # Tạo bảng tóm tắt
    summary_df = generate_summary_table(all_metrics, output_dir)
    
    # In kết quả tóm tắt
    print("\n=== KẾT QUẢ TÓM TẮT ===")
    print(summary_df)
    
    print(f"\nQuá trình phân tích đã hoàn tất. Kết quả được lưu trong thư mục: {output_dir}")
    
    return summary_df

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='QTrust Attack Comparison Analysis')
    
    parser.add_argument('--num-shards', type=int, default=4, help='So luong shard')
    parser.add_argument('--nodes-per-shard', type=int, default=10, help='So nut tren moi shard')
    parser.add_argument('--steps', type=int, default=500, help='So buoc mo phong cho moi loai tan cong')
    parser.add_argument('--tx-per-step', type=int, default=20, help='So giao dich moi buoc')
    parser.add_argument('--malicious', type=float, default=20, help='Ty le phan tram nut doc hai (mac dinh)')
    parser.add_argument('--output-dir', type=str, default='results_comparison', help='Thu muc luu ket qua')
    parser.add_argument('--attack-subset', type=str, nargs='+', 
                      choices=['none', '51_percent', 'sybil', 'eclipse', 'selfish_mining', 
                              'bribery', 'ddos', 'finney', 'mixed'], 
                      default=None, 
                      help='Tuy chon tap con cac tan cong de phan tich')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_comprehensive_analysis(args) 