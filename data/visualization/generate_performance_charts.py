#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs('docs/exported_charts', exist_ok=True)

def load_data():
    """Tải dữ liệu từ file CSV."""
    csv_path = 'data/sources/comparative_analysis.csv'
    return pd.read_csv(csv_path)

def create_performance_comparison_chart(df):
    """Tạo biểu đồ so sánh hiệu năng."""
    plt.figure(figsize=(12, 7))
    
    # Lấy dữ liệu
    systems = df['System']
    throughput = df['Throughput (tx/s)']
    
    # Tạo biểu đồ cột
    bars = plt.bar(systems, throughput, color=['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D'])
    
    # Thêm giá trị trên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('So sánh thông lượng giữa các hệ thống Blockchain', fontsize=16, fontweight='bold')
    plt.ylabel('Thông lượng (Giao dịch/giây)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(throughput) * 1.15)  # Thêm khoảng trống cho số liệu trên cột
    
    # Thêm nguồn dữ liệu
    references = df['Reference'].tolist()
    source_text = "Nguồn: " + "; ".join(set([ref.split(',')[0] for ref in references]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Lưu biểu đồ
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Performance comparison chart created: docs/exported_charts/performance_comparison.png")
    plt.close()

def create_radar_chart(df):
    """Tạo biểu đồ radar để so sánh đa chiều."""
    # Chuẩn bị dữ liệu
    categories = ['Throughput (tx/s)', 'Latency (s)', 'Energy Consumption', 'Security', 'Attack Resistance']
    
    # Chuẩn hóa dữ liệu (0-1)
    normalized_df = df.copy()
    
    # Đảo ngược giá trị cho latency và energy consumption (thấp hơn là tốt hơn)
    max_latency = normalized_df['Latency (s)'].max()
    max_energy = normalized_df['Energy Consumption'].max()
    normalized_df['Latency (s)'] = (max_latency - normalized_df['Latency (s)']) / max_latency
    normalized_df['Energy Consumption'] = (max_energy - normalized_df['Energy Consumption']) / max_energy
    
    # Chuẩn hóa các cột khác
    for col in ['Throughput (tx/s)', 'Security', 'Attack Resistance']:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    # Số lượng biến
    N = len(categories)
    
    # Góc cho mỗi trục
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Đóng biểu đồ
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Thêm mỗi hệ thống vào biểu đồ
    colors = ['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D']
    for i, system in enumerate(normalized_df['System']):
        values = normalized_df.loc[i, categories].values.tolist()
        values += values[:1]  # Đóng biểu đồ
        ax.plot(angles, values, linewidth=2, label=system, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Thiết lập các trục
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    # Đặt y-ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
    ax.grid(True)
    
    # Thêm legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Tiêu đề
    plt.title('So sánh đa chiều giữa các hệ thống Blockchain', size=15, fontweight='bold', y=1.1)
    
    # Thêm nguồn dữ liệu
    references = df['Reference'].tolist()
    source_text = "Nguồn: " + "; ".join(set([ref.split(',')[0] for ref in references]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig('docs/exported_charts/radar_chart.png', dpi=300, bbox_inches='tight')
    print("Radar chart created: docs/exported_charts/radar_chart.png")
    plt.close()

def create_latency_chart(df):
    """Tạo biểu đồ so sánh độ trễ."""
    plt.figure(figsize=(12, 7))
    
    # Lấy dữ liệu
    systems = df['System']
    latency = df['Latency (s)']
    
    # Tạo biểu đồ cột ngang
    bars = plt.barh(systems, latency, color=['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D'])
    
    # Thêm giá trị bên cạnh mỗi cột
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                 f'{width}s', va='center', fontweight='bold')
    
    plt.title('So sánh độ trễ giữa các hệ thống Blockchain', fontsize=16, fontweight='bold')
    plt.xlabel('Độ trễ (giây)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, max(latency) * 1.2)  # Thêm khoảng trống cho số liệu
    
    # Thêm nguồn dữ liệu
    references = df['Reference'].tolist()
    source_text = "Nguồn: " + "; ".join(set([ref.split(',')[0] for ref in references]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Lưu biểu đồ
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/latency_chart.png', dpi=300, bbox_inches='tight')
    print("Latency chart created: docs/exported_charts/latency_chart.png")
    plt.close()

def create_security_chart(df):
    """Tạo biểu đồ so sánh bảo mật và khả năng chống tấn công."""
    plt.figure(figsize=(12, 8))
    
    # Lấy dữ liệu
    systems = df['System']
    security = df['Security']
    attack_resistance = df['Attack Resistance']
    
    # Thiết lập biểu đồ
    x = np.arange(len(systems))
    width = 0.35
    
    # Tạo biểu đồ cột ghép
    bar1 = plt.bar(x - width/2, security, width, label='Bảo mật', color='#2C82C9')
    bar2 = plt.bar(x + width/2, attack_resistance, width, label='Khả năng chống tấn công', color='#EF4836')
    
    # Thêm giá trị trên mỗi cột
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('So sánh khả năng bảo mật giữa các hệ thống Blockchain', fontsize=16, fontweight='bold')
    plt.ylabel('Điểm đánh giá (0-1)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(x, systems)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Thêm nguồn dữ liệu
    references = df['Reference'].tolist()
    source_text = "Nguồn: " + "; ".join(set([ref.split(',')[0] for ref in references[:3]]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Lưu biểu đồ
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/security_chart.png', dpi=300, bbox_inches='tight')
    print("Security comparison chart created: docs/exported_charts/security_chart.png")
    plt.close()

def create_energy_chart(df):
    """Tạo biểu đồ so sánh tiêu thụ năng lượng."""
    plt.figure(figsize=(10, 6))
    
    # Lấy dữ liệu
    systems = df['System']
    energy = df['Energy Consumption']
    
    # Tạo biểu đồ
    bars = plt.bar(systems, energy, color=['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D'])
    
    # Thêm giá trị trên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('So sánh tiêu thụ năng lượng tương đối', fontsize=16, fontweight='bold')
    plt.ylabel('Tiêu thụ năng lượng (đơn vị tương đối)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(energy) * 1.15)
    
    # Thêm nguồn dữ liệu
    references = df['Reference'].tolist()
    source_text = "Nguồn: " + "; ".join(set([ref.split(',')[0] for ref in references[:2]]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Lưu biểu đồ
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/energy_chart.png', dpi=300, bbox_inches='tight')
    print("Energy efficiency chart created: docs/exported_charts/energy_chart.png")
    plt.close()

def main():
    """Run all chart generation functions."""
    df = load_data()
    create_performance_comparison_chart(df)
    create_radar_chart(df)
    create_latency_chart(df)
    create_security_chart(df)
    create_energy_chart(df)
    print("All performance charts have been generated successfully.")

if __name__ == "__main__":
    main() 