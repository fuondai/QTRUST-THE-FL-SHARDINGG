#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blockchain Comparison Utilities - Các hàm tiện ích để so sánh blockchain

File này cung cấp các hàm để tạo biểu đồ và báo cáo so sánh hiệu suất của
các hệ thống blockchain khác nhau.
"""

import os
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Any

def plot_heatmap_comparison(systems: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Tạo biểu đồ heatmap so sánh các hệ thống blockchain.
    
    Args:
        systems: Dict chứa dữ liệu các hệ thống blockchain
        output_dir: Thư mục đầu ra để lưu biểu đồ
    """
    # Chuyển đổi dữ liệu sang DataFrame để dễ xử lý
    df = pd.DataFrame(systems).T
    
    # Chuẩn hóa dữ liệu để so sánh công bằng
    normalized_df = df.copy()
    for col in df.columns:
        if col in ['latency', 'energy']:
            # Đối với các chỉ số mà giá trị thấp hơn là tốt hơn
            # Sử dụng 1 - normalized để đảo ngược (giá trị cao = tốt hơn)
            normalized_df[col] = 1 - ((df[col] - df[col].min()) / 
                                    (df[col].max() - df[col].min()) 
                                    if df[col].max() != df[col].min() else 0)
        else:
            # Đối với các chỉ số mà giá trị cao hơn là tốt hơn
            normalized_df[col] = ((df[col] - df[col].min()) / 
                                (df[col].max() - df[col].min())
                                if df[col].max() != df[col].min() else 1.0)
    
    # Tạo figure lớn
    plt.figure(figsize=(14, 10))
    
    # Tạo bảng màu - từ đỏ (thấp) đến xanh lá (cao)
    cmap = LinearSegmentedColormap.from_list('green_red', ['#d73027', '#f46d43', '#fdae61', 
                                                         '#fee08b', '#d9ef8b', '#a6d96a', 
                                                         '#66bd63', '#1a9850'])
    
    # Vẽ heatmap
    ax = sns.heatmap(normalized_df, annot=df.round(2), fmt=".2f", 
                   cmap=cmap, linewidths=0.5, 
                   cbar_kws={'label': 'Điểm số (đã chuẩn hóa)'})
    
    # Tùy chỉnh
    plt.title('Ma trận so sánh các hệ thống Blockchain', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Các chỉ số hiệu suất', fontsize=12)
    plt.ylabel('Hệ thống', fontsize=12)
    
    # Đổi tên cột cho đẹp hơn
    ax.set_xticklabels([col.replace('_', ' ').title() for col in df.columns], rotation=45, ha='right')
    
    # Lưu biểu đồ
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"blockchain_heatmap_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Lưu DataFrame đã chuẩn hóa để tham khảo
    normalized_df.to_csv(os.path.join(output_dir, f"normalized_metrics_{timestamp}.csv"))

def plot_relationship_comparison(systems: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Tạo biểu đồ scatter thể hiện mối quan hệ giữa throughput, latency và security.
    
    Args:
        systems: Dict chứa dữ liệu các hệ thống blockchain
        output_dir: Thư mục đầu ra để lưu biểu đồ
    """
    # Tạo DataFrame
    data = []
    for system, values in systems.items():
        data.append({
            'System': system,
            'Throughput': values['throughput'],
            'Latency': values['latency'],
            'Security': values['security'],
            'Energy': values['energy'],
            'Decentralization': values['decentralization']
        })
    
    df = pd.DataFrame(data)
    
    # Tạo biểu đồ scatter
    plt.figure(figsize=(14, 10))
    
    # Sử dụng kích thước điểm dựa trên điểm bảo mật
    sizes = df['Security'] * 1000
    
    # Màu sắc dựa trên mức độ phi tập trung
    colors = df['Decentralization']
    
    # Nếu throughput có sự chênh lệch quá lớn, sử dụng thang logarit
    if max(df['Throughput']) / min(df['Throughput']) > 100:
        scatter = plt.scatter(np.log10(df['Throughput']), df['Latency'], 
                           s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='k')
        plt.xlabel('Throughput (log10 tx/s)', fontsize=12)
    else:
        scatter = plt.scatter(df['Throughput'], df['Latency'], 
                           s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='k')
        plt.xlabel('Throughput (tx/s)', fontsize=12)
    
    # Tương tự cho latency
    if max(df['Latency']) / min(df['Latency']) > 100:
        plt.yscale('log')
        plt.ylabel('Latency (log ms)', fontsize=12)
    else:
        plt.ylabel('Latency (ms)', fontsize=12)
    
    # Thêm nhãn cho mỗi điểm
    for i, system in enumerate(df['System']):
        plt.annotate(system, 
                   (np.log10(df['Throughput'][i]) if max(df['Throughput']) / min(df['Throughput']) > 100 
                    else df['Throughput'][i], 
                    df['Latency'][i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Thêm colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Decentralization Score', fontsize=10)
    
    # Thêm chú thích cho kích thước
    plt.annotate('Bubble size represents Security Score', xy=(0.05, 0.95),
               xycoords='axes fraction', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Mối quan hệ giữa Throughput, Latency, và Security', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Lưu biểu đồ
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"blockchain_relationship_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_table(systems: Dict[str, Dict[str, float]], output_dir: str) -> pd.DataFrame:
    """
    Tạo bảng so sánh các hệ thống blockchain và tính điểm tổng hợp.
    
    Args:
        systems: Dict chứa dữ liệu các hệ thống blockchain
        output_dir: Thư mục đầu ra để lưu bảng
        
    Returns:
        DataFrame chứa bảng so sánh đã tính điểm
    """
    # Tạo DataFrame
    df = pd.DataFrame(systems).T
    
    # Đổi tên cột
    df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Thêm cột Performance Score với trọng số cho mỗi chỉ số
    df['Performance Score'] = (
        # Throughput: 25%
        df['Throughput'] / df['Throughput'].max() * 0.25 + 
        # Latency: 25% (giá trị thấp hơn là tốt hơn)
        (1 - df['Latency'] / df['Latency'].max()) * 0.25 + 
        # Security: 20%
        df['Security'] * 0.20 + 
        # Energy: 10% (giá trị thấp hơn là tốt hơn)
        (1 - df['Energy'] / df['Energy'].max()) * 0.10 +
        # Scalability: 10%
        df['Scalability'] * 0.10 +
        # Cross-shard efficiency: 5%
        df['Cross Shard Efficiency'] * 0.05 +
        # Attack resistance: 5%
        df['Attack Resistance'] * 0.05
    )
    
    # Tính thứ hạng tổng thể
    df['Rank'] = df['Performance Score'].rank(ascending=False).astype(int)
    
    # Sắp xếp theo Performance Score
    df = df.sort_values('Performance Score', ascending=False)
    
    # Làm tròn các giá trị
    df = df.round(3)
    
    # Lưu bảng dưới dạng CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"blockchain_comparison_table_{timestamp}.csv")
    df.to_csv(csv_path)
    
    # Tạo HTML với định dạng đẹp
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Blockchain Systems Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
            .highlight {{ background-color: #e6ffe6; }}
            .top-score {{ font-weight: bold; color: #006600; }}
            caption {{ font-size: 1.5em; margin-bottom: 10px; }}
            .footer {{ font-size: 0.8em; margin-top: 20px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>So sánh hiệu suất các hệ thống Blockchain</h1>
        <p>Tạo ngày: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        {df.to_html(classes='highlight')}
        <p class="footer">
            Ghi chú: Performance Score được tính dựa trên trọng số của các chỉ số:<br>
            Throughput (25%), Latency (25%), Security (20%), Energy (10%), 
            Scalability (10%), Cross-shard Efficiency (5%), Attack Resistance (5%)
        </p>
    </body>
    </html>
    """
    
    # Lưu HTML
    html_path = os.path.join(output_dir, f"blockchain_comparison_table_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
    
    return df

def run_all_comparisons(output_dir: Optional[str] = None) -> None:
    """
    Chạy tất cả các phân tích so sánh và tạo báo cáo tổng hợp.
    
    Args:
        output_dir: Thư mục đầu ra để lưu báo cáo
    """
    from qtrust.benchmark.blockchain_comparison import (
        BLOCKCHAIN_SYSTEMS, ATTACK_RESISTANCE, generate_comparison_report
    )
    
    # Tạo báo cáo so sánh đầy đủ
    generate_comparison_report(output_dir)
    
    # Tạo thông báo hoàn thành
    print("Đã hoàn thành tất cả các phân tích so sánh blockchain!")

if __name__ == "__main__":
    # Chạy tất cả các phân tích khi gọi trực tiếp file
    run_all_comparisons() 