#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kiểm tra đơn giản cho HTDCM - Script để kiểm tra khả năng phát hiện nút độc hại của HTDCM.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any
import time

# Thêm thư mục hiện tại vào PYTHONPATH để có thể import các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.trust.htdcm import HTDCM, HTDCMNode

def test_htdcm_malicious_detection():
    """
    Kiểm tra khả năng phát hiện nút độc hại của HTDCM.
    """
    print("=== Bắt đầu kiểm tra HTDCM ===")
    
    # Cấu hình mạng đơn giản
    num_shards = 3
    nodes_per_shard = 8
    total_nodes = num_shards * nodes_per_shard
    malicious_percentage = 25  # 25% nút độc hại
    
    # Tạo đồ thị mạng
    G = nx.Graph()
    
    # Tạo danh sách các shard và các nút trong mỗi shard
    shards = []
    malicious_nodes = []
    
    print(f"Tạo mạng với {num_shards} shard, mỗi shard có {nodes_per_shard} nút")
    
    for s in range(num_shards):
        shard_nodes = []
        for n in range(nodes_per_shard):
            node_id = s * nodes_per_shard + n
            # Quyết định xem node có độc hại không
            is_malicious = np.random.rand() < (malicious_percentage / 100)
            
            # Thêm node vào đồ thị
            G.add_node(node_id, 
                      shard_id=s, 
                      trust_score=0.7,
                      is_malicious=is_malicious)
            
            if is_malicious:
                malicious_nodes.append(node_id)
                
            shard_nodes.append(node_id)
        shards.append(shard_nodes)
    
    print(f"Đã tạo {total_nodes} nút, có {len(malicious_nodes)} nút độc hại")
    
    # Thêm các kết nối trong shard (đầy đủ kết nối)
    for shard_nodes in shards:
        for i in range(len(shard_nodes)):
            for j in range(i + 1, len(shard_nodes)):
                G.add_edge(shard_nodes[i], shard_nodes[j], weight=1)
    
    # Thêm các kết nối giữa các shard (kết nối ngẫu nhiên)
    cross_shard_connections = int(total_nodes * 0.3)  # 30% số lượng node
    
    for _ in range(cross_shard_connections):
        shard1 = np.random.randint(0, num_shards)
        shard2 = np.random.randint(0, num_shards)
        while shard1 == shard2:
            shard2 = np.random.randint(0, num_shards)
            
        node1 = np.random.choice(shards[shard1])
        node2 = np.random.choice(shards[shard2])
        
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2, weight=2)  # Kết nối xuyên shard có trọng số 2
    
    # Thiết lập HTDCM
    htdcm = HTDCM(network=G, shards=shards)
    
    # Mô phỏng các giao dịch và phát hiện nút độc hại
    sim_steps = 50
    tx_per_step = 10
    detected_history = []
    trust_score_history = {node_id: [] for node_id in G.nodes()}
    
    print(f"Mô phỏng {sim_steps} bước, mỗi bước {tx_per_step} giao dịch")
    
    for step in range(sim_steps):
        # Tạo các giao dịch
        for _ in range(tx_per_step):
            # Chọn ngẫu nhiên nguồn và đích
            source_shard = np.random.randint(0, num_shards)
            target_shard = np.random.randint(0, num_shards)
            
            source_node = np.random.choice(shards[source_shard])
            target_node = np.random.choice(shards[target_shard])
            
            # Quyết định xem giao dịch có thành công không
            # Các node độc hại có xác suất thất bại cao hơn
            source_is_malicious = G.nodes[source_node].get('is_malicious', False)
            target_is_malicious = G.nodes[target_node].get('is_malicious', False)
            
            # Xác suất thành công dựa trên độc hại
            if source_is_malicious or target_is_malicious:
                success_prob = 0.3  # Nút độc hại có xác suất thành công thấp
            else:
                success_prob = 0.9  # Nút bình thường có xác suất thành công cao
            
            # Quyết định kết quả giao dịch
            tx_success = np.random.rand() < success_prob
            
            # Mô phỏng thời gian phản hồi
            if source_is_malicious or target_is_malicious:
                # Node độc hại có thời gian phản hồi biến động
                response_time = np.random.uniform(15, 30)
            else:
                # Node bình thường có thời gian phản hồi ổn định
                response_time = np.random.uniform(5, 15)
            
            # Cập nhật thông tin tin cậy cho các node
            htdcm.update_node_trust(source_node, tx_success, response_time, True)
            htdcm.update_node_trust(target_node, tx_success, response_time, False)
        
        # Lưu điểm tin cậy
        for node_id in G.nodes():
            trust_score_history[node_id].append(htdcm.nodes[node_id].trust_score)
        
        # Phát hiện các node độc hại
        detected_nodes = htdcm.identify_malicious_nodes(min_malicious_activities=1, advanced_filtering=True)
        detected_history.append(len(detected_nodes))
        
        # Báo cáo tiến trình
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Bước {step + 1}/{sim_steps}: {len(detected_nodes)} nút độc hại được phát hiện")
            
            # Tính các chỉ số hiệu quả
            true_malicious = set(malicious_nodes)
            detected_malicious = set(detected_nodes)
            
            true_positives = len(true_malicious.intersection(detected_malicious))
            false_positives = len(detected_malicious - true_malicious)
            false_negatives = len(true_malicious - detected_malicious)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  - Độ chính xác: {precision:.2f}, Độ nhạy: {recall:.2f}, F1: {f1_score:.2f}")
    
    # Phân tích cuối cùng
    final_detected = htdcm.identify_malicious_nodes(min_malicious_activities=2, advanced_filtering=True)
    true_malicious = set(malicious_nodes)
    detected_malicious = set(final_detected)
    
    # Tính chỉ số hiệu quả cuối cùng
    true_positives = len(true_malicious.intersection(detected_malicious))
    false_positives = len(detected_malicious - true_malicious)
    false_negatives = len(true_malicious - detected_malicious)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Kết quả cuối cùng ===")
    print(f"Tổng số nút độc hại thực tế: {len(malicious_nodes)}")
    print(f"Số nút độc hại được phát hiện: {len(final_detected)}")
    print(f"Độ chính xác: {precision:.2f}")
    print(f"Độ nhạy: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    
    # Vẽ kết quả
    plt.figure(figsize=(12, 8))
    
    # 1. Biểu đồ số lượng nút độc hại được phát hiện
    plt.subplot(2, 2, 1)
    plt.plot(range(sim_steps), detected_history, marker='o')
    plt.axhline(y=len(malicious_nodes), color='r', linestyle='--', label='Thực tế')
    plt.xlabel('Bước mô phỏng')
    plt.ylabel('Số nút độc hại phát hiện được')
    plt.title('Phát hiện nút độc hại theo thời gian')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Biểu đồ điểm tin cậy trung bình của nút độc hại và không độc hại
    malicious_trust = np.mean([np.array(trust_score_history[n]) for n in malicious_nodes], axis=0)
    non_malicious = [n for n in G.nodes() if n not in malicious_nodes]
    non_malicious_trust = np.mean([np.array(trust_score_history[n]) for n in non_malicious], axis=0)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(sim_steps), malicious_trust, 'r-', label='Nút độc hại')
    plt.plot(range(sim_steps), non_malicious_trust, 'g-', label='Nút bình thường')
    plt.xlabel('Bước mô phỏng')
    plt.ylabel('Điểm tin cậy trung bình')
    plt.title('Điểm tin cậy trung bình theo thời gian')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Histogram điểm tin cậy cuối cùng
    plt.subplot(2, 2, 3)
    malicious_final_trust = [trust_score_history[n][-1] for n in malicious_nodes]
    non_malicious_final_trust = [trust_score_history[n][-1] for n in non_malicious]
    
    plt.hist([non_malicious_final_trust, malicious_final_trust], bins=10,
            label=['Nút bình thường', 'Nút độc hại'], alpha=0.7, color=['g', 'r'])
    plt.axvline(x=htdcm.malicious_threshold, color='k', linestyle='--', label='Ngưỡng độc hại')
    plt.xlabel('Điểm tin cậy')
    plt.ylabel('Số lượng nút')
    plt.title('Phân bố điểm tin cậy cuối cùng')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Biểu đồ chỉ số hiệu quả theo thời gian
    precision_history = []
    recall_history = []
    f1_history = []
    
    for step in range(sim_steps):
        # Lấy điểm tin cậy tại thời điểm step
        trust_at_step = {node_id: trust_score_history[node_id][step] for node_id in G.nodes()}
        
        # Xác định node bị phát hiện tại step này (phần bị thay thế hoàn toàn)
        min_activities = 1 if step < sim_steps // 2 else 2
        detected_at_step = []
        
        for node_id, score in trust_at_step.items():
            if score < htdcm.malicious_threshold and node_id in htdcm.nodes:
                node = htdcm.nodes[node_id]
                
                # Kiểm tra nâng cao
                enough_malicious_activities = node.malicious_activities >= min_activities
                
                # Kiểm tra tỉ lệ thành công
                total_txs = node.successful_txs + node.failed_txs
                low_success_rate = False
                if total_txs >= 5:
                    success_rate = node.successful_txs / total_txs if total_txs > 0 else 0
                    low_success_rate = success_rate < 0.4
                
                # Kiểm tra thời gian phản hồi
                high_response_time = False
                if node.response_times and len(node.response_times) >= 3:
                    avg_response_time = np.mean(node.response_times)
                    high_response_time = avg_response_time > 20
                
                # Kiểm tra feedback từ peer
                poor_peer_rating = False
                if hasattr(node, 'peer_ratings') and node.peer_ratings:
                    avg_peer_rating = np.mean(list(node.peer_ratings.values()))
                    poor_peer_rating = avg_peer_rating < 0.4
                
                # Tính điểm minh chứng
                evidence_count = sum([
                    enough_malicious_activities,
                    low_success_rate,
                    high_response_time,
                    poor_peer_rating
                ])
                
                if evidence_count >= 2:
                    detected_at_step.append(node_id)
        
        # Tính các chỉ số
        true_positives = len(set(malicious_nodes).intersection(set(detected_at_step)))
        false_positives = len(set(detected_at_step) - set(malicious_nodes))
        false_negatives = len(set(malicious_nodes) - set(detected_at_step))
        
        p = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        r = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision_history.append(p)
        recall_history.append(r)
        f1_history.append(f1)
    
    plt.subplot(2, 2, 4)
    plt.plot(range(sim_steps), precision_history, 'b-', label='Độ chính xác')
    plt.plot(range(sim_steps), recall_history, 'g-', label='Độ nhạy')
    plt.plot(range(sim_steps), f1_history, 'r-', label='F1 Score')
    plt.xlabel('Bước mô phỏng')
    plt.ylabel('Chỉ số')
    plt.title('Chỉ số hiệu quả theo thời gian')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('htdcm_test_results.png', dpi=300)
    plt.show()
    
    print(f"Đã lưu biểu đồ kết quả vào htdcm_test_results.png")

if __name__ == "__main__":
    test_htdcm_malicious_detection() 