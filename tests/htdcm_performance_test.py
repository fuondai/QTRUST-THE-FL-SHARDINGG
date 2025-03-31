#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List, Tuple
from datetime import datetime
import argparse
from collections import defaultdict

# Thêm thư mục hiện tại vào PYTHONPATH để có thể import các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.trust.htdcm import HTDCM, HTDCMNode

def simulate_network_with_attacks(
    num_shards: int = 4,
    nodes_per_shard: int = 10,
    malicious_percentage: float = 20,
    attack_type: str = "sybil",
    simulation_steps: int = 100,
    tx_per_step: int = 10
) -> Dict[str, Any]:
    """
    Mô phỏng một mạng blockchain với HTDCM và các tấn công khác nhau.
    
    Args:
        num_shards: Số lượng shard trong mạng
        nodes_per_shard: Số lượng node trên mỗi shard
        malicious_percentage: Tỷ lệ phần trăm nút độc hại
        attack_type: Loại tấn công ('sybil', 'eclipse', '51_percent', 'selfish_mining', 'ddos', 'mixed')
        simulation_steps: Số bước mô phỏng
        tx_per_step: Số giao dịch mỗi bước
        
    Returns:
        Dict chứa các metrics của mô phỏng
    """
    print(f"Mô phỏng mạng với {num_shards} shard, {nodes_per_shard} node/shard, {malicious_percentage}% nút độc hại, tấn công {attack_type}")
    
    # Tạo đồ thị mạng blockchain
    G = nx.Graph()
    
    # Tạo danh sách các shard và các nút trong mỗi shard
    total_nodes = num_shards * nodes_per_shard
    shards = []
    
    # Theo dõi các nút độc hại
    malicious_nodes = []
    
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
                      is_malicious=is_malicious,
                      attack_type=attack_type if is_malicious else None)
            
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
    
    # Thực hiện các cấu hình tấn công cụ thể
    if attack_type == "eclipse":
        # Tấn công Eclipse: Cô lập một số node
        if malicious_nodes:
            # Chọn ngẫu nhiên các nút cần cô lập
            non_malicious = [n for n in G.nodes() if not G.nodes[n].get('is_malicious', False)]
            targets = np.random.choice(non_malicious, 
                                      size=min(3, len(non_malicious)), 
                                      replace=False)
            
            # Cô lập các nút target bằng cách chỉ kết nối chúng với các nút độc hại
            for target in targets:
                # Xóa tất cả các kết nối hiện tại
                target_edges = list(G.edges(target))
                for u, v in target_edges:
                    G.remove_edge(u, v)
                
                # Thêm kết nối mới chỉ đến các nút độc hại
                for m_node in malicious_nodes:
                    G.add_edge(target, m_node, weight=1)
                print(f"Node {target} đã bị cô lập trong tấn công Eclipse")
    
    elif attack_type == "51_percent":
        # Tấn công 51%: Đảm bảo có ít nhất 51% nút độc hại
        malicious_count = len(malicious_nodes)
        if malicious_count < total_nodes * 0.51:
            # Chọn thêm các nút ngẫu nhiên để đạt 51%
            non_malicious = [n for n in G.nodes() if not G.nodes[n].get('is_malicious', False)]
            additional_needed = int(total_nodes * 0.51) - malicious_count
            
            if additional_needed > 0 and len(non_malicious) > 0:
                additional = np.random.choice(non_malicious, 
                                            size=min(additional_needed, len(non_malicious)), 
                                            replace=False)
                
                for node in additional:
                    G.nodes[node]['is_malicious'] = True
                    G.nodes[node]['attack_type'] = attack_type
                    malicious_nodes.append(node)
                
                print(f"Thêm {len(additional)} nút độc hại để đạt 51%, tổng số: {len(malicious_nodes)}")

def run_htdcm_performance_analysis(args):
    """
    Phân tích hiệu năng của HTDCM dưới nhiều kịch bản và cấu hình.
    
    Args:
        args: Tham số dòng lệnh
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Bắt đầu phân tích hiệu năng HTDCM ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Thiết lập các kịch bản mô phỏng - thay đổi tùy vào args
    if args.mode == "attack_comparison":
        # So sánh giữa các loại tấn công - giảm bớt số lượng tấn công để gỡ lỗi
        scenarios = []
        for attack_type in ["no_attack", "sybil", "eclipse"]:
            malicious_pct = 20
            if attack_type == "51_percent":
                malicious_pct = 51
            elif attack_type == "sybil":
                malicious_pct = 30
                
            scenarios.append({
                "num_shards": args.num_shards,
                "nodes_per_shard": args.nodes_per_shard,
                "malicious_percentage": malicious_pct,
                "attack_type": attack_type,
                "simulation_steps": args.steps,
                "tx_per_step": args.tx_per_step
            }) 