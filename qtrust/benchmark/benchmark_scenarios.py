#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark Scenarios - Tập hợp các kịch bản kiểm chuẩn cho QTrust

File này định nghĩa các kịch bản benchmark chuẩn để đánh giá hiệu suất của QTrust 
trong các điều kiện mạng thực tế và so sánh với các hệ thống blockchain khác.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any

@dataclass
class NetworkCondition:
    """Mô tả điều kiện mạng cho kịch bản benchmark."""
    latency_base: float = 10.0  # Độ trễ cơ bản (ms)
    latency_variance: float = 5.0  # Biến thiên độ trễ (ms)
    packet_loss_rate: float = 0.01  # Tỷ lệ mất gói (0-1)
    bandwidth_limit: float = 1000.0  # Giới hạn băng thông (MB/s)
    congestion_probability: float = 0.05  # Xác suất tắc nghẽn (0-1)
    jitter: float = 2.0  # Độ dao động độ trễ (ms)

@dataclass
class AttackProfile:
    """Mô tả tấn công trong kịch bản benchmark."""
    attack_type: str = "none"  # Loại tấn công: none, ddos, 51_percent, sybil, eclipse, mixed
    malicious_node_percentage: float = 0.0  # Phần trăm node độc hại (0-1)
    attack_intensity: float = 0.0  # Cường độ tấn công (0-1)
    attack_target: str = "random"  # Mục tiêu tấn công: random, specific_shard, validators
    attack_duration: int = 0  # Thời lượng tấn công (số bước)
    attack_start_step: int = 0  # Bước bắt đầu tấn công

@dataclass
class WorkloadProfile:
    """Mô tả khối lượng công việc cho kịch bản benchmark."""
    transactions_per_step_base: int = 100  # Số giao dịch cơ bản mỗi bước
    transactions_per_step_variance: int = 20  # Biến thiên số giao dịch mỗi bước
    cross_shard_transaction_ratio: float = 0.3  # Tỷ lệ giao dịch xuyên shard (0-1)
    transaction_value_mean: float = 25.0  # Giá trị trung bình của giao dịch
    transaction_value_variance: float = 10.0  # Biến thiên giá trị giao dịch
    transaction_size_mean: float = 1.0  # Kích thước trung bình của giao dịch (KB)
    transaction_size_variance: float = 0.2  # Biến thiên kích thước giao dịch (KB)
    bursty_traffic: bool = False  # Có lưu lượng đột biến hay không
    burst_interval: int = 50  # Khoảng thời gian giữa các đợt tăng đột biến (số bước)
    burst_multiplier: float = 3.0  # Hệ số nhân cho lưu lượng đột biến

@dataclass
class NodeProfile:
    """Mô tả cấu hình node trong kịch bản benchmark."""
    processing_power_mean: float = 1.0  # Sức mạnh xử lý trung bình
    processing_power_variance: float = 0.2  # Biến thiên sức mạnh xử lý
    energy_efficiency_mean: float = 0.8  # Hiệu suất năng lượng trung bình (0-1)
    energy_efficiency_variance: float = 0.1  # Biến thiên hiệu suất năng lượng
    reliability_mean: float = 0.95  # Độ tin cậy trung bình (0-1)
    reliability_variance: float = 0.05  # Biến thiên độ tin cậy
    node_failure_rate: float = 0.01  # Tỷ lệ lỗi node (0-1)
    node_recovery_rate: float = 0.8  # Tỷ lệ phục hồi node (0-1)

@dataclass
class BenchmarkScenario:
    """Định nghĩa đầy đủ một kịch bản benchmark."""
    id: str  # ID duy nhất cho kịch bản
    name: str  # Tên mô tả kịch bản
    description: str  # Mô tả chi tiết về kịch bản
    num_shards: int  # Số lượng shard
    nodes_per_shard: int  # Số node trên mỗi shard
    max_steps: int  # Số bước mô phỏng tối đa
    network_conditions: NetworkCondition  # Điều kiện mạng
    attack_profile: AttackProfile  # Thông tin tấn công
    workload_profile: WorkloadProfile  # Thông tin khối lượng công việc
    node_profile: NodeProfile  # Thông tin cấu hình node
    enable_dynamic_resharding: bool = True  # Cho phép resharding động
    min_shards: int = 4  # Số lượng shard tối thiểu
    max_shards: int = 32  # Số lượng shard tối đa
    enable_adaptive_consensus: bool = True  # Sử dụng đồng thuận thích ứng
    enable_bls: bool = True  # Sử dụng BLS signature aggregation
    enable_adaptive_pos: bool = True  # Sử dụng Adaptive PoS
    enable_lightweight_crypto: bool = True  # Sử dụng mã hóa nhẹ
    enable_federated: bool = False  # Sử dụng Federated Learning
    seed: Optional[int] = 42  # Seed cho tính tái lập
    
    def get_command_line_args(self) -> str:
        """Chuyển đổi kịch bản thành tham số dòng lệnh cho main.py."""
        args = []
        args.append(f"--num-shards {self.num_shards}")
        args.append(f"--nodes-per-shard {self.nodes_per_shard}")
        args.append(f"--max-steps {self.max_steps}")
        
        if self.attack_profile.attack_type != "none":
            args.append(f"--attack-scenario {self.attack_profile.attack_type}")
            
        if self.enable_bls:
            args.append("--enable-bls")
            
        if self.enable_adaptive_pos:
            args.append("--enable-adaptive-pos")
            
        if self.enable_lightweight_crypto:
            args.append("--enable-lightweight-crypto")
            
        if self.enable_federated:
            args.append("--enable-federated")
            
        if self.seed is not None:
            args.append(f"--seed {self.seed}")
            
        return " ".join(args)

# Định nghĩa các kịch bản benchmark chuẩn
BENCHMARK_SCENARIOS = {
    # Kịch bản cơ bản - hệ thống ổn định với lưu lượng trung bình
    "basic": BenchmarkScenario(
        id="basic",
        name="Cơ bản",
        description="Mô phỏng cơ bản với điều kiện mạng ổn định và lưu lượng trung bình",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=10.0,
            latency_variance=5.0,
            packet_loss_rate=0.01,
            bandwidth_limit=1000.0,
            congestion_probability=0.05,
            jitter=2.0
        ),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=200,
            transactions_per_step_variance=50,
            cross_shard_transaction_ratio=0.3
        ),
        node_profile=NodeProfile()
    ),
    
    # Kịch bản tải cao - kiểm tra hiệu suất dưới tải cao
    "high_load": BenchmarkScenario(
        id="high_load",
        name="Tải cao",
        description="Mô phỏng với lưu lượng giao dịch cao để kiểm tra khả năng mở rộng",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=15.0,
            latency_variance=8.0,
            packet_loss_rate=0.02,
            bandwidth_limit=800.0,
            congestion_probability=0.1,
            jitter=3.0
        ),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=500,
            transactions_per_step_variance=100,
            cross_shard_transaction_ratio=0.4,
            bursty_traffic=True,
            burst_interval=100,
            burst_multiplier=3.0
        ),
        node_profile=NodeProfile(
            processing_power_variance=0.3,
            node_failure_rate=0.02
        )
    ),
    
    # Kịch bản tấn công DDoS
    "ddos_attack": BenchmarkScenario(
        id="ddos_attack",
        name="Tấn công DDoS",
        description="Mô phỏng dưới tấn công từ chối dịch vụ phân tán",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=20.0,
            latency_variance=10.0,
            packet_loss_rate=0.05,
            bandwidth_limit=600.0,
            congestion_probability=0.2,
            jitter=5.0
        ),
        attack_profile=AttackProfile(
            attack_type="ddos",
            malicious_node_percentage=0.1,
            attack_intensity=0.8,
            attack_target="random",
            attack_duration=300,
            attack_start_step=200
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=300,
            cross_shard_transaction_ratio=0.35
        ),
        node_profile=NodeProfile(
            reliability_mean=0.9,
            node_failure_rate=0.03
        )
    ),
    
    # Kịch bản 51% attack
    "fifty_one_percent": BenchmarkScenario(
        id="fifty_one_percent",
        name="Tấn công 51%",
        description="Mô phỏng dưới tấn công 51% tập trung vào một shard",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(),
        attack_profile=AttackProfile(
            attack_type="51_percent",
            malicious_node_percentage=0.15,
            attack_intensity=0.9,
            attack_target="specific_shard",
            attack_duration=400,
            attack_start_step=300
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=250,
            cross_shard_transaction_ratio=0.35,
            transaction_value_mean=40.0
        ),
        node_profile=NodeProfile()
    ),
    
    # Kịch bản tấn công Sybil
    "sybil_attack": BenchmarkScenario(
        id="sybil_attack",
        name="Tấn công Sybil",
        description="Mô phỏng dưới tấn công Sybil với nhiều node độc hại giả mạo danh tính",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(),
        attack_profile=AttackProfile(
            attack_type="sybil",
            malicious_node_percentage=0.2,
            attack_intensity=0.7,
            attack_target="validators",
            attack_duration=500,
            attack_start_step=250
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=200
        ),
        node_profile=NodeProfile(
            reliability_variance=0.15
        )
    ),
    
    # Kịch bản tấn công Eclipse
    "eclipse_attack": BenchmarkScenario(
        id="eclipse_attack",
        name="Tấn công Eclipse",
        description="Mô phỏng dưới tấn công Eclipse nhằm cô lập một shard khỏi mạng",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_variance=15.0,
            packet_loss_rate=0.08
        ),
        attack_profile=AttackProfile(
            attack_type="eclipse",
            malicious_node_percentage=0.25,
            attack_intensity=0.85,
            attack_target="specific_shard",
            attack_duration=350,
            attack_start_step=400
        ),
        workload_profile=WorkloadProfile(
            cross_shard_transaction_ratio=0.5
        ),
        node_profile=NodeProfile()
    ),
    
    # Kịch bản mô phỏng điều kiện mạng thực tế - độ trễ cao, packet loss
    "real_world_conditions": BenchmarkScenario(
        id="real_world_conditions",
        name="Điều kiện mạng thực tế",
        description="Mô phỏng trong điều kiện mạng thực tế với độ trễ cao và mất gói",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=50.0,
            latency_variance=25.0,
            packet_loss_rate=0.1,
            bandwidth_limit=500.0,
            congestion_probability=0.15,
            jitter=15.0
        ),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=150,
            cross_shard_transaction_ratio=0.4,
            bursty_traffic=True
        ),
        node_profile=NodeProfile(
            processing_power_variance=0.4,
            reliability_mean=0.85,
            node_failure_rate=0.05
        )
    ),
    
    # Kịch bản quy mô lớn - kiểm tra với số lượng shard và node cao
    "large_scale": BenchmarkScenario(
        id="large_scale",
        name="Quy mô lớn",
        description="Mô phỏng quy mô lớn với số lượng shard và node cao",
        num_shards=32,
        nodes_per_shard=30,
        max_steps=500,
        network_conditions=NetworkCondition(),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=1000,
            transactions_per_step_variance=200,
            cross_shard_transaction_ratio=0.45
        ),
        node_profile=NodeProfile(),
        enable_federated=True
    ),
    
    # Kịch bản tấn công kết hợp - nhiều loại tấn công cùng lúc
    "mixed_attack": BenchmarkScenario(
        id="mixed_attack",
        name="Tấn công kết hợp",
        description="Mô phỏng dưới nhiều loại tấn công kết hợp (DDoS + Sybil)",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=30.0,
            packet_loss_rate=0.07
        ),
        attack_profile=AttackProfile(
            attack_type="mixed",
            malicious_node_percentage=0.3,
            attack_intensity=0.9,
            attack_target="random",
            attack_duration=600,
            attack_start_step=200
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=200
        ),
        node_profile=NodeProfile(
            node_failure_rate=0.04
        )
    )
}

def get_scenario(scenario_id: str) -> BenchmarkScenario:
    """Lấy kịch bản benchmark theo ID."""
    if scenario_id not in BENCHMARK_SCENARIOS:
        raise ValueError(f"Không tìm thấy kịch bản benchmark với ID: {scenario_id}")
    return BENCHMARK_SCENARIOS[scenario_id]

def get_all_scenario_ids() -> List[str]:
    """Lấy danh sách ID của tất cả các kịch bản benchmark."""
    return list(BENCHMARK_SCENARIOS.keys())

def get_all_scenarios() -> Dict[str, BenchmarkScenario]:
    """Lấy tất cả các kịch bản benchmark."""
    return BENCHMARK_SCENARIOS

if __name__ == "__main__":
    # Hiển thị tất cả các kịch bản và lệnh tương ứng
    for scenario_id, scenario in BENCHMARK_SCENARIOS.items():
        print(f"Kịch bản: {scenario.name} ({scenario_id})")
        print(f"Mô tả: {scenario.description}")
        print(f"Lệnh: py -3.10 -m main {scenario.get_command_line_args()}")
        print("-" * 80) 