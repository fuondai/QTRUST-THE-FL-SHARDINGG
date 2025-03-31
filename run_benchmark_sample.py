#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script mẫu để chạy kịch bản benchmark QTrust

Chạy script này để kiểm tra cài đặt benchmark và chạy kịch bản cơ bản.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from qtrust.benchmark.benchmark_runner import run_benchmark, get_all_scenario_ids
from qtrust.benchmark.benchmark_scenarios import get_all_scenarios

def parse_args():
    parser = argparse.ArgumentParser(description="Chạy kịch bản benchmark mẫu cho QTrust")
    parser.add_argument("--scenario", type=str, default="basic", 
                      help="ID của kịch bản benchmark (mặc định: basic)")
    parser.add_argument("--list", action="store_true", 
                      help="Liệt kê tất cả kịch bản có sẵn và thoát")
    parser.add_argument("--output-dir", type=str, 
                      help="Thư mục đầu ra (mặc định: benchmark_results/[scenario_id]_[timestamp])")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.list:
        print("Các kịch bản benchmark có sẵn:")
        print("-" * 80)
        
        for scenario_id, scenario in get_all_scenarios().items():
            print(f"ID: {scenario_id}")
            print(f"Tên: {scenario.name}")
            print(f"Mô tả: {scenario.description}")
            print("-" * 80)
        return
    
    scenario_id = args.scenario
    valid_scenarios = get_all_scenario_ids()
    
    if scenario_id not in valid_scenarios:
        print(f"Lỗi: Kịch bản '{scenario_id}' không tồn tại.")
        print(f"Các kịch bản hợp lệ: {', '.join(valid_scenarios)}")
        return 1
    
    print(f"Bắt đầu benchmark cho kịch bản: {scenario_id}")
    start_time = time.time()
    
    try:
        result = run_benchmark(scenario_id, args.output_dir, verbose=True)
        end_time = time.time()
        
        print("\nTổng kết benchmark:")
        print(f"- Kịch bản: {result['scenario_name']} ({scenario_id})")
        print(f"- Thời gian thực thi: {result['execution_time']:.2f} giây")
        print(f"- Thư mục kết quả: {result['output_dir']}")
        
        # Hiển thị các metric chính nếu có
        metrics = result.get("results", {})
        if metrics:
            print("\nCác metric chính:")
            print(f"- Throughput: {metrics.get('average_throughput', 'N/A')} tx/s")
            print(f"- Latency: {metrics.get('average_latency', 'N/A')} ms")
            print(f"- Energy: {metrics.get('average_energy', 'N/A')} mJ/tx")
            print(f"- Security Score: {metrics.get('security_score', 'N/A')}")
        
        return 0
    except Exception as e:
        print(f"Lỗi khi chạy benchmark: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 