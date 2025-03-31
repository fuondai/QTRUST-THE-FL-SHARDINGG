#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run All Tests - Chạy tất cả các bài test của QTrust
Kết quả sẽ được lưu vào thư mục results
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

def create_directories():
    """Tạo các thư mục cần thiết để lưu kết quả."""
    directories = [
        'results',
        'results/cache',
        'results/attack',
        'results/benchmark',
        'results/htdcm',
        'results/energy',
        'results/charts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

def run_command(command, description=None, save_output=True, output_file=None):
    """Chạy lệnh và ghi nhật ký."""
    print(f"\n{'='*80}")
    if description:
        print(f"{description}")
    print(f"Lệnh: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    if save_output and output_file:
        with open(output_file, 'w') as f:
            result = subprocess.run(
                command,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
    else:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.stderr:
            print(f"Lỗi: {result.stderr}")
    
    elapsed_time = time.time() - start_time
    print(f"\nĐã hoàn thành trong {elapsed_time:.2f} giây. Mã thoát: {result.returncode}\n")
    
    return result.returncode == 0

def run_all_tests(skip_long_tests=False):
    """Chạy tất cả các bài test."""
    create_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_tests = []
    success_count = 0
    
    # Test đơn giản về nhập khẩu
    test = {
        'name': 'Import Test',
        'command': f'py -3.10 tests/test_import.py',
        'output': f'results/import_test_{timestamp}.log',
        'description': 'Kiểm tra nhập khẩu các module'
    }
    all_tests.append(test)
    
    # Test về hiệu suất caching
    test = {
        'name': 'Cache Optimization Test',
        'command': f'py -3.10 tests/cache_optimization_test.py --shards 24 --nodes 20 --steps 20 --tx 200',
        'output': f'results/cache/cache_optimization_{timestamp}.log',
        'description': 'Kiểm tra hiệu suất caching với 24 shards, 20 nodes mỗi shard'
    }
    all_tests.append(test)
    
    test = {
        'name': 'Detailed Caching Test',
        'command': f'py -3.10 tests/test_caching.py --agent rainbow --episodes 5',
        'output': f'results/cache/detailed_caching_{timestamp}.log',
        'description': 'Kiểm tra hiệu suất caching chi tiết với Rainbow DQN'
    }
    all_tests.append(test)
    
    # Test HTDCM
    test = {
        'name': 'HTDCM Test',
        'command': f'py -3.10 tests/htdcm_test.py',
        'output': f'results/htdcm/htdcm_test_{timestamp}.log',
        'description': 'Kiểm tra cơ chế HTDCM'
    }
    all_tests.append(test)
    
    test = {
        'name': 'HTDCM Performance Test',
        'command': f'py -3.10 tests/htdcm_performance_test.py',
        'output': f'results/htdcm/htdcm_performance_{timestamp}.log',
        'description': 'Kiểm tra hiệu suất HTDCM'
    }
    all_tests.append(test)
    
    # Test Rainbow DQN trên CartPole
    test = {
        'name': 'Rainbow DQN CartPole Test',
        'command': f'py -3.10 tests/test_rainbow_cartpole.py',
        'output': f'results/rainbow_cartpole_{timestamp}.log',
        'description': 'Kiểm tra Rainbow DQN trên môi trường CartPole'
    }
    all_tests.append(test)
    
    # Test mô phỏng tấn công
    if not skip_long_tests:
        test = {
            'name': 'Attack Simulation',
            'command': f'py -3.10 tests/attack_simulation_runner.py --num-shards 16 --nodes-per-shard 12 --attack-type all --output-dir results/attack',
            'output': f'results/attack/attack_simulation_{timestamp}.log',
            'description': 'Mô phỏng các loại tấn công'
        }
        all_tests.append(test)
    
    # Test so sánh benchmark
    test = {
        'name': 'Benchmark Comparison',
        'command': f'py -3.10 tests/benchmark_comparison_systems.py --output-dir results/benchmark',
        'output': f'results/benchmark/benchmark_comparison_{timestamp}.log',
        'description': 'So sánh hiệu suất với các hệ thống blockchain khác'
    }
    all_tests.append(test)
    
    # Test plot attack comparison
    test = {
        'name': 'Plot Attack Comparison',
        'command': f'py -3.10 tests/plot_attack_comparison.py --output-dir results/charts',
        'output': f'results/attack/plot_attack_{timestamp}.log',
        'description': 'Vẽ biểu đồ so sánh các loại tấn công'
    }
    all_tests.append(test)
    
    # Chạy tất cả các bài test
    for test in all_tests:
        print(f"\n\n{'#'*100}")
        print(f"Đang chạy test: {test['name']}")
        print(f"{'#'*100}\n")
        
        success = run_command(
            test['command'],
            test['description'],
            save_output=True,
            output_file=test['output']
        )
        
        if success:
            success_count += 1
        
    # In kết quả tổng quan
    print(f"\n\n{'#'*100}")
    print(f"Tổng quan kết quả:")
    print(f"{'#'*100}\n")
    print(f"Tổng số test: {len(all_tests)}")
    print(f"Thành công: {success_count}")
    print(f"Thất bại: {len(all_tests) - success_count}")
    print(f"Tỷ lệ thành công: {success_count/len(all_tests)*100:.2f}%")
    
    return success_count == len(all_tests)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy tất cả các bài test của QTrust.')
    parser.add_argument('--skip-long', action='store_true', help='Bỏ qua các bài test chạy lâu')
    
    args = parser.parse_args()
    
    success = run_all_tests(skip_long_tests=args.skip_long)
    
    sys.exit(0 if success else 1) 