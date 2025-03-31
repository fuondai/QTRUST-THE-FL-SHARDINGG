#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Final Benchmark - Chạy benchmark cuối cùng cho QTrust và lưu kết quả vào thư mục cleaned_results
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def create_directories():
    """Tạo các thư mục cần thiết để lưu kết quả."""
    directories = [
        'cleaned_results',
        'cleaned_results/cache',
        'cleaned_results/attack',
        'cleaned_results/benchmark',
        'cleaned_results/htdcm',
        'cleaned_results/charts'
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
        with open(output_file, 'w', encoding='utf-8') as f:
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
    
    return result.returncode == 0, result.stdout if not save_output else None

def clean_old_results():
    """Dọn dẹp các kết quả cũ."""
    dirs_to_clean = [
        'results',
        'logs',
        'benchmark_results',
        'energy_results'
    ]
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Đã xóa thư mục: {directory}")
            except Exception as e:
                print(f"Không thể xóa thư mục {directory}: {e}")
                
    # Xóa các file log và file png tạm thời
    for file in os.listdir('.'):
        if file.endswith('.log') or (file.endswith('.png') and file != 'qtrust_logo.png'):
            try:
                os.remove(file)
                print(f"Đã xóa file: {file}")
            except Exception as e:
                print(f"Không thể xóa file {file}: {e}")

def run_final_benchmark():
    """Chạy benchmark cuối cùng và lưu kết quả."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_directories()
    
    # Dọn dẹp kết quả cũ
    clean_old_results()
    
    # Tạo thư mục results tạm thời
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/cache", exist_ok=True)
    os.makedirs("results/attack", exist_ok=True)
    os.makedirs("results/benchmark", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)
    
    all_results = {
        "timestamp": timestamp,
        "tests": {}
    }
    
    # 1. Chạy cache optimization test - cấu hình tối ưu
    command = "py -3.10 tests/cache_optimization_test.py --shards 32 --nodes 24 --steps 50 --tx 500"
    description = "Chạy cache optimization test với cấu hình tối ưu (32 shards, 24 nodes/shard)"
    output_file = f"results/cache/cache_optimization_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["cache_optimization"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Sao chép biểu đồ cache nếu tồn tại
    cache_charts = [f for f in os.listdir('.') if f.startswith('cache_') and f.endswith('.png')]
    for chart in cache_charts:
        try:
            shutil.copy(chart, f"cleaned_results/cache/{timestamp}_{chart}")
            all_results["tests"]["cache_optimization"]["charts"] = [f"cleaned_results/cache/{timestamp}_{chart}"]
        except Exception as e:
            print(f"Không thể sao chép biểu đồ cache {chart}: {e}")
    
    # 2. Chạy benchmark so sánh với các hệ thống khác
    command = "py -3.10 tests/benchmark_comparison_systems.py --output-dir results/benchmark"
    description = "So sánh QTrust với các hệ thống blockchain khác"
    output_file = f"results/benchmark/benchmark_comparison_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["benchmark_comparison"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Sao chép kết quả benchmark
    benchmark_files = [f for f in os.listdir('results/benchmark') if f.endswith('.png') or f.endswith('.json') or f.endswith('.csv')]
    for file in benchmark_files:
        try:
            shutil.copy(f"results/benchmark/{file}", f"cleaned_results/benchmark/{timestamp}_{file}")
            if "charts" not in all_results["tests"]["benchmark_comparison"]:
                all_results["tests"]["benchmark_comparison"]["charts"] = []
            all_results["tests"]["benchmark_comparison"]["charts"].append(f"cleaned_results/benchmark/{timestamp}_{file}")
        except Exception as e:
            print(f"Không thể sao chép file benchmark {file}: {e}")
    
    # 3. Chạy mô phỏng tấn công
    command = "py -3.10 tests/attack_simulation_runner.py --num-shards 32 --nodes-per-shard 24 --attack-type all --output-dir results/attack"
    description = "Mô phỏng các loại tấn công trên QTrust (32 shards, 24 nodes/shard)"
    output_file = f"results/attack/attack_simulation_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["attack_simulation"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # 4. Chạy plot attack comparison
    command = "py -3.10 tests/plot_attack_comparison.py --output-dir results/charts"
    description = "Vẽ biểu đồ so sánh các loại tấn công"
    output_file = f"results/attack/plot_attack_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["attack_comparison"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Sao chép kết quả phân tích tấn công
    attack_files = []
    if os.path.exists('results/attack'):
        attack_files = [f for f in os.listdir('results/attack') if f.endswith('.png') or f.endswith('.json') or f.endswith('.csv')]
    chart_files = []
    if os.path.exists('results/charts'):
        chart_files = [f for f in os.listdir('results/charts') if f.endswith('.png') or f.endswith('.json') or f.endswith('.csv')]
    
    for file in attack_files:
        try:
            shutil.copy(f"results/attack/{file}", f"cleaned_results/attack/{timestamp}_{file}")
            if "charts" not in all_results["tests"]["attack_simulation"]:
                all_results["tests"]["attack_simulation"]["charts"] = []
            all_results["tests"]["attack_simulation"]["charts"].append(f"cleaned_results/attack/{timestamp}_{file}")
        except Exception as e:
            print(f"Không thể sao chép file tấn công {file}: {e}")
            
    for file in chart_files:
        try:
            shutil.copy(f"results/charts/{file}", f"cleaned_results/charts/{timestamp}_{file}")
            if "charts" not in all_results["tests"]["attack_comparison"]:
                all_results["tests"]["attack_comparison"]["charts"] = []
            all_results["tests"]["attack_comparison"]["charts"].append(f"cleaned_results/charts/{timestamp}_{file}")
        except Exception as e:
            print(f"Không thể sao chép file biểu đồ {file}: {e}")
    
    # 5. Chạy HTDCM performance test
    command = "py -3.10 tests/htdcm_performance_test.py"
    description = "Kiểm tra hiệu suất HTDCM"
    output_file = f"results/htdcm/htdcm_performance_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["htdcm_performance"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Lưu tổng hợp kết quả
    with open(f"cleaned_results/benchmark_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    # Tạo file README trong thư mục cleaned_results
    with open("cleaned_results/README.md", 'w', encoding='utf-8') as f:
        f.write(f"# QTrust Benchmark Results\n\n")
        f.write(f"Results from benchmark run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Test | Status | Description |\n")
        f.write("|------|--------|-------------|\n")
        
        for test_name, test_info in all_results["tests"].items():
            status = "✅ Success" if test_info["success"] else "❌ Failed"
            description = test_info["command"]
            f.write(f"| {test_name} | {status} | `{description}` |\n")
        
        f.write("\n## Details\n\n")
        for test_name, test_info in all_results["tests"].items():
            f.write(f"### {test_name}\n\n")
            f.write(f"- Command: `{test_info['command']}`\n")
            f.write(f"- Status: {'Success' if test_info['success'] else 'Failed'}\n")
            
            if "charts" in test_info and test_info["charts"]:
                f.write("\n#### Generated Charts\n\n")
                for chart in test_info["charts"]:
                    f.write(f"- [{os.path.basename(chart)}]({chart})\n")
            
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"Benchmark đã hoàn thành. Kết quả đã được lưu vào thư mục cleaned_results.")
    print(f"Tổng hợp kết quả: cleaned_results/benchmark_summary_{timestamp}.json")
    print(f"{'='*80}\n")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy benchmark cuối cùng cho QTrust.')
    parser.add_argument('--clean-only', action='store_true', help='Chỉ dọn dẹp kết quả cũ không chạy benchmark')
    
    args = parser.parse_args()
    
    if args.clean_only:
        clean_old_results()
    else:
        run_final_benchmark() 