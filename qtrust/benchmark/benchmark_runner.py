#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark Runner - Công cụ chạy các kịch bản benchmark cho QTrust

File này cung cấp các công cụ để thực thi các kịch bản benchmark đã định nghĩa
và thu thập kết quả để phân tích và so sánh.
"""

import os
import sys
import time
import subprocess
import argparse
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình encoding cho output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục gốc vào PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from qtrust.benchmark.benchmark_scenarios import (
    get_scenario, get_all_scenario_ids, get_all_scenarios,
    BenchmarkScenario, NetworkCondition, AttackProfile,
    WorkloadProfile, NodeProfile
)

# Thư mục lưu kết quả
RESULTS_DIR = os.path.join(project_root, "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmark(scenario_id: str, output_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Chạy một kịch bản benchmark và thu thập kết quả.
    
    Args:
        scenario_id: ID của kịch bản cần chạy
        output_dir: Thư mục đầu ra, nếu không chỉ định sẽ tạo thư mục con trong RESULTS_DIR
        verbose: In thông tin chi tiết trong quá trình chạy
        
    Returns:
        Dict chứa metadata và kết quả của benchmark
    """
    start_time = time.time()
    scenario = get_scenario(scenario_id)
    
    if verbose:
        print(f"Chạy kịch bản benchmark: {scenario.name} ({scenario_id})")
        print(f"Mô tả: {scenario.description}")
        print(f"Tham số: {scenario.get_command_line_args()}")
        print("-" * 80)
    
    # Tạo thư mục đầu ra nếu chưa chỉ định
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, f"{scenario_id}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo file JSON chứa thông tin cấu hình kịch bản
    scenario_config_path = os.path.join(output_dir, "scenario_config.json")
    with open(scenario_config_path, "w", encoding="utf-8") as f:
        # Chuyển đổi dataclasses thành dict
        scenario_dict = {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "num_shards": scenario.num_shards,
            "nodes_per_shard": scenario.nodes_per_shard,
            "max_steps": scenario.max_steps,
            "network_conditions": {
                "latency_base": scenario.network_conditions.latency_base,
                "latency_variance": scenario.network_conditions.latency_variance,
                "packet_loss_rate": scenario.network_conditions.packet_loss_rate,
                "bandwidth_limit": scenario.network_conditions.bandwidth_limit,
                "congestion_probability": scenario.network_conditions.congestion_probability,
                "jitter": scenario.network_conditions.jitter
            },
            "attack_profile": {
                "attack_type": scenario.attack_profile.attack_type,
                "malicious_node_percentage": scenario.attack_profile.malicious_node_percentage,
                "attack_intensity": scenario.attack_profile.attack_intensity,
                "attack_target": scenario.attack_profile.attack_target,
                "attack_duration": scenario.attack_profile.attack_duration,
                "attack_start_step": scenario.attack_profile.attack_start_step
            },
            "workload_profile": {
                "transactions_per_step_base": scenario.workload_profile.transactions_per_step_base,
                "transactions_per_step_variance": scenario.workload_profile.transactions_per_step_variance,
                "cross_shard_transaction_ratio": scenario.workload_profile.cross_shard_transaction_ratio,
                "transaction_value_mean": scenario.workload_profile.transaction_value_mean,
                "transaction_value_variance": scenario.workload_profile.transaction_value_variance,
                "transaction_size_mean": scenario.workload_profile.transaction_size_mean,
                "transaction_size_variance": scenario.workload_profile.transaction_size_variance,
                "bursty_traffic": scenario.workload_profile.bursty_traffic,
                "burst_interval": scenario.workload_profile.burst_interval,
                "burst_multiplier": scenario.workload_profile.burst_multiplier
            },
            "node_profile": {
                "processing_power_mean": scenario.node_profile.processing_power_mean,
                "processing_power_variance": scenario.node_profile.processing_power_variance,
                "energy_efficiency_mean": scenario.node_profile.energy_efficiency_mean,
                "energy_efficiency_variance": scenario.node_profile.energy_efficiency_variance,
                "reliability_mean": scenario.node_profile.reliability_mean,
                "reliability_variance": scenario.node_profile.reliability_variance,
                "node_failure_rate": scenario.node_profile.node_failure_rate,
                "node_recovery_rate": scenario.node_profile.node_recovery_rate
            },
            "enable_dynamic_resharding": scenario.enable_dynamic_resharding,
            "min_shards": scenario.min_shards,
            "max_shards": scenario.max_shards,
            "enable_adaptive_consensus": scenario.enable_adaptive_consensus,
            "enable_bls": scenario.enable_bls,
            "enable_adaptive_pos": scenario.enable_adaptive_pos,
            "enable_lightweight_crypto": scenario.enable_lightweight_crypto,
            "enable_federated": scenario.enable_federated,
            "seed": scenario.seed
        }
        json.dump(scenario_dict, f, indent=4)
    
    # Xây dựng lệnh chạy
    save_path = os.path.join(output_dir, "results")
    
    cmd = [
        "py", "-3.10", "-m", "main",
        "--eval",  # Chế độ đánh giá
        "--save-dir", save_path,
    ]
    
    # Thêm tham số từ kịch bản
    scenario_args = scenario.get_command_line_args().split()
    cmd.extend(scenario_args)
    
    # Log file
    log_file_path = os.path.join(output_dir, "benchmark_log.txt")
    
    # Chạy lệnh và ghi log
    if verbose:
        print(f"Đang chạy lệnh: {' '.join(cmd)}")
        print(f"Log file: {log_file_path}")
    
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
        
        # Đọc output và ghi vào log file
        for line in process.stdout:
            log_file.write(line)
            if verbose:
                print(line.strip())
        
        process.wait()
    
    # Đọc kết quả từ file JSON (nếu có)
    results_json_path = os.path.join(save_path, "final_metrics.json")
    results = {}
    
    if os.path.exists(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    # Thêm thông tin về thời gian chạy
    end_time = time.time()
    execution_time = end_time - start_time
    
    benchmark_results = {
        "scenario_id": scenario_id,
        "scenario_name": scenario.name,
        "execution_time": execution_time,
        "timestamp": timestamp,
        "output_dir": output_dir,
        "results": results,
        "exit_code": process.returncode
    }
    
    # Lưu kết quả tổng hợp
    with open(os.path.join(output_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)
    
    if verbose:
        print(f"Benchmark hoàn thành trong {execution_time:.2f} giây")
        print(f"Kết quả đã được lưu vào: {output_dir}")
    
    return benchmark_results

def run_all_benchmarks(
    scenario_ids: Optional[List[str]] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Chạy nhiều kịch bản benchmark, có thể song song.
    
    Args:
        scenario_ids: Danh sách ID kịch bản cần chạy, nếu None sẽ chạy tất cả
        parallel: Cho phép chạy song song
        max_workers: Số lượng worker tối đa nếu chạy song song
        verbose: In thông tin chi tiết trong quá trình chạy
        
    Returns:
        Dict chứa kết quả của tất cả các benchmark, với key là scenario_id
    """
    if scenario_ids is None:
        scenario_ids = get_all_scenario_ids()
    
    if verbose:
        print(f"Chuẩn bị chạy {len(scenario_ids)} kịch bản benchmark")
        if parallel:
            num_workers = multiprocessing.cpu_count() if max_workers is None else max_workers
            print(f"Chế độ song song với {num_workers} worker")
    
    # Tạo thư mục đầu ra cho batch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = os.path.join(RESULTS_DIR, f"batch_{timestamp}")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    all_results = {}
    
    if parallel:
        # Chạy song song với ProcessPoolExecutor
        num_workers = multiprocessing.cpu_count() if max_workers is None else max_workers
        
        def _run_benchmark_wrapper(scenario_id):
            output_dir = os.path.join(batch_output_dir, scenario_id)
            return run_benchmark(scenario_id, output_dir, verbose=False)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_scenario = {
                executor.submit(_run_benchmark_wrapper, scenario_id): scenario_id
                for scenario_id in scenario_ids
            }
            
            for future in future_to_scenario:
                scenario_id = future_to_scenario[future]
                try:
                    result = future.result()
                    all_results[scenario_id] = result
                    if verbose:
                        print(f"Hoàn thành: {scenario_id} trong {result['execution_time']:.2f} giây")
                except Exception as e:
                    if verbose:
                        print(f"Lỗi khi chạy {scenario_id}: {str(e)}")
    else:
        # Chạy tuần tự
        for scenario_id in scenario_ids:
            output_dir = os.path.join(batch_output_dir, scenario_id)
            try:
                result = run_benchmark(scenario_id, output_dir, verbose)
                all_results[scenario_id] = result
            except Exception as e:
                if verbose:
                    print(f"Lỗi khi chạy {scenario_id}: {str(e)}")
    
    # Lưu tổng hợp tất cả kết quả
    summary_path = os.path.join(batch_output_dir, "all_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    
    if verbose:
        print(f"Tất cả benchmark đã hoàn thành. Tổng hợp kết quả tại: {summary_path}")
    
    return all_results

def generate_comparison_report(results_dir: Optional[str] = None, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Tạo báo cáo so sánh từ các kết quả benchmark đã chạy.
    
    Args:
        results_dir: Thư mục chứa kết quả (thư mục batch), nếu None sẽ sử dụng thư mục batch mới nhất
        output_file: Đường dẫn để lưu báo cáo, nếu None sẽ tạo tên file dựa trên timestamp
        
    Returns:
        DataFrame chứa dữ liệu so sánh
    """
    if results_dir is None:
        # Tìm thư mục batch mới nhất
        batch_dirs = [d for d in os.listdir(RESULTS_DIR) if d.startswith("batch_")]
        if not batch_dirs:
            raise ValueError("Không tìm thấy thư mục kết quả batch nào")
        batch_dirs.sort(reverse=True)  # Sắp xếp theo thời gian giảm dần
        results_dir = os.path.join(RESULTS_DIR, batch_dirs[0])
    
    # Đọc file tổng hợp kết quả
    summary_path = os.path.join(results_dir, "all_results_summary.json")
    if not os.path.exists(summary_path):
        # Tìm các thư mục con chứa kết quả riêng lẻ
        result_data = {}
        for scenario_dir in os.listdir(results_dir):
            scenario_path = os.path.join(results_dir, scenario_dir)
            if os.path.isdir(scenario_path):
                result_file = os.path.join(scenario_path, "benchmark_results.json")
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as f:
                        result_data[scenario_dir] = json.load(f)
    else:
        with open(summary_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
    
    # Tạo DataFrame từ kết quả
    data = []
    for scenario_id, result in result_data.items():
        scenario_name = result.get("scenario_name", scenario_id)
        execution_time = result.get("execution_time", 0)
        
        # Trích xuất các metrics chính từ kết quả
        metrics = result.get("results", {})
        throughput = metrics.get("average_throughput", 0)
        latency = metrics.get("average_latency", 0)
        energy = metrics.get("average_energy", 0)
        security_score = metrics.get("security_score", 0)
        cross_shard_ratio = metrics.get("cross_shard_ratio", 0)
        
        # Các thông tin về cấu hình kịch bản
        config_file = os.path.join(results_dir, scenario_id, "scenario_config.json")
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        attack_type = config.get("attack_profile", {}).get("attack_type", "none")
        num_shards = config.get("num_shards", 0)
        nodes_per_shard = config.get("nodes_per_shard", 0)
        
        # Tạo dòng dữ liệu
        row = {
            "Scenario ID": scenario_id,
            "Scenario Name": scenario_name,
            "Num Shards": num_shards,
            "Nodes per Shard": nodes_per_shard,
            "Attack Type": attack_type,
            "Throughput (tx/s)": throughput,
            "Latency (ms)": latency,
            "Energy (mJ/tx)": energy,
            "Security Score": security_score,
            "Cross-Shard Ratio": cross_shard_ratio,
            "Execution Time (s)": execution_time
        }
        data.append(row)
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    # Lưu báo cáo nếu chỉ định output_file
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"benchmark_comparison_{timestamp}.csv")
    
    df.to_csv(output_file, index=False)
    print(f"Đã lưu báo cáo so sánh vào: {output_file}")
    
    # Vẽ biểu đồ so sánh
    output_dir = os.path.dirname(output_file)
    plot_comparison_charts(df, output_dir)
    
    return df

def plot_comparison_charts(df: pd.DataFrame, output_dir: str):
    """
    Vẽ biểu đồ so sánh từ dữ liệu benchmark.
    
    Args:
        df: DataFrame chứa dữ liệu benchmark
        output_dir: Thư mục đầu ra để lưu biểu đồ
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Biểu đồ so sánh Throughput
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Throughput (tx/s)", ascending=False), 
                x="Scenario Name", y="Throughput (tx/s)")
    plt.xticks(rotation=45, ha="right")
    plt.title("So sánh Throughput giữa các kịch bản", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 2. Biểu đồ so sánh Latency
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Latency (ms)"), 
                x="Scenario Name", y="Latency (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.title("So sánh Latency giữa các kịch bản", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"latency_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 3. Biểu đồ so sánh Security Score
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Security Score", ascending=False), 
                x="Scenario Name", y="Security Score")
    plt.xticks(rotation=45, ha="right")
    plt.title("So sánh Security Score giữa các kịch bản", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"security_score_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 4. Biểu đồ so sánh Energy
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Energy (mJ/tx)"), 
                x="Scenario Name", y="Energy (mJ/tx)")
    plt.xticks(rotation=45, ha="right")
    plt.title("So sánh Energy giữa các kịch bản", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"energy_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 5. Biểu đồ Ma trận nhiệt
    plt.figure(figsize=(16, 12))
    metrics = ["Throughput (tx/s)", "Latency (ms)", "Energy (mJ/tx)", "Security Score", "Cross-Shard Ratio"]
    
    # Chuẩn hóa dữ liệu cho ma trận nhiệt
    df_heatmap = df[["Scenario Name"] + metrics].set_index("Scenario Name")
    for col in df_heatmap.columns:
        if col in ["Latency (ms)", "Energy (mJ/tx)"]:  # Các chỉ số mà thấp hơn là tốt hơn
            df_heatmap[col] = (df_heatmap[col].max() - df_heatmap[col]) / (df_heatmap[col].max() - df_heatmap[col].min())
        else:  # Các chỉ số mà cao hơn là tốt hơn
            df_heatmap[col] = (df_heatmap[col] - df_heatmap[col].min()) / (df_heatmap[col].max() - df_heatmap[col].min())
    
    sns.heatmap(df_heatmap, annot=True, cmap="viridis", linewidths=.5)
    plt.title("Ma trận nhiệt của các metric giữa các kịch bản (đã chuẩn hóa)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metrics_heatmap_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 6. Biểu đồ scatter 3D: Throughput vs Latency vs Security
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = df["Throughput (tx/s)"]
        y = df["Latency (ms)"]
        z = df["Security Score"]
        
        ax.scatter(x, y, z, c=df["Energy (mJ/tx)"], cmap="plasma", s=100, alpha=0.7)
        
        for i, scenario in enumerate(df["Scenario Name"]):
            ax.text(x[i], y[i], z[i], scenario, fontsize=8)
        
        ax.set_xlabel("Throughput (tx/s)")
        ax.set_ylabel("Latency (ms)")
        ax.set_zlabel("Security Score")
        plt.title("3D Scatter: Throughput vs Latency vs Security", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"3d_scatter_{timestamp}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Không thể tạo biểu đồ 3D: {str(e)}")

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Benchmark Runner cho QTrust")
    
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")
    
    # Lệnh list-scenarios
    list_parser = subparsers.add_parser("list-scenarios", help="Liệt kê các kịch bản benchmark có sẵn")
    
    # Lệnh run
    run_parser = subparsers.add_parser("run", help="Chạy một kịch bản benchmark")
    run_parser.add_argument("scenario_id", help="ID của kịch bản cần chạy")
    run_parser.add_argument("--output-dir", help="Thư mục đầu ra tùy chọn")
    run_parser.add_argument("--quiet", action="store_true", help="Chế độ im lặng (không in chi tiết)")
    
    # Lệnh run-all
    run_all_parser = subparsers.add_parser("run-all", help="Chạy tất cả các kịch bản benchmark")
    run_all_parser.add_argument("--scenario-ids", nargs="+", help="Danh sách ID kịch bản cần chạy")
    run_all_parser.add_argument("--parallel", action="store_true", help="Chạy song song các kịch bản")
    run_all_parser.add_argument("--max-workers", type=int, help="Số lượng worker tối đa nếu chạy song song")
    run_all_parser.add_argument("--quiet", action="store_true", help="Chế độ im lặng (không in chi tiết)")
    
    # Lệnh compare
    compare_parser = subparsers.add_parser("compare", help="Tạo báo cáo so sánh từ kết quả benchmark")
    compare_parser.add_argument("--results-dir", help="Thư mục chứa kết quả batch")
    compare_parser.add_argument("--output-file", help="Đường dẫn để lưu báo cáo CSV")
    
    return parser.parse_args()

def main():
    """Điểm vào chính của công cụ benchmark."""
    args = parse_args()
    
    if args.command == "list-scenarios":
        # Liệt kê các kịch bản
        print("Các kịch bản benchmark có sẵn:")
        print("-" * 80)
        
        for scenario_id, scenario in get_all_scenarios().items():
            print(f"ID: {scenario_id}")
            print(f"Tên: {scenario.name}")
            print(f"Mô tả: {scenario.description}")
            print(f"Tham số: {scenario.get_command_line_args()}")
            print("-" * 80)
    
    elif args.command == "run":
        # Chạy một kịch bản
        verbose = not args.quiet
        run_benchmark(args.scenario_id, args.output_dir, verbose)
    
    elif args.command == "run-all":
        # Chạy tất cả các kịch bản
        verbose = not args.quiet
        scenario_ids = args.scenario_ids
        run_all_benchmarks(scenario_ids, args.parallel, args.max_workers, verbose)
    
    elif args.command == "compare":
        # Tạo báo cáo so sánh
        generate_comparison_report(args.results_dir, args.output_file)
    
    else:
        print("Lệnh không hợp lệ. Chạy 'benchmark_runner.py -h' để xem trợ giúp.")

if __name__ == "__main__":
    main() 