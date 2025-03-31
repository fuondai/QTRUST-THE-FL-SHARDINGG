#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run All - Chạy tất cả các bước từ đầu đến cuối cho QTrust
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

def run_command(command, description=None):
    """Chạy lệnh và ghi nhật ký."""
    print(f"\n{'='*80}")
    if description:
        print(f"{description}")
    print(f"Lệnh: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
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

def run_all(args):
    """Chạy toàn bộ workflow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Bắt đầu chạy toàn bộ workflow QTrust lúc {timestamp}")
    
    # Dọn dẹp nếu cần
    if args.clean:
        run_command("py -3.10 run_final_benchmark.py --clean-only", "Dọn dẹp thư mục kết quả")
    
    # 1. Chạy tất cả các test
    if not args.skip_tests:
        success = run_command("py -3.10 tests/run_all_tests.py", "Chạy tất cả các test")
        if not success and not args.ignore_failures:
            print("Các test có lỗi. Dừng tiến trình.")
            return False
    
    # 2. Chạy benchmark cuối cùng
    if not args.skip_benchmark:
        success = run_command("py -3.10 run_final_benchmark.py", "Chạy benchmark cuối cùng")
        if not success and not args.ignore_failures:
            print("Benchmark có lỗi. Dừng tiến trình.")
            return False
    
    # 3. Tạo biểu đồ kết quả
    if not args.skip_charts:
        success = run_command("py -3.10 generate_final_charts.py", "Tạo biểu đồ kết quả")
        if not success and not args.ignore_failures:
            print("Tạo biểu đồ có lỗi. Dừng tiến trình.")
            return False
    
    # 4. Hiển thị thông tin tổng quan
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"QTrust đã hoàn thành tất cả các bước.")
    print(f"Bắt đầu: {timestamp}")
    print(f"Kết thúc: {end_timestamp}")
    print(f"{'='*80}")
    
    print("\nTài liệu quan trọng:")
    print("- README.md: Tổng quan dự án")
    print("- docs/architecture/qtrust_architecture.md: Kiến trúc QTrust")
    print("- docs/methodology/qtrust_methodology.md: Phương pháp nghiên cứu")
    print("- docs/exported_charts/index.html: Biểu đồ kết quả")
    print("- cleaned_results/README.md: Kết quả benchmark")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy tất cả các bước từ đầu đến cuối cho QTrust.')
    parser.add_argument('--clean', action='store_true', help='Dọn dẹp thư mục kết quả trước khi chạy')
    parser.add_argument('--skip-tests', action='store_true', help='Bỏ qua việc chạy tests')
    parser.add_argument('--skip-benchmark', action='store_true', help='Bỏ qua việc chạy benchmark')
    parser.add_argument('--skip-charts', action='store_true', help='Bỏ qua việc tạo biểu đồ')
    parser.add_argument('--ignore-failures', action='store_true', help='Tiếp tục ngay cả khi có lỗi')
    
    args = parser.parse_args()
    
    success = run_all(args)
    sys.exit(0 if success else 1) 