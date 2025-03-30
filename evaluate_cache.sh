#!/bin/bash

# Script đánh giá hiệu suất của caching trong dự án QTrust
# Chạy từng agent và ghi lại kết quả để so sánh

echo "========== ĐÁNH GIÁ HIỆU SUẤT CACHING QTRUST =========="
echo "Thời gian bắt đầu: $(date)"
echo

# Kiểm tra Python
python_cmd="python"
if command -v py &> /dev/null; then
    python_cmd="py -3.10"
fi

# Tạo thư mục cho kết quả
results_dir="cache_results"
mkdir -p "$results_dir"

# Chạy test với DQN Agent
echo "===== Đánh giá DQN Agent ====="
echo "Đang chạy với caching..."
$python_cmd test_caching.py --agent dqn --episodes 10 > "$results_dir/dqn_with_cache.log"
echo "Đang chạy không có caching..."
$python_cmd test_caching.py --agent dqn --episodes 10 --disable-cache > "$results_dir/dqn_without_cache.log"

# Chạy test với Rainbow DQN Agent
echo "===== Đánh giá Rainbow DQN Agent ====="
echo "Đang chạy với caching..."
$python_cmd test_caching.py --agent rainbow --episodes 10 > "$results_dir/rainbow_with_cache.log"
echo "Đang chạy không có caching..."
$python_cmd test_caching.py --agent rainbow --episodes 10 --disable-cache > "$results_dir/rainbow_without_cache.log"

# Chạy test với Actor-Critic Agent
echo "===== Đánh giá Actor-Critic Agent ====="
echo "Đang chạy với caching..."
$python_cmd test_caching.py --agent actor-critic --episodes 10 > "$results_dir/actor_critic_with_cache.log"
echo "Đang chạy không có caching..."
$python_cmd test_caching.py --agent actor-critic --episodes 10 --disable-cache > "$results_dir/actor_critic_without_cache.log"

# Chạy so sánh tổng hợp
echo "===== So sánh tổng hợp ====="
$python_cmd test_caching.py --compare-from-logs "$results_dir" > "$results_dir/comparison_results.log"

echo
echo "Đánh giá hoàn tất! Các kết quả được lưu trong thư mục $results_dir"
echo "Xem báo cáo tổng hợp tại: $results_dir/comparison_results.log"
echo "Thời gian kết thúc: $(date)"

# Hiển thị tóm tắt nhanh
echo
echo "===== TÓM TẮT NHANH ====="
if [ -f "$results_dir/comparison_results.log" ]; then
    grep "speedup" "$results_dir/comparison_results.log"
    grep "Cache Hit Ratio" "$results_dir/comparison_results.log"
else
    echo "Chưa có báo cáo tổng hợp"
fi 