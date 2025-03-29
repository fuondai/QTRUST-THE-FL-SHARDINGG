# QTrust - Tối Ưu Hóa Blockchain Sharding bằng Deep Reinforcement Learning

## Tổng Quan

QTrust là một framework nghiên cứu toàn diện nhằm tối ưu hóa hiệu suất blockchain dựa trên kỹ thuật sharding bằng Deep Reinforcement Learning (DRL). Dự án này tập trung vào các yếu tố quan trọng như hiệu suất xử lý giao dịch (throughput), độ trễ, tiêu thụ năng lượng và bảo mật.

### Đặc Điểm Chính

- **Tối ưu hóa dựa trên DRL**: Sử dụng các thuật toán DRL hiện đại để tối ưu hóa các quyết định routing và đồng thuận trong blockchain.
- **Mô phỏng blockchain sharding**: Môi trường mô phỏng blockchain với nhiều shard và các giao dịch xuyên shard để huấn luyện và đánh giá.
- **Cơ chế đồng thuận thích ứng**: Tự động chọn giao thức đồng thuận tối ưu dựa trên điều kiện mạng và yêu cầu bảo mật.
- **Cơ chế tin cậy phân cấp (HTDCM)**: Theo dõi và đánh giá độ tin cậy của các node trong mạng.
- **Học liên hợp (Federated Learning)**: Hỗ trợ huấn luyện các node phân tán mà không cần chia sẻ dữ liệu cục bộ.

## Các Thành Phần

### 1. BlockchainEnvironment

Môi trường mô phỏng blockchain với khả năng tái tạo nhiều tình huống mạng khác nhau, bao gồm:
- Mô phỏng nhiều shard trong mạng blockchain
- Tạo và xử lý giao dịch xuyên shard
- Mô hình hóa độ trễ mạng và tiêu thụ năng lượng
- Tính toán phần thưởng dựa trên hiệu suất xử lý, độ trễ, tiêu thụ năng lượng và bảo mật

### 2. DQNAgent

Agent học tăng cường sâu triển khai các kỹ thuật tiên tiến:
- Double DQN cho độ ổn định học tập
- Kiến trúc mạng Dueling Network để ước tính giá trị trạng thái
- Prioritized Experience Replay để đẩy nhanh học tập
- Noisy Networks cho exploration hiệu quả

### 3. Cơ Chế HTDCM

Cơ chế tin cậy phân cấp (Hierarchical Trust-based Data Center Mechanism) giúp:
- Theo dõi độ tin cậy của node dựa trên lịch sử hoạt động
- Phát hiện hành vi bất thường trong mạng
- Hỗ trợ ra quyết định routing và đồng thuận

### 4. AdaptiveConsensus

Module chọn giao thức đồng thuận tối ưu dựa trên trạng thái mạng:
- Fast BFT: Cho độ trễ thấp khi mạng ổn định
- PBFT: Cân bằng giữa hiệu suất và bảo mật
- Robust BFT: Tối ưu bảo mật khi mạng không ổn định

### 5. MADRAPIDRouter

Router thông minh để điều hướng giao dịch giữa các shard:
- Cân bằng tải trên các shard
- Giảm thiểu giao dịch xuyên shard khi có thể
- Tối ưu hóa độ trễ và tiêu thụ năng lượng

### 6. Federated Learning

Hỗ trợ huấn luyện phân tán với:
- Các phương pháp tổng hợp khác nhau (FedAvg, FedTrust, FedAdam)
- Cá nhân hóa mô hình cho từng client
- Tích hợp điểm tin cậy cho trọng số client

### 7. Visualization

Công cụ trực quan hóa kết quả và hiệu suất:
- Biểu đồ cho phần thưởng, throughput, độ trễ và tiêu thụ năng lượng
- Dashboard tương tác dựa trên Streamlit
- So sánh hiệu suất giữa các hệ thống khác nhau

## Cài Đặt

```bash
# Clone repository
git clone https://github.com/username/qtrust.git
cd qtrust

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng Poetry nếu có
poetry install
```

## Sử Dụng

### Huấn Luyện DQN Agent

```python
from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.agents.dqn.agent import DQNAgent

# Khởi tạo môi trường
env = BlockchainEnvironment(num_shards=4, num_nodes_per_shard=10)

# Khởi tạo agent
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.nvec[0],
    prioritized_replay=True,
    dueling=True
)

# Huấn luyện agent
agent.train(env, n_episodes=1000)
```

### Chạy Mô Phỏng

```python
# Chạy mô phỏng với agent đã huấn luyện
agent.load("models/best_model.pth")
total_reward = agent.evaluate(env, n_episodes=10, render=True)
print(f"Average reward: {total_reward}")
```

## Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Hãy tham khảo file `CONTRIBUTING.md` để biết thêm chi tiết.

## Giấy Phép

Dự án này được cấp phép theo [MIT License](LICENSE). 