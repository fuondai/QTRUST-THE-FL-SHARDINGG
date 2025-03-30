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

# HƯỚNG DẪN TRIỂN KHAI DỰ ÁN QTRUST-FL-SHARDING

## GIỚI THIỆU

Dự án QTrust-FL-Sharding là dự án cải tiến blockchain sharding sử dụng Deep Reinforcement Learning (DRL) và Federated Learning (FL). Tài liệu này cung cấp hướng dẫn chi tiết cho AI về cách triển khai các cải tiến theo kế hoạch, tránh hiện tượng ảo giác AI và đảm bảo tập trung vào nhiệm vụ.

## PHƯƠNG PHÁP TRIỂN KHAI

### 1. Quy trình kiểm tra trước khi triển khai

AI cần thực hiện các bước sau trước khi triển khai bất kỳ thay đổi nào:

1. **Xác minh tệp nguồn**: Kiểm tra nội dung hiện tại của tệp trước khi sửa đổi
2. **Xác nhận cấu trúc dự án**: Xác minh vị trí và quan hệ của tệp trong dự án
3. **Kiểm tra tính nhất quán**: Đảm bảo thay đổi phù hợp với codebase hiện tại
4. **Tham chiếu tài liệu kỹ thuật**: Tham khảo `.cursorrules` và `qtrust_improvement_plan.md`

### 2. Quy trình cập nhật Knowledge Graph

AI cần duy trì Knowledge Graph để theo dõi tiến độ và tránh lặp lại hoặc bỏ sót công việc:

1. **Khởi tạo công việc**: Tạo entity cho mỗi nhiệm vụ cần thực hiện
2. **Cập nhật trạng thái**: Cập nhật trạng thái công việc (chưa bắt đầu, đang thực hiện, hoàn thành)
3. **Ghi nhận thay đổi**: Lưu lại các thay đổi đã thực hiện và kết quả đạt được
4. **Liên kết tài nguyên**: Liên kết giữa tệp, chức năng và nhiệm vụ

## HƯỚNG DẪN CHI TIẾT TỪNG GIAI ĐOẠN

### Giai đoạn 1: Tối ưu cấu trúc mạng

#### 1.1. Cải tiến Blockchain Environment

```
Tệp: qtrust/simulation/blockchain_environment.py
```

**Các bước thực hiện**:
1. Xác định cấu trúc hiện tại của lớp Environment
2. Sửa đổi để hỗ trợ 24-32 shard (hiện tại 12)
3. Thêm cơ chế resharding động khi phát hiện tắc nghẽn
4. Cập nhật reward function để phản ánh hiệu suất sharding

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định về tên biến hoặc phương thức chưa xác minh
- LUÔN đọc code hiện tại trước khi sửa đổi
- XÁC MINH các tham số đầu vào và kiểu dữ liệu trước khi thực hiện thay đổi

#### 1.2. Cải tiến MAD-RAPID Router

```
Tệp: qtrust/routing/mad_rapid.py
```

**Các bước thực hiện**:
1. Phân tích logic routing hiện tại
2. Thêm proximity-aware routing để giảm latency
3. Thêm cơ chế dynamic mesh connections
4. Triển khai predictive routing dựa trên lịch sử

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định các hàm tiện ích có sẵn
- XÁC MINH giao diện với các module khác
- KIỂM TRA tính tương thích với môi trường mô phỏng

### Giai đoạn 2: Cải tiến cơ chế đồng thuận

#### 2.1. Nâng cấp Adaptive Consensus

```
Tệp: qtrust/consensus/adaptive_consensus.py
```

**Các bước thực hiện**:
1. Xác định thuật toán đồng thuận hiện tại
2. Thêm các cơ chế đồng thuận nhẹ hơn
3. Triển khai BLS signature aggregation
4. Tối ưu hóa quá trình lựa chọn đồng thuận

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định các thư viện mã hóa có sẵn
- KIỂM TRA tỷ lệ trade-off giữa bảo mật và hiệu suất
- CẬP NHẬT Knowledge Graph sau mỗi thay đổi lớn

#### 2.2. Nâng cấp HTDCM Trust

```
Tệp: qtrust/trust/htdcm.py
```

**Các bước thực hiện**:
1. Xác định cơ chế trust hiện tại
2. Thêm ML-based anomaly detection
3. Triển khai reputation-based validator selection
4. Tăng cường phát hiện các loại tấn công

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định các thuật toán ML đã được triển khai
- KIỂM TRA ảnh hưởng của thay đổi đến hiệu suất tổng thể
- LƯU TRỮ kết quả benchmark trước và sau khi thay đổi

### Giai đoạn 3: Nâng cấp kiến trúc DRL

#### 3.1. Cải tiến DQN Agent

```
Tệp: qtrust/agents/dqn/agent.py
```

**Các bước thực hiện**:
1. Nâng cấp từ DQN thông thường lên Rainbow DQN
2. Thêm Actor-Critic architecture
3. Thêm Meta-learning để học nhanh hơn

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định kiến trúc mạng nơ-ron
- KIỂM TRA cách DQN Agent hiện tại tương tác với môi trường
- XÁC MINH cấu trúc lớp và interface trước khi sửa đổi

#### 3.2. Cải tiến Federated Learning

```
Tệp: qtrust/federated/federated_learning.py
```

**Các bước thực hiện**:
1. Triển khai Federated Reinforcement Learning
2. Tích hợp Privacy-preserving techniques
3. Tối ưu hóa quá trình model aggregation

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định cấu trúc dữ liệu hiện tại
- KIỂM TRA cách client/server tương tác
- XÁC MINH các phụ thuộc và thư viện cần thiết

### Giai đoạn 4: Tối ưu song song và xử lý

#### 4.1. Cải tiến Large Scale Simulation

```
Tệp: large_scale_simulation.py
```

**Các bước thực hiện**:
1. Tối ưu hóa code cho xử lý song song
2. Triển khai multi-threading và pipeline processing
3. Thêm cơ chế phân tích hiệu suất chi tiết

**Tránh hiện tượng ảo giác**:
- KHÔNG giả định cách simulation hiện đang chạy
- KIỂM TRA cách mô hình được khởi tạo và đánh giá
- XÁC MINH hệ thống ghi nhật ký và đo lường hiệu suất

## LỆNH THỰC THI THAM KHẢO

Khi chạy và kiểm tra các thay đổi, sử dụng các lệnh sau:

```
# Chạy với cấu hình mạng cơ bản
py -3.10 -m main --num-shards 24 --nodes-per-shard 20 --batch-size 256 --episodes 10 --max-steps 500

# Chạy training tối ưu
py -3.10 optimized_training.py --num-shards 24 --nodes-per-shard 20 --batch-size 512 --hidden-size 1024

# Mô phỏng với cấu hình đột phá
py -3.10 -m main --num-shards 32 --nodes-per-shard 24 --batch-size 512 --enable-federated
```

## DANH SÁCH KIỂM TRA TRƯỚC KHI HOÀN THÀNH

Trước khi hoàn thành mỗi nhiệm vụ, hãy đảm bảo rằng:

1. ✅ Đã kiểm tra code có lỗi cú pháp hoặc lỗi logic
2. ✅ Đã kiểm tra hiệu suất trước và sau khi thay đổi
3. ✅ Đã cập nhật Knowledge Graph với trạng thái mới
4. ✅ Đã đạt được các chỉ số hiệu suất của milestone tương ứng
5. ✅ Đã ghi chép lại các thử thách và giải pháp

## THEO DÕI TIẾN ĐỘ

Theo dõi tiến độ thực hiện qua các milestone:

1. **Milestone 1**: Hoàn thành cải tiến cấu trúc mạng
   - Throughput > 1000 tx/s
   - Latency < 150 ms

2. **Milestone 2**: Hoàn thành cải tiến cơ chế đồng thuận
   - Energy < 1000 mJ/tx
   - Security > 0.85

3. **Milestone 3**: Hoàn thành nâng cấp DRL
   - Throughput > 5000 tx/s
   - Latency < 50 ms

4. **Milestone 4**: Hoàn thành tối ưu xử lý
   - Throughput > 8000 tx/s
   - Latency < 20 ms

5. **Milestone 5**: Hoàn thành đánh giá
   - So sánh với các blockchain hiện đại
   - Tài liệu kỹ thuật hoàn chỉnh

## KẾ HOẠCH DỰ PHÒNG

Khi gặp thách thức không lường trước, tham khảo các giải pháp dự phòng:

- **Vấn đề throughput**: Tăng số lượng shard, tối ưu cross-shard communication
- **Vấn đề latency**: Triển khai hierarchical consensus
- **Vấn đề bảo mật**: Tăng cường HTDCM, thêm cơ chế phát hiện tấn công
- **Quá trình tối ưu chậm**: Xây dựng pipeline CI/CD

## KẾT LUẬN

Tuân thủ hướng dẫn này sẽ giúp AI tránh hiện tượng ảo giác và đảm bảo triển khai chính xác kế hoạch cải tiến QTrust. Luôn tham khảo `.cursorrules` và `qtrust_improvement_plan.md` khi cần thêm thông tin chi tiết. 