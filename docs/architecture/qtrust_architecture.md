# Kiến trúc QTrust - Blockchain Sharding tối ưu hóa bằng DRL và FL

## Tổng quan kiến trúc

QTrust được thiết kế theo kiến trúc module với sự tương tác chặt chẽ giữa các thành phần. Dưới đây là mô hình kiến trúc tổng quan:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                           QTrust Framework                                │
│                                                                           │
└───────────────┬───────────────────────────────────┬───────────────────────┘
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐       ┌─────────────────────────────────┐
│                               │       │                                 │
│   BlockchainEnvironment       │◄──────┤        DQN Agents              │
│   - Sharding Simulation       │       │   - Rainbow DQN                │
│   - Dynamic Resharding        │       │   - Actor-Critic               │
│   - Cross-Shard Transactions  │       │   - Policy Networks            │
│   - Performance Monitoring    │       │   - Prioritized Experience     │
│                               │       │                                 │
└──────────────┬────────────────┘       └──────────────┬──────────────────┘
               │                                       │
               │                                       │
               ▼                                       │
┌──────────────────────────────┐                       │
│                              │                       │
│    AdaptiveConsensus         │◄──────────────────────┘
│    - Fast BFT                │                       
│    - PBFT                    │                       ┌────────────────────────────┐
│    - Robust BFT              │◄─────────────────────►│                            │
│    - Protocol Selection      │                       │     FederatedLearning      │
└──────────────┬───────────────┘                       │     - FedAvg               │
               │                                       │     - FedTrust             │
               │                                       │     - Secure Aggregation   │
               ▼                                       │                            │
┌──────────────────────────────┐                       └──────────────┬─────────────┘
│                              │                                      │
│     MADRAPIDRouter           │◄─────────────────────────────────────┘
│     - Transaction Routing    │                                      
│     - Load Balancing         │                       ┌────────────────────────────┐
│     - Congestion Avoidance   │◄─────────────────────►│                            │
│     - Predictive Routing     │                       │        HTDCM               │
└──────────────────────────────┘                       │     - Trust Evaluation     │
                                                       │     - Anomaly Detection    │
                                                       │     - Security Monitoring  │
                                                       │                            │
                                                       └────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                           Caching System                                  │
│       ┌─────────────┐     ┌─────────────┐      ┌──────────────┐          │
│       │  LRU Cache  │     │  TTL Cache  │      │ Tensor Cache │          │
│       └─────────────┘     └─────────────┘      └──────────────┘          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Mô tả chi tiết các thành phần

### 1. BlockchainEnvironment

Môi trường mô phỏng blockchain với khả năng tái tạo nhiều tình huống mạng khác nhau. Thành phần này triển khai:

- **Sharding Framework**: Mô hình hóa hệ thống với nhiều shard (24-32)
- **Network Simulation**: Mô phỏng độ trễ, băng thông và kết nối P2P
- **Transaction Generator**: Tạo và xử lý giao dịch xuyên shard
- **Dynamic Resharding**: Cơ chế resharding động dựa trên tải mạng
- **Performance Metrics**: Thu thập và báo cáo các chỉ số hiệu suất

### 2. DQN Agents

Các agent học tăng cường sâu (DRL) triển khai các kỹ thuật tiên tiến để tối ưu hóa quyết định:

- **Rainbow DQN**: Kết hợp 6 cải tiến DQN (Double DQN, Dueling, PER, Multi-step, Distributional RL, Noisy Nets)
- **Actor-Critic**: Kiến trúc học đồng thời chính sách và giá trị
- **Multi-objective Optimization**: Tối ưu hóa đa mục tiêu (throughput, latency, energy, security)
- **Experience Replay**: Lưu trữ và tái sử dụng trải nghiệm cho hiệu quả học tập

### 3. AdaptiveConsensus

Module lựa chọn động giao thức đồng thuận dựa trên trạng thái mạng:

- **Fast BFT**: Giao thức nhanh cho mạng ổn định và giao dịch giá trị thấp
- **PBFT**: Cân bằng giữa hiệu suất và bảo mật
- **Robust BFT**: Tối ưu cho bảo mật cao khi mạng không ổn định
- **Consensus Selection Algorithm**: Thuật toán quyết định giao thức tối ưu
- **BLS Signature Aggregation**: Giảm kích thước chữ ký và chi phí truyền thông

### 4. MADRAPIDRouter

Router thông minh để điều hướng giao dịch giữa các shard:

- **Proximity-aware Routing**: Định tuyến dựa trên khoảng cách mạng
- **Dynamic Mesh Connections**: Kết nối linh hoạt giữa các shard
- **Predictive Routing**: Dự đoán tắc nghẽn và điều chỉnh trước
- **Cross-shard Optimization**: Tối ưu hóa giao dịch xuyên shard
- **Load Balancing**: Cân bằng tải giữa các shard

### 5. HTDCM (Hierarchical Trust-based Data Center Mechanism)

Cơ chế tin cậy phân cấp để đánh giá độ tin cậy của các node:

- **Multi-level Trust Evaluation**: Đánh giá tin cậy ở cấp node, shard và mạng
- **ML-based Anomaly Detection**: Phát hiện hành vi bất thường bằng học máy
- **Attack Classification**: Phân loại nhiều dạng tấn công
- **Trust Scoring**: Chấm điểm tin cậy cho các node và shard
- **Reputation Management**: Quản lý danh tiếng dài hạn

### 6. FederatedLearning

Hệ thống huấn luyện phân tán với bảo vệ quyền riêng tư:

- **Federated DRL**: Học tăng cường liên hợp giữa các node
- **FedTrust**: Tổng hợp mô hình dựa trên điểm tin cậy
- **Secure Aggregation**: Tổng hợp an toàn không tiết lộ dữ liệu cục bộ
- **Differential Privacy**: Bảo vệ quyền riêng tư trong quá trình học
- **Model Personalization**: Điều chỉnh mô hình phù hợp với từng node

### 7. Caching System

Hệ thống caching thông minh để tối ưu hiệu suất:

- **LRU Cache**: Lưu trữ các giá trị được sử dụng gần đây
- **TTL Cache**: Caching với thời gian hết hạn
- **Tensor Cache**: Đặc biệt tối ưu cho các tensor PyTorch
- **Cache Statistics**: Theo dõi tỷ lệ cache hit/miss
- **Intelligent Eviction**: Chiến lược loại bỏ thông minh

## Luồng dữ liệu và tương tác

1. **BlockchainEnvironment** cung cấp trạng thái mạng hiện tại cho DQN Agents
2. **DQN Agents** đưa ra quyết định về routing và giao thức đồng thuận
3. **AdaptiveConsensus** lựa chọn giao thức đồng thuận dựa trên quyết định của agent
4. **MADRAPIDRouter** định tuyến giao dịch dựa trên quyết định của agent
5. **HTDCM** cung cấp thông tin tin cậy cho router và consensus
6. **FederatedLearning** tổng hợp kinh nghiệm học tập từ nhiều node
7. **Caching System** hỗ trợ tất cả các module với caching thông minh

## Ưu điểm kiến trúc

1. **Mô-đun hóa cao**: Dễ dàng thay thế hoặc cải tiến từng thành phần
2. **Khả năng mở rộng**: Hỗ trợ từ vài chục đến hàng nghìn node
3. **Thích ứng linh hoạt**: Tự điều chỉnh dựa trên điều kiện mạng
4. **Phi tập trung**: Không có điểm thất bại đơn lẻ
5. **Bảo mật cao**: Nhiều lớp phát hiện và ngăn chặn tấn công
6. **Tiết kiệm năng lượng**: Tối ưu hóa tiêu thụ năng lượng
7. **Học liên tục**: Liên tục cải thiện thông qua học tăng cường và federated learning 