# Module DQN (Deep Q-Network)

Mô-đun này chứa cài đặt của thuật toán DQN (Deep Q-Network) với nhiều cải tiến hiện đại:

- Double DQN
- Dueling Network Architecture
- Prioritized Experience Replay
- Noisy Networks

## Cấu trúc thư mục

```
qtrust/agents/dqn/
├── __init__.py        # Xuất lớp DQNAgent
├── agent.py           # Chứa lớp DQNAgent chính
├── networks.py        # Các kiến trúc mạng neural (QNetwork, NoisyLinear, ResidualBlock)
├── replay_buffer.py   # Các lớp replay buffer (PrioritizedReplayBuffer, ReplayBuffer, EfficientReplayBuffer)
├── utils.py           # Các hàm tiện ích
├── train.py           # Các hàm huấn luyện và đánh giá
└── README.md          # Tài liệu này
```

## Cách sử dụng

### Sử dụng cơ bản

```python
from qtrust.agents import DQNAgent
import gym

# Tạo môi trường
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Tạo DQN Agent
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    hidden_layers=[128, 64],
    prioritized_replay=True,   # Sử dụng PER
    noisy_nets=False           # Không sử dụng Noisy Networks
)

# Huấn luyện agent
from qtrust.agents.dqn.train import train_dqn

results = train_dqn(
    agent=agent,
    env=env,
    n_episodes=1000,
    max_t=1000,
    early_stopping=True,
    patience=50
)

# Đánh giá agent
from qtrust.agents.dqn.train import evaluate_dqn

avg_reward = evaluate_dqn(
    agent=agent,
    env=env,
    n_episodes=10,
    render=True
)

print(f"Average reward: {avg_reward}")

# Vẽ biểu đồ phần thưởng
from qtrust.agents.dqn.train import plot_dqn_rewards

plot_dqn_rewards(
    rewards=results['rewards'],
    val_rewards=results['validation_rewards'],
    save_path='dqn_rewards.png'
)
```

### So sánh các biến thể DQN

```python
from qtrust.agents.dqn.train import compare_dqn_variants
import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Định nghĩa các biến thể
variants = [
    {
        'name': 'Vanilla DQN',
        'state_size': state_size,
        'action_size': action_size,
        'prioritized_replay': False,
        'noisy_nets': False,
        'dueling': False
    },
    {
        'name': 'Double DQN + PER',
        'state_size': state_size,
        'action_size': action_size,
        'prioritized_replay': True,
        'noisy_nets': False,
        'dueling': False
    },
    {
        'name': 'Dueling DQN',
        'state_size': state_size,
        'action_size': action_size,
        'prioritized_replay': False,
        'noisy_nets': False,
        'dueling': True
    },
    {
        'name': 'Noisy DQN',
        'state_size': state_size,
        'action_size': action_size,
        'prioritized_replay': False,
        'noisy_nets': True,
        'dueling': False
    },
    {
        'name': 'Rainbow DQN',
        'state_size': state_size,
        'action_size': action_size,
        'prioritized_replay': True,
        'noisy_nets': True,
        'dueling': True
    }
]

# So sánh các biến thể
results = compare_dqn_variants(env, variants, n_episodes=500)
```

## Tham số DQNAgent

DQNAgent nhận nhiều tham số cho phép tùy chỉnh hành vi:

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `state_size` | Kích thước không gian trạng thái | (bắt buộc) |
| `action_size` | Kích thước không gian hành động | (bắt buộc) |
| `seed` | Seed cho các giá trị ngẫu nhiên | 42 |
| `buffer_size` | Kích thước tối đa của replay buffer | 100000 |
| `batch_size` | Kích thước batch khi lấy mẫu từ replay buffer | 64 |
| `gamma` | Hệ số chiết khấu | 0.99 |
| `tau` | Tỷ lệ cập nhật tham số cho mạng target | 1e-3 |
| `learning_rate` | Tốc độ học | 5e-4 |
| `update_every` | Tần suất cập nhật mạng (số bước) | 4 |
| `prioritized_replay` | Sử dụng Prioritized Experience Replay | True |
| `alpha` | Hệ số alpha cho PER | 0.6 |
| `beta_start` | Giá trị beta ban đầu cho PER | 0.4 |
| `dueling` | Sử dụng kiến trúc mạng Dueling | True |
| `noisy_nets` | Sử dụng Noisy Networks | False |
| `hidden_layers` | Kích thước các lớp ẩn | [128, 64] |
| `device` | Thiết bị sử dụng ('cpu', 'cuda', 'auto') | 'auto' |
| `min_epsilon` | Giá trị epsilon tối thiểu | 0.01 |
| `epsilon_decay` | Tốc độ giảm epsilon | 0.995 |
| `use_efficient_buffer` | Sử dụng EfficientReplayBuffer | False |
| `clip_gradients` | Giới hạn gradient | True |
| `grad_clip_value` | Giá trị giới hạn gradient | 1.0 |
| `warm_up_steps` | Số bước warm-up trước khi bắt đầu học | 1000 |
| `save_dir` | Thư mục lưu model | 'models' | 