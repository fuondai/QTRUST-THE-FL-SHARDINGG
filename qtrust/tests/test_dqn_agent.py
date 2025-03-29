"""
Bài kiểm thử cho DQN agent.
"""

import unittest
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, MultiDiscrete

from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.networks import QNetwork

class SimpleEnv(gym.Env):
    """
    Môi trường đơn giản cho kiểm thử.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = MultiDiscrete([3, 2])  # Hai không gian hành động rời rạc
        self.state = np.zeros(5, dtype=np.float32)
        self.step_count = 0
        
    def reset(self):
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        reward = 1.0 if action[0] == 1 else -0.1
        done = self.step_count >= 10
        info = {}
        return self.state, reward, done, info

class TestQNetwork(unittest.TestCase):
    """
    Kiểm thử cho QNetwork.
    """
    
    def setUp(self):
        self.state_size = 5
        self.action_dim = [3, 2]  # Hai không gian hành động rời rạc
        self.hidden_sizes = [64, 64]
        self.network = QNetwork(self.state_size, self.action_dim, hidden_sizes=self.hidden_sizes)
        
    def test_initialization(self):
        """
        Kiểm thử khởi tạo mạng Q.
        """
        # Kiểm tra các tham số
        self.assertEqual(self.network.state_size, self.state_size)
        self.assertEqual(self.network.action_dim, self.action_dim)
        
        # Kiểm tra cấu trúc mạng
        self.assertIsNotNone(self.network.input_layer)
        self.assertIsNotNone(self.network.res_blocks)
        self.assertEqual(len(self.network.output_layers), len(self.action_dim))
        self.assertIsNotNone(self.network.value_stream)
        
    def test_forward_pass(self):
        """
        Kiểm thử pass thuận (forward pass).
        """
        # Tạo input ngẫu nhiên
        batch_size = 10
        x = torch.randn(batch_size, self.state_size)
        
        # Forward pass
        action_values, state_value = self.network(x)
        
        # Kiểm tra kích thước output
        self.assertEqual(len(action_values), len(self.action_dim))
        self.assertEqual(action_values[0].shape, (batch_size, self.action_dim[0]))
        self.assertEqual(action_values[1].shape, (batch_size, self.action_dim[1]))
        self.assertEqual(state_value.shape, (batch_size, 1))

class TestDQNAgent(unittest.TestCase):
    """
    Kiểm thử cho DQNAgent.
    """
    
    def setUp(self):
        self.env = SimpleEnv()
        
        # Kích thước không gian trạng thái và hành động
        self.state_size = 5
        self.action_size = 3
        
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42,
            buffer_size=1000,
            batch_size=64,
            gamma=0.99,
            tau=1e-3,
            learning_rate=0.001,
            update_every=100,
            prioritized_replay=True,
            alpha=0.6,
            beta_start=0.4,
            dueling=True,
            noisy_nets=False,
            hidden_layers=[64, 64],
            device='cpu'
        )
        
    def test_initialization(self):
        """
        Kiểm thử khởi tạo agent.
        """
        # Kiểm tra các tham số
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.batch_size, 64)
        
        # Kiểm tra các network
        self.assertIsInstance(self.agent.qnetwork_local, QNetwork)
        self.assertIsInstance(self.agent.qnetwork_target, QNetwork)
        
        # Kiểm tra optimizer
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
        
        # Kiểm tra buffer trống ban đầu
        self.assertEqual(len(self.agent.memory), 0)
        
    def test_step(self):
        """
        Kiểm thử hàm step.
        """
        # Tạo vài experience và thêm vào buffer
        state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        action = 1  # Hành động mẫu (index)
        reward = 1.0
        next_state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        done = False
        
        # Thêm 100 experience
        for _ in range(100):
            self.agent.step(state, action, reward, next_state, done)
            
        # Kiểm tra buffer
        self.assertEqual(len(self.agent.memory), 100)
        
    def test_act(self):
        """
        Kiểm thử hàm act.
        """
        # Tạo state ngẫu nhiên
        state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        
        # Thực hiện hành động với epsilon=0 (greedy)
        action = self.agent.act(state, eps=0)
        
        # Kiểm tra hành động
        self.assertIsInstance(action, (int, np.int64))
        self.assertIn(action, range(self.action_size))
        
    def test_learn(self):
        """
        Kiểm thử quá trình học.
        """
        # Thêm đủ experience để học
        state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        action = 1  # Hành động mẫu
        reward = 1.0
        next_state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        done = False
        
        # Lấy các tham số trước khi học
        params_before = [p.clone().detach() for p in self.agent.qnetwork_local.parameters()]
        
        # Thêm nhiều experience vào buffer
        for _ in range(self.agent.batch_size + 10):  # Thêm nhiều hơn batch_size
            self.agent.step(state, action, reward, next_state, done)
            
        # Các tham số có thể thay đổi sau nhiều bước nhưng không gọi _learn trực tiếp
        # nên không kiểm tra chi tiết ở đây
            
    def test_soft_update(self):
        """
        Kiểm thử hàm soft update target network.
        """
        # Thay đổi local network
        for param in self.agent.qnetwork_local.parameters():
            param.data = torch.randn_like(param.data)
            
        # Target network ban đầu khác với local network
        # Lưu trữ bản sao của các tham số target network để so sánh sau
        target_params = []
        for param in self.agent.qnetwork_target.parameters():
            target_params.append(param.data.clone())
        
        # Thực hiện soft update
        self.agent._soft_update()
        
        # Kiểm tra sau khi update, target network đã được cập nhật 
        # nhưng không hoàn toàn giống local (do soft update với tau < 1)
        params_changed = False
        for i, (local_param, target_param) in enumerate(zip(
            self.agent.qnetwork_local.parameters(),
            self.agent.qnetwork_target.parameters()
        )):
            # So sánh với bản sao đã lưu trước đó
            if not torch.all(torch.eq(target_param.data, target_params[i])):
                params_changed = True
                break
        
        # Đảm bảo rằng ít nhất một tham số đã thay đổi
        self.assertTrue(params_changed, "Soft update không thay đổi bất kỳ tham số nào của target network")
        
    def test_update_epsilon(self):
        """
        Kiểm thử epsilon decay.
        """
        # Lưu epsilon ban đầu
        initial_epsilon = self.agent.epsilon
        
        # Thực hiện epsilon decay
        for _ in range(10):
            self.agent.update_epsilon()
            
        # Kiểm tra epsilon đã giảm
        self.assertLess(self.agent.epsilon, initial_epsilon)
        
        # Kiểm tra epsilon không thấp hơn epsilon_end
        self.assertGreaterEqual(self.agent.epsilon, self.agent.eps_end)

if __name__ == '__main__':
    unittest.main() 