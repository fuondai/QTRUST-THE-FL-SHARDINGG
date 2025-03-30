"""
Bộ nhớ kinh nghiệm sử dụng trong DQN Agent.

File này chứa các lớp:
- ReplayBuffer: Bộ nhớ kinh nghiệm tiêu chuẩn cho DQN
- PrioritizedReplayBuffer: Bộ nhớ kinh nghiệm ưu tiên cho DQN
- EfficientReplayBuffer: Bộ nhớ tối ưu hiệu suất
- NStepReplayBuffer: Bộ nhớ hỗ trợ N-step returns cho Rainbow DQN
- NStepPrioritizedReplayBuffer: Kết hợp N-step và PER cho Rainbow DQN
"""

import numpy as np
import torch
import random
from collections import deque, namedtuple
from typing import Tuple, Optional, List, Union, Dict, Any

# Định nghĩa namedtuple Experience để lưu trữ kinh nghiệm trong replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Bộ nhớ kinh nghiệm tiêu chuẩn.
    """
    def __init__(self, buffer_size: int, batch_size: int, device: str = 'cpu'):
        """
        Khởi tạo ReplayBuffer.
        
        Args:
            buffer_size: Kích thước tối đa của buffer
            batch_size: Kích thước batch khi lấy mẫu
            device: Thiết bị để chuyển tensor (cpu/cuda)
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        """
        Lưu trữ một kinh nghiệm trong bộ nhớ.
        
        Args:
            state: Trạng thái
            action: Hành động
            reward: Phần thưởng
            next_state: Trạng thái kế tiếp
            done: Trạng thái kết thúc (boolean)
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self):
        """
        Lấy mẫu ngẫu nhiên từ bộ nhớ.
        
        Returns:
            Tuple gồm (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences], dtype=np.uint8)).to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Trả về số lượng kinh nghiệm trong bộ nhớ."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """
    Bộ nhớ kinh nghiệm ưu tiên (Prioritized Experience Replay)
    Ưu tiên lấy mẫu những kinh nghiệm có TD error cao
    """
    def __init__(self, buffer_size: int, batch_size: int, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_end: float = 1.0, 
                 beta_frames: int = 100000, device: str = 'cpu'):
        """
        Khởi tạo Prioritized Replay Buffer
        
        Args:
            buffer_size: Kích thước tối đa của buffer
            batch_size: Kích thước batch khi sampling
            alpha: Hệ số quyết định mức độ ưu tiên (0 = sampling đều, 1 = sampling hoàn toàn theo ưu tiên)
            beta_start: Giá trị beta ban đầu cho importance sampling weight
            beta_end: Giá trị beta cuối cùng
            beta_frames: Số frame để beta tăng từ start đến end
            device: Thiết bị để chuyển tensor (cpu/cuda)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.beta_increment = (beta_end - beta_start) / beta_frames
        self.frame = 0
        self.device = device
        
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        
    def update_beta(self, batch_size: Optional[int] = None):
        """
        Cập nhật beta theo số frame hiện tại
        
        Args:
            batch_size: Kích thước batch mới nếu cần thay đổi
        """
        self.frame += 1
        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        
        if batch_size is not None:
            self.batch_size = batch_size
            
    def push(self, state, action, reward, next_state, done, error=None):
        """
        Thêm một kinh nghiệm vào buffer với mức độ ưu tiên
        
        Args:
            state: Trạng thái
            action: Hành động
            reward: Phần thưởng
            next_state: Trạng thái tiếp theo
            done: Cờ kết thúc
            error: TD error hoặc mức ưu tiên, nếu None thì sử dụng max priority
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience
            
        if error is None:
            # Nếu không có error thì dùng max priority hoặc 1 nếu buffer trống
            max_priority = self.priorities.max() if self.memory else 1.0
            self.priorities[self.pos] = max_priority
        else:
            # Thêm một lượng nhỏ để đảm bảo mọi kinh nghiệm đều có cơ hội được lấy mẫu
            self.priorities[self.pos] = (abs(error) + 1e-5) ** self.alpha
            
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self):
        """
        Lấy mẫu một batch kinh nghiệm theo độ ưu tiên
        
        Returns:
            Tuple chứa batch states, actions, rewards, next_states, dones, indices và weights
        """
        N = len(self.memory)
        if N == 0:
            return None
        
        if N < self.batch_size:
            batch_size = N
        else:
            batch_size = self.batch_size
            
        # Tính toán xác suất lấy mẫu dựa trên ưu tiên
        if N < self.buffer_size:
            priorities = self.priorities[:N]
        else:
            priorities = self.priorities
            
        probs = priorities[:N] / priorities[:N].sum()
        
        # Lấy mẫu theo xác suất
        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        
        # Tính toán importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = [self.memory[idx] for idx in indices]
        
        # Tách batch thành các thành phần
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Chuyển thành tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """
        Cập nhật mức độ ưu tiên cho các kinh nghiệm
        
        Args:
            indices: Các chỉ số cần cập nhật
            errors: TD errors mới
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    
    def __len__(self):
        """Trả về số lượng kinh nghiệm trong bộ nhớ."""
        return len(self.memory)


class EfficientReplayBuffer:
    """
    Bộ nhớ kinh nghiệm hiệu quả về bộ nhớ, sử dụng numpy array thay vì deque.
    Thực hiện tối ưu hóa về mặt hiệu năng và bộ nhớ so với ReplayBuffer thông thường.
    """
    def __init__(self, buffer_size: int, batch_size: int, 
                 state_shape: tuple, action_shape: tuple,
                 device: str = 'cpu'):
        """
        Khởi tạo EfficientReplayBuffer.
        
        Args:
            buffer_size: Kích thước tối đa của buffer
            batch_size: Kích thước batch khi lấy mẫu
            state_shape: Hình dạng của state (ví dụ: (84, 84, 4) cho Atari)
            action_shape: Hình dạng của action
            device: Thiết bị để chuyển tensor (cpu/cuda)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.count = 0
        self.current = 0
        
        # Khởi tạo buffer bằng numpy array
        self.states = np.zeros((buffer_size,) + state_shape, dtype=np.float32)
        if isinstance(action_shape, int):
            self.actions = np.zeros((buffer_size,), dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size,) + state_shape, dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.uint8)
    
    def push(self, state, action, reward, next_state, done):
        """
        Lưu trữ một kinh nghiệm trong bộ nhớ.
        
        Args:
            state: Trạng thái
            action: Hành động
            reward: Phần thưởng
            next_state: Trạng thái kế tiếp
            done: Trạng thái kết thúc (boolean)
        """
        # Lưu trữ transition
        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_states[self.current] = next_state
        self.dones[self.current] = done
        
        self.current = (self.current + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)
    
    def sample(self):
        """
        Lấy mẫu ngẫu nhiên từ bộ nhớ.
        
        Returns:
            Tuple gồm (states, actions, rewards, next_states, dones)
        """
        # Lấy mẫu ngẫu nhiên từ buffer
        if self.count < self.batch_size:
            indices = np.random.choice(self.count, self.count, replace=False)
        else:
            indices = np.random.choice(self.count, self.batch_size, replace=False)
        
        # Lấy dữ liệu từ buffer và chuyển thành tensor
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Trả về số lượng kinh nghiệm trong bộ nhớ."""
        return self.count 

class NStepReplayBuffer:
    """
    Bộ nhớ kinh nghiệm hỗ trợ N-step returns.
    Lưu trữ chuỗi n bước liên tiếp để tính toán giá trị return n-step.
    """
    def __init__(self, buffer_size: int, batch_size: int, n_step: int = 3, 
                 gamma: float = 0.99, device: str = 'cpu'):
        """
        Khởi tạo NStepReplayBuffer.
        
        Args:
            buffer_size: Kích thước tối đa của buffer
            batch_size: Kích thước batch khi lấy mẫu
            n_step: Số bước cho n-step returns
            gamma: Hệ số chiết khấu
            device: Thiết bị để chuyển tensor (cpu/cuda)
        """
        self.memory = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
    
    def _get_n_step_info(self):
        """
        Tính toán phần thưởng n-step và trạng thái kết thúc sau n bước.
        
        Returns:
            Tuple: (reward, next_state, done)
        """
        reward, next_state, done = self.n_step_buffer[-1][2:]
        
        # Nếu đã kết thúc, không cần tính toán thêm
        if done:
            return reward, next_state, done
            
        # Tính n-step return: reward + gamma * reward' + gamma^2 * reward'' + ...
        for idx in range(len(self.n_step_buffer) - 1, 0, -1):
            r, s, d = self.n_step_buffer[idx-1][2:]
            reward = r + self.gamma * (1 - d) * reward
            next_state, done = (s, d) if d else (next_state, done)
            
        return reward, next_state, done
        
    def push(self, state, action, reward, next_state, done):
        """
        Lưu trữ một kinh nghiệm trong bộ nhớ n-step.
        
        Args:
            state: Trạng thái
            action: Hành động
            reward: Phần thưởng
            next_state: Trạng thái kế tiếp
            done: Trạng thái kết thúc (boolean)
        """
        experience = (state, action, reward, next_state, done)
        self.n_step_buffer.append(experience)
        
        # Nếu n_step_buffer chưa đủ n phần tử, chưa thể tính n-step return
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Tính toán phần thưởng n-step và trạng thái sau n bước
        reward, next_state, done = self._get_n_step_info()
        
        # Lấy trạng thái và hành động ban đầu
        state, action = self.n_step_buffer[0][:2]
        
        # Lưu trữ kinh nghiệm n-step
        self.memory.append((state, action, reward, next_state, done))
        
        # Nếu kết thúc, xóa buffer n-step
        if done:
            self.n_step_buffer.clear()
    
    def sample(self):
        """
        Lấy mẫu ngẫu nhiên từ bộ nhớ.
        
        Returns:
            Tuple gồm (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in experiences], dtype=np.uint8)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Trả về số lượng kinh nghiệm trong bộ nhớ."""
        return len(self.memory)


class NStepPrioritizedReplayBuffer:
    """
    Kết hợp N-step returns và Prioritized Experience Replay.
    Lưu trữ chuỗi n bước liên tiếp và ưu tiên lấy mẫu những kinh nghiệm có TD error cao.
    """
    def __init__(self, buffer_size: int, batch_size: int, n_step: int = 3,
                 alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 100000, gamma: float = 0.99, device: str = 'cpu'):
        """
        Khởi tạo NStepPrioritizedReplayBuffer.
        
        Args:
            buffer_size: Kích thước tối đa của buffer
            batch_size: Kích thước batch khi lấy mẫu
            n_step: Số bước cho n-step returns
            alpha: Hệ số quyết định mức độ ưu tiên
            beta_start: Giá trị beta ban đầu cho importance sampling weight
            beta_end: Giá trị beta cuối cùng
            beta_frames: Số frame để beta tăng từ start đến end
            gamma: Hệ số chiết khấu
            device: Thiết bị để chuyển tensor (cpu/cuda)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.beta_increment = (beta_end - beta_start) / beta_frames
        self.frame = 0
        self.gamma = gamma
        self.device = device
        
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        
    def update_beta(self, batch_size: Optional[int] = None):
        """
        Cập nhật beta theo số frame hiện tại
        
        Args:
            batch_size: Kích thước batch mới nếu cần thay đổi
        """
        self.frame += 1
        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        
        if batch_size is not None:
            self.batch_size = batch_size
    
    def _get_n_step_info(self):
        """
        Tính toán phần thưởng n-step và trạng thái kết thúc sau n bước.
        
        Returns:
            Tuple: (reward, next_state, done)
        """
        reward, next_state, done = self.n_step_buffer[-1][2:]
        
        # Nếu đã kết thúc, không cần tính toán thêm
        if done:
            return reward, next_state, done
            
        # Tính n-step return: reward + gamma * reward' + gamma^2 * reward'' + ...
        for idx in range(len(self.n_step_buffer) - 1, 0, -1):
            r, s, d = self.n_step_buffer[idx-1][2:]
            reward = r + self.gamma * (1 - d) * reward
            next_state, done = (s, d) if d else (next_state, done)
            
        return reward, next_state, done
            
    def push(self, state, action, reward, next_state, done, error=None):
        """
        Thêm một kinh nghiệm vào buffer với mức độ ưu tiên
        
        Args:
            state: Trạng thái
            action: Hành động
            reward: Phần thưởng
            next_state: Trạng thái tiếp theo
            done: Cờ kết thúc
            error: TD error hoặc mức ưu tiên, nếu None thì sử dụng max priority
        """
        # Lưu trữ kinh nghiệm 1-step vào n_step_buffer
        experience = (state, action, reward, next_state, done)
        self.n_step_buffer.append(experience)
        
        # Nếu n_step_buffer chưa đủ n phần tử, chưa thể tính n-step return
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Tính toán phần thưởng n-step và trạng thái sau n bước
        n_reward, n_next_state, n_done = self._get_n_step_info()
        
        # Lấy trạng thái và hành động ban đầu
        n_state, n_action = self.n_step_buffer[0][:2]
        
        # Lưu trữ kinh nghiệm n-step vào buffer chính
        n_experience = (n_state, n_action, n_reward, n_next_state, n_done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(n_experience)
        else:
            self.memory[self.pos] = n_experience
            
        if error is None:
            # Nếu không có error thì dùng max priority hoặc 1 nếu buffer trống
            max_priority = self.priorities.max() if self.memory else 1.0
            self.priorities[self.pos] = max_priority
        else:
            # Thêm một lượng nhỏ để đảm bảo mọi kinh nghiệm đều có cơ hội được lấy mẫu
            self.priorities[self.pos] = (abs(error) + 1e-5) ** self.alpha
            
        self.pos = (self.pos + 1) % self.buffer_size
        
        # Nếu kết thúc, xóa buffer n-step
        if done:
            self.n_step_buffer.clear()
    
    def sample(self):
        """
        Lấy mẫu một batch kinh nghiệm theo độ ưu tiên
        
        Returns:
            Tuple chứa batch states, actions, rewards, next_states, dones, indices và weights
        """
        N = len(self.memory)
        if N == 0:
            return None
        
        if N < self.batch_size:
            batch_size = N
        else:
            batch_size = self.batch_size
            
        # Tính toán xác suất lấy mẫu dựa trên ưu tiên
        if N < self.buffer_size:
            priorities = self.priorities[:N]
        else:
            priorities = self.priorities
            
        probs = priorities[:N] / priorities[:N].sum()
        
        # Lấy mẫu theo xác suất
        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        
        # Tính toán importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = [self.memory[idx] for idx in indices]
        
        # Tách batch thành các thành phần
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Chuyển thành tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """
        Cập nhật mức độ ưu tiên cho các kinh nghiệm
        
        Args:
            indices: Các chỉ số cần cập nhật
            errors: TD errors mới
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    
    def __len__(self):
        """Trả về số lượng kinh nghiệm trong bộ nhớ."""
        return len(self.memory) 