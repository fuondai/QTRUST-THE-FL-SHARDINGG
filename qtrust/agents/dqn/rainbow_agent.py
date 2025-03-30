"""
Rainbow DQN Agent - cài đặt trọn vẹn của Rainbow DQN.

Module này chứa cài đặt của Rainbow DQN Agent với đầy đủ các cải tiến:
- Double DQN
- Dueling Network Architecture
- Prioritized Experience Replay
- Noisy Networks for Exploration
- Distributional RL (Categorical DQN)
- Multi-step learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from qtrust.agents.dqn.networks import CategoricalQNetwork
from qtrust.agents.dqn.replay_buffer import NStepPrioritizedReplayBuffer
from qtrust.agents.dqn.utils import (
    soft_update, hard_update, calculate_td_error, calculate_huber_loss,
    exponential_decay, linear_decay, generate_timestamp, create_save_directory,
    plot_learning_curve, format_time, get_device, logger, SAVE_DIR
)

class RainbowDQNAgent:
    """
    Rainbow DQN Agent - triển khai đầy đủ các cải tiến của Rainbow DQN.
    """
    def __init__(self, 
                state_size: int, 
                action_size: int, 
                seed: int = 42,
                buffer_size: int = 100000,
                batch_size: int = 64,
                gamma: float = 0.99,
                tau: float = 1e-3,
                learning_rate: float = 5e-4,
                update_every: int = 4,
                n_step: int = 3,
                alpha: float = 0.6,
                beta_start: float = 0.4,
                n_atoms: int = 51,
                v_min: float = -10.0,
                v_max: float = 10.0,
                hidden_layers: List[int] = [512, 256],
                device: str = 'auto',
                min_epsilon: float = 0.01,
                epsilon_decay: float = 0.995,
                clip_gradients: bool = True,
                grad_clip_value: float = 1.0,
                warm_up_steps: int = 1000,
                save_dir: str = SAVE_DIR):
        """
        Khởi tạo Rainbow DQN Agent.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_size: Kích thước không gian hành động
            seed: Seed cho các giá trị ngẫu nhiên
            buffer_size: Kích thước tối đa của replay buffer
            batch_size: Kích thước batch khi lấy mẫu từ replay buffer
            gamma: Hệ số chiết khấu
            tau: Tỷ lệ cập nhật tham số cho mạng target
            learning_rate: Tốc độ học
            update_every: Tần suất cập nhật mạng (số bước)
            n_step: Số bước cho n-step returns
            alpha: Hệ số alpha cho PER
            beta_start: Giá trị beta ban đầu cho PER
            n_atoms: Số atom trong phân phối categorical DQN
            v_min: Giá trị Q tối thiểu cho categorical DQN
            v_max: Giá trị Q tối đa cho categorical DQN
            hidden_layers: Kích thước các lớp ẩn
            device: Thiết bị sử dụng ('cpu', 'cuda', 'auto')
            min_epsilon: Giá trị epsilon tối thiểu
            epsilon_decay: Tốc độ giảm epsilon
            clip_gradients: Có giới hạn gradient trong quá trình học hay không
            grad_clip_value: Giá trị giới hạn gradient
            warm_up_steps: Số bước warm-up trước khi bắt đầu học
            save_dir: Thư mục lưu model
        """
        self.device = get_device(device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.n_step = n_step
        self.clip_gradients = clip_gradients
        self.grad_clip_value = grad_clip_value
        self.warm_up_steps = warm_up_steps
        self.save_dir = create_save_directory(save_dir)
        
        # Tham số cho Categorical DQN
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Khởi tạo mạng Q - Luôn sử dụng Noisy Networks và Dueling trong Rainbow
        self.qnetwork_local = CategoricalQNetwork(
            state_size, [action_size], n_atoms=n_atoms, v_min=v_min, v_max=v_max,
            hidden_sizes=hidden_layers, noisy=True
        ).to(self.device)
        
        self.qnetwork_target = CategoricalQNetwork(
            state_size, [action_size], n_atoms=n_atoms, v_min=v_min, v_max=v_max,
            hidden_sizes=hidden_layers, noisy=True
        ).to(self.device)
        
        # Sao chép tham số từ local sang target
        hard_update(self.qnetwork_target, self.qnetwork_local)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Thêm learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Khởi tạo replay memory - Luôn sử dụng N-step và PER trong Rainbow
        self.memory = NStepPrioritizedReplayBuffer(
            buffer_size, batch_size, n_step=n_step, alpha=alpha, beta_start=beta_start,
            gamma=gamma, device=self.device
        )
            
        # Biến đếm số bước học
        self.t_step = 0
        self.total_steps = 0
        
        # Khởi tạo epsilon cho exploration
        self.eps_start = 1.0
        self.eps_end = min_epsilon
        self.eps_decay = epsilon_decay
        self.epsilon = self.eps_start
        
        # Thêm biến để theo dõi performance
        self.training_rewards = []
        self.validation_rewards = []
        self.loss_history = []
        self.best_score = -float('inf')
        self.best_model_path = None
        self.train_start_time = None

    def step(self, state, action, reward, next_state, done):
        """
        Thực hiện một bước học: lưu kinh nghiệm và học nếu đến lúc
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái mới
            done: Cờ kết thúc
        """
        self.total_steps += 1
        
        # Lưu kinh nghiệm vào replay memory với tính toán TD error
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Tính current Q distribution
            action_distributions, _ = self.qnetwork_local(state_tensor)
            action_dist = action_distributions[0][:, action, :]  # [batch_size, n_atoms]
            
            # Tính expected Q distribution với double DQN
            next_action_distributions, _ = self.qnetwork_local(next_state_tensor)
            next_actions = next_action_distributions[0].sum(dim=2).argmax(dim=1)  # Lấy hành động từ local network
            
            next_dist_target, _ = self.qnetwork_target(next_state_tensor)
            next_dist = next_dist_target[0].gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms))
            next_dist = next_dist.squeeze(1)  # [batch_size, n_atoms]
            
            # Tính TD target
            if done:
                # Khi kết thúc, phân phối target là delta function tại reward
                target_idx = torch.clamp(torch.floor((reward - self.v_min) / self.delta_z), 0, self.n_atoms - 1).long()
                target_dist = torch.zeros(self.n_atoms).to(self.device)
                target_dist[target_idx] = 1.0
            else:
                # Khi chưa kết thúc, phân phối target là projection của phân phối Bellman
                # Vì đây chỉ là TD error estimate cho priority, nên dùng cách đơn giản hơn
                target_dist = next_dist
            
            # Tính KL divergence làm TD error
            td_error = F.kl_div(action_dist.log(), target_dist, reduction='sum').item()
        
        # Thêm kinh nghiệm vào buffer với ưu tiên dựa trên TD error
        self.memory.push(state, action, reward, next_state, done, td_error)
        
        # Học mỗi update_every bước
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size and self.total_steps > self.warm_up_steps:
            experiences = self.memory.sample()
            self._learn(experiences)
    
    def act(self, state, eps=None):
        """
        Lựa chọn hành động dựa trên policy hiện tại
        
        Args:
            state: Trạng thái hiện tại
            eps: Epsilon cho epsilon-greedy, mặc định là None (sử dụng epsilon nội bộ)
            
        Returns:
            int: Hành động được chọn
        """
        # Chuyển state thành tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Tắt chế độ học cho mạng
        self.qnetwork_local.eval()
        with torch.no_grad():
            # Lấy phân phối Q
            action_distributions, _ = self.qnetwork_local(state)
            action_dist = action_distributions[0]  # [batch_size, action_size, n_atoms]
            
            # Tính expected value
            expected_q = torch.sum(action_dist * self.support.unsqueeze(0).unsqueeze(0), dim=2)
            
        # Bật lại chế độ học cho mạng
        self.qnetwork_local.train()
        
        # Reset noise cho Noisy Networks
        self.qnetwork_local.reset_noise()
        
        # Chọn hành động dựa trên expected Q values
        return np.argmax(expected_q.cpu().data.numpy())
    
    def _learn(self, experiences):
        """
        Cập nhật giá trị Q dựa trên batch kinh nghiệm.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences
        
        # Tính toán phân phối Q cho trạng thái hiện tại
        action_distributions, _ = self.qnetwork_local(states)
        action_dist = action_distributions[0]  # [batch_size, action_size, n_atoms]
        
        # Lấy phân phối cho các hành động đã thực hiện
        action_indices = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
        current_dist = action_dist.gather(1, action_indices).squeeze(1)  # [batch_size, n_atoms]
        
        # Tính toán phân phối Q target - Double DQN
        next_action_distributions, _ = self.qnetwork_local(next_states)
        next_actions = next_action_distributions[0].sum(dim=2).argmax(dim=1)  # Lấy hành động từ local network
        
        next_dist_target, _ = self.qnetwork_target(next_states)
        next_dist = next_dist_target[0].gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms))
        next_dist = next_dist.squeeze(1)  # [batch_size, n_atoms]
        
        # Tính phân phối target - Distributional RL
        target_dist = torch.zeros_like(current_dist)
        
        for idx in range(self.batch_size):
            if dones[idx]:
                # Khi kết thúc, phân phối target là delta function tại reward
                target_idx = torch.clamp(torch.floor((rewards[idx] - self.v_min) / self.delta_z), 0, self.n_atoms - 1).long()
                target_dist[idx, target_idx] = 1.0
            else:
                # Projection của phân phối Bellman
                # p(Tz) = p(z)
                
                # Tz_j = r + gamma * z_j
                Tz = rewards[idx] + self.gamma * self.support
                
                # Tính projection
                Tz = torch.clamp(Tz, self.v_min, self.v_max)
                b = (Tz - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Phân phối xác suất thành 2 bins gần nhất
                for j in range(self.n_atoms):
                    target_dist[idx, l[j]] += next_dist[idx, j] * (u[j] - b[j])
                    target_dist[idx, u[j]] += next_dist[idx, j] * (b[j] - l[j])
        
        # Tính loss - Cross entropy loss
        log_current_dist = current_dist.clamp(1e-10, 1.0).log()
        loss = -(target_dist * log_current_dist).sum(1)
        
        # Áp dụng importance sampling weights từ PER
        loss = (loss * weights).mean()
        
        # Cập nhật priorities
        td_errors = np.abs(loss.detach().cpu().numpy())
        self.memory.update_priorities(indices, td_errors)
        
        # Update beta
        self.memory.update_beta()
        
        # Tối ưu hóa loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Giới hạn gradient nếu cần
        if self.clip_gradients:
            for param in self.qnetwork_local.parameters():
                param.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Cập nhật mạng target
        soft_update(self.qnetwork_target, self.qnetwork_local, self.tau)
        
        # Lưu giá trị loss
        self.loss_history.append(loss.item())
    
    def save(self, filepath=None, episode=None):
        """
        Lưu mô hình
        
        Args:
            filepath: Đường dẫn file lưu model, nếu None thì tạo filepath mặc định
            episode: Số episode hiện tại, dùng để đặt tên file
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if filepath is None:
            timestamp = generate_timestamp()
            if episode is not None:
                filepath = os.path.join(self.save_dir, f"rainbow_dqn_ep{episode}_{timestamp}.pth")
            else:
                filepath = os.path.join(self.save_dir, f"rainbow_dqn_{timestamp}.pth")
        
        torch.save({
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'training_rewards': self.training_rewards,
            'validation_rewards': self.validation_rewards,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'hyperparams': {
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'tau': self.tau,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'n_atoms': self.n_atoms,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'n_step': self.n_step
            }
        }, filepath)
        
        logger.info(f"Model đã được lưu tại: {filepath}")
        return filepath
    
    def load(self, filepath):
        """
        Tải mô hình từ file
        
        Args:
            filepath: Đường dẫn file model
            
        Returns:
            bool: True nếu tải thành công, False nếu thất bại
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Không tìm thấy file model: {filepath}")
                return False
                
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
                
            if 'training_rewards' in checkpoint:
                self.training_rewards = checkpoint['training_rewards']
                
            if 'validation_rewards' in checkpoint:
                self.validation_rewards = checkpoint['validation_rewards']
                
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
                
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
            
            logger.info(f"Model đã được tải từ: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {str(e)}")
            return False 