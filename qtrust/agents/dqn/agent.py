"""
Lớp DQNAgent chính.

Module này chứa cài đặt của DQNAgent với các cải tiến như:
- Double DQN
- Dueling Network Architecture
- Prioritized Experience Replay
- Noisy Networks for Exploration
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

from qtrust.agents.dqn.networks import QNetwork
from qtrust.agents.dqn.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, EfficientReplayBuffer
from qtrust.agents.dqn.utils import (
    soft_update, hard_update, calculate_td_error, calculate_huber_loss,
    exponential_decay, linear_decay, generate_timestamp, create_save_directory,
    plot_learning_curve, format_time, get_device, logger, SAVE_DIR
)

class DQNAgent:
    """
    Deep Q-Network Agent với các cải tiến:
    - Double DQN
    - Dueling Network
    - Prioritized Experience Replay
    - Noisy Networks
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
                prioritized_replay: bool = True,
                alpha: float = 0.6,
                beta_start: float = 0.4,
                dueling: bool = True,
                noisy_nets: bool = False,
                hidden_layers: List[int] = [128, 64],
                device: str = 'auto',
                min_epsilon: float = 0.01,
                epsilon_decay: float = 0.995,
                use_efficient_buffer: bool = False,
                clip_gradients: bool = True,
                grad_clip_value: float = 1.0,
                warm_up_steps: int = 1000,
                save_dir: str = SAVE_DIR):
        """
        Khởi tạo DQNAgent.
        
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
            prioritized_replay: Sử dụng Prioritized Experience Replay hay không
            alpha: Hệ số alpha cho PER
            beta_start: Giá trị beta ban đầu cho PER
            dueling: Sử dụng kiến trúc mạng Dueling hay không
            noisy_nets: Sử dụng Noisy Networks hay không
            hidden_layers: Kích thước các lớp ẩn
            device: Thiết bị sử dụng ('cpu', 'cuda', 'auto')
            min_epsilon: Giá trị epsilon tối thiểu
            epsilon_decay: Tốc độ giảm epsilon
            use_efficient_buffer: Sử dụng EfficientReplayBuffer thay vì buffer thông thường
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
        self.prioritized_replay = prioritized_replay
        self.noisy_nets = noisy_nets
        self.clip_gradients = clip_gradients
        self.grad_clip_value = grad_clip_value
        self.warm_up_steps = warm_up_steps
        self.save_dir = create_save_directory(save_dir)
        
        # Khởi tạo mạng Q
        self.qnetwork_local = QNetwork(state_size, [action_size], hidden_sizes=hidden_layers, noisy=noisy_nets).to(self.device)
        self.qnetwork_target = QNetwork(state_size, [action_size], hidden_sizes=hidden_layers, noisy=noisy_nets).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Thêm learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Khởi tạo replay memory
        if use_efficient_buffer:
            if prioritized_replay:
                logger.warning("EfficientReplayBuffer không hỗ trợ prioritized replay. Sử dụng buffer PER thông thường.")
                self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta_start, 1.0, device=self.device)
            else:
                self.memory = EfficientReplayBuffer(buffer_size, batch_size, (state_size,), (1,), device=self.device)
        else:
            if prioritized_replay:
                self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta_start, 1.0, device=self.device)
            else:
                self.memory = ReplayBuffer(buffer_size, batch_size, device=self.device)
            
        # Biến đếm số bước học
        self.t_step = 0
        self.total_steps = 0
        
        # Khởi tạo epsilon cho exploration (chỉ sử dụng khi không dùng noisy networks)
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
        
        # Lưu kinh nghiệm vào replay memory
        if self.prioritized_replay:
            # Tính TD error cho ưu tiên ban đầu
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Tính current Q value
                q_values, _ = self.qnetwork_local(state_tensor)
                current_q = q_values[0][0, action].item()
                
                # Tính target Q value với double DQN
                next_q_values, _ = self.qnetwork_local(next_state_tensor)
                next_action = next_q_values[0].argmax(dim=1).item()
                next_q_target_values, _ = self.qnetwork_target(next_state_tensor)
                target_q = next_q_target_values[0][0, next_action].item()
                
                # Tính TD target
                td_target = reward if done else reward + self.gamma * target_q
                
                # Tính TD error
                td_error = td_target - current_q
                
                self.memory.push(state, action, reward, next_state, done, td_error)
        else:
            self.memory.push(state, action, reward, next_state, done)
        
        # Học mỗi update_every bước
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size and self.total_steps > self.warm_up_steps:
            if self.prioritized_replay:
                self._learn_prioritized()
            else:
                experiences = self._sample_from_memory()
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
            action_values, _ = self.qnetwork_local(state)
            
            # Lấy giá trị Q của chiều hành động đầu tiên
            action_values = action_values[0]
            
        # Bật lại chế độ học cho mạng
        self.qnetwork_local.train()
        
        # Nếu sử dụng noisy networks, reset noise và không cần epsilon-greedy
        if self.noisy_nets:
            self.qnetwork_local.reset_noise()
            return np.argmax(action_values.cpu().data.numpy())
        
        # Epsilon-greedy action selection
        if eps is None:
            eps = self.epsilon
            
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _sample_from_memory(self):
        """
        Lấy mẫu ngẫu nhiên từ replay memory (sử dụng cho non-prioritized)
        
        Returns:
            Tuple: experiences (state, action, reward, next_state, done)
        """
        return self.memory.sample()
    
    def _learn(self, experiences):
        """
        Cập nhật giá trị Q dựa trên batch kinh nghiệm.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Lấy Q values của next_state từ TARGET network
        with torch.no_grad():
            next_state_values, _ = self.qnetwork_target(next_states)
            
            # Double DQN: lấy hành động tốt nhất từ LOCAL network, sử dụng Q-values từ TARGET network
            next_q_values, _ = self.qnetwork_local(next_states)
            next_actions = next_q_values[0].argmax(dim=1, keepdim=True)
            q_targets_next = next_state_values[0].gather(1, next_actions)
            
            # Tính TD target
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # Lấy Q values hiện tại của các hành động đã thực hiện
        q_values, state_values = self.qnetwork_local(states)
        q_values = q_values[0].gather(1, actions)
        
        # Tính loss và cập nhật weights
        loss = F.mse_loss(q_values, q_targets)
        
        # Thêm vào lịch sử loss
        self.loss_history.append(loss.item())
        
        # Thực hiện backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clipping gradients
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip_value)
            
        self.optimizer.step()
        
        # Cập nhật mạng target (soft update)
        self._soft_update()
        
        # Cập nhật learning rate
        self.scheduler.step()
    
    def _soft_update(self):
        """
        Cập nhật tham số của mạng target (qnetwork_target) bằng cách trộn một phần nhỏ
        từ mạng local (qnetwork_local) sử dụng tham số tau.
        
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        # Đảm bảo tau > 0 để soft update có hiệu lực
        if self.tau <= 0:
            self.tau = 1e-3  # Đặt giá trị mặc định nếu tau không hợp lệ
            
        # Thực hiện soft update từ local network sang target network
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def _learn_prioritized(self):
        """
        Cập nhật tham số mạng sử dụng Prioritized Experience Replay
        """
        # Lấy batch từ PER
        result = self.memory.sample()
        
        if result is None:
            return
            
        states, actions, rewards, next_states, dones, indices, weights = result
        
        # Double DQN
        # Lấy Q values từ mạng local cho trạng thái tiếp theo
        next_q_values, _ = self.qnetwork_local(next_states)
        next_actions = next_q_values[0].detach().argmax(dim=1, keepdim=True)
        
        # Lấy Q values từ mạng target cho next_state, next_action
        next_q_target_values, _ = self.qnetwork_target(next_states)
        # Gather Q values cho hành động đã chọn từ mạng local
        next_q_targets = next_q_target_values[0].gather(1, next_actions)
        
        # Tính Q targets
        q_targets = rewards.unsqueeze(1) + (self.gamma * next_q_targets * (1 - dones.unsqueeze(1)))
        
        # Lấy Q value hiện tại
        q_values, _ = self.qnetwork_local(states)
        q_expected = q_values[0].gather(1, actions.unsqueeze(1))
        
        # Tính TD errors cho việc cập nhật ưu tiên
        with torch.no_grad():
            td_errors = torch.abs(q_targets - q_expected).squeeze().cpu().numpy()
        
        # Cập nhật ưu tiên trong buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Tính loss với importance sampling weights
        loss = calculate_huber_loss(q_expected, q_targets, weights)
        self.loss_history.append(loss.item())
        
        # Cập nhật tham số mạng
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip_value)
        
        self.optimizer.step()
        
        # Cập nhật learning rate
        self.scheduler.step()
        
        # Cập nhật mạng target
        self._soft_update()

    def update_epsilon(self, epsilon_decay_type: str = 'exponential'):
        """
        Cập nhật epsilon cho chính sách epsilon-greedy
        
        Args:
            epsilon_decay_type: Loại suy giảm epsilon ('exponential' hoặc 'linear')
        """
        if epsilon_decay_type == 'exponential':
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        elif epsilon_decay_type == 'linear':
            self.epsilon = linear_decay(self.eps_start, self.eps_end, 1000, self.total_steps)
        else:
            self.epsilon = exponential_decay(self.eps_start, self.eps_end, self.eps_decay, self.total_steps // 100)
    
    def save(self, path: str = None):
        """
        Lưu mô hình
        
        Args:
            path: Đường dẫn để lưu mô hình, nếu None thì sử dụng đường dẫn mặc định
        """
        if path is None:
            timestamp = generate_timestamp()
            path = os.path.join(self.save_dir, f"dqn_model_{timestamp}.pth")
            
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'training_rewards': self.training_rewards,
            'validation_rewards': self.validation_rewards,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, path)
        
        logger.info(f"Mô hình đã được lưu tại: {path}")
        return path
    
    def load(self, path: str):
        """
        Tải mô hình
        
        Args:
            path: Đường dẫn đến file mô hình
            
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        """
        if not os.path.exists(path):
            logger.error(f"Không tìm thấy file model tại {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                
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
                
            logger.info(f"Đã tải model từ {path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
            return False
    
    def save_checkpoint(self, episode: int, reward: float, is_best: bool = False):
        """
        Lưu checkpoint của model
        
        Args:
            episode: Số episode hiện tại
            reward: Phần thưởng trung bình
            is_best: Có phải là model tốt nhất không
            
        Returns:
            str: Đường dẫn đến file checkpoint
        """
        timestamp = generate_timestamp()
        filename = os.path.join(self.save_dir, f"dqn_checkpoint_ep{episode}_{timestamp}.pth")
        
        self.save(filename)
        
        if is_best:
            best_filename = os.path.join(self.save_dir, "dqn_best_model.pth")
            self.best_model_path = best_filename
            
            # Sao chép file hoặc lưu trực tiếp
            torch.save({
                'qnetwork_local': self.qnetwork_local.state_dict(),
                'qnetwork_target': self.qnetwork_target.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'loss_history': self.loss_history,
                'training_rewards': self.training_rewards,
                'validation_rewards': self.validation_rewards,
                'epsilon': self.epsilon,
                'total_steps': self.total_steps,
                'episode': episode,
                'reward': reward,
                'timestamp': timestamp
            }, best_filename)
            
            logger.info(f"Đã lưu model tốt nhất với phần thưởng {reward:.2f} tại episode {episode}")
        
        return filename

    def load_best_model(self):
        """
        Tải model tốt nhất nếu có
        
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        """
        best_model_path = os.path.join(self.save_dir, "dqn_best_model.pth")
        if os.path.exists(best_model_path):
            success = self.load(best_model_path)
            if success:
                try:
                    checkpoint = torch.load(best_model_path, map_location=self.device)
                    if 'episode' in checkpoint and 'reward' in checkpoint:
                        logger.info(f"Đã tải model tốt nhất từ episode {checkpoint['episode']} với phần thưởng {checkpoint['reward']:.2f}")
                    else:
                        logger.info(f"Đã tải model tốt nhất")
                except Exception as e:
                    logger.error(f"Lỗi khi đọc thông tin model tốt nhất: {e}")
                return True
        logger.warning("Không tìm thấy model tốt nhất")
        return False

    def evaluate(self, env, n_episodes: int = 5, max_t: int = 1000, render: bool = False):
        """
        Đánh giá agent
        
        Args:
            env: Môi trường tương tác
            n_episodes: Số lượng episodes để đánh giá
            max_t: Số bước tối đa trong mỗi episode
            render: Có render môi trường hay không
            
        Returns:
            float: Phần thưởng trung bình trên tất cả các episodes
        """
        from .train import evaluate_dqn
        return evaluate_dqn(self, env, n_episodes, max_t, render)
    
    def plot_rewards(self, save_path: Optional[str] = None):
        """
        Vẽ biểu đồ phần thưởng huấn luyện
        
        Args:
            save_path: Đường dẫn để lưu biểu đồ, nếu None thì hiển thị
        """
        from .train import plot_dqn_rewards
        plot_dqn_rewards(
            rewards=self.training_rewards, 
            val_rewards=self.validation_rewards,
            title="DQN Training Rewards", 
            save_path=save_path
        )
