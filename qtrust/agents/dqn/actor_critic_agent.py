"""
Actor-Critic Agent - cài đặt kiến trúc Actor-Critic cho học sâu.

Module này chứa cài đặt của Actor-Critic Agent với các tính năng:
- Actor-Critic Architecture
- Advantage Actor-Critic (A2C)
- Entropy regularization để khuyến khích khám phá
- Noisy Networks (tùy chọn) cho exploration
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

from qtrust.agents.dqn.networks import ActorNetwork, CriticNetwork
from qtrust.agents.dqn.replay_buffer import ReplayBuffer, NStepReplayBuffer
from qtrust.agents.dqn.utils import (
    soft_update, hard_update, calculate_td_error, calculate_huber_loss,
    exponential_decay, linear_decay, generate_timestamp, create_save_directory,
    plot_learning_curve, format_time, get_device, logger, SAVE_DIR
)
from qtrust.utils.cache import tensor_cache, lru_cache, compute_hash

class ActorCriticAgent:
    """
    Actor-Critic Agent - kết hợp Actor và Critic cho học tăng cường sâu.
    """
    def __init__(self, 
                state_size: int, 
                action_size: int, 
                seed: int = 42,
                buffer_size: int = 100000,
                batch_size: int = 64,
                gamma: float = 0.99,
                tau: float = 1e-3,
                actor_lr: float = 1e-4,
                critic_lr: float = 5e-4,
                update_every: int = 4,
                use_n_step: bool = True,
                n_step: int = 3,
                hidden_layers: List[int] = [512, 256],
                device: str = 'auto',
                entropy_coef: float = 0.01,
                value_loss_coef: float = 0.5,
                use_noisy_nets: bool = False,
                clip_gradients: bool = True,
                grad_clip_value: float = 1.0,
                warm_up_steps: int = 1000,
                distributional: bool = False,
                n_atoms: int = 51,
                v_min: float = -10.0,
                v_max: float = 10.0,
                save_dir: str = SAVE_DIR):
        """
        Khởi tạo Actor-Critic Agent.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_size: Kích thước không gian hành động
            seed: Seed cho các giá trị ngẫu nhiên
            buffer_size: Kích thước tối đa của replay buffer
            batch_size: Kích thước batch khi lấy mẫu từ replay buffer
            gamma: Hệ số chiết khấu
            tau: Tỷ lệ cập nhật tham số cho mạng target
            actor_lr: Tốc độ học cho Actor
            critic_lr: Tốc độ học cho Critic
            update_every: Tần suất cập nhật mạng (số bước)
            use_n_step: Sử dụng n-step returns hay không
            n_step: Số bước cho n-step returns
            hidden_layers: Kích thước các lớp ẩn
            device: Thiết bị sử dụng ('cpu', 'cuda', 'auto')
            entropy_coef: Hệ số cho entropy regularization
            value_loss_coef: Hệ số cho value loss
            use_noisy_nets: Sử dụng Noisy Networks cho exploration
            clip_gradients: Có giới hạn gradient trong quá trình học hay không
            grad_clip_value: Giá trị giới hạn gradient
            warm_up_steps: Số bước warm-up trước khi bắt đầu học
            distributional: Sử dụng Distributional RL cho Critic hay không
            n_atoms: Số atom trong phân phối (chỉ dùng khi distributional=True)
            v_min: Giá trị Q tối thiểu (chỉ dùng khi distributional=True)
            v_max: Giá trị Q tối đa (chỉ dùng khi distributional=True)
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
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.use_noisy_nets = use_noisy_nets
        self.clip_gradients = clip_gradients
        self.grad_clip_value = grad_clip_value
        self.warm_up_steps = warm_up_steps
        self.save_dir = create_save_directory(save_dir)
        self.distributional = distributional
        
        # Tham số cho Distributional RL (nếu sử dụng)
        if distributional:
            self.n_atoms = n_atoms
            self.v_min = v_min
            self.v_max = v_max
            self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Khởi tạo mạng Actor
        self.actor = ActorNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets
        ).to(self.device)
        
        # Khởi tạo mạng Critic
        self.critic = CriticNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets,
            n_atoms=n_atoms if distributional else 1, v_min=v_min, v_max=v_max
        ).to(self.device)
        
        # Tạo target networks
        self.critic_target = CriticNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets,
            n_atoms=n_atoms if distributional else 1, v_min=v_min, v_max=v_max
        ).to(self.device)
        
        # Sao chép tham số từ critic sang critic_target
        hard_update(self.critic_target, self.critic)
        
        # Khởi tạo optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Thêm learning rate scheduler
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)
        
        # Khởi tạo replay memory
        if use_n_step:
            self.memory = NStepReplayBuffer(buffer_size, batch_size, n_step=n_step, gamma=gamma, device=self.device)
        else:
            self.memory = ReplayBuffer(buffer_size, batch_size, device=self.device)
            
        # Biến đếm số bước học
        self.t_step = 0
        self.total_steps = 0
        
        # Thêm biến để theo dõi performance
        self.training_rewards = []
        self.validation_rewards = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.best_score = -float('inf')
        self.best_model_path = None
        self.train_start_time = None

        # Cache cho các hành động và giá trị
        self.action_cache = {}
        self.value_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000  # Giới hạn kích thước cache
        
        # Thống kê huấn luyện
        self.train_count = 0
        self.loss_history = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': []
        }

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
        self.memory.push(state, action, reward, next_state, done)
        
        # Học mỗi update_every bước
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size and self.total_steps > self.warm_up_steps:
            experiences = self.memory.sample()
            self._learn(experiences)
    
    def act(self, state, explore=True, use_target=False):
        """
        Lựa chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            use_target: Sử dụng mạng target để lấy hành động
            
        Returns:
            Tuple: Hành động được chọn và log xác suất
        """
        # Chuyển sang tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Đảm bảo state có kích thước [batch_size, state_size]
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        state = state.to(self.device)
        
        # Tính hash của state để kiểm tra cache
        state_hash = None
        if not explore:
            if isinstance(state, torch.Tensor):
                state_np = state.cpu().numpy()
                state_hash = compute_hash(state_np.tobytes())
            else:
                state_hash = compute_hash(state.tobytes())
                
            # Kiểm tra trong cache
            if state_hash in self.action_cache:
                self.cache_hits += 1
                return self.action_cache[state_hash]
            
            self.cache_misses += 1
        
        # Lấy xác suất hành động
        with torch.no_grad():
            if use_target:
                action_probs = self.actor_target(state) if hasattr(self, 'actor_target') else self.actor(state)
            else:
                action_probs = self._get_action_probs(state)
        
        # Khởi tạo phân phối hành động
        if self.distributional:
            # Phân phối qua mỗi atom của phân phối
            dist = torch.distributions.Categorical(action_probs.squeeze())
        else:
            # Phân phối đơn giản qua các hành động
            dist = torch.distributions.Categorical(action_probs)
        
        if explore:
            # Lấy mẫu ngẫu nhiên từ phân phối
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Chọn hành động có xác suất cao nhất
            action = torch.argmax(action_probs, dim=1)
            log_prob = dist.log_prob(action)
        
        # Lưu vào cache nếu không thăm dò
        if not explore and state_hash is not None:
            action_item = action.item() if action.numel() == 1 else action.cpu().numpy()
            log_prob_item = log_prob.item() if log_prob.numel() == 1 else log_prob.cpu().numpy()
            
            self.action_cache[state_hash] = (action_item, log_prob_item)
            
            # Giới hạn kích thước cache
            if len(self.action_cache) > self.max_cache_size:
                # Xóa một mục ngẫu nhiên
                random_key = random.choice(list(self.action_cache.keys()))
                del self.action_cache[random_key]
        
        if action.numel() == 1 and log_prob.numel() == 1:
            return action.item(), log_prob.item()
        else:
            return action.cpu().numpy(), log_prob.cpu().numpy()
    
    def _learn(self, experiences):
        """
        Cập nhật Actor và Critic dựa trên batch kinh nghiệm.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.size(0)
        
        # Tạo one-hot encoding cho actions
        actions_one_hot = F.one_hot(actions, self.action_size).float()
        
        # === Cập nhật Critic ===
        if self.distributional:
            # Lấy phân phối Q từ critic
            q_distributions, q_values = self.critic(states, [actions_one_hot])
            
            # Lấy phân phối Q tiếp theo từ target critic
            with torch.no_grad():
                # Tính toán phân phối hành động tiếp theo từ actor
                next_action_probs = self.actor(next_states)
                next_action_probs = next_action_probs[0]
                
                # Lấy phân phối Q target cho hành động tiếp theo
                next_q_distributions, _ = self.critic_target(next_states, [next_action_probs])
                next_q_dist = next_q_distributions[0]  # [batch_size, n_atoms]
                
                # Tính toán phân phối target
                target_dist = torch.zeros_like(q_distributions[0])
                for idx in range(batch_size):
                    if dones[idx]:
                        # Khi kết thúc, phân phối target là delta function tại reward
                        target_idx = torch.clamp(torch.floor((rewards[idx] - self.v_min) / self.delta_z), 0, self.n_atoms - 1).long()
                        target_dist[idx, target_idx] = 1.0
                    else:
                        # Projection của phân phối Bellman
                        Tz = rewards[idx] + self.gamma * self.support
                        Tz = torch.clamp(Tz, self.v_min, self.v_max)
                        b = (Tz - self.v_min) / self.delta_z
                        l = b.floor().long()
                        u = b.ceil().long()
                        
                        # Phân phối xác suất thành 2 bins gần nhất
                        for j in range(self.n_atoms):
                            target_dist[idx, l[j]] += next_q_dist[idx, j] * (u[j] - b[j])
                            target_dist[idx, u[j]] += next_q_dist[idx, j] * (b[j] - l[j])
                
                # Tính Critic loss - Cross entropy loss
                log_q_dist = q_distributions[0].clamp(1e-10, 1.0).log()
                critic_loss = -(target_dist * log_q_dist).sum(1).mean()
        else:
            # Lấy giá trị Q từ critic
            q_values = self.critic(states, [actions_one_hot])
            q_value = q_values[0]  # [batch_size, 1]
            
            # Tính giá trị Q target
            with torch.no_grad():
                # Tính toán phân phối hành động tiếp theo từ actor
                next_action_probs = self.actor(next_states)
                next_action_probs = next_action_probs[0]
                
                # Lấy giá trị Q target cho hành động tiếp theo
                next_q_values = self.critic_target(next_states, [next_action_probs])
                next_q_value = next_q_values[0]
                
                # Tính TD target
                q_target = rewards + (self.gamma * next_q_value * (1 - dones))
            
            # Tính Critic loss (MSE)
            critic_loss = F.mse_loss(q_value, q_target)
        
        # Cập nhật Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Giới hạn gradient cho Critic nếu cần
        if self.clip_gradients:
            for param in self.critic.parameters():
                param.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)
        
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        
        # === Cập nhật Actor ===
        # Tính toán phân phối hành động từ actor
        action_probs = self.actor(states)
        action_probs = action_probs[0]
        
        # Tạo distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Lấy mẫu hành động mới và tính log prob
        sampled_actions = dist.sample()
        log_probs = dist.log_prob(sampled_actions)
        
        # Tính entropy của phân phối
        entropy = dist.entropy().mean()
        
        # Tạo one-hot encoding cho hành động được lấy mẫu
        sampled_actions_one_hot = F.one_hot(sampled_actions, self.action_size).float()
        
        # Lấy giá trị Q cho hành động được lấy mẫu
        if self.distributional:
            _, sampled_q_values = self.critic(states, [sampled_actions_one_hot])
            advantage = sampled_q_values[0].detach()
        else:
            sampled_q_values = self.critic(states, [sampled_actions_one_hot])
            advantage = sampled_q_values[0].detach()
        
        # Tính Actor loss
        actor_loss = -(log_probs * advantage).mean() - self.entropy_coef * entropy
        
        # Cập nhật Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Giới hạn gradient cho Actor nếu cần
        if self.clip_gradients:
            for param in self.actor.parameters():
                param.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)
        
        self.actor_optimizer.step()
        self.actor_scheduler.step()
        
        # Cập nhật mạng target
        soft_update(self.critic_target, self.critic, self.tau)
        
        # Lưu giá trị loss
        self.critic_loss_history.append(critic_loss.item())
        self.actor_loss_history.append(actor_loss.item())
        self.entropy_history.append(entropy.item())
        
        self.train_count += 1
        
        # Xóa cache định kỳ
        if self.train_count % 1000 == 0:
            self.clear_cache()
    
    def clear_cache(self):
        """Xóa cache hành động và giá trị."""
        self.action_cache.clear()
        self.value_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    @tensor_cache
    def _get_action_probs(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Tính xác suất các hành động cho trạng thái đã cho.
        
        Args:
            state_tensor: Tensor trạng thái đầu vào
            
        Returns:
            torch.Tensor: Xác suất các hành động
        """
        return self.actor(state_tensor)
    
    @tensor_cache
    def _get_state_value(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Tính giá trị trạng thái cho trạng thái đã cho.
        
        Args:
            state_tensor: Tensor trạng thái đầu vào
            
        Returns:
            torch.Tensor: Giá trị trạng thái
        """
        return self.critic(state_tensor)
    
    @tensor_cache
    def _get_target_value(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Tính giá trị trạng thái target cho trạng thái đã cho.
        
        Args:
            state_tensor: Tensor trạng thái đầu vào
            
        Returns:
            torch.Tensor: Giá trị trạng thái từ mạng target
        """
        return self.critic_target(state_tensor)
    
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
                filepath = os.path.join(self.save_dir, f"actor_critic_ep{episode}_{timestamp}.pth")
            else:
                filepath = os.path.join(self.save_dir, f"actor_critic_{timestamp}.pth")
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'actor_loss_history': self.actor_loss_history,
            'critic_loss_history': self.critic_loss_history,
            'entropy_history': self.entropy_history,
            'training_rewards': self.training_rewards,
            'validation_rewards': self.validation_rewards,
            'total_steps': self.total_steps,
            'hyperparams': {
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'tau': self.tau,
                'entropy_coef': self.entropy_coef,
                'value_loss_coef': self.value_loss_coef,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'distributional': self.distributional
            },
            'loss_history': self.loss_history,
            'train_count': self.train_count
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
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            if 'actor_scheduler_state_dict' in checkpoint:
                self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
                
            if 'critic_scheduler_state_dict' in checkpoint:
                self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
                
            if 'actor_loss_history' in checkpoint:
                self.actor_loss_history = checkpoint['actor_loss_history']
                
            if 'critic_loss_history' in checkpoint:
                self.critic_loss_history = checkpoint['critic_loss_history']
                
            if 'entropy_history' in checkpoint:
                self.entropy_history = checkpoint['entropy_history']
                
            if 'training_rewards' in checkpoint:
                self.training_rewards = checkpoint['training_rewards']
                
            if 'validation_rewards' in checkpoint:
                self.validation_rewards = checkpoint['validation_rewards']
                
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
            
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            
            if 'train_count' in checkpoint:
                self.train_count = checkpoint['train_count']
            
            logger.info(f"Model đã được tải từ: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {str(e)}")
            return False 

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Trả về thống kê hiệu suất của agent.
        
        Returns:
            Dict[str, Any]: Thống kê hiệu suất
        """
        return {
            'train_count': self.train_count,
            'actor_loss': np.mean(self.loss_history['actor_loss'][-100:]) if self.loss_history['actor_loss'] else None,
            'critic_loss': np.mean(self.loss_history['critic_loss'][-100:]) if self.loss_history['critic_loss'] else None,
            'entropy': np.mean(self.loss_history['entropy'][-100:]) if self.loss_history['entropy'] else None,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        } 