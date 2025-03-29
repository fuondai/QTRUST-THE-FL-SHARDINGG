"""
Các hàm tiện ích và hằng số cho DQN Agent.

Module này định nghĩa các hàm tiện ích và hằng số sử dụng trong DQN Agent.
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dqn_agent')

# Cấu trúc hằng số
SAVE_DIR = 'models'  # Thư mục mặc định để lưu model

def create_save_directory(dir_path: str = SAVE_DIR) -> str:
    """
    Tạo thư mục lưu model nếu không tồn tại.
    
    Args:
        dir_path: Đường dẫn thư mục
        
    Returns:
        str: Đường dẫn thư mục đã tạo
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Đã tạo thư mục {dir_path} để lưu model")
    return dir_path

def soft_update(target_model: torch.nn.Module, 
               local_model: torch.nn.Module, 
               tau: float = 1e-3):
    """
    Cập nhật mềm tham số mạng target: θ_target = τ*θ_local + (1 - τ)*θ_target
    
    Args:
        target_model: Mạng target được cập nhật
        local_model: Mạng local chứa tham số mới
        tau: Hệ số interpolation
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
def hard_update(target_model: torch.nn.Module, local_model: torch.nn.Module):
    """
    Cập nhật hoàn toàn tham số mạng target bằng tham số mạng local
    
    Args:
        target_model: Mạng target được cập nhật
        local_model: Mạng local chứa tham số mới
    """
    target_model.load_state_dict(local_model.state_dict())

def calculate_td_error(current_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """
    Tính TD error giữa giá trị Q hiện tại và giá trị Q mục tiêu
    
    Args:
        current_q: Giá trị Q hiện tại
        target_q: Giá trị Q mục tiêu
        
    Returns:
        torch.Tensor: TD error
    """
    return torch.abs(target_q - current_q)

def calculate_huber_loss(current_q: torch.Tensor, 
                         target_q: torch.Tensor, 
                         weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Tính Huber loss giữa giá trị Q hiện tại và giá trị Q mục tiêu
    
    Args:
        current_q: Giá trị Q hiện tại
        target_q: Giá trị Q mục tiêu
        weights: Trọng số cho từng mẫu (cho prioritized replay)
        
    Returns:
        torch.Tensor: Huber loss
    """
    elementwise_loss = F.huber_loss(current_q, target_q, reduction='none')
    
    if weights is not None:
        # Áp dụng trọng số nếu có
        loss = torch.mean(elementwise_loss * weights)
    else:
        loss = torch.mean(elementwise_loss)
        
    return loss

def exponential_decay(start_value: float, end_value: float, decay_rate: float, step: int) -> float:
    """
    Tính giá trị theo suy giảm hàm mũ
    
    Args:
        start_value: Giá trị ban đầu
        end_value: Giá trị tối thiểu
        decay_rate: Tỷ lệ suy giảm (0 < decay_rate < 1)
        step: Bước hiện tại
    
    Returns:
        float: Giá trị sau khi suy giảm
    """
    return max(end_value, start_value * (decay_rate ** step))

def linear_decay(start_value: float, end_value: float, decay_steps: int, step: int) -> float:
    """
    Tính giá trị theo suy giảm tuyến tính
    
    Args:
        start_value: Giá trị ban đầu
        end_value: Giá trị tối thiểu
        decay_steps: Số bước để đạt giá trị tối thiểu
        step: Bước hiện tại
    
    Returns:
        float: Giá trị sau khi suy giảm
    """
    fraction = min(float(step) / float(decay_steps), 1.0)
    return start_value + fraction * (end_value - start_value)

def generate_timestamp() -> str:
    """
    Tạo chuỗi timestamp để dùng trong tên file
    
    Returns:
        str: Chuỗi timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_learning_curve(rewards: List[float], 
                        avg_window: int = 20,
                        title: str = "Learning Curve", 
                        save_path: Optional[str] = None):
    """
    Vẽ đồ thị học tập
    
    Args:
        rewards: Danh sách phần thưởng
        avg_window: Cửa sổ để tính trung bình động
        title: Tiêu đề đồ thị
        save_path: Đường dẫn để lưu đồ thị, nếu None thì hiển thị
    """
    plt.figure(figsize=(10, 6))
    
    # Vẽ phần thưởng từng episode
    plt.plot(rewards, label='Reward per Episode', alpha=0.3)
    
    # Tính và vẽ trung bình động
    avg_rewards = []
    for i in range(len(rewards)):
        if i < avg_window:
            avg_rewards.append(np.mean(rewards[:i+1]))
        else:
            avg_rewards.append(np.mean(rewards[i-avg_window+1:i+1]))
    
    plt.plot(avg_rewards, label=f'Average Reward (window={avg_window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Đồ thị đã được lưu tại: {save_path}")
    else:
        plt.show()
        
def format_time(seconds: float) -> str:
    """
    Định dạng thời gian từ số giây
    
    Args:
        seconds: Số giây
        
    Returns:
        str: Chuỗi định dạng thời gian
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{s:.1f}s"

def get_device(device: str = 'auto') -> torch.device:
    """
    Lấy thiết bị để chạy pytorch (CPU/CUDA)
    
    Args:
        device: auto/cpu/cuda
        
    Returns:
        torch.device: Thiết bị được chọn
    """
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) 