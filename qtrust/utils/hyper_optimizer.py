import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Callable
import os
import json
from datetime import datetime
import torch

class BayesianOptimizer:
    """
    Tối ưu hóa Bayesian cho hyperparameters.
    """
    def __init__(self,
                param_ranges: Dict[str, Tuple[float, float]],
                objective_function: Callable,
                n_initial_points: int = 5,
                exploration_weight: float = 0.1,
                log_dir: str = 'logs/hyperopt',
                maximize: bool = True):
        """
        Khởi tạo BayesianOptimizer.
        
        Args:
            param_ranges: Dictionary chứa phạm vi của các hyperparameter
            objective_function: Hàm mục tiêu cần tối ưu hóa
            n_initial_points: Số điểm khởi tạo ban đầu
            exploration_weight: Trọng số cho việc khám phá
            log_dir: Thư mục lưu logs
            maximize: True nếu tối đa hóa mục tiêu, False nếu tối thiểu hóa
        """
        self.param_ranges = param_ranges
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        self.log_dir = log_dir
        self.maximize = maximize
        
        # Tạo thư mục logs
        os.makedirs(log_dir, exist_ok=True)
        
        # Lưu lịch sử tìm kiếm
        self.X = []  # Các điểm đã thử
        self.y = []  # Giá trị mục tiêu tương ứng
        
        # Tham số mô hình GP
        self.length_scale = 1.0
        self.noise = 1e-5
        
        # Thông tin tối ưu
        self.best_params = None
        self.best_value = -float('inf') if maximize else float('inf')
    
    def optimize(self, n_iterations: int = 30) -> Dict[str, float]:
        """
        Thực hiện tối ưu hóa Bayesian.
        
        Args:
            n_iterations: Số vòng lặp tối ưu hóa
            
        Returns:
            Dict[str, float]: Hyperparameters tốt nhất
        """
        # Khởi tạo với các điểm ban đầu
        self._initialize()
        
        # Vòng lặp chính cho tối ưu hóa Bayesian
        for i in range(n_iterations):
            # Chọn điểm tiếp theo để đánh giá
            next_point = self._select_next_point()
            
            # Đánh giá hàm mục tiêu tại điểm đó
            params_dict = self._vector_to_params(next_point)
            value = self.objective_function(params_dict)
            
            # Cập nhật lịch sử
            self.X.append(next_point)
            self.y.append(value)
            
            # Cập nhật điểm tốt nhất
            if (self.maximize and value > self.best_value) or \
               (not self.maximize and value < self.best_value):
                self.best_value = value
                self.best_params = params_dict
            
            # Log quá trình
            self._log_iteration(i, params_dict, value)
            
            # Cập nhật trọng số khám phá
            self.exploration_weight *= 0.95  # Giảm dần việc khám phá
        
        # Lưu và trả về kết quả tốt nhất
        self._save_results()
        return self.best_params
    
    def _initialize(self):
        """
        Khởi tạo với các điểm ban đầu ngẫu nhiên.
        """
        for _ in range(self.n_initial_points):
            # Tạo một điểm ngẫu nhiên trong không gian tham số
            point = self._sample_random_point()
            
            # Đánh giá hàm mục tiêu
            params_dict = self._vector_to_params(point)
            value = self.objective_function(params_dict)
            
            # Cập nhật lịch sử
            self.X.append(point)
            self.y.append(value)
            
            # Cập nhật điểm tốt nhất
            if (self.maximize and value > self.best_value) or \
               (not self.maximize and value < self.best_value):
                self.best_value = value
                self.best_params = params_dict
    
    def _sample_random_point(self) -> np.ndarray:
        """
        Lấy mẫu một điểm ngẫu nhiên trong không gian tham số.
        
        Returns:
            np.ndarray: Điểm ngẫu nhiên
        """
        point = []
        for param_name, (lower, upper) in self.param_ranges.items():
            # Lấy mẫu theo phân phối đều
            value = np.random.uniform(lower, upper)
            point.append(value)
        
        return np.array(point)
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Chuyển đổi vector thành dict tham số.
        
        Args:
            vector: Vector các giá trị tham số
            
        Returns:
            Dict[str, float]: Dictionary tham số
        """
        params = {}
        for i, (param_name, _) in enumerate(self.param_ranges.items()):
            params[param_name] = vector[i]
        
        return params
    
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """
        Chuyển đổi dict tham số thành vector.
        
        Args:
            params: Dictionary tham số
            
        Returns:
            np.ndarray: Vector các giá trị tham số
        """
        vector = []
        for param_name in self.param_ranges.keys():
            vector.append(params[param_name])
        
        return np.array(vector)
    
    def _select_next_point(self) -> np.ndarray:
        """
        Chọn điểm tiếp theo để đánh giá dựa trên acquisition function.
        
        Returns:
            np.ndarray: Điểm tiếp theo
        """
        if len(self.X) == 0:
            return self._sample_random_point()
        
        # Chuyển đổi sang numpy arrays
        X = np.array(self.X)
        y = np.array(self.y)
        
        # Nếu tối thiểu hóa, đảo dấu y
        if not self.maximize:
            y = -y
        
        # Triển khai thuật toán hỗ trợ quyết định dựa trên mô hình
        # Sử dụng Expected Improvement (EI) làm acquisition function
        
        # Lấy điểm ngẫu nhiên
        n_samples = 1000
        candidates = np.random.uniform(
            low=[lower for _, (lower, _) in self.param_ranges.items()],
            high=[upper for _, (_, upper) in self.param_ranges.items()],
            size=(n_samples, len(self.param_ranges))
        )
        
        # Tính toán GP posterior cho mỗi ứng viên
        ei_values = []
        for candidate in candidates:
            mean, std = self._gp_posterior(candidate, X, y)
            
            # Tính Expected Improvement
            improvement = mean - np.max(y)
            Z = improvement / (std + 1e-9)
            ei = improvement * (0.5 + 0.5 * np.tanh(Z / np.sqrt(2))) + \
                 std * np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
            
            # Thêm hạng phạt để khuyến khích khám phá
            distance_penalty = -self._min_distance(candidate, X)
            
            # Kết hợp
            acquisition = ei + self.exploration_weight * distance_penalty
            ei_values.append(acquisition)
        
        # Chọn điểm tốt nhất
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]
    
    def _gp_posterior(self, x: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Tính toán posterior từ Gaussian Process.
        
        Args:
            x: Điểm cần tính posterior
            X: Điểm đã đánh giá
            y: Giá trị đã đánh giá
            
        Returns:
            Tuple[float, float]: (mean, standard deviation)
        """
        # Tính ma trận kernel
        K = self._rbf_kernel(X, X)
        K += np.eye(len(X)) * self.noise  # Add noise
        
        # Tính kernel giữa x và X
        k = self._rbf_kernel(x.reshape(1, -1), X).flatten()
        
        # Tính kernel của x với chính nó
        kxx = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1)).flatten()[0]
        
        # Tính posterior
        K_inv = np.linalg.inv(K)
        mean = k.dot(K_inv).dot(y)
        var = kxx - k.dot(K_inv).dot(k)
        std = np.sqrt(max(1e-8, var))  # Tránh giá trị âm do sai số số học
        
        return mean, std
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Hàm kernel RBF.
        
        Args:
            X1: Tập điểm thứ nhất
            X2: Tập điểm thứ hai
            
        Returns:
            np.ndarray: Ma trận kernel
        """
        sq_dists = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sq_dists / self.length_scale**2)
    
    def _min_distance(self, x: np.ndarray, X: np.ndarray) -> float:
        """
        Tính khoảng cách tối thiểu từ x đến tập X.
        
        Args:
            x: Điểm cần tính
            X: Tập điểm
            
        Returns:
            float: Khoảng cách tối thiểu
        """
        if len(X) == 0:
            return float('inf')
        
        dists = np.sqrt(np.sum((X - x)**2, axis=1))
        return np.min(dists)
    
    def _log_iteration(self, iteration: int, params: Dict[str, float], value: float):
        """
        Log thông tin cho mỗi vòng lặp.
        
        Args:
            iteration: Số vòng lặp
            params: Tham số đã đánh giá
            value: Giá trị mục tiêu
        """
        print(f"Iteration {iteration+1}:")
        for param_name, param_value in params.items():
            print(f"  {param_name} = {param_value:.6f}")
        print(f"  Value = {value:.6f}")
        if (self.maximize and value == self.best_value) or (not self.maximize and value == self.best_value):
            print(f"  (New best value!)")
        print()
    
    def _save_results(self):
        """
        Lưu kết quả tối ưu hóa.
        """
        # Tạo timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu thông tin tối ưu
        results = {
            'best_params': self.best_params,
            'best_value': float(self.best_value),
            'history': {
                'params': [self._vector_to_params(x) for x in self.X],
                'values': [float(y) for y in self.y]
            },
            'timestamp': timestamp,
            'maximize': self.maximize
        }
        
        # Lưu ra file
        results_path = os.path.join(self.log_dir, f"hyperopt_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Lưu biểu đồ
        self._plot_optimization_history(os.path.join(self.log_dir, f"hyperopt_history_{timestamp}.png"))
        
        print(f"Kết quả tối ưu hóa đã được lưu tại {results_path}")
    
    def _plot_optimization_history(self, save_path: str):
        """
        Vẽ đồ thị lịch sử tối ưu hóa.
        
        Args:
            save_path: Đường dẫn để lưu hình
        """
        # Sao chép giá trị y
        y_values = self.y.copy()
        
        # Nếu tối thiểu hóa, đảo dấu y
        if not self.maximize:
            best_vals = [-float('inf')]
            for y in y_values:
                best_vals.append(min(best_vals[-1], y) if best_vals[-1] != -float('inf') else y)
            best_vals = best_vals[1:]
        else:
            best_vals = [-float('inf')]
            for y in y_values:
                best_vals.append(max(best_vals[-1], y) if best_vals[-1] != -float('inf') else y)
            best_vals = best_vals[1:]
        
        plt.figure(figsize=(12, 6))
        
        # Vẽ giá trị ở mỗi vòng lặp
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(y_values) + 1), y_values, 'o-', label='Observed values')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('Objective Value at Each Iteration')
        plt.grid(True)
        
        # Vẽ giá trị tốt nhất qua các vòng lặp
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(best_vals) + 1), best_vals, 'o-', color='green', label='Best value')
        plt.xlabel('Iteration')
        plt.ylabel('Best objective value')
        plt.title('Best Objective Value Over Iterations')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class HyperParameterOptimizer:
    """
    Tối ưu hóa hyperparameter cho DQN Agent.
    """
    def __init__(self, 
                env_creator: Callable,
                agent_creator: Callable,
                param_ranges: Dict[str, Tuple[float, float]] = None,
                n_episodes_per_trial: int = 200,
                log_dir: str = 'logs/hyperopt',
                n_eval_episodes: int = 20,
                maximize: bool = True,
                device: str = 'auto'):
        """
        Khởi tạo HyperParameterOptimizer.
        
        Args:
            env_creator: Hàm tạo môi trường
            agent_creator: Hàm tạo agent với tham số
            param_ranges: Dictionary chứa phạm vi của các hyperparameter
            n_episodes_per_trial: Số episode cho mỗi lần thử nghiệm
            log_dir: Thư mục lưu logs
            n_eval_episodes: Số episode để đánh giá
            maximize: True nếu tối đa hóa điểm, False nếu tối thiểu hóa loss
            device: Thiết bị sử dụng ('cpu', 'cuda', 'auto')
        """
        self.env_creator = env_creator
        self.agent_creator = agent_creator
        self.n_episodes_per_trial = n_episodes_per_trial
        self.log_dir = log_dir
        self.n_eval_episodes = n_eval_episodes
        self.maximize = maximize
        
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Phạm vi tham số mặc định nếu không được cung cấp
        if param_ranges is None:
            self.param_ranges = {
                'learning_rate': (1e-5, 1e-2),
                'gamma': (0.9, 0.999),
                'tau': (1e-4, 1e-2),
                'alpha': (0.3, 0.9),
                'beta_start': (0.2, 0.6),
                'eps_end': (0.01, 0.2),
                'eps_decay': (0.9, 0.999)
            }
        else:
            self.param_ranges = param_ranges
        
        # Tạo thư mục logs
        os.makedirs(log_dir, exist_ok=True)
        
        # Tạo optimizer
        self.optimizer = BayesianOptimizer(
            param_ranges=self.param_ranges,
            objective_function=self._objective_function,
            log_dir=log_dir,
            maximize=maximize
        )
    
    def optimize(self, n_iterations: int = 20) -> Dict[str, float]:
        """
        Thực hiện tối ưu hóa hyperparameter.
        
        Args:
            n_iterations: Số vòng lặp tối ưu hóa
            
        Returns:
            Dict[str, float]: Hyperparameters tốt nhất
        """
        return self.optimizer.optimize(n_iterations)
    
    def _objective_function(self, params: Dict[str, float]) -> float:
        """
        Hàm mục tiêu để tối ưu hóa.
        
        Args:
            params: Dictionary chứa hyperparameters
            
        Returns:
            float: Giá trị mục tiêu (phần thưởng hoặc negative loss)
        """
        # Tạo môi trường
        env = self.env_creator()
        
        # Tạo agent với params
        agent = self.agent_creator(params, self.device)
        
        # Huấn luyện agent
        total_rewards = []
        
        for episode in range(self.n_episodes_per_trial):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.step(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            # Cập nhật epsilon
            agent.update_epsilon()
        
        # Đánh giá agent
        eval_rewards = []
        
        for _ in range(self.n_eval_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state, 0.0)  # Không khám phá trong đánh giá
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        
        # Log kết quả
        print(f"Params: {params}")
        print(f"Average Training Reward: {np.mean(total_rewards):.2f}")
        print(f"Average Evaluation Reward: {avg_eval_reward:.2f}")
        
        if not self.maximize:
            # Nếu tối thiểu hóa loss, sử dụng negative reward
            return -avg_eval_reward
        
        return avg_eval_reward

def optimize_dqn_hyperparameters(env_creator, agent_creator, param_ranges=None, 
                               n_iterations=20, n_episodes_per_trial=200,
                               log_dir='logs/hyperopt'):
    """
    Hàm helper để tối ưu hóa hyperparameters cho DQN.
    
    Args:
        env_creator: Hàm tạo môi trường
        agent_creator: Hàm tạo agent
        param_ranges: Phạm vi tham số
        n_iterations: Số vòng lặp tối ưu hóa
        n_episodes_per_trial: Số episode cho mỗi lần thử
        log_dir: Thư mục lưu logs
        
    Returns:
        Dict[str, float]: Hyperparameters tốt nhất
    """
    optimizer = HyperParameterOptimizer(
        env_creator=env_creator,
        agent_creator=agent_creator,
        param_ranges=param_ranges,
        n_episodes_per_trial=n_episodes_per_trial,
        log_dir=log_dir
    )
    
    best_params = optimizer.optimize(n_iterations)
    
    # Lưu kết quả ra file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(log_dir, f"best_dqn_params_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Hyperparameters tốt nhất đã được lưu tại {results_path}")
    return best_params 