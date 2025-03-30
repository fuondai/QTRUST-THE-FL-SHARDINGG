"""
Công cụ caching cho ứng dụng QTrust.

Module này cung cấp các lớp và hàm tiện ích để thực hiện caching, giảm thiểu
tính toán trùng lặp và cải thiện hiệu suất của toàn bộ hệ thống.
"""

import time
import functools
import hashlib
import pickle
import logging
from typing import Dict, Any, Callable, Tuple, List, Union, Optional, TypeVar, Generic

import numpy as np
import torch

# Thiết lập logger
logger = logging.getLogger("qtrust.cache")

# Type variable cho các generic
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class LRUCache(Generic[K, V]):
    """
    Triển khai bộ nhớ cache LRU (Least Recently Used).
    
    Lớp này quản lý cache theo thuật toán LRU, loại bỏ các mục được sử dụng lâu nhất
    khi kích thước cache đạt đến giới hạn.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Khởi tạo LRUCache.
        
        Args:
            capacity: Số lượng mục tối đa trong cache
        """
        self.capacity = capacity
        self.cache: Dict[K, V] = {}
        self.usage_order: List[K] = []
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Lấy giá trị từ cache. Cập nhật thứ tự sử dụng.
        
        Args:
            key: Khóa cần truy xuất
            default: Giá trị mặc định nếu khóa không tồn tại
            
        Returns:
            Giá trị được cache hoặc giá trị mặc định
        """
        if key not in self.cache:
            return default
        
        # Cập nhật thứ tự sử dụng
        self.usage_order.remove(key)
        self.usage_order.append(key)
        
        return self.cache[key]
    
    def put(self, key: K, value: V) -> None:
        """
        Thêm hoặc cập nhật giá trị trong cache.
        
        Args:
            key: Khóa
            value: Giá trị
        """
        if key in self.cache:
            # Cập nhật thứ tự sử dụng
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Xóa phần tử ít được sử dụng nhất
            oldest = self.usage_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.usage_order.append(key)
    
    def __contains__(self, key: K) -> bool:
        """Kiểm tra xem khóa có trong cache không."""
        return key in self.cache
    
    def __len__(self) -> int:
        """Số lượng phần tử trong cache."""
        return len(self.cache)
    
    def clear(self) -> None:
        """Xóa tất cả các phần tử trong cache."""
        self.cache.clear()
        self.usage_order.clear()

class TTLCache(Generic[K, V]):
    """
    Triển khai bộ nhớ cache TTL (Time-To-Live).
    
    Lớp này quản lý cache với thời gian sống cho mỗi mục, tự động
    loại bỏ các mục hết hạn.
    """
    
    def __init__(self, ttl: float = 300.0, capacity: int = 1000):
        """
        Khởi tạo TTLCache.
        
        Args:
            ttl: Thời gian sống (giây) cho mỗi mục
            capacity: Số lượng mục tối đa trong cache
        """
        self.ttl = ttl
        self.capacity = capacity
        self.cache: Dict[K, Tuple[V, float]] = {}  # (value, expiration_time)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Lấy giá trị từ cache nếu chưa hết hạn.
        
        Args:
            key: Khóa cần truy xuất
            default: Giá trị mặc định nếu khóa không tồn tại hoặc đã hết hạn
            
        Returns:
            Giá trị được cache hoặc giá trị mặc định
        """
        if key not in self.cache:
            return default
        
        value, expiration_time = self.cache[key]
        
        # Kiểm tra xem mục có hết hạn chưa
        if time.time() > expiration_time:
            del self.cache[key]
            return default
        
        return value
    
    def put(self, key: K, value: V) -> None:
        """
        Thêm hoặc cập nhật giá trị trong cache với thời gian hết hạn mới.
        
        Args:
            key: Khóa
            value: Giá trị
        """
        # Xóa các mục đã hết hạn
        self._clean_expired()
        
        # Kiểm tra xem cache đã đầy chưa
        if len(self.cache) >= self.capacity and key not in self.cache:
            # Xóa mục hết hạn sớm nhất
            self._remove_oldest()
        
        # Tính thời gian hết hạn mới
        expiration_time = time.time() + self.ttl
        
        # Thêm hoặc cập nhật mục trong cache
        self.cache[key] = (value, expiration_time)
    
    def _clean_expired(self) -> None:
        """Xóa tất cả các mục đã hết hạn khỏi cache."""
        current_time = time.time()
        expired_keys = [k for k, (_, exp_time) in self.cache.items() if current_time > exp_time]
        for key in expired_keys:
            del self.cache[key]
    
    def _remove_oldest(self) -> None:
        """Xóa mục có thời gian hết hạn sớm nhất."""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        del self.cache[oldest_key]
    
    def __contains__(self, key: K) -> bool:
        """Kiểm tra xem khóa có trong cache không và chưa hết hạn."""
        if key not in self.cache:
            return False
        
        _, expiration_time = self.cache[key]
        if time.time() > expiration_time:
            del self.cache[key]
            return False
        
        return True
    
    def __len__(self) -> int:
        """Số lượng phần tử trong cache (kể cả các mục đã hết hạn)."""
        return len(self.cache)
    
    def clear(self) -> None:
        """Xóa tất cả các phần tử trong cache."""
        self.cache.clear()

def compute_hash(obj: Any) -> str:
    """
    Tính toán mã hash cho một đối tượng.
    
    Hỗ trợ các đối tượng NumPy và PyTorch.
    
    Args:
        obj: Đối tượng cần tính hash
        
    Returns:
        str: Chuỗi hash
    """
    # Trường hợp đặc biệt cho tensors
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    
    # Trường hợp đặc biệt cho các mảng NumPy
    if isinstance(obj, np.ndarray):
        obj_bytes = obj.tobytes()
    else:
        try:
            # Thử serialize đối tượng
            obj_bytes = pickle.dumps(obj)
        except (pickle.PickleError, TypeError):
            # Fallback để xử lý các đối tượng không thể pickle
            obj_bytes = str(obj).encode('utf-8')
    
    # Tính hash
    return hashlib.md5(obj_bytes).hexdigest()

def lru_cache(maxsize: int = 128):
    """
    Decorator để cache kết quả của hàm với LRU cache.
    
    Args:
        maxsize: Số lượng kết quả tối đa để cache
        
    Returns:
        Decorator
    """
    cache = LRUCache(capacity=maxsize)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Tạo khóa từ tham số
            key_parts = [compute_hash(arg) for arg in args]
            key_parts.extend(f"{k}:{compute_hash(v)}" for k, v in sorted(kwargs.items()))
            key = f"{func.__name__}:{'-'.join(key_parts)}"
            
            # Kiểm tra cache
            if key in cache:
                return cache.get(key)
            
            # Tính toán kết quả
            result = func(*args, **kwargs)
            
            # Lưu trong cache
            cache.put(key, result)
            
            return result
        
        # Thêm tham chiếu tới cache để có thể xóa cache nếu cần
        wrapper.cache = cache
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": cache.capacity}
        wrapper.cache_clear = cache.clear
        
        return wrapper
    
    return decorator

def ttl_cache(ttl: float = 300.0, maxsize: int = 128):
    """
    Decorator để cache kết quả của hàm với TTL cache.
    
    Args:
        ttl: Thời gian sống (giây) cho mỗi mục
        maxsize: Số lượng kết quả tối đa để cache
        
    Returns:
        Decorator
    """
    cache = TTLCache(ttl=ttl, capacity=maxsize)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Tạo khóa từ tham số
            key_parts = [compute_hash(arg) for arg in args]
            key_parts.extend(f"{k}:{compute_hash(v)}" for k, v in sorted(kwargs.items()))
            key = f"{func.__name__}:{'-'.join(key_parts)}"
            
            # Kiểm tra cache
            if key in cache:
                return cache.get(key)
            
            # Tính toán kết quả
            result = func(*args, **kwargs)
            
            # Lưu trong cache
            cache.put(key, result)
            
            return result
        
        # Thêm tham chiếu tới cache để có thể xóa cache nếu cần
        wrapper.cache = cache
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": cache.capacity, "ttl": cache.ttl}
        wrapper.cache_clear = cache.clear
        
        return wrapper
    
    return decorator

def tensor_cache(func: Callable):
    """
    Decorator đặc biệt để cache kết quả tính toán PyTorch Tensor.
    
    Args:
        func: Hàm cần cache
        
    Returns:
        Hàm đã được wrapper
    """
    # Sử dụng LRU cache với kích thước mặc định
    cache = LRUCache(capacity=128)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Chuyển đổi tất cả Tensor thành NumPy để tính hash
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Chỉ sử dụng shape và một phần dữ liệu để tạo hash
                shape = arg.shape
                dtype = str(arg.dtype)
                sample = arg.detach().flatten()[:min(100, arg.numel())].cpu().numpy()
                processed_args.append((shape, dtype, sample))
            else:
                processed_args.append(arg)
        
        processed_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # Tương tự như args
                shape = v.shape
                dtype = str(v.dtype)
                sample = v.detach().flatten()[:min(100, v.numel())].cpu().numpy()
                processed_kwargs[k] = (shape, dtype, sample)
            else:
                processed_kwargs[k] = v
        
        # Tạo khóa từ tham số đã xử lý
        key_parts = [compute_hash(arg) for arg in processed_args]
        key_parts.extend(f"{k}:{compute_hash(v)}" for k, v in sorted(processed_kwargs.items()))
        key = f"{func.__name__}:{'-'.join(key_parts)}"
        
        # Kiểm tra cache
        if key in cache:
            return cache.get(key)
        
        # Tính toán kết quả
        result = func(*args, **kwargs)
        
        # Chỉ cache nếu kết quả là tensor và không yêu cầu gradient
        if isinstance(result, torch.Tensor) and not result.requires_grad:
            cache.put(key, result.clone().detach())
        elif isinstance(result, (list, tuple)) and all(isinstance(r, torch.Tensor) and not r.requires_grad for r in result):
            cloned_result = tuple(r.clone().detach() for r in result)
            cache.put(key, cloned_result)
        else:
            # Không cache kết quả có gradient hoặc không phải tensor
            return result
        
        return result
    
    # Thêm tham chiếu tới cache để có thể xóa cache nếu cần
    wrapper.cache = cache
    wrapper.cache_info = lambda: {"size": len(cache), "maxsize": cache.capacity}
    wrapper.cache_clear = cache.clear
    
    return wrapper 