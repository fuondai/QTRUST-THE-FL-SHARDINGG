"""
Module cung cấp hệ thống logging chuẩn cho QTrust.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import io
from pathlib import Path
from typing import Optional, Union, Dict, Any
import datetime

# Thiết lập thư mục logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Định dạng log tiêu chuẩn
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Mức độ log mặc định
DEFAULT_LOG_LEVEL = logging.INFO


class CustomFormatter(logging.Formatter):
    """
    Formatter tùy chỉnh hỗ trợ màu sắc cho console output.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    detailed_fmt = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: blue + detailed_fmt + reset,
        logging.INFO: green + log_fmt + reset,
        logging.WARNING: yellow + detailed_fmt + reset,
        logging.ERROR: red + detailed_fmt + reset,
        logging.CRITICAL: bold_red + detailed_fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class UTF8StreamHandler(logging.StreamHandler):
    """
    StreamHandler tùy chỉnh hỗ trợ UTF-8 encoding cho Windows
    """
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
        self.stream = io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='backslashreplace')

    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.stream.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, 
               level: int = DEFAULT_LOG_LEVEL, 
               console: bool = True, 
               file: bool = True,
               log_dir: Optional[Union[str, Path]] = None, 
               log_format: str = DEFAULT_FORMAT,
               max_file_size_mb: int = 10,
               backup_count: int = 5) -> logging.Logger:
    """
    Tạo và cấu hình logger.
    
    Args:
        name: Tên của logger
        level: Cấp độ logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Có log ra console hay không
        file: Có log ra file hay không
        log_dir: Thư mục lưu file log, nếu None sẽ dùng LOG_DIR
        log_format: Định dạng log
        max_file_size_mb: Kích thước tối đa của file log (MB)
        backup_count: Số lượng file log backup giữ lại
        
    Returns:
        Logger đã được cấu hình
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Tránh thêm handler trùng lặp
    if logger.handlers:
        return logger
    
    # Tạo log directory nếu không tồn tại
    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
    
    # Thêm console handler
    if console:
        # Sử dụng UTF8StreamHandler thay vì StreamHandler mặc định
        console_handler = UTF8StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    # Thêm file handler
    if file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# Tạo logger cho toàn bộ ứng dụng
app_logger = get_logger("qtrust")
simulation_logger = get_logger("qtrust.simulation")
consensus_logger = get_logger("qtrust.consensus")
dqn_logger = get_logger("qtrust.dqn")
trust_logger = get_logger("qtrust.trust")
federated_logger = get_logger("qtrust.federated") 