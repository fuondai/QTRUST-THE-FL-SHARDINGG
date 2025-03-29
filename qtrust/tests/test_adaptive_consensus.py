"""
Bài kiểm thử cho adaptive consensus.
"""

import unittest
import numpy as np
from typing import Dict, List

from qtrust.consensus.adaptive_consensus import (
    ConsensusProtocol,
    FastBFT,
    PBFT,
    RobustBFT,
    AdaptiveConsensus
)

class TestConsensusProtocols(unittest.TestCase):
    """
    Kiểm thử cho các giao thức đồng thuận.
    """
    
    def test_fastbft(self):
        """
        Kiểm thử giao thức FastBFT.
        """
        # Khởi tạo FastBFT
        protocol = FastBFT(latency_factor=0.2, energy_factor=0.2, security_factor=0.5)
        
        # Kiểm tra các thuộc tính
        self.assertEqual(protocol.name, "FastBFT")
        self.assertEqual(protocol.latency_factor, 0.2)
        self.assertEqual(protocol.energy_factor, 0.2)
        self.assertEqual(protocol.security_factor, 0.5)
        
        # Thực thi giao thức với tin cậy cao
        trust_scores = {1: 0.9, 2: 0.85, 3: 0.95}
        result, latency, energy = protocol.execute(transaction_value=10.0, trust_scores=trust_scores)
        
        # Kiểm tra kết quả
        self.assertTrue(result)  # Với tin cậy cao, kết quả phải thành công
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
        
        # Thực thi giao thức với tin cậy thấp
        low_trust_scores = {1: 0.2, 2: 0.3, 3: 0.1}
        result, latency, energy = protocol.execute(transaction_value=10.0, trust_scores=low_trust_scores)
        
        # Với FastBFT và tin cậy thấp, có khả năng thất bại
        # Không kiểm tra result vì tính ngẫu nhiên, nhưng kiểm tra các giá trị khác
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
    
    def test_pbft(self):
        """
        Kiểm thử giao thức PBFT.
        """
        # Khởi tạo PBFT
        protocol = PBFT(latency_factor=0.5, energy_factor=0.5, security_factor=0.8)
        
        # Kiểm tra các thuộc tính
        self.assertEqual(protocol.name, "PBFT")
        self.assertEqual(protocol.latency_factor, 0.5)
        self.assertEqual(protocol.energy_factor, 0.5)
        self.assertEqual(protocol.security_factor, 0.8)
        
        # Thực thi giao thức
        trust_scores = {1: 0.7, 2: 0.6, 3: 0.8}
        result, latency, energy = protocol.execute(transaction_value=30.0, trust_scores=trust_scores)
        
        # Kiểm tra kết quả
        self.assertIsInstance(result, bool)
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
        
        # So sánh với FastBFT, PBFT nên có độ trễ và năng lượng cao hơn
        fast_protocol = FastBFT(latency_factor=0.2, energy_factor=0.2, security_factor=0.5)
        _, fast_latency, fast_energy = fast_protocol.execute(transaction_value=30.0, trust_scores=trust_scores)
        
        # Trong cài đặt, PBFT nên có độ trễ và năng lượng cao hơn
        self.assertGreater(latency, fast_latency)
        self.assertGreater(energy, fast_energy)
    
    def test_robustbft(self):
        """
        Kiểm thử giao thức RobustBFT.
        """
        # Khởi tạo RobustBFT
        protocol = RobustBFT(latency_factor=0.8, energy_factor=0.8, security_factor=0.95)
        
        # Kiểm tra các thuộc tính
        self.assertEqual(protocol.name, "RobustBFT")
        self.assertEqual(protocol.latency_factor, 0.8)
        self.assertEqual(protocol.energy_factor, 0.8)
        self.assertEqual(protocol.security_factor, 0.95)
        
        # Thực thi giao thức
        trust_scores = {1: 0.5, 2: 0.4, 3: 0.6}
        result, latency, energy = protocol.execute(transaction_value=80.0, trust_scores=trust_scores)
        
        # Kiểm tra kết quả
        self.assertIsInstance(result, bool)
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
        
        # So sánh với PBFT, RobustBFT nên có độ trễ và năng lượng cao hơn
        pbft_protocol = PBFT(latency_factor=0.5, energy_factor=0.5, security_factor=0.8)
        _, pbft_latency, pbft_energy = pbft_protocol.execute(transaction_value=80.0, trust_scores=trust_scores)
        
        # Trong cài đặt, RobustBFT nên có độ trễ và năng lượng cao hơn
        self.assertGreater(latency, pbft_latency)
        self.assertGreater(energy, pbft_energy)

class TestAdaptiveConsensus(unittest.TestCase):
    """
    Kiểm thử cho hệ thống Adaptive Consensus.
    """
    
    def setUp(self):
        """
        Thiết lập trước mỗi bài kiểm thử.
        """
        self.consensus = AdaptiveConsensus(
            transaction_threshold_low=10.0,
            transaction_threshold_high=50.0,
            congestion_threshold=0.7,
            min_trust_threshold=0.3
        )
        
    def test_initialization(self):
        """
        Kiểm thử khởi tạo Adaptive Consensus.
        """
        # Kiểm tra các thuộc tính
        self.assertEqual(self.consensus.transaction_threshold_low, 10.0)
        self.assertEqual(self.consensus.transaction_threshold_high, 50.0)
        self.assertEqual(self.consensus.congestion_threshold, 0.7)
        self.assertEqual(self.consensus.min_trust_threshold, 0.3)
        
        # Kiểm tra các giao thức đã được khởi tạo
        self.assertIsInstance(self.consensus.fast_bft, FastBFT)
        self.assertIsInstance(self.consensus.pbft, PBFT)
        self.assertIsInstance(self.consensus.robust_bft, RobustBFT)
    
    def test_select_protocol_by_value(self):
        """
        Kiểm thử lựa chọn giao thức dựa trên giá trị giao dịch.
        """
        # Cài đặt trust scores cao để loại bỏ ảnh hưởng
        trust_scores = {1: 0.9, 2: 0.9, 3: 0.9}
        congestion = 0.3  # Thấp
        
        # Giá trị thấp -> FastBFT
        low_value = 5.0
        protocol = self.consensus.select_protocol(low_value, congestion, trust_scores)
        self.assertIsInstance(protocol, FastBFT)
        
        # Giá trị trung bình -> PBFT
        medium_value = 30.0
        protocol = self.consensus.select_protocol(medium_value, congestion, trust_scores)
        self.assertIsInstance(protocol, PBFT)
        
        # Giá trị cao -> RobustBFT
        high_value = 80.0
        protocol = self.consensus.select_protocol(high_value, congestion, trust_scores)
        self.assertIsInstance(protocol, RobustBFT)
    
    def test_select_protocol_by_congestion(self):
        """
        Kiểm thử lựa chọn giao thức dựa trên mức độ tắc nghẽn.
        """
        # Cài đặt giá trị trung bình để tập trung vào congestion
        value = 30.0
        trust_scores = {1: 0.9, 2: 0.9, 3: 0.9}
        
        # Tắc nghẽn thấp -> PBFT (mặc định cho giá trị trung bình)
        low_congestion = 0.2
        protocol = self.consensus.select_protocol(value, low_congestion, trust_scores)
        self.assertIsInstance(protocol, PBFT)
        
        # Tắc nghẽn cao -> FastBFT (để giảm độ trễ)
        high_congestion = 0.9
        protocol = self.consensus.select_protocol(value, high_congestion, trust_scores)
        self.assertIsInstance(protocol, FastBFT)
    
    def test_select_protocol_by_trust(self):
        """
        Kiểm thử lựa chọn giao thức dựa trên tin cậy.
        """
        # Cài đặt giá trị trung bình và tắc nghẽn thấp
        value = 30.0
        congestion = 0.3
        
        # Tin cậy cao -> PBFT (mặc định cho giá trị trung bình)
        high_trust_scores = {1: 0.9, 2: 0.85, 3: 0.95}
        protocol = self.consensus.select_protocol(value, congestion, high_trust_scores)
        self.assertIsInstance(protocol, PBFT)
        
        # Tin cậy thấp -> RobustBFT (để tăng cường bảo mật)
        low_trust_scores = {1: 0.2, 2: 0.15, 3: 0.25}
        protocol = self.consensus.select_protocol(value, congestion, low_trust_scores)
        self.assertIsInstance(protocol, RobustBFT)
    
    def test_execute_consensus(self):
        """
        Kiểm thử thực thi đồng thuận.
        """
        # Cài đặt các tham số
        value = 30.0
        congestion = 0.3
        trust_scores = {1: 0.8, 2: 0.7, 3: 0.9}
        
        # Thực hiện đồng thuận
        result, protocol_name, latency, energy = self.consensus.execute_consensus(
            value, congestion, trust_scores
        )
        
        # Kiểm tra kết quả
        self.assertIsInstance(result, bool)
        self.assertIsInstance(protocol_name, str)
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
        
        # Chạy nhiều lần để kiểm tra tính ổn định
        results = []
        for _ in range(10):
            res, _, _, _ = self.consensus.execute_consensus(value, congestion, trust_scores)
            results.append(res)
            
        # Với tin cậy cao, ít nhất một số lần chạy nên thành công
        self.assertGreaterEqual(sum(results), 5)  # Ít nhất 5/10 lần thành công
    
    def test_get_protocol_factors(self):
        """
        Kiểm thử lấy các factor của giao thức.
        """
        # Lấy factors của FastBFT
        latency, energy, security = self.consensus.get_protocol_factors("FastBFT")
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertIsInstance(security, float)
        
        # Kiểm tra giá trị
        self.assertEqual(latency, self.consensus.fast_bft.latency_factor)
        self.assertEqual(energy, self.consensus.fast_bft.energy_factor)
        self.assertEqual(security, self.consensus.fast_bft.security_factor)
        
        # Thử với giao thức không tồn tại
        with self.assertRaises(ValueError):
            self.consensus.get_protocol_factors("NonExistentProtocol")
    
if __name__ == '__main__':
    unittest.main() 