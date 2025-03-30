import unittest
import random
import time
import numpy as np
from qtrust.consensus.adaptive_pos import AdaptivePoSManager, ValidatorStakeInfo
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus


class TestValidatorStakeInfo(unittest.TestCase):
    """Kiểm thử lớp ValidatorStakeInfo."""
    
    def test_init(self):
        """Kiểm tra khởi tạo validator."""
        validator = ValidatorStakeInfo(id=1, initial_stake=100.0, max_energy=200.0)
        self.assertEqual(validator.id, 1)
        self.assertEqual(validator.stake, 100.0)
        self.assertEqual(validator.max_energy, 200.0)
        self.assertEqual(validator.current_energy, 200.0)
        self.assertTrue(validator.active)
    
    def test_update_stake(self):
        """Kiểm tra cập nhật stake."""
        validator = ValidatorStakeInfo(id=1, initial_stake=100.0)
        validator.update_stake(50.0)
        self.assertEqual(validator.stake, 150.0)
        validator.update_stake(-200.0)
        self.assertEqual(validator.stake, 0.0)  # Không âm
    
    def test_consume_energy(self):
        """Kiểm tra tiêu thụ năng lượng."""
        validator = ValidatorStakeInfo(id=1, initial_stake=100.0, max_energy=100.0)
        self.assertTrue(validator.consume_energy(50.0))
        self.assertEqual(validator.current_energy, 50.0)
        self.assertTrue(validator.consume_energy(50.0))
        self.assertEqual(validator.current_energy, 0.0)
        self.assertFalse(validator.consume_energy(10.0))
        self.assertEqual(validator.current_energy, 0.0)
    
    def test_recharge_energy(self):
        """Kiểm tra nạp lại năng lượng."""
        validator = ValidatorStakeInfo(id=1, max_energy=100.0)
        validator.consume_energy(80.0)
        self.assertEqual(validator.current_energy, 20.0)
        validator.recharge_energy(30.0)
        self.assertEqual(validator.current_energy, 50.0)
        validator.recharge_energy(None)  # Nạp đầy
        self.assertEqual(validator.current_energy, 100.0)
    
    def test_update_performance(self):
        """Kiểm tra cập nhật hiệu suất."""
        validator = ValidatorStakeInfo(id=1)
        initial_score = validator.performance_score
        validator.update_performance(True)  # Thành công
        self.assertGreater(validator.performance_score, initial_score)
        self.assertEqual(validator.successful_validations, 1)
        
        high_score = validator.performance_score
        validator.update_performance(False)  # Thất bại
        self.assertLess(validator.performance_score, high_score)
        self.assertEqual(validator.failed_validations, 1)
    
    def test_average_energy_consumption(self):
        """Kiểm tra tính toán mức tiêu thụ năng lượng trung bình."""
        validator = ValidatorStakeInfo(id=1, max_energy=100.0)
        self.assertEqual(validator.get_average_energy_consumption(), 0.0)
        
        validator.consume_energy(10.0)
        validator.consume_energy(20.0)
        self.assertEqual(validator.get_average_energy_consumption(), 15.0)


class TestAdaptivePoSManager(unittest.TestCase):
    """Kiểm thử lớp AdaptivePoSManager."""
    
    def setUp(self):
        """Chuẩn bị môi trường kiểm thử."""
        self.pos_manager = AdaptivePoSManager(
            num_validators=20,
            active_validator_ratio=0.7,
            rotation_period=50,
            min_stake=10.0,
            energy_threshold=30.0,
            performance_threshold=0.3,
            seed=42
        )
    
    def test_init(self):
        """Kiểm tra khởi tạo PoS manager."""
        self.assertEqual(len(self.pos_manager.validators), 20)
        self.assertEqual(self.pos_manager.num_active_validators, 14)  # 70% của 20
        self.assertEqual(len(self.pos_manager.active_validators), 14)
    
    def test_select_validator_for_block(self):
        """Kiểm tra chọn validator cho block."""
        selected = self.pos_manager.select_validator_for_block()
        self.assertIsNotNone(selected)
        self.assertIn(selected, self.pos_manager.active_validators)
        
        # Kiểm tra với trust scores
        trust_scores = {v_id: random.random() for v_id in self.pos_manager.validators}
        selected_with_trust = self.pos_manager.select_validator_for_block(trust_scores)
        self.assertIsNotNone(selected_with_trust)
        self.assertIn(selected_with_trust, self.pos_manager.active_validators)
    
    def test_select_validators_for_committee(self):
        """Kiểm tra chọn committee validators."""
        committee = self.pos_manager.select_validators_for_committee(5)
        self.assertEqual(len(committee), 5)
        for v_id in committee:
            self.assertIn(v_id, self.pos_manager.active_validators)
        
        # Kiểm tra với committee_size lớn
        large_committee = self.pos_manager.select_validators_for_committee(30)
        self.assertLessEqual(len(large_committee), len(self.pos_manager.active_validators))
    
    def test_update_validator_energy(self):
        """Kiểm tra cập nhật năng lượng validator."""
        validator_id = next(iter(self.pos_manager.active_validators))
        initial_energy = self.pos_manager.validators[validator_id].current_energy
        initial_stake = self.pos_manager.validators[validator_id].stake
        
        # Cập nhật với giao dịch thành công
        self.pos_manager.update_validator_energy(validator_id, 10.0, True)
        self.assertLess(self.pos_manager.validators[validator_id].current_energy, initial_energy)
        self.assertGreater(self.pos_manager.validators[validator_id].stake, initial_stake)
        
        # Cập nhật validator không tồn tại (không lỗi)
        self.pos_manager.update_validator_energy(999, 10.0, True)
    
    def test_rotate_validators(self):
        """Kiểm tra luân chuyển validator."""
        # Khởi tạo PoS mới với rotation_period nhỏ để dễ kiểm thử
        pos = AdaptivePoSManager(
            num_validators=20,
            rotation_period=1,  # Luân chuyển mỗi vòng
            energy_threshold=30.0
        )
        
        # Khiến một số validator có năng lượng thấp
        for v_id in list(pos.active_validators)[:3]:
            pos.validators[v_id].consume_energy(80.0)  # Năng lượng còn 20
        
        # Thực hiện luân chuyển
        rotations = pos.rotate_validators()
        self.assertGreaterEqual(rotations, 1)  # Ít nhất một validator được luân chuyển
    
    def test_update_energy_recharge(self):
        """Kiểm tra nạp lại năng lượng cho validators."""
        # Khiến một số validator không hoạt động
        active_ids = list(self.pos_manager.active_validators)
        for v_id in active_ids[:3]:
            self.pos_manager.validators[v_id].active = False
            self.pos_manager.active_validators.remove(v_id)
            self.pos_manager.validators[v_id].current_energy = 10.0
        
        # Nạp năng lượng
        self.pos_manager.update_energy_recharge(0.1)  # 10% mỗi lần
        
        # Kiểm tra validator không hoạt động đã được nạp năng lượng
        for v_id in active_ids[:3]:
            self.assertGreater(self.pos_manager.validators[v_id].current_energy, 10.0)
    
    def test_get_energy_statistics(self):
        """Kiểm tra lấy thống kê năng lượng."""
        stats = self.pos_manager.get_energy_statistics()
        self.assertIn("total_energy", stats)
        self.assertIn("active_energy", stats)
        self.assertIn("inactive_energy", stats)
        self.assertIn("avg_energy", stats)
        self.assertIn("avg_active_energy", stats)
        self.assertIn("energy_saved", stats)
    
    def test_get_validator_statistics(self):
        """Kiểm tra lấy thống kê validator."""
        stats = self.pos_manager.get_validator_statistics()
        self.assertEqual(stats["total_validators"], 20)
        self.assertEqual(stats["active_validators"], 14)
        self.assertEqual(stats["inactive_validators"], 6)
    
    def test_simulate_round(self):
        """Kiểm tra mô phỏng một round của PoS."""
        result = self.pos_manager.simulate_round()
        self.assertIn("success", result)
        self.assertIn("validator", result)
        self.assertIn("stake", result)
        self.assertIn("energy_consumed", result)
        self.assertIn("performance_score", result)


class TestAdaptiveConsensusWithPoS(unittest.TestCase):
    """Kiểm thử tích hợp AdaptiveConsensus với AdaptivePoS."""
    
    def setUp(self):
        """Chuẩn bị môi trường kiểm thử."""
        self.consensus = AdaptiveConsensus(
            enable_adaptive_pos=True,
            num_validators_per_shard=10,
            active_validator_ratio=0.6,
            rotation_period=5
        )
    
    def test_execute_consensus_with_pos(self):
        """Kiểm tra thực hiện đồng thuận với AdaptivePoS."""
        # Tạo trust scores mô phỏng
        trust_scores = {i: random.random() for i in range(1, 21)}
        
        # Thực hiện đồng thuận
        result, protocol, latency, energy = self.consensus.execute_consensus(
            transaction_value=20.0,
            congestion=0.3,
            trust_scores=trust_scores,
            shard_id=0  # Shard ID 0
        )
        
        # Kiểm tra kết quả trả về
        self.assertIsInstance(result, bool)
        self.assertIsInstance(protocol, str)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
    
    def test_pos_statistics(self):
        """Kiểm tra thống kê về AdaptivePoS."""
        # Thực hiện một số đồng thuận
        trust_scores = {i: random.random() for i in range(1, 21)}
        for _ in range(10):
            self.consensus.execute_consensus(
                transaction_value=15.0,
                congestion=0.4,
                trust_scores=trust_scores,
                shard_id=0
            )
        
        # Lấy thống kê
        stats = self.consensus.get_pos_statistics()
        
        # Kiểm tra cấu trúc thống kê
        self.assertTrue(stats["enabled"])
        self.assertIn("total_energy_saved", stats)
        self.assertIn("total_rotations", stats)
        self.assertIn("shard_stats", stats)
        self.assertIn(0, stats["shard_stats"])
    
    def test_select_committee_for_shard(self):
        """Kiểm tra chọn ủy ban validator cho một shard."""
        committee = self.consensus.select_committee_for_shard(
            shard_id=0,
            committee_size=5,
            trust_scores={i: random.random() for i in range(1, 21)}
        )
        
        # Kiểm tra kết quả
        self.assertEqual(len(committee), 5)
        for v_id in committee:
            self.assertGreaterEqual(v_id, 1)
            self.assertLessEqual(v_id, 10)  # num_validators_per_shard = 10


if __name__ == "__main__":
    unittest.main() 