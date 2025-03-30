import random
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any
import os
import sys
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.consensus.adaptive_pos import AdaptivePoSManager


class EnergySimulation:
    """
    Mô phỏng để kiểm tra hiệu quả tiết kiệm năng lượng của Adaptive PoS.
    """
    def __init__(self, 
                 simulation_rounds: int = 1000,
                 num_shards: int = 5,
                 validators_per_shard: int = 15,
                 active_ratio_pos: float = 0.7,
                 rotation_period: int = 50,
                 transaction_rate: float = 10.0,
                 plot_results: bool = True,
                 save_dir: str = "results"):
        """
        Khởi tạo mô phỏng.
        
        Args:
            simulation_rounds: Số vòng mô phỏng
            num_shards: Số lượng shard
            validators_per_shard: Số lượng validator mỗi shard
            active_ratio_pos: Tỷ lệ validator hoạt động trong Adaptive PoS
            rotation_period: Chu kỳ luân chuyển validator
            transaction_rate: Tốc độ giao dịch trung bình (giao dịch/vòng)
            plot_results: Vẽ biểu đồ kết quả
            save_dir: Thư mục lưu kết quả
        """
        self.simulation_rounds = simulation_rounds
        self.num_shards = num_shards
        self.validators_per_shard = validators_per_shard
        self.active_ratio_pos = active_ratio_pos
        self.rotation_period = rotation_period
        self.transaction_rate = transaction_rate
        self.plot_results = plot_results
        self.save_dir = save_dir
        
        # Khởi tạo cơ chế đồng thuận
        self.adaptive_consensus = AdaptiveConsensus(
            num_validators_per_shard=validators_per_shard,
            enable_adaptive_pos=True,
            active_validator_ratio=active_ratio_pos,
            rotation_period=rotation_period
        )
        
        self.standard_consensus = AdaptiveConsensus(
            num_validators_per_shard=validators_per_shard,
            enable_adaptive_pos=False
        )
        
        # Khởi tạo trust scores cho mỗi validator trong mỗi shard
        self.trust_scores = {}
        for shard in range(num_shards):
            for validator in range(1, validators_per_shard + 1):
                # Validator ID: shard_id * 100 + validator_id
                validator_id = shard * 100 + validator
                self.trust_scores[validator_id] = 0.5 + random.random() * 0.5  # 0.5-1.0
        
        # Khởi tạo mức độ tắc nghẽn cho mỗi shard
        self.congestion_levels = {shard: 0.2 + random.random() * 0.3 for shard in range(num_shards)}
        
        # Biến theo dõi kết quả
        self.results = {
            "rounds": [],
            "adaptive_pos_energy": [],
            "standard_energy": [],
            "adaptive_pos_success_rate": [],
            "standard_success_rate": [],
            "energy_saved": [],
            "rotations": [],
            "active_validators": []
        }
        
        # Tạo thư mục lưu kết quả nếu chưa tồn tại
        if plot_results and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def simulate_transaction_batch(self, consensus, shard_id: int, num_transactions: int) -> Dict[str, Any]:
        """
        Mô phỏng một loạt giao dịch trên một shard.
        
        Args:
            consensus: Đối tượng AdaptiveConsensus
            shard_id: ID của shard
            num_transactions: Số lượng giao dịch cần xử lý
            
        Returns:
            Dict[str, Any]: Kết quả mô phỏng
        """
        # Lọc trust scores cho validators trong shard này
        shard_validators = {v_id: score for v_id, score in self.trust_scores.items() 
                           if v_id // 100 == shard_id}
        
        total_energy = 0.0
        successful_txs = 0
        
        for _ in range(num_transactions):
            # Tạo một giao dịch ngẫu nhiên
            tx_value = random.expovariate(1.0 / 20.0)  # Giá trị giao dịch trung bình 20
            is_cross_shard = random.random() < 0.3  # 30% giao dịch xuyên shard
            
            # Thực hiện đồng thuận
            result, protocol, latency, energy = consensus.execute_consensus(
                transaction_value=tx_value,
                congestion=self.congestion_levels[shard_id],
                trust_scores=shard_validators,
                network_stability=0.7,
                cross_shard=is_cross_shard,
                shard_id=shard_id
            )
            
            # Ghi nhận kết quả
            total_energy += energy
            if result:
                successful_txs += 1
        
        return {
            "total_energy": total_energy,
            "successful_txs": successful_txs,
            "total_txs": num_transactions,
            "success_rate": successful_txs / num_transactions if num_transactions > 0 else 0
        }
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Chạy mô phỏng cho toàn bộ hệ thống.
        
        Returns:
            Dict[str, Any]: Kết quả mô phỏng
        """
        print(f"Bắt đầu mô phỏng với {self.simulation_rounds} vòng...")
        
        for round_num in range(1, self.simulation_rounds + 1):
            # Cập nhật mức độ tắc nghẽn (có dao động theo thời gian)
            for shard in range(self.num_shards):
                # Dao động 10% mỗi vòng
                delta = (random.random() - 0.5) * 0.1
                self.congestion_levels[shard] = max(0.1, min(0.9, self.congestion_levels[shard] + delta))
            
            # Tính số giao dịch cho mỗi shard (dựa trên mức độ tắc nghẽn)
            transactions_per_shard = {
                shard: int(self.transaction_rate * (0.8 + self.congestion_levels[shard]))
                for shard in range(self.num_shards)
            }
            
            # Mô phỏng với Adaptive PoS
            adaptive_pos_results = {
                shard: self.simulate_transaction_batch(
                    self.adaptive_consensus, shard, transactions_per_shard[shard]
                ) for shard in range(self.num_shards)
            }
            
            # Mô phỏng với cơ chế tiêu chuẩn (không dùng Adaptive PoS)
            standard_results = {
                shard: self.simulate_transaction_batch(
                    self.standard_consensus, shard, transactions_per_shard[shard]
                ) for shard in range(self.num_shards)
            }
            
            # Tính tổng năng lượng và tỷ lệ thành công
            adaptive_pos_energy = sum(r["total_energy"] for r in adaptive_pos_results.values())
            standard_energy = sum(r["total_energy"] for r in standard_results.values())
            
            adaptive_pos_success = sum(r["successful_txs"] for r in adaptive_pos_results.values())
            standard_success = sum(r["successful_txs"] for r in standard_results.values())
            
            total_txs = sum(r["total_txs"] for r in adaptive_pos_results.values())
            
            adaptive_pos_success_rate = adaptive_pos_success / total_txs if total_txs > 0 else 0
            standard_success_rate = standard_success / total_txs if total_txs > 0 else 0
            
            # Lấy thống kê từ Adaptive PoS
            pos_stats = self.adaptive_consensus.get_pos_statistics()
            
            # Tính tổng số validator hoạt động qua tất cả các shard
            active_validators = 0
            for shard_id in range(self.num_shards):
                if shard_id in pos_stats["shard_stats"]:
                    active_validators += pos_stats["shard_stats"][shard_id]["validators"]["active_validators"]
            
            # Lưu kết quả
            self.results["rounds"].append(round_num)
            self.results["adaptive_pos_energy"].append(adaptive_pos_energy)
            self.results["standard_energy"].append(standard_energy)
            self.results["adaptive_pos_success_rate"].append(adaptive_pos_success_rate)
            self.results["standard_success_rate"].append(standard_success_rate)
            self.results["energy_saved"].append(pos_stats["total_energy_saved"])
            self.results["rotations"].append(pos_stats["total_rotations"])
            self.results["active_validators"].append(active_validators)
            
            # In tiến độ
            if round_num % 100 == 0 or round_num == 1:
                energy_saving = (1 - adaptive_pos_energy / standard_energy) * 100 if standard_energy > 0 else 0
                print(f"Vòng {round_num}/{self.simulation_rounds}: Tiết kiệm năng lượng {energy_saving:.2f}%, "
                      f"Số validator hoạt động: {active_validators}/{self.num_shards * self.validators_per_shard}")
        
        # Vẽ biểu đồ kết quả
        if self.plot_results:
            self.plot_simulation_results()
        
        # Tính tổng kết quả
        total_adaptive_energy = sum(self.results["adaptive_pos_energy"])
        total_standard_energy = sum(self.results["standard_energy"])
        energy_saving_percent = (1 - total_adaptive_energy / total_standard_energy) * 100 if total_standard_energy > 0 else 0
        
        avg_adaptive_success = np.mean(self.results["adaptive_pos_success_rate"]) * 100
        avg_standard_success = np.mean(self.results["standard_success_rate"]) * 100
        
        final_results = {
            "total_rounds": self.simulation_rounds,
            "total_adaptive_energy": total_adaptive_energy,
            "total_standard_energy": total_standard_energy,
            "energy_saving_percent": energy_saving_percent,
            "avg_adaptive_success": avg_adaptive_success,
            "avg_standard_success": avg_standard_success,
            "total_pos_energy_saved": self.results["energy_saved"][-1],
            "total_rotations": self.results["rotations"][-1],
            "final_active_validators": self.results["active_validators"][-1]
        }
        
        print("\nKết quả mô phỏng:")
        print(f"Tổng số vòng: {final_results['total_rounds']}")
        print(f"Tổng năng lượng (Adaptive PoS): {final_results['total_adaptive_energy']:.2f}")
        print(f"Tổng năng lượng (Standard): {final_results['total_standard_energy']:.2f}")
        print(f"Tiết kiệm năng lượng: {final_results['energy_saving_percent']:.2f}%")
        print(f"Tỷ lệ thành công (Adaptive PoS): {final_results['avg_adaptive_success']:.2f}%")
        print(f"Tỷ lệ thành công (Standard): {final_results['avg_standard_success']:.2f}%")
        print(f"Tổng số lần luân chuyển validator: {final_results['total_rotations']}")
        print(f"Số validator hoạt động cuối cùng: {final_results['final_active_validators']}")
        
        return final_results
    
    def plot_simulation_results(self):
        """Vẽ biểu đồ kết quả mô phỏng."""
        # Tạo figure với 4 subplots
        plt.figure(figsize=(16, 14))
        
        # 1. Biểu đồ so sánh năng lượng tiêu thụ
        plt.subplot(2, 2, 1)
        plt.plot(self.results["rounds"], self.results["adaptive_pos_energy"], 
                 label="Adaptive PoS", color="green")
        plt.plot(self.results["rounds"], self.results["standard_energy"], 
                 label="Standard", color="red")
        plt.xlabel("Vòng mô phỏng")
        plt.ylabel("Năng lượng tiêu thụ")
        plt.title("So sánh năng lượng tiêu thụ")
        plt.legend()
        plt.grid(True)
        
        # 2. Biểu đồ năng lượng tiết kiệm được
        plt.subplot(2, 2, 2)
        plt.plot(self.results["rounds"], self.results["energy_saved"], color="blue")
        plt.xlabel("Vòng mô phỏng")
        plt.ylabel("Năng lượng tiết kiệm")
        plt.title("Năng lượng tiết kiệm được (Adaptive PoS)")
        plt.grid(True)
        
        # 3. Biểu đồ số lần luân chuyển
        plt.subplot(2, 2, 3)
        plt.plot(self.results["rounds"], self.results["rotations"], color="purple")
        plt.xlabel("Vòng mô phỏng")
        plt.ylabel("Số lần luân chuyển")
        plt.title("Tổng số lần luân chuyển validator")
        plt.grid(True)
        
        # 4. Biểu đồ tỷ lệ thành công
        plt.subplot(2, 2, 4)
        plt.plot(self.results["rounds"], [rate * 100 for rate in self.results["adaptive_pos_success_rate"]], 
                 label="Adaptive PoS", color="green")
        plt.plot(self.results["rounds"], [rate * 100 for rate in self.results["standard_success_rate"]], 
                 label="Standard", color="red")
        plt.xlabel("Vòng mô phỏng")
        plt.ylabel("Tỷ lệ thành công (%)")
        plt.title("So sánh tỷ lệ thành công")
        plt.legend()
        plt.grid(True)
        
        # Lưu biểu đồ
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "energy_optimization_results.png"))
        plt.close()
        
        print(f"Đã lưu biểu đồ kết quả vào {self.save_dir}/energy_optimization_results.png")


def main():
    """Chạy mô phỏng từ dòng lệnh."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mô phỏng tối ưu hóa năng lượng cho QTrust')
    parser.add_argument('--rounds', type=int, default=200, help='Số vòng mô phỏng')
    parser.add_argument('--shards', type=int, default=2, help='Số lượng shard')
    parser.add_argument('--validators', type=int, default=10, help='Số validator mỗi shard')
    parser.add_argument('--active-ratio', type=float, default=0.7, help='Tỷ lệ validator hoạt động')
    parser.add_argument('--rotation-period', type=int, default=20, help='Chu kỳ luân chuyển validator')
    parser.add_argument('--tx-rate', type=float, default=10.0, help='Tốc độ giao dịch')
    parser.add_argument('--no-plot', action='store_true', help='Không vẽ biểu đồ')
    parser.add_argument('--save-dir', type=str, default='results', help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Chạy mô phỏng
    sim = EnergySimulation(
        simulation_rounds=args.rounds,
        num_shards=args.shards,
        validators_per_shard=args.validators,
        active_ratio_pos=args.active_ratio,
        rotation_period=args.rotation_period,
        transaction_rate=args.tx_rate,
        plot_results=not args.no_plot,
        save_dir=args.save_dir
    )
    
    results = sim.run_simulation()
    return results


if __name__ == "__main__":
    main() 