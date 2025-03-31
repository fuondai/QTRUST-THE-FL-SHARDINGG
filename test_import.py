import sys
sys.path.insert(0, '.')

try:
    print("Thử import DQNAgent...")
    from qtrust.agents.dqn.agent import DQNAgent
    print("✓ Import DQNAgent thành công!")
except Exception as e:
    print(f"✗ Lỗi khi import DQNAgent: {e}")

try:
    print("\nThử import RainbowDQNAgent...")
    from qtrust.agents.dqn.rainbow_agent import RainbowDQNAgent
    print("✓ Import RainbowDQNAgent thành công!")
except Exception as e:
    print(f"✗ Lỗi khi import RainbowDQNAgent: {e}")

try:
    print("\nThử import ActorCriticAgent...")
    from qtrust.agents.dqn.actor_critic_agent import ActorCriticAgent
    print("✓ Import ActorCriticAgent thành công!")
except Exception as e:
    print(f"✗ Lỗi khi import ActorCriticAgent: {e}")

print("\nKiểm tra các dependencies và imports của RainbowDQNAgent:")
try:
    print("Thử import numpy, torch...")
    import numpy as np
    import torch
    print("✓ Import numpy và torch thành công!")
except Exception as e:
    print(f"✗ Lỗi khi import numpy, torch: {e}")

try:
    print("\nThử import CategoricalQNetwork...")
    from qtrust.agents.dqn.networks import CategoricalQNetwork
    print("✓ Import CategoricalQNetwork thành công!")
except Exception as e:
    print(f"✗ Lỗi khi import CategoricalQNetwork: {e}")

try:
    print("\nThử import NStepPrioritizedReplayBuffer...")
    from qtrust.agents.dqn.replay_buffer import NStepPrioritizedReplayBuffer
    print("✓ Import NStepPrioritizedReplayBuffer thành công!")
except Exception as e:
    print(f"✗ Lỗi khi import NStepPrioritizedReplayBuffer: {e}")

print("\nKiểm tra hoàn tất!") 