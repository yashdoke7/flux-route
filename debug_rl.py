import torch
import numpy as np
from environment.env import RoutingEnv
from environment.simulator.network import GLOBAL_MAX_DEGREE
from agent.train_dqn import load_policy
from environment.models import Action
import os

def debug():
    env = RoutingEnv()
    checkpoint_path = 'agent/checkpoints/policy_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Train first with:")
        print("  python -m agent.train_dqn --episodes 3000")
        return

    model = load_policy(checkpoint_path)
    model.eval()

    print(f"Model obs_dim={model.fc1.in_features}, action_dim={model.max_degree}")
    print(f"Env obs_size={env.observation_size}")

    if model.fc1.in_features != env.observation_size:
        print(f"ERROR: Model expects {model.fc1.in_features} dims but env produces {env.observation_size}")
        print("You need to retrain. Old checkpoints are incompatible.")
        return

    obs = env.reset(task_id='easy_static_mesh', seed=42)
    pkt = env._current_packet
    print(f"Initial: Source={pkt.source}, Dest={pkt.destination}")
    print(f"Distance: {env._network.get_distance(pkt.source, pkt.destination)} hops")

    for i in range(50):
        flat_obs = env.obs_to_flat(obs)
        mask = obs.action_mask

        action_idx = model.select_action(flat_obs, mask, epsilon=0.0)

        with torch.no_grad():
            state_v = torch.FloatTensor(flat_obs).unsqueeze(0)
            mask_v = torch.FloatTensor(mask).unsqueeze(0)
            q_values = model.forward(state_v, mask_v)

        action_val = q_values[0][action_idx].item()
        step_result = env.step(Action(next_hop_index=action_idx))

        pkt = env._current_packet
        curr_node = pkt.source if pkt else "Done"
        dist = env._network.get_distance(pkt.source, pkt.destination) if pkt else 0

        print(
            f"Step {i:2d}: Action={action_idx} (Q={action_val:.2f}) | "
            f"Node={curr_node}, Dist={dist}, Reward={step_result.reward:.2f} "
            f"{'DELIVERED' if step_result.info.delivered else ''}"
            f"{'DROPPED' if step_result.info.drop_occurred else ''}"
        )

        obs = step_result.observation
        if step_result.done:
            print(f"Episode finished")
            break

if __name__ == "__main__":
    debug()
