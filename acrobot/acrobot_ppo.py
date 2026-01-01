import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


ENV_ID = "Acrobot-v1"
Total_Timesteps = 300_000   
Num_Envs = 8                
Learning_Rate = 3e-4
Save_Dir = "weights"
os.makedirs(Save_Dir, exist_ok=True)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

def save_ppo_actor_as_pt(sb3_model, save_path, obs_dim, act_dim):
    custom_model = QNetwork(obs_dim, act_dim)
    
    with torch.no_grad():
        # Layer 1
        custom_model.net[0].weight.data = sb3_model.policy.mlp_extractor.policy_net[0].weight.data.clone()
        custom_model.net[0].bias.data = sb3_model.policy.mlp_extractor.policy_net[0].bias.data.clone()
        # Layer 2
        custom_model.net[2].weight.data = sb3_model.policy.mlp_extractor.policy_net[2].weight.data.clone()
        custom_model.net[2].bias.data = sb3_model.policy.mlp_extractor.policy_net[2].bias.data.clone()
        # Output Layer (Action Net)
        custom_model.net[4].weight.data = sb3_model.policy.action_net.weight.data.clone()
        custom_model.net[4].bias.data = sb3_model.policy.action_net.bias.data.clone()

    torch.save(custom_model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

class CheckpointCallback(BaseCallback):
    def __init__(self, env_id, obs_dim, act_dim, verbose=1):
        super().__init__(verbose)
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.medium_saved = False
        self.expert_saved = False
        self.reward_history = [] 

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                ep_rew = info['episode']['r']
                self.reward_history.append(ep_rew)
                
                if len(self.reward_history) >= 50:
                    mean_reward = np.mean(self.reward_history[-50:])
                    
                    # Medium 
                    if not self.medium_saved and mean_reward > -200:
                        print(f"\nMedium (Avg: {mean_reward:.1f})")
                        save_path = f"{Save_Dir}/{self.env_id}_medium.pt"
                        save_ppo_actor_as_pt(self.model, save_path, self.obs_dim, self.act_dim)
                        self.medium_saved = True

                    # Expert 
                    if not self.expert_saved and mean_reward > -90:
                        print(f"\nExpert (Avg: {mean_reward:.1f})")
                        save_path = f"{Save_Dir}/{self.env_id}_expert.pt"
                        save_ppo_actor_as_pt(self.model, save_path, self.obs_dim, self.act_dim)
                        self.expert_saved = True

        return True


def plot_learning_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.title(f"{ENV_ID} Training Reward Curve")
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw')
    
    window = 50
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(avg, color='red', label=f'Moving Avg ({window})')
    
    plt.axhline(-100, color='green', linestyle='--', label='Expert (-100)')
    plt.axhline(-200, color='blue', linestyle='--', label='Medium (-200)')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def watch_agent(model_path, obs_dim, act_dim):
    if not os.path.exists(model_path):
        return

    print(f"\nWatching Agent: {model_path}")
    env = gym.make(ENV_ID, render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QNetwork(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(3):
        obs, _ = env.reset()
        done = False
        total_rew = 0
        while not done:
            state_t = torch.tensor(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(state_t)
                action = torch.argmax(logits, dim=1).item()
            
            obs, rew, terminated, truncated, _ = env.step(action)
            total_rew += rew
            done = terminated or truncated
        print(f"Episode {ep+1} Score: {total_rew:.1f}")
    
    env.close()

if __name__ == "__main__":
    env = make_vec_env(ENV_ID, n_envs=Num_Envs)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"System: {ENV_ID} | Obs: {obs_dim} | Act: {act_dim}")


    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=Learning_Rate,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.0, 
        tensorboard_log=f"./{ENV_ID}_tb_log/"
    )

    callback = CheckpointCallback(ENV_ID, obs_dim, act_dim)
    print("Training Started")
    model.learn(total_timesteps=Total_Timesteps, callback=callback)
    print("Training Finished.")

    final_path = f"{Save_Dir}/{ENV_ID}_expert.pt"
    if not callback.expert_saved:
        print("Save last model.")
        save_ppo_actor_as_pt(model, final_path, obs_dim, act_dim)

    env.close()

    if callback.reward_history:
        plot_learning_curve(callback.reward_history)

    watch_agent(final_path, obs_dim, act_dim)