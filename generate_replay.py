from gymnasium.utils.env_checker import check_env
from gym_cityflow.envs.grid_1x1.grid_1x1_demo_env import Grid1x1DemoEnv
from stable_baselines3 import PPO


print("Loading environment...")
env = Grid1x1DemoEnv(max_timesteps=600)

print("Loading model...")
model = PPO.load("models/ppo_traffic_new_reward_200000_steps", env=env)

print("Simulating...")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render(mode="terminal")
    if terminated or truncated:
        break

print("Simulation complete.")
print("Replay file saved at:", f"{env.replay_files_dir_path}/replay_{env.current_episode}.txt")

env.close()