from gymnasium.utils.env_checker import check_env
from gym_cityflow import CityflowEnv
from stable_baselines3 import PPO


print("Loading environment...")
env = CityflowEnv(max_timesteps=600, save_replay=True)

print("Loading model...")
model = PPO.load("models/ppo_traffic_agent_900000_steps.zip", env=env)

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