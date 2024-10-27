from gymnasium.utils.env_checker import check_env
from gym_cityflow import CityflowEnv
from stable_baselines3 import PPO


print("Loading environment...")
env_name = "example"
env = CityflowEnv(max_timesteps=100, save_replay=True, env_name=env_name, terminal_logs=True)

print("Loading model...")
model = PPO.load("ppo_traffic_final_model_[example]_100000_timesteps.zip", env=env)

print("Simulating...")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print("Simulation complete.")
print("Replay file saved at:", f"{env.replay_files_dir_path}/replay_{env.current_episode}.txt")

env.close()