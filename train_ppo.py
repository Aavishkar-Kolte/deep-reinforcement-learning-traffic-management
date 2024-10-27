import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_cityflow import CityflowEnv


parser = argparse.ArgumentParser(description="Train a PPO model on CityFlow traffic environment.")
parser.add_argument("--env_name", type=str, required=True, help="Name of the environment configuration.")
parser.add_argument("--max_timesteps", type=int, default=200, help="Maximum timesteps per episode.")
parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps for training (default: 1000 * max_timesteps).")
args = parser.parse_args()

env_name = args.env_name
max_timesteps = args.max_timesteps
total_timesteps = args.total_timesteps or 1000 * max_timesteps

env = CityflowEnv(max_timesteps=max_timesteps, save_replay=False, env_name=env_name, terminal_logs=True)
check_env(env)

env = Monitor(env)
env = DummyVecEnv([lambda: env])

save_freq = 250 * max_timesteps

checkpoint_callback = CheckpointCallback(save_freq, save_path='./models/', name_prefix=f"ppo_traffic_checkpoint_[{env_name}]")

model = PPO("MlpPolicy", env, ent_coef=0.0003, verbose=1, n_steps=1000, tensorboard_log="./ppo_tensorboard/")

model.learn(total_timesteps, callback=checkpoint_callback, progress_bar=True)
model.save(f"ppo_traffic_final_model_[{env_name}]_{total_timesteps}_timesteps")
