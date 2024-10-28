import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_cityflow import CityflowEnv


def get_next_run_number(dir_path):
    """Gets the next available run number in the specified directory."""
    existing_runs = [
        int(name.split("_")[-1]) for name in os.listdir(dir_path) if name.startswith("PPO_") and name.split("_")[-1].isdigit()
    ]
    return max(existing_runs, default=0) + 1


parser = argparse.ArgumentParser(description="Train a PPO model on CityFlow traffic environment.")
parser.add_argument("--env_name", type=str, required=True, help="Name of the environment configuration.")
parser.add_argument("--max_timesteps", type=int, default=200, help="Maximum timesteps per episode.")
parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps for training (default: 1000 * max_timesteps).")
args = parser.parse_args()

env_name = args.env_name
max_timesteps = args.max_timesteps
total_timesteps = args.total_timesteps or 1000 * max_timesteps

# Paths
model_base_dir = f"./models/{env_name}"
tensorboard_log_path = f"./ppo_tensorboard/{env_name}"

os.makedirs(model_base_dir, exist_ok=True)
os.makedirs(tensorboard_log_path, exist_ok=True)

run_num = get_next_run_number(model_base_dir)

model_save_path = os.path.join(model_base_dir, ("PPO_" + str(run_num)))

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(tensorboard_log_path, exist_ok=True)

print(f"Model will be saved to: {model_save_path}")
print(f"TensorBoard logs will be saved to: {tensorboard_log_path}")

# Environment setup
env = CityflowEnv(max_timesteps=max_timesteps, save_replay=False, env_name=env_name, terminal_logs=True)
check_env(env)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Checkpoint callback setup
save_freq = 250 * max_timesteps
checkpoint_callback = CheckpointCallback(save_freq, save_path=model_save_path, name_prefix=f"ppo_checkpoint")

# Model training
model = PPO("MlpPolicy", env, ent_coef=0.0003, verbose=1, n_steps=1000, tensorboard_log=tensorboard_log_path)
model.learn(total_timesteps, callback=checkpoint_callback, progress_bar=True)
model.save(os.path.join(model_save_path, f"ppo_final_model_{total_timesteps}_timesteps"))

print("Training complete.")
