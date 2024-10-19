from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_cityflow.envs.grid_1x1.grid_1x1_demo_env import Grid1x1DemoEnv


env = Grid1x1DemoEnv(max_timesteps=900)
check_env(env)

env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Define a checkpoint callback to save models during training
checkpoint_callback = CheckpointCallback(save_freq=900000, save_path='./models/', name_prefix='ppo_traffic_agent')

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

model.learn(total_timesteps=900000, callback=checkpoint_callback)
model.save("ppo_traffic")