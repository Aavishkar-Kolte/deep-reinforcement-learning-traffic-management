import cityflow
import os
import json
from datetime import datetime

# Parameters for the simulation
# env_name = "example"
# env_name = "manhattan_16x3"
env_name = "syn_4x4_gaussian_500_1h"
thread_num = 1  
max_timesteps = 2000 

# Create directories for storing replay files
replay_files_base_path = os.path.join(os.getcwd(), "replay_files", env_name)
os.makedirs(replay_files_base_path, exist_ok=True)

# Create a timestamped subdirectory for the current run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
replay_files_dir_path = os.path.join(replay_files_base_path, timestamp)
os.makedirs(replay_files_dir_path, exist_ok=True)

config_file_path = os.path.join(os.getcwd(), 'gym_cityflow', 'config_files', 'config.json')
with open(config_file_path, 'r') as file:
    config_data = json.load(file)

config_data['dir'] = ''
config_data['roadnetFile'] = os.path.join('gym_cityflow', "config_files", env_name, "roadnet.json")
config_data['flowFile'] = os.path.join('gym_cityflow', "config_files", env_name, "flow.json")
config_data['roadnetLogFile'] = os.path.join(replay_files_base_path, 'replay_roadnet.json')
config_data['saveReplay'] = True

with open(config_file_path, 'w') as file:
    json.dump(config_data, file, indent=4)


engine = cityflow.Engine(config_file=config_file_path, thread_num=thread_num)
engine.set_save_replay(open=True)

current_timestep = 1

engine.set_replay_file(os.path.join(replay_files_dir_path, f"replay.txt"))

# Run the simulation
while current_timestep <= max_timesteps:
    engine.next_step()  
    print(f"Timestep {current_timestep} completed.")
    current_timestep += 1

engine.set_save_replay(open=False)
print("Simulation complete.")

