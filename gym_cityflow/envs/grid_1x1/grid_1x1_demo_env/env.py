import os
import gymnasium as gym
from gymnasium.spaces import Box
import cityflow
import numpy as np
import json
from datetime import datetime


class Grid1x1DemoEnv(gym.Env):
    def __init__(self, thread_num=1, time_steps=3600):
        # Set up the environment configuration
        self.env_name = "Grid1x1DemoEnv"
        os.makedirs("replay_files", exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), "replay_files", self.env_name), exist_ok=True)

        # Get the current timestamp
        timestamp = datetime.now()
        timestamp_string = timestamp.strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(os.getcwd(), "replay_files", self.env_name, timestamp_string), exist_ok=True)

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(self.current_dir, 'config_files', 'config.json')
        self.engine = cityflow.Engine(config_file=config_file_path, thread_num=thread_num)

        self.time_steps = time_steps
        self.current_episode = 0

        self.replay_files_dir_path = os.path.join(os.getcwd(), "replay_files", self.env_name, timestamp_string)
        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        # Load the road network data
        roadnet_file_path = os.path.join(self.current_dir, "config_files", "roadnet.json")
        with open(roadnet_file_path, 'r') as file:
            roadnet_data = json.load(file)

        # Process intersections to set up action space
        self.non_peripheral_intersections = []
        high_value_arr = []

        for intersection in roadnet_data["intersections"]:
            if not intersection["virtual"]:
                self.non_peripheral_intersections.append(intersection)
                high_value_arr.append(len(intersection["trafficLight"]["lightphases"]) - 1)

        # For manual testing
        # print("Non-peripheral intersections:", self.non_peripheral_intersections)
        # print("High value array:", np.array(high_value_arr, dtype=np.int64))

        # Define action space
        self.action_space = Box(
            low=np.zeros(len(high_value_arr), dtype=np.int64),
            high=np.array(high_value_arr, dtype=np.int64),
            shape=(len(high_value_arr),),
            dtype=np.int64
        )

        # Define observation space
        self.observation_space_length = len(self.engine.get_lane_waiting_vehicle_count())
        int64_max_value = np.iinfo(np.int64).max
        self.observation_space = Box(
            low=0,
            high=int64_max_value,
            shape=(self.observation_space_length,),
            dtype=np.int64
        )        


    def step(self, action):
        pass


    def reset(self):
        pass


    def render(self, mode='human'):
        pass


    def close(self):
        pass

