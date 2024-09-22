import os
import gymnasium as gym
from gymnasium.spaces import Box
import cityflow
import numpy as np
import json
from datetime import datetime


class Grid1x1DemoEnv(gym.Env):
    def __init__(self, thread_num=1, max_timesteps=3600):
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

        self.max_timesteps = max_timesteps
        self.current_timestep = 0
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

        for ind in range(len(action)):
            self.engine.set_tl_phase(self.non_peripheral_intersections[ind]["id"], action[ind])

        self.engine.next_step()

        terminated = True if self.current_timestep >= self.max_timesteps else False
        truncated = False
        
        reward = (-1*self.engine.get_average_travel_time()) + (-0.2*sum(self.engine.get_lane_waiting_vehicle_count().values()))

        lane_waiting_vehicle_count = list(self.engine.get_lane_waiting_vehicle_count().values())
        observation = np.array(lane_waiting_vehicle_count, dtype=np.int64)

        info = {}

        self.current_timestep += 1
        
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.current_episode += 1
        self.current_timestep = 0
        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        lane_waiting_vehicle_count = list(self.engine.get_lane_waiting_vehicle_count().values())
        observation = np.array(lane_waiting_vehicle_count, dtype=np.int64)
        info = {}

        return observation, info
    

    def render(self, mode='human'):
        pass


    def close(self):
        pass

