import os
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import cityflow
import numpy as np
from datetime import datetime
import json

class CityflowEnv(gym.Env):
    env_name = ""
    metadata = {
        "render_modes" : [
            "terminal"
        ]
    }
    
    
    def __init__(self, thread_num=1, max_timesteps=3600, save_replay=False, env_name="env", terminal_logs=False):
        self.env_name = env_name
        os.makedirs("replay_files", exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), "replay_files", self.env_name), exist_ok=True)

        timestamp = datetime.now()
        timestamp_string = timestamp.strftime("%Y%m%d_%H%M%S")
        if save_replay:
            os.makedirs(os.path.join(os.getcwd(), "replay_files", self.env_name, timestamp_string), exist_ok=True)

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(self.current_dir, 'config_files', 'config.json')

        with open(config_file_path, 'r') as file:
            data = json.load(file)

        data['dir'] = ""
        data['rlTrafficLight'] = True
        data['roadnetFile'] = os.path.join(self.current_dir, 'config_files', f"{self.env_name}", 'roadnet.json')
        data['flowFile'] = os.path.join(self.current_dir, 'config_files', f"{self.env_name}", 'flow.json')
        data['roadnetLogFile'] = os.path.join('replay_files', self.env_name, 'replay_roadnet.json')
        data['saveReplay'] = save_replay

        with open(config_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        self.engine = cityflow.Engine(config_file=config_file_path, thread_num=thread_num)

        self.max_timesteps = max_timesteps
        self.current_timestep = 1
        self.current_episode = 0
        self._terminal_logs = terminal_logs

        self.replay_files_dir_path = os.path.join(os.getcwd(), "replay_files", self.env_name, timestamp_string)
        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        roadnet_file_path = os.path.join(self.current_dir, "config_files", f"{self.env_name}", "roadnet.json")
        with open(roadnet_file_path, 'r') as file:
            roadnet_data = json.load(file)

        self.non_peripheral_intersections = []
        self.intersection_phases = []
        self.current_phases = {}

        for intersection in roadnet_data["intersections"]:
            if not intersection["virtual"]:
                self.non_peripheral_intersections.append(intersection)
                self.intersection_phases.append(len(intersection["trafficLight"]["lightphases"]))
                self.current_phases[intersection["id"]] = 0  # Initialize to phase 0 or the default phase

        self._roads_data = {}
        for road in roadnet_data["roads"]:
            self._roads_data[road["id"]] = {
                "lanes": road["lanes"]
            }
        
        
        # Flattened observation space
        lane_max_vehicle_count = 1000
        non_peripheral_intersection_count = len(self.non_peripheral_intersections)
        lane_count = len(self.engine.get_lane_waiting_vehicle_count().values())

        # print(f"Non-peripheral intersections: {self.non_peripheral_intersections}")
        # print(f"count: {len(self.non_peripheral_intersections)}")
        # print(f"lane_count: {lane_count}")

        self.observation_space = Box(low=0,high=lane_max_vehicle_count, shape=((lane_count + non_peripheral_intersection_count),), dtype=int)
        self._last_observation = np.array([0]*(lane_count + non_peripheral_intersection_count), dtype=int)
        self._last_action = np.array([0]*non_peripheral_intersection_count, dtype=int)
        self._last_reward = 0

        # Action space
        self.action_space = MultiDiscrete([phase + 1 for phase in self.intersection_phases])
        # print(self.action_space)


    def step(self, action):
        self._last_action = action
        
        for ind, intersection in enumerate(self.non_peripheral_intersections):
            intersection_id = intersection["id"]

            # If the action is different from the previous phase, set the new phase
            if action[ind] < self.intersection_phases[ind] and action[ind] != self.current_phases[intersection_id]:
                self.engine.set_tl_phase(intersection_id, action[ind])
                self.current_phases[intersection_id] = action[ind]  # Update stored phase

        prev_avg_travel_time = self.engine.get_average_travel_time()
        reward = 0

        for _ in range(10):
            self.engine.next_step()
            reward += (sum(self.engine.get_vehicle_speed().values()) / len(self.engine.get_vehicle_speed().values()) )

        terminated = self.current_timestep >= self.max_timesteps
        truncated = False
        
        curr_avg_travel_time = self.engine.get_average_travel_time()
        reward /= 10

        self._last_reward = reward

        # Flatten the observation
        observation_list = list(self.engine.get_lane_waiting_vehicle_count().values())
        for phase in action:
            observation_list.append(phase)
        observation = np.array(observation_list)
        self._last_observation = observation

        info = {}

        self.current_timestep += 1

        if self._terminal_logs:
            self.render()

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()

        self.current_episode += 1
        self.current_timestep = 1

        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        # Initialize observation and reset phase tracking
        observation = self._last_observation

        info = {}

        return observation, info
    

    def render(self, mode="terminal"):
        if mode == "terminal" :
            print("-"*20)
            print(f"Current timestep: {self.current_timestep}")
            print("Action: ", self._last_action)
            print(f"Reward: {self._last_reward}")
            print("-"*20)
        else:
            pass


    def close(self):
        pass

    def _reward_func_10(self):
        return sum(self.engine.get_vehicle_speed().values()) / len(self.engine.get_vehicle_speed().values())  
