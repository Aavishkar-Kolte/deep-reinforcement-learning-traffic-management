import os
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import cityflow
import numpy as np
from datetime import datetime
import json

class Grid1x1DemoEnv(gym.Env):
    env_name = "Grid1x1DemoEnv"
    metadata = {
        "render_modes" : [
            "terminal"
        ]
    }
    
    
    def __init__(self, thread_num=1, max_timesteps=3600, save_replay=False):
        os.makedirs("replay_files", exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), "replay_files", self.env_name), exist_ok=True)

        timestamp = datetime.now()
        timestamp_string = timestamp.strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(os.getcwd(), "replay_files", self.env_name, timestamp_string), exist_ok=True)

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(self.current_dir, 'config_files', 'config.json')

        with open(config_file_path, 'r') as file:
            data = json.load(file)

        data['dir'] = ""
        data['roadnetFile'] = os.path.join(self.current_dir, 'config_files', 'roadnet.json')
        data['flowFile'] = os.path.join(self.current_dir, 'config_files', 'flow.json')
        data['roadnetLogFile'] = os.path.join('replay_files', self.env_name, 'replay_roadnet.json')
        data['saveReplay'] = save_replay

        with open(config_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        self.engine = cityflow.Engine(config_file=config_file_path, thread_num=thread_num)

        self.max_timesteps = max_timesteps
        self.current_timestep = 1
        self.current_episode = 0


        self.replay_files_dir_path = os.path.join(os.getcwd(), "replay_files", self.env_name, timestamp_string)
        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        roadnet_file_path = os.path.join(self.current_dir, "config_files", "roadnet.json")
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
        self.observation_space = Box(low=0,high=lane_max_vehicle_count, shape=((lane_count + non_peripheral_intersection_count),), dtype=int)
        self._last_observation = np.array([0]*(lane_count + 1), dtype=int)

        # Action space
        self.action_space = MultiDiscrete([phase + 1 for phase in self.intersection_phases])
        self._avg_avg_tt = 0

    def step(self, action):
        for ind, intersection in enumerate(self.non_peripheral_intersections):
            intersection_id = intersection["id"]

            # If the action is different from the previous phase, set the new phase
            if action[ind] < self.intersection_phases[ind] and action[ind] != self.current_phases[intersection_id]:
                print(f"Setting phase for intersection {intersection_id}: {action[ind]}")
                self.engine.set_tl_phase(intersection_id, action[ind])
                self.current_phases[intersection_id] = action[ind]  # Update stored phase
            else:
                print(f"Phase for intersection {intersection_id} remains unchanged: {self.current_phases[intersection_id]}")

        self.engine.next_step()

        terminated = self.current_timestep >= self.max_timesteps
        truncated = False

        reward = 0
        if(self.engine.get_average_travel_time() < self._avg_avg_tt):
            reward += (self._avg_avg_tt - self.engine.get_average_travel_time())
        else:
            reward -= (self.engine.get_average_travel_time() - self._avg_avg_tt)

        print(f"Average travel time: {self.engine.get_average_travel_time()} avg_avg_tt: {self._avg_avg_tt}")
        print(f"Reward: {reward}")

        self._avg_avg_tt = (self._avg_avg_tt * (self.current_timestep - 1)+ self.engine.get_average_travel_time()) / self.current_timestep
        
        # Flatten the observation
        observation_list = list(self.engine.get_lane_waiting_vehicle_count().values())
        for phase in action:
            observation_list.append(phase)
        observation = np.array(observation_list)
        self._last_observation = observation

        info = {}

        self.current_timestep += 1

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()

        self.current_episode += 1
        self.current_timestep = 1
        self._avg_avg_tt = 0

        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        # Initialize observation and reset phase tracking
        observation = self._last_observation

        info = {}

        return observation, info


    def render(self, mode="terminal"):
        if mode == "terminal" :
            print(f"{self.engine.get_average_travel_time()}   {sum(self.engine.get_lane_waiting_vehicle_count().values())}")
            print("Reward: ", (-1 * self.engine.get_average_travel_time()) + (-2 * sum(self.engine.get_lane_waiting_vehicle_count().values())))
        pass

    def close(self):
        pass
