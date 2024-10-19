import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict, Discrete
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
    
    
    def __init__(self, thread_num=1, max_timesteps=3600):
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

        with open(config_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        self.engine = cityflow.Engine(config_file=config_file_path, thread_num=thread_num)

        self.max_timesteps = max_timesteps
        self.current_timestep = 0
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
        self.observation_space = Dict({
            f"{intersection['id']}_phase": Discrete(len(intersection["trafficLight"]["lightphases"])) for intersection in self.non_peripheral_intersections
        })
        for intersection in self.non_peripheral_intersections:
            for road in intersection["roads"]:
                for ind in range(len(self._roads_data[road]["lanes"])):
                    lane_id = f"{road}_{ind}"
                    self.observation_space.spaces[f"{intersection['id']}_{lane_id}_running_vehicle_count"] = Discrete(1000)
                    self.observation_space.spaces[f"{intersection['id']}_{lane_id}_waiting_vehicle_count"] = Discrete(1000)

        # Action space
        self.action_space = MultiDiscrete([phase + 1 for phase in self.intersection_phases])


    def step(self, action):
        for ind, intersection in enumerate(self.non_peripheral_intersections):
            intersection_id = intersection["id"]

            # If the action is different from the previous phase, set the new phase
            if action[ind] < self.intersection_phases[ind] and action[ind] != self.current_phases[intersection_id]:
                # print(f"Setting phase for intersection {intersection_id}: {action[ind]}")
                self.engine.set_tl_phase(intersection_id, action[ind])
                self.current_phases[intersection_id] = action[ind]  # Update stored phase
            # else:
                # print(f"Phase for intersection {intersection_id} remains unchanged: {self.current_phases[intersection_id]}")

        self.engine.next_step()

        terminated = self.current_timestep >= self.max_timesteps
        truncated = False

        reward = (-1 * self.engine.get_average_travel_time()) + (-2 * sum(self.engine.get_lane_waiting_vehicle_count().values()))
        
        # Flatten the observation
        observation = {}
        for intersection in self.non_peripheral_intersections:
            intersection_id = intersection["id"]

            # Store current phase
            observation[f"{intersection_id}_phase"] = self.current_phases[intersection_id]

            # Get lane-specific data
            for road_id in intersection["roads"]:
                road = self._roads_data.get(road_id)

                for ind, _ in enumerate(road["lanes"]):
                    lane_id = f"{road_id}_{ind}"
                    observation[f"{intersection_id}_{lane_id}_running_vehicle_count"] = self.engine.get_lane_vehicle_count().get(lane_id, 0)
                    observation[f"{intersection_id}_{lane_id}_waiting_vehicle_count"] = self.engine.get_lane_waiting_vehicle_count().get(lane_id, 0)

        info = {}

        self.current_timestep += 1

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()

        self.current_episode += 1
        self.current_timestep = 0

        self.engine.set_replay_file(os.path.join(self.replay_files_dir_path, f"replay_{self.current_episode}.txt"))

        # Initialize observation and reset phase tracking
        observation = {}
        for intersection in self.non_peripheral_intersections:
            intersection_id = intersection["id"]
            self.current_phases[intersection_id] = 0  # Reset to default phase

            # Store current phase
            observation[f"{intersection_id}_phase"] = self.current_phases[intersection_id]

            # Get lane-specific data
            for road_id in intersection["roads"]:
                road = self._roads_data.get(road_id)

                for ind, _ in enumerate(road["lanes"]):
                    lane_id = f"{road_id}_{ind}"
                    observation[f"{intersection_id}_{lane_id}_running_vehicle_count"] = self.engine.get_lane_vehicle_count().get(lane_id, 0)
                    observation[f"{intersection_id}_{lane_id}_waiting_vehicle_count"] = self.engine.get_lane_waiting_vehicle_count().get(lane_id, 0)

        info = {}

        return observation, info


    def render(self, mode="terminal"):
        # if mode == "terminal" :
        #     print(f"{self.engine.get_average_travel_time()}   {sum(self.engine.get_lane_waiting_vehicle_count().values())}")
        #     print("Reward: ", (-1 * self.engine.get_average_travel_time()) + (-2 * sum(self.engine.get_lane_waiting_vehicle_count().values())))
        pass

    def close(self):
        pass
