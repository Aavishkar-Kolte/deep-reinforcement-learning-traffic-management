from gymnasium.utils.env_checker import check_env
from gym_cityflow import CityflowEnv
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt

print("Loading environment...")
env_name = "example"
model_path = "models/example/PPO_3/ppo_checkpoint_250_steps.zip"
model_num = model_path.split("/")[-2]
replay_config = {
    "save_replay": True,
    "model_num": model_num
}
env = CityflowEnv(max_timesteps=100, env_name=env_name, replay_config=replay_config)
check_env(env)

print("Loading model...")
model = PPO.load(model_path, env=env)

print("Simulating...")
obs, info = env.reset()
replay_file_dir = info["replay_file_dir"]

print("Replay file directory:", replay_file_dir)

charts_file_path = os.path.join(replay_file_dir, "charts.txt")
with open(charts_file_path, "w") as charts_file:
    charts_file.write("AvgTravelTime\tAvgSpeed\tNumVehicles\tNumWaitingVehicles\tNumRunningVehicles\n")

    time_step = 0
    while True:
        # action = env.action_space.sample()
        action, _ = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)

        for i in range(10):
            charts_file.write(
                f"{info['avg_travel_time'][i]}\t{info['avg_speed'][i]}\t{info['num_vehicles'][i]}\t"
                f"{info['num_waiting_vehicles'][i]}\t{info['num_running_vehicles'][i]}\n"
            )
            time_step += 1

        if terminated or truncated:
            break

print("Simulation complete.")
print("Replay file saved at:", f"{env.replay_files_dir_path}/replay_{env.current_episode}.txt")

env.close()

# Plotting data from charts.txt
def plot_metrics_from_file(file_path, save_dir):
    avg_travel_time = []
    avg_speed = []
    num_vehicles = []
    num_waiting_vehicles = []
    num_running_vehicles = []


    with open(file_path, "r") as file:
        next(file)
        for line in file:
            data = line.strip().split("\t")
            avg_travel_time.append(float(data[0]))
            avg_speed.append(float(data[1]))
            num_vehicles.append(float(data[2]))
            num_waiting_vehicles.append(float(data[3]))
            num_running_vehicles.append(float(data[4]))

    time_steps = list(range(len(avg_travel_time)))

    metrics = {
        "Average Travel Time": (avg_travel_time, "Time (s)"),
        "Average Speed": (avg_speed, "Speed (m/s)"),
        "Number of Vehicles": (num_vehicles, "Count"),
        "Number of Waiting Vehicles": (num_waiting_vehicles, "Count"),
        "Number of Running Vehicles": (num_running_vehicles, "Count")
    }

    for title, (data, y_label) in metrics.items():
        plt.figure()
        plt.plot(time_steps, data, label=title)
        plt.xlabel("Time Step")
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray') 
        plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png"))
        plt.close()

    plt.figure()
    plt.plot(time_steps, metrics["Number of Vehicles"][0], label="Number of Vehicles")
    plt.plot(time_steps, metrics["Number of Waiting Vehicles"][0], label="Number of Waiting Vehicles")
    plt.plot(time_steps, metrics["Number of Running Vehicles"][0], label="Number of Running Vehicles") 
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.title("Vehicle Counts")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(os.path.join(save_dir, "vehicle_counts.png"))
    plt.close()


    

plot_metrics_from_file(charts_file_path, replay_file_dir)

print("Charts generated and saved in:", replay_file_dir)

