import os
import matplotlib.pyplot as plt

def read_metrics(file_path):
    """Reads metrics from a single charts.txt file."""
    metrics = {
        "Average Travel Time": [],
        "Average Speed": [],
        "Number of Vehicles": [],
        "Number of Waiting Vehicles": [],
        "Number of Running Vehicles": []
    }

    with open(file_path, "r") as file:
        next(file)  # Skip the header row
        for line in file:
            data = line.strip().split("\t")
            metrics["Average Travel Time"].append(float(data[0]))
            metrics["Average Speed"].append(float(data[1]))
            metrics["Number of Vehicles"].append(float(data[2]))
            metrics["Number of Waiting Vehicles"].append(float(data[3]))
            metrics["Number of Running Vehicles"].append(float(data[4]))
    
    return metrics

def plot_comparative_metrics(files, labels, output_dir):
    """Generates comparative plots for each metric using multiple files."""
    # Prepare data structures
    all_metrics_data = {metric: [] for metric in ["Average Travel Time", "Average Speed", "Number of Vehicles", "Number of Waiting Vehicles", "Number of Running Vehicles"]}

    # Read data from each file
    for file_path in files:
        file_metrics = read_metrics(file_path)
        for metric, values in file_metrics.items():
            all_metrics_data[metric].append(values)
    
    # Define titles and labels for each metric
    y_labels = {
        "Average Travel Time": "Time (s)",
        "Average Speed": "Speed (m/s)",
        "Number of Vehicles": "Count",
        "Number of Waiting Vehicles": "Count",
        "Number of Running Vehicles": "Count"
    }

    # Plot each metric with data from all files
    for metric, data_sets in all_metrics_data.items():
        plt.figure()
        
        for idx, data in enumerate(data_sets):
            time_steps = list(range(len(data)))
            plt.plot(time_steps, data, label=labels[idx])

        plt.xlabel("Time Step")
        plt.ylabel(y_labels[metric])
        plt.title(f"Comparative {metric}")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, color="gray") 
        plt.savefig(os.path.join(output_dir, f"comparative_{metric.replace(' ', '_').lower()}.png"))
        plt.close()

files = [
    "path/to/first/charts.txt",
    "path/to/second/charts.txt",
    "path/to/third/charts.txt"
]
labels = ["Run 1", "Run 2", "Run 3"]
output_dir = "path/to/output/directory"

os.makedirs(output_dir, exist_ok=True)

plot_comparative_metrics(files, labels, output_dir)

print("Comparative charts generated and saved in:", output_dir)
