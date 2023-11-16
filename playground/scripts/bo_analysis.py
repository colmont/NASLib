import json
import os
import statistics
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

def extract_valid_acc_from_dir(directory):
    """Extract 'valid_acc' lists from errors.json files within a given directory."""
    valid_acc_lists = []
    all_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    numeric_dirs = [d for d in all_dirs if d.isdigit()]
    print(f"Found {len(numeric_dirs)} trials in {directory}") 

    for dir_name in numeric_dirs:
        trial_dir = os.path.join(directory, dir_name)
        errors_path = os.path.join(trial_dir, "errors.json")
        
        try:
            with open(errors_path, 'r') as file:
                content = json.load(file)
                for item in content:
                    if "valid_acc" in item:
                        valid_acc_lists.append(item["valid_acc"])
                        break
        except FileNotFoundError:
            print(f"File not found: {errors_path}")
        except json.JSONDecodeError:
            print(f"Could not decode JSON from: {errors_path}")

    return valid_acc_lists

def extract_k_from_dir(directory):
    """Extract the value of 'k' from errors.json files within a given directory."""
    k_values = []
    all_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    numeric_dirs = [d for d in all_dirs if d.isdigit()]

    for dir_name in numeric_dirs:
        trial_dir = os.path.join(directory, dir_name)
        errors_path = os.path.join(trial_dir, "errors.json")

        try:
            with open(errors_path, 'r') as file:
                content = json.load(file)

                for item in content:
                    if "search" in item and "k" in item["search"]:
                        k_values.append(item["search"]["k"])
                        break
        except FileNotFoundError:
            print(f"File not found: {errors_path}")
        except json.JSONDecodeError:
            print(f"Could not decode JSON from: {errors_path}")

    return k_values

def find_lowest_number_of_trials(all_data):
    min_trials = float('inf')
    for _, _, all_valid_accs, _ in all_data:
        for valid_accs in all_valid_accs:
            min_trials = min(min_trials, len(valid_accs))
    print("Lowest number of trials:", min_trials)
    return min_trials

def process_directories(nasbench101_dir):
    dirs_to_process = [d for d in os.listdir(nasbench101_dir) if os.path.isdir(os.path.join(nasbench101_dir, d))]
    print("Found directories:", dirs_to_process)
    all_valid_accs = []
    all_k_values = []

    for directory in dirs_to_process:
        valid_accs = extract_valid_acc_from_dir(os.path.join(nasbench101_dir, directory))
        all_valid_accs.append(valid_accs)
        k_values = extract_k_from_dir(os.path.join(nasbench101_dir, directory))
        all_k_values += k_values

    return dirs_to_process, all_valid_accs, all_k_values

def calculate_common_k(all_k_values):
    return all_k_values[0] if len(set(all_k_values)) == 1 else None

def prepare_data_for_plotting(all_valid_accs, min_trials):

    for i in range(len(all_valid_accs)):
        all_valid_accs[i] = all_valid_accs[i][:min_trials]
    
    return all_valid_accs

def process_directories_for_multiple_timestamps(location, timestamps):
    all_data = []
    for timestamp in timestamps:
        print("Currently handling timestamp:", timestamp)
        nasbench101_dir = os.path.join(location, timestamp, "cifar10", "nas_predictors", "nasbench101")
        dirs_to_process, valid_accs, k_values = process_directories(nasbench101_dir)
        all_data.append((timestamp, dirs_to_process, valid_accs, k_values))

    min_trials = find_lowest_number_of_trials(all_data)
    all_data_prepared = []

    for timestamp, dirs_to_process, valid_accs, k_values in all_data:
        common_k = calculate_common_k(k_values)
        valid_accs_prepared = prepare_data_for_plotting(valid_accs, min_trials)
        all_data_prepared.append((timestamp, dirs_to_process, valid_accs_prepared, common_k))

    return all_data_prepared

def plot_combined_validation_error_graph(all_data):
    plt.figure(figsize=(12, 6))
    cmap = cm.get_cmap('tab10')
    color_index = 0

    for timestamp_data in all_data:
        timestamp, dirs_to_process, all_valid_accs, common_k = timestamp_data
        labels = [f"{timestamp}-{dir}" for dir in dirs_to_process]

        for i, valid_accs in enumerate(all_valid_accs):
            color = cmap(color_index % cmap.N)
            # Increment color_index after each plot
            color_index += 1

            problematic_indices = []
            lengths = [len(acc) for acc in valid_accs]
            mode_length = statistics.mode(lengths)

            for j in range(len(valid_accs)):
                if len(valid_accs[j]) != mode_length:
                    problematic_indices.append(j)

            valid_accs = [valid_accs[j] for j in range(len(valid_accs)) if j not in problematic_indices]
            error = 100 - np.median(valid_accs, axis=0)
            error_stderr = np.std(valid_accs, axis=0) / np.sqrt(len(valid_accs))
            adjusted_time_steps = [j * common_k for j in range(len(error))]

            plt.plot(adjusted_time_steps, error, label=f"{labels[i]} - Median Error", color=color)
            plt.fill_between(adjusted_time_steps, error - error_stderr, error + error_stderr, color=color, alpha=0.2) # label=f"{labels[i]} - Standard Error" #TODO: add?

    plt.title("Adjusted Median Validation Errors with Standard Error (Combined)")
    plt.xlabel(f"Adjusted Time Step")
    plt.ylabel("Validation Error (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # filename = all_data[1][0]
    filename = "combined"
    output_path = f"/cluster/scratch/cdoumont/playground/bo_graphs/{filename}.pdf"
    plt.savefig(output_path, format='pdf')
    plt.close()


def generate_combined_validation_error_graph(location, timestamps):
    all_data_prepared = process_directories_for_multiple_timestamps(location, timestamps)
    plot_combined_validation_error_graph(all_data_prepared)

location = "/cluster/scratch/cdoumont/playground/runs/bo"
reference_timestamp = '20231010_120707'
timestamps = ['20231108_233044', '20231110_110941', '20231110_220314', '20231111_163035', '20231112_125318', '20231113_095926']
# for timestamp in timestamps:
#     generate_combined_validation_error_graph(location, [reference_timestamp, timestamp])
all_timestamps = [reference_timestamp] + timestamps
generate_combined_validation_error_graph(location, all_timestamps)