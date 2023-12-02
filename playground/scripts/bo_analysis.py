import csv
import json
import os
import statistics
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import auc

from naslib.utils.log import setup_logger

# Constants
ERRORS_JSON = "errors.json"
CIFAR10_DIR = "cifar10"
NAS_PREDICTORS_DIR = "nas_predictors"
NASBENCH101_DIR = "nasbench101"  # FIXME
OUTPUT_PATH = "/cluster/scratch/cdoumont/playground/bo_graphs/"
CSV_FILENAME = "/cluster/home/cdoumont/NASLib/playground/scripts/ranked_labels_and_auc.csv"
ACC_TYPE = "test_acc"  # valid_acc or test_acc

# Configure logger
logger = setup_logger(name="bo_analysis")
logger.setLevel(logging.INFO)


def extract_data_from_dir(directory, data_key, search_key=None):
    """Extract specified data from errors.json files within a given directory."""
    data_lists = []
    all_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    numeric_dirs = [d for d in all_dirs if d.isdigit()]
    logger.info(f"Found {len(numeric_dirs)} trials in {directory}")

    for dir_name in numeric_dirs:
        trial_dir = os.path.join(directory, dir_name)
        errors_path = os.path.join(trial_dir, ERRORS_JSON)

        try:
            with open(errors_path, 'r') as file:
                content = json.load(file)
                for item in content:
                    if data_key in item and (not search_key or search_key in item[data_key]):
                        data_lists.append(item[data_key] if not search_key else item[data_key][search_key])
                        break
        except FileNotFoundError as e:
            logger.warning(f"File not found: {errors_path} - {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not decode JSON from: {errors_path} - {e}")

    return data_lists


def create_all_trajectories(location, timestamps):
    """Create all trajectories from the specified directory and timestamps."""
    all_trajectories = []
    for timestamp in timestamps:
        nasbench101_dir = os.path.join(location, timestamp, CIFAR10_DIR, NAS_PREDICTORS_DIR, NASBENCH101_DIR)

        for subdir in os.listdir(nasbench101_dir):
            if os.path.isdir(os.path.join(nasbench101_dir, subdir)):
                accs_trials = extract_data_from_dir(os.path.join(nasbench101_dir, subdir), ACC_TYPE)
                k_values_trials = extract_data_from_dir(os.path.join(nasbench101_dir, subdir), "search", "k")
                all_trajectories.append((timestamp, subdir, accs_trials, k_values_trials))
    return all_trajectories


def calculate_best_accs_so_far(accs_trials):
    """ Calculate the best (highest) accuracies so far for each trial. """
    best_accs_so_far = []
    for trial in accs_trials:
        best_accs = []
        current_best = 0
        for acc in trial:
            current_best = max(current_best, acc)
            best_accs.append(current_best)
        best_accs_so_far.append(best_accs)
    return best_accs_so_far


def update_trajectories_with_best_accs(all_trajectories):
    """Update all trajectories with the best valid accuracies so far."""
    for i, (timestamp, subdir, accs_trials, k_values_trials) in enumerate(all_trajectories):
        accs_trials = calculate_best_accs_so_far(accs_trials)
        updated_trajectory = (timestamp, subdir, accs_trials, k_values_trials)
        all_trajectories[i] = updated_trajectory
    return all_trajectories


def adjust_trial_counts(all_trajectories, min_trials=None):
    """Adjust trial counts to ensure uniformity across all trajectories."""
    if min_trials is None:
        min_trials = float('inf')
        for _, _, accs_trials, _ in all_trajectories:
            min_trials = min(min_trials, len(accs_trials))
        logger.info("Lowest number of trials: {}".format(min_trials))

    for i, (timestamp, subdir, accs_trials, k_values_trials) in enumerate(all_trajectories):
        updated_accs_trials = accs_trials[:min_trials]
        updated_trajectory = (timestamp, subdir, updated_accs_trials, k_values_trials)
        all_trajectories[i] = updated_trajectory
    return all_trajectories


def remove_problematic_trials(all_trajectories):
    """Remove problematic trials from all trajectories."""
    all_lengths = []
    for _, _, accs_trials, _ in all_trajectories:
        all_lengths.extend([len(accs) for accs in accs_trials])
    mode_length = statistics.mode(all_lengths)

    # Safe removal of problematic trials
    for _, _, accs_trials, _ in all_trajectories:
        accs_trials[:] = [accs for accs in accs_trials if len(accs) == mode_length]
    return all_trajectories


def validate_k_values(all_trajectories):
    """Validate that all k_values are the same across trajectories."""
    for _, _, _, k_values_trials in all_trajectories:
        k_values_set = set(k for k in k_values_trials)
        if len(k_values_set) != 1:
            raise ValueError("Not all k values are equal")
    return k_values_set.pop()


def calculate_errors_and_rank(all_trajectories, common_k):
    """Calculate median and standard error of the mean, and rank the trajectories."""
    labels_for_ranking = []
    errors_for_ranking = []

    for timestamp, subdir, accs_trials, _ in all_trajectories:
        error = 100 - np.median(accs_trials, axis=0)
        error_stderr = np.std(accs_trials, axis=0) / np.sqrt(len(accs_trials))
        adjusted_time_steps = [j * common_k for j in range(len(error))]

        labels_for_ranking.append(f"{timestamp}-{subdir}")
        errors_for_ranking.append(error)

    auc_list = [auc(np.arange(len(error)), error) for error in errors_for_ranking]
    sorted_indices = np.argsort(auc_list)
    sorted_labels = [labels_for_ranking[i] for i in sorted_indices]
    sorted_auc = [auc_list[i] for i in sorted_indices]

    return sorted_labels, sorted_auc, errors_for_ranking, adjusted_time_steps, error_stderr


def save_ranked_labels_to_csv(sorted_labels, sorted_auc):
    """Save ranked labels and AUC to a CSV file."""
    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Label', 'AUC'])
        for label, auc_value in zip(sorted_labels, sorted_auc):
            csvwriter.writerow([label, auc_value])
    logger.info(f"Saved ranked labels and AUC to {CSV_FILENAME}")


def plot_validation_errors(errors_for_ranking, labels_for_ranking, adjusted_time_steps, error_stderr):
    """Plot the combined validation error graph."""
    cmap = cm.get_cmap('tab10')
    color_index = 0

    for i, error in enumerate(errors_for_ranking):
        color = cmap(color_index % cmap.N)
        color_index += 1

        plt.plot(adjusted_time_steps, error, label=f"{labels_for_ranking[i]}", color=color)
        plt.fill_between(adjusted_time_steps, error - error_stderr, error + error_stderr, color=color, alpha=0.2)

    if ACC_TYPE == "valid_acc":
        acc_type = "Validation"
    elif ACC_TYPE == "test_acc":
        acc_type = "Test"

    if NASBENCH101_DIR == "nasbench101":
        dataset = "NAS-Bench-101"
    elif NASBENCH101_DIR == "nasbench201":
        dataset = "NAS-Bench-201"

    plt.title(f"Median {acc_type} Errors (with Standard Error) on {dataset}")
    plt.xlabel("Number of Architectures Evaluated")
    plt.ylabel("Validation Error (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_PATH, "combined.pdf")
    plt.savefig(output_path, format='pdf')
    plt.close()
    logger.info(f"Saved combined validation error graph to {output_path}")


def find_lowest_error(labels_for_ranking, errors_for_ranking):
    """Find and log the lowest error found overall."""
    lowest_error = float('inf')
    lowest_error_label = None
    for label, error in zip(labels_for_ranking, errors_for_ranking):
        if min(error) < lowest_error:
            lowest_error = min(error)
            lowest_error_label = label
    logger.info(f"Lowest error found: {lowest_error} ({lowest_error_label})")


def main():
    location = "/cluster/scratch/cdoumont/playground/runs/bo"
    timestamps = ['20231129_000209']

    all_trajectories = create_all_trajectories(location, timestamps)
    all_trajectories = update_trajectories_with_best_accs(all_trajectories)
    all_trajectories = adjust_trial_counts(all_trajectories)
    all_trajectories = remove_problematic_trials(all_trajectories)
    common_k = validate_k_values(all_trajectories)

    sorted_labels, sorted_auc, errors_for_ranking, adjusted_time_steps, error_stderr = calculate_errors_and_rank(all_trajectories, common_k)

    save_ranked_labels_to_csv(sorted_labels, sorted_auc)
    plot_validation_errors(errors_for_ranking, sorted_labels, adjusted_time_steps, error_stderr)
    find_lowest_error(sorted_labels, errors_for_ranking)


if __name__ == "__main__":
    main()
