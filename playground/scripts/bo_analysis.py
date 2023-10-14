import json
import os
from matplotlib import pyplot as plt

import numpy as np


def extract_valid_acc_from_dir(directory):
    """Extract 'valid_acc' lists from errors.json files within a given directory."""
    valid_acc_lists = []
    
    for trial in range(100):
        trial_dir = os.path.join(directory, str(trial))
        errors_path = os.path.join(trial_dir, "errors.json")
        
        with open(errors_path, 'r') as file:
            content = json.load(file)
            
            for item in content:
                if "valid_acc" in item:
                    valid_acc_lists.append(item["valid_acc"])
                    break
    
    return valid_acc_lists


def extract_k_from_dir(directory):
    """Extract the value of 'k' from errors.json files within a given directory."""
    k_values = []
    
    for trial in range(100):
        trial_dir = os.path.join(directory, str(trial))
        errors_path = os.path.join(trial_dir, "errors.json")
        
        with open(errors_path, 'r') as file:
            content = json.load(file)
            
            for item in content:
                if "search" in item and "k" in item["search"]:
                    k_values.append(item["search"]["k"])
                    break
    
    return k_values


def generate_validation_error_graph(timestamped_dir_path, output_path):
    """Generate the validation error graph from a timestamped directory and save it to the specified output path."""
    # Extract 'valid_acc' lists
    nasbench101_dir = os.path.join(timestamped_dir_path, "cifar10", "nas_predictors", "nasbench101")
    gpwl_valid_accs = extract_valid_acc_from_dir(os.path.join(nasbench101_dir, "gpwl"))
    gp_heat_valid_accs = extract_valid_acc_from_dir(os.path.join(nasbench101_dir, "gp_heat"))
    
    # Check the consistency of 'k' values
    all_k_values = extract_k_from_dir(os.path.join(nasbench101_dir, "gpwl")) + extract_k_from_dir(os.path.join(nasbench101_dir, "gp_heat"))
    common_k = all_k_values[0] if len(set(all_k_values)) == 1 else None

    # Compute validation error and standard error
    gpwl_error = 100 - np.median(gpwl_valid_accs, axis=0)
    gp_heat_error = 100 - np.median(gp_heat_valid_accs, axis=0)

    gpwl_error_stderr = np.std(gpwl_valid_accs, axis=0) / np.sqrt(100)
    gp_heat_error_stderr = np.std(gp_heat_valid_accs, axis=0) / np.sqrt(100)

    # Adjust the time steps using the value of k
    adjusted_time_steps = [i * common_k for i in range(len(gpwl_error))]

    # Plot the validation error graph
    plt.figure(figsize=(12, 6))
    plt.plot(adjusted_time_steps, gpwl_error, label="gpwl - Median Error", color="blue")
    plt.fill_between(adjusted_time_steps, 
                     gpwl_error - gpwl_error_stderr, 
                     gpwl_error + gpwl_error_stderr, 
                     color="blue", alpha=0.2, label="gpwl - Standard Error")

    plt.plot(adjusted_time_steps, gp_heat_error, label="gp_heat - Median Error", color="red")
    plt.fill_between(adjusted_time_steps, 
                     gp_heat_error - gp_heat_error_stderr, 
                     gp_heat_error + gp_heat_error_stderr, 
                     color="red", alpha=0.2, label="gp_heat - Standard Error")

    plt.title("Adjusted Median Validation Errors with Standard Error")
    plt.xlabel(f"Adjusted Time Step (x{common_k})")
    plt.ylabel("Validation Error (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure to the specified output path
    plt.savefig(output_path, format='pdf')
    plt.close()

# Save the graph as a PDF file
timestamp_dir = "/cluster/scratch/cdoumont/playground/bo_example/"
output_pdf_path = "/cluster/scratch/cdoumont/playground/bo_analysis.pdf"
generate_validation_error_graph(timestamp_dir, output_pdf_path)
