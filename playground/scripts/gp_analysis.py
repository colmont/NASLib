import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

def extract_train_size_single(configs_dir):
    train_size_single_values = []
    for config_file in os.listdir(configs_dir):
        with open(os.path.join(configs_dir, config_file), 'r') as f:
            yaml_content = yaml.safe_load(f)
            train_size_single_values.append(yaml_content['train_size_single'])
    unique_train_size_single = set(train_size_single_values)
    if len(unique_train_size_single) != 1:
        raise ValueError(f"Inconsistent train_size_single values in {configs_dir}")
    return unique_train_size_single.pop()

def extract_metric_values(model_dir, metric):
    metric_values = []
    for number_dir in os.listdir(model_dir):
        error_file_path = os.path.join(model_dir, number_dir, 'errors.json')
        if os.path.exists(error_file_path):
            with open(error_file_path, 'r') as f:
                json_content = json.load(f)
                metric_values.append(json_content[1][metric])
    return metric_values

def plot_data(timestamps_dir, output_path, metric):
    data = {
        'gp_heat': {'train_sizes': [], 'medians': [], 'std_errors': []},
        'gpwl': {'train_sizes': [], 'medians': [], 'std_errors': []}
    }

    for timestamped_dir in os.listdir(timestamps_dir):
        if timestamped_dir == '.DS_Store':
            continue
        configs_dir = os.path.join(timestamps_dir, timestamped_dir, 'cifar10', 'configs', 'predictors')
        train_size_single = extract_train_size_single(configs_dir)

        for model_type in ['gp_heat', 'gpwl']:
            model_dir = os.path.join(timestamps_dir, timestamped_dir, 'cifar10', 'predictors', model_type)
            metric_values = extract_metric_values(model_dir, metric)
            median_metric = np.median(metric_values)
            std_error_metric = np.std(metric_values) / np.sqrt(len(metric_values))
            data[model_type]['train_sizes'].append(train_size_single)
            data[model_type]['medians'].append(median_metric)
            data[model_type]['std_errors'].append(std_error_metric)

    # Sorting the data by train_size_single values before plotting
    for model_type in ['gp_heat', 'gpwl']:
        sorted_indices = np.argsort(data[model_type]['train_sizes'])
        data[model_type]['train_sizes'] = np.array(data[model_type]['train_sizes'])[sorted_indices].tolist()
        data[model_type]['medians'] = np.array(data[model_type]['medians'])[sorted_indices].tolist()
        data[model_type]['std_errors'] = np.array(data[model_type]['std_errors'])[sorted_indices].tolist()

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for model_type, color in [('gp_heat', 'blue'), ('gpwl', 'red')]:
        plt.errorbar(
            data[model_type]['train_sizes'], 
            data[model_type]['medians'], 
            yerr=data[model_type]['std_errors'], 
            fmt='-o', label=model_type, color=color
        )
    ax.set_xlabel('Train Size Single')
    ax.set_ylabel(f'Median {metric}')
    ax.set_title(f'Median {metric} vs. Train Size Single for gp_heat and gpwl')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')

# Save the graph as a PDF file
metric = "rmse"
timestamps_dir = "/cluster/scratch/cdoumont/playground/gp_example/"
output_pdf_path = "/cluster/scratch/cdoumont/playground/gp_analysis.pdf"
plot_data(timestamps_dir, output_pdf_path, metric)
