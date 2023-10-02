import os
import json

import numpy as np

# Path to the parent directory
dir_path = 'playground/runs/20230713_110156/cifar10/predictors/gpwl'

# This list will contain all the json content
json_content = []

# os.walk gives you a generator that produces file names in a directory tree
# by walking the tree either top-down or bottom-up
for dirpath, dirnames, filenames in os.walk(dir_path):
    for file in filenames:
        # Check if the file is 'errors.json'
        if file == 'errors.json':
            # Construct full file path
            file_path = os.path.join(dirpath, file)
            
            # Open, read and append json file content to json_content list
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Assuming that data is a list
                json_content.append(data)

# Now json_content list contains all json content from 'errors.json' files
print(len(json_content))

# Create a numpy array of size len(json_content)
rmses = np.zeros(len(json_content))

# Iterate over json_content list
for i, json in enumerate(json_content):
    # Iterate over each json file
    for key, value in json[1].items():
        # Check if the key is 'rmse'
        if key == 'spearman':
            # Append the value to numpy array
            rmses[i] = value

# Print the mean of the numpy array
print(np.mean(rmses))

# Print the standard deviation of the numpy array
print(np.std(rmses))

# Print the minimum of the numpy array
print(np.min(rmses))

# Print the maximum of the numpy array
print(np.max(rmses))

# Print the median of the numpy array
print(np.median(rmses))

# Print all indices of the numpy array where the value is higher than 2.0
indices = np.where(rmses > 2.0)
print(indices)

# Print all values of the numpy array where the value is higher than 2.0
values = print(rmses[indices])