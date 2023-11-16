import os

def delete_pth_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                file_path = os.path.join(root, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)

# Replace 'start_directory' with the path of your starting directory.
start_directory = "/cluster/scratch/cdoumont/playground/runs/bo/"
delete_pth_files(start_directory)