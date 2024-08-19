import json
import os

def main():
    # Specify the parent directory
    parent_dir = "viz/real images"
    json_file_path = "facescrub_idx_to_class.json"

    with open(json_file_path, 'r') as json_file:
        # Load the JSON data
        data = json.load(json_file)

    # Iterate through all directories and subdirectories
    for directory in os.listdir(parent_dir):
        new_name = None

        # Create the full path of the current subdirectory
        current_path = os.path.join(parent_dir, directory)

        for key, value in data.items():
            if value == directory:
                print(f"Key for value '{directory}': {key}")
                new_name = "ID_" + f"{key}"
                break
        else:
            print(f"Value '{directory}' not found in the dictionary.")
        
        # Create the new full path with the new name
        if new_name:
            new_path = os.path.join(parent_dir, new_name)
            
            try:
                # Rename the subdirectory
                os.rename(current_path, new_path)
                print(f"Renamed: {current_path} -> {new_path}")
            except OSError as e:
                print(f"Error renaming {current_path}: {e}")
        else:
            continue

if __name__ == '__main__':
    main()