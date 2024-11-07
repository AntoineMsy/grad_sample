import re
import os
import yaml

def get_overlap_runs(output_prefix, key = "fid_ev"):
    
    # Regular expression to match filenames in the format fid_ev_N.npz
    pattern = re.compile(r"%s(\d+)_state\.npz"%key)

    # List to store all the found numbers
    numbers = []

    # Iterate over all files in the folder
    for filename in os.listdir(output_prefix):
        match = pattern.match(filename)
        if match:
            # Extract the number and convert it to an integer
            N = int(match.group(1))
            numbers.append(N)

    # Output the list of numbers
    return sorted(numbers)

def load_yaml_to_vars(yaml_path):
    # Open and parse the YAML file
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create variables dynamically
    for key, value in config.items():
        globals()[key] = value
        print(f"{key} = {globals()[key]}")  # Print each variable for confirmation

def get_unique_run_name_from_logs(directory, base_name):
    """
    Scan a folder to find all files matching the pattern 'base_name_N' 
    and return the smallest available integer N.
    
    Args:
    - directory (str): The path to the folder to search (e.g., './logs').
    - base_name (str): The base name to look for in the filenames (e.g., 'name').
    
    Returns:
    - str: A unique run name in the format 'base_name_N'.
    """
    # Regular expression pattern to match files like 'name_N' (where N is an integer)
    pattern = re.compile(rf'^{re.escape(base_name)}_(\d+)(\..+)?$')

    # Initialize a set to hold the used numbers
    used_numbers = set()

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the number N and add it to the set
            used_numbers.add(int(match.group(1)))

    # Find the smallest unused number
    N = 0
    while N in used_numbers:
        N += 1

    # Return the unique run name
    return f"{base_name}_{N}"
