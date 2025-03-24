#!/bin/bash

# Function to launch commands for given devices
launch_jobs() {
    local config_name=$1
    local devices=("${!2}")
    
    for device in "${devices[@]}"; do
        cmd="python main.py --config-name=$config_name device='$device'"
        echo "Launching: $cmd"
        eval "$cmd" &
    done
    wait
}

# Array of devices
devices=(4 5 6)

# Accept config-name as an argument (default to 'qchem_fs' if not provided)
config_name=${1:-qchem_fs}

# Launch the jobs
for _ in {1..4}; do
    launch_jobs "$config_name" devices[@]
done