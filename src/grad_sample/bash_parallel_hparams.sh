#!/bin/bash

# Fixed variables
sample_size=9
n_iter=8000

# Values to iterate over
is_modes=(2.0 1.0 0.0 0.5 hpsi)
lrs=(0.0001 0.0005 0.001 0.005 0.01 0.05)
diag_shifts=(1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8 1e-8, 5e-9, 1e-9)

# Function to generate and run commands for a specific device and is_mode
run_is_mode_on_device() {
    local device="$1"
    local is_mode="$2"
    for lr in "${lrs[@]}"; do
        for diag_shift in "${diag_shifts[@]}"; do
            cmd="python main.py sample_size=$sample_size device='$device' n_iter=$n_iter is_mode=$is_mode lr=$lr diag_shift=$diag_shift"
            echo "Launching: $cmd"
            eval "$cmd"
        done
    done
}

# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done

# Wait for all background processes to complete
wait

echo "All tasks completed."
