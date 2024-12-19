#!/bin/bash

# Define device-specific commands as strings, separated by newlines
# device_1_commands="python main.py sample_size=9 device='1' n_iter=6000 is_mode=0.005
# python main.py sample_size=9 device='1' n_iter=6000 is=0.005"

# device_2_commands="python main.py sample_size=9 device='2' n_iter=6000 is_mode=0.5
# python main.py sample_size=9 device='2' n_iter=8000 is_mode=1.3"

# device_3_commands="python main.py sample_size=9 device='3' n_iter=6000 is_mode=1.
# python main.py sample_size=9 device='3' n_iter=6000 is_mode=1."

device_4_commands="python main.py sample_size=9 device='4' n_iter=6000 is_mode=0.8
python main.py sample_size=8 device='4' n_iter=6000 is_mode=0.8"

device_5_commands="python main.py sample_size=9 device='5' n_iter=6000 is_mode=2.0
python main.py sample_size=9 device='5' n_iter=6000 is_mode=1.7"

device_6_commands="python main.py sample_size=9 device='6' n_iter=6000 is_mode=hpsi
python main.py sample_size=9 device='6' n_iter=6000 is_mode=0."

# Function to process commands for a single device
run_device_commands() {
    local commands="$1"  # String of commands
    while IFS= read -r cmd; do
        echo "Launching: $cmd"
        eval "$cmd"
    done <<< "$commands"
}

# Launch each device's commands in parallel
run_device_commands "$device_1_commands" &
run_device_commands "$device_2_commands" &
run_device_commands "$device_3_commands" &
run_device_commands "$device_4_commands" &
run_device_commands "$device_5_commands" &
run_device_commands "$device_6_commands" &

# Wait for all background processes to complete
wait

echo "All tasks completed."
