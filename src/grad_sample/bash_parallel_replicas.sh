#!/bin/bash
# Fixed variables
sample_size=9
n_iter=8000

# Function to convert scientific notation to decimal format
convert_to_decimal() {
    printf "%.*f\n" 10 "$1"
}

# Function to generate a list of points
generate_list() {
    local start=$1
    local end=$2
    local points=$3
    local -a list=()

    # Convert start and end to decimal if necessary
    start=$(convert_to_decimal "$start")
    end=$(convert_to_decimal "$end")

    # Calculate the interval between points
    interval=$(echo "($end - $start) / ($points - 1)" | bc -l)

    # Generate the list of points
    for ((i=0; i<points; i++)); do
        value=$(echo "$start + $i * $interval" | bc -l)
        list+=("$value")
    done

    # Output the generated list
    echo "${list[@]}"
}


# Example usage:
# Generate the `lrs` list
lrs_start=0.0001
lrs_end=0.01
lrs_points=10
lrs=($(generate_list $lrs_start $lrs_end $lrs_points))


# Print the lists
echo "Generated lrs: ${lrs[@]}"
# Values to iterate over


# Function to generate and run commands for a specific device and is_mode
# run_is_mode_on_device() {
#     local device="$1"
#     local is_mode="$2"
#     for lr in "${lrs[@]}"; do
#         for diag_shift in "${diag_shifts[@]}"; do
#             cmd="python main.py sample_size=$sample_size device='$device' n_iter=$n_iter is_mode=$is_mode lr=$lr diag_shift=$diag_shift"
#             echo "Launching: $cmd"
#             eval "$cmd"
#         done
#     done
# }
run_is_mode_on_device() {
    local device="$1"
    local is_mode="$2"
    for lr in "${lrs[@]}"; do
        cmd="python main.py --config-name=train_is sample_size=$sample_size device='$device' n_iter=$n_iter is_mode=$is_mode"
        echo "Launching: $cmd"
        eval "$cmd"
    done
}

# is_modes=(2.0 1.0 0.0 0.5)
# # Assign each is_mode to a specific device
# for i in "${!is_modes[@]}"; do
#     device=$((i + 3))  # Devices are numbered 1, 2, ...
#     is_mode="${is_modes[$i]}"
#     run_is_mode_on_device "$device" "$is_mode" &
# done
# wait

# is_modes=(-1 0.0 0.5 1.0)
# # Assign each is_mode to a specific device
# for i in "${!is_modes[@]}"; do
#     device=$((i + 3))  # Devices are numbered 1, 2, ...
#     is_mode="${is_modes[$i]}"
#     run_is_mode_on_device "$device" "$is_mode" &
# done

is_modes=(2.0 1.8 1.6 1.4 1.2)
# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done
wait
is_modes=(1.0 0.8 0.6 0.5 0.4)
# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done
wait
is_modes=(0.2 0.1 0.01 0.001 0.0)
# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done
sample_size=10

is_modes=(2.0 1.8 1.6 1.4 1.2)
# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done
wait
is_modes=(1.0 0.8 0.6 0.5 0.4)
# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done
wait
is_modes=(0.2 0.1 0.01 0.001 0.0)
# Assign each is_mode to a specific device
for i in "${!is_modes[@]}"; do
    device=$((i + 1))  # Devices are numbered 1, 2, ...
    is_mode="${is_modes[$i]}"
    run_is_mode_on_device "$device" "$is_mode" &
done
wait



# is_modes=(2.0 0.2 0.6 0.8)
# wait
# # Assign each is_mode to a specific device
# for i in "${!is_modes[@]}"; do
#     device=$((i + 3))  # Devices are numbered 1, 2, ...
#     is_mode="${is_modes[$i]}"
#     run_is_mode_on_device "$device" "$is_mode" &
# done
# wait
# is_modes=(1.4 1.6 1.8)
# # Assign each is_mode to a specific device
# for i in "${!is_modes[@]}"; do
#     device=$((i + 3))  # Devices are numbered 1, 2, ...
#     is_mode="${is_modes[$i]}"
#     run_is_mode_on_device "$device" "$is_mode" &
# done
# # Wait for all background processes to complete

# echo "All tasks completed."
