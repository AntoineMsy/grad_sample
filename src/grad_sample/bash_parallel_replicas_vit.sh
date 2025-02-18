#!/bin/bash

# Fixed variables
sample_size=10
n_iter=6000

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
lrs_points=2
lrs=($(generate_list $lrs_start $lrs_end $lrs_points))

# Print the lists
echo "Generated lrs: ${lrs[@]}"
# Values to iterate over
#!/bin/bash

run_is_mode_on_device() {
    local device="$1"
    local is_mode="$2"
    local J_value="$3"
    local sample_size="$4"  # New argument to pass J override value
    # for lr in "${lrs[@]}"; do
    # cmd="python main.py --config-name=vit_large_nosym sample_size=$sample_size device='$device' n_iter=$n_iter is_mode=$is_mode $J_value"
    cmd="python main.py --config-name=vit_large_nosym sample_size=$sample_size device='$device' n_iter=$n_iter is_mode=$is_mode"
    echo "Launching: $cmd"
    eval "$cmd"
    # done
}

run_group() {
    local is_modes=("$@")  # Capture all arguments as an array
    for i in "${!is_modes[@]}"; do
        run_is_mode_on_device "$((i + 1))" "${is_modes[i]}" "$J_value" &
    done
    wait
}

run_group_samples() {
    local sample_sizes=("$@")  # Capture all arguments as an array
    for i in "${!sample_sizes[@]}"; do
        run_is_mode_on_device "$((i + 1))" "$is_mode" "$J_value" "${sample_sizes[i]}" &
    done
    wait
}
# Values to iterate over for J
J_values=("model.J=[1.0,0.4]" "model.J=[1.0,0.8]")
J_values=("model.J=[1.0,0.5]")
is_mode=1.0
for J_value in "${J_values[@]}"; do
    run_group_samples 8 9 10
done
is_mode=0.5
for J_value in "${J_values[@]}"; do
    run_group_samples 8 9 10
done

is_mode=1.8
for J_value in "${J_values[@]}"; do
    run_group_samples 8 9 10
done
# First batch
# sample_size=10
# for J_value in "${J_values[@]}"; do
#     run_group 2.0 1.8 1.5 1.0 0.5 0.1
# done

# # Second batch with updated sample_size
# sample_size=11
# for J_value in "${J_values[@]}"; do
#     run_group 2.0 1.8 1.5 1.0 0.5 0.1
# done


# Second batch with updated sample_size

# for J_value in "${J_values[@]}"; do
#     run_group 2.0 1.8 1.5 1.0 0.5 0.1
# done
