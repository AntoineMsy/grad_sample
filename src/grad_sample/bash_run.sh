#!/bin/bash
devices=(4 , 5 , 6)
for device in "${devices[@]}"; do
        cmd="python main.py --config-name=qchem_fs device='$device' "
        echo "Launching: $cmd"
        eval "$cmd" &
    done
wait
for device in "${devices[@]}"; do
        cmd="python main.py --config-name=qchem_fs device='$device' "
        echo "Launching: $cmd"
        eval "$cmd" &
    done
wait
for device in "${devices[@]}"; do
        cmd="python main.py --config-name=qchem_fs device='$device' "
        echo "Launching: $cmd"
        eval "$cmd" &
    done
wait
for device in "${devices[@]}"; do
        cmd="python main.py --config-name=qchem_fs device='$device' "
        echo "Launching: $cmd"
        eval "$cmd" &
    done