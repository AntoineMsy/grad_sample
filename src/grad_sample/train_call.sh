#!/bin/bash

alphas=(1 2 3 4 5 6 7)
model="heisenberg1d"
device="3"
lr=0.001
n_iters=(2000 2000 2000 4000 4000 6000 6000)
L=14
task="train"
for i in "${!alphas[@]}"; do
alpha=${alphas[i]}
n_iter=${n_iters[i]}
echo "Running: python main.py ansatz.alpha=${alpha} model=${model} model.L=${L} task=${task} device=${device} lr=${lr} n_iter=${n_iter}"
python main.py ansatz.alpha=${alpha} model=${model} model.L=${L} task=${task} device=${device} lr=${lr} n_iter=${n_iter}
done