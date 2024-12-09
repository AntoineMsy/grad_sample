# python main.py ansatz.alpha=2

# python main.py ansatz.alpha=1 model.L=4 task=train device="3" lr=0.01 n_iter=500
# python main.py ansatz.alpha=2 model.L=4 task=train device="3" lr=0.01 n_iter=500
# python main.py ansatz.alpha=3 model.L=4 task=train device="3" lr=0.01 n_iter=500
# python main.py ansatz.alpha=4 model.L=4 task=train device="3" lr=0.01 n_iter=500
# python main.py ansatz.alpha=5 model.L=4 task=train device="3" lr=0.01 n_iter=500
# python main.py ansatz.alpha=6 model.L=4 task=train device="3" lr=0.01 n_iter=500
# python main.py ansatz.alpha=7 model.L=4 task=train device="3" lr=0.01 n_iter=500

# python main.py ansatz.alpha=1 task=train n_iter=2000
# python main.py ansatz.alpha=2 task=train n_iter=2000
# python main.py ansatz.alpha=3 task=train n_iter=4000
# python main.py ansatz.alpha=4 task=train n_iter=4000
# python main.py ansatz.alpha=5 task=train n_iter=6000
# python main.py ansatz.alpha=6 task=train n_iter=6000
# python main.py ansatz.alpha=7 task=train n_iter=6000

# python main.py ansatz.alpha=8 task=train n_iter=6000

# python main.py ansatz.alpha=1
# python main.py ansatz.alpha=2
# python main.py ansatz.alpha=3
# python main.py ansatz.alpha=4  
# python main.py ansatz.alpha=5 chunk_size_vmap=2
# python main.py ansatz.alpha=6 chunk_size_vmap=4
# python main.py ansatz.alpha=7 chunk_size_vmap=4
# python main.py ansatz.alpha=8 chunk_size_vmap=4

# python main.py ansatz.alpha=1 task=analysis_state
# python main.py ansatz.alpha=2 task=analysis_state
# python main.py ansatz.alpha=3 task=analysis_state
# python main.py ansatz.alpha=4 task=analysis_state 
# python main.py ansatz.alpha=5 task=analysis_state chunk_size_vmap=2
# python main.py ansatz.alpha=6 task=analysis_state chunk_size_vmap=4
# python main.py ansatz.alpha=7 task=analysis_state chunk_size_vmap=4
# python main.py ansatz.alpha=8 task=analysis_state chunk_size_vmap=4

# python main.py model=heisenberg1d model.sign_rule=False model.L=10 lr=0.001 n_iter=5000

# python main.py ansatz.alpha=1 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=2000
# python main.py ansatz.alpha=2 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=4000
# python main.py ansatz.alpha=3 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=4000
# python main.py ansatz.alpha=4 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=4000
# python main.py ansatz.alpha=5 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=6000
# python main.py ansatz.alpha=6 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=8000
# python main.py ansatz.alpha=7 model=heisenberg1d model.sign_rule=False task=train device="4" n_iter=8000

# python main.py ansatz.alpha=1 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=2000
# python main.py ansatz.alpha=2 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=4000
# python main.py ansatz.alpha=3 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=4000
# python main.py ansatz.alpha=4 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=4000
# python main.py ansatz.alpha=5 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=6000
# python main.py ansatz.alpha=6 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=8000
# python main.py ansatz.alpha=7 model=heisenberg1d model.sign_rule=True task=train device="4" n_iter=8000

# python main.py ansatz.alpha=1 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=2000
# python main.py ansatz.alpha=2 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=4000
# python main.py ansatz.alpha=3 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=4000
# python main.py ansatz.alpha=4 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=4000
# python main.py ansatz.alpha=5 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=6000
# python main.py ansatz.alpha=6 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=8000
# python main.py ansatz.alpha=7 model=heisenberg1d model.sign_rule=False task=analysis_dp device="4" n_iter=8000

# python main.py ansatz.alpha=1 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=2000
# python main.py ansatz.alpha=2 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=4000
# python main.py ansatz.alpha=3 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=4000
# python main.py ansatz.alpha=4 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=4000
# python main.py ansatz.alpha=5 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=6000
# python main.py ansatz.alpha=6 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=8000
# python main.py ansatz.alpha=7 model=heisenberg1d model.sign_rule=True task=analysis_dp device="4" n_iter=8000

# python main.py ansatz.alpha=2 model=heisenberg1d model.sign_rule=False model.L=14 task=analysis_dp device="4" chunk_size_vmap=100

# #!/bin/bash

# Define the ranges and parameters
# alphas=(1 2 3 4 5)
# iterations=(4000 6000 6000 6000 6000)
# # sign_rules=("False" "True")
# tasks=("train")

# # Loop over sign_rules, tasks, and alphas
# for sign_rule in "${sign_rules[@]}"; do
#   for task in "${tasks[@]}"; do
#     for i in "${!alphas[@]}"; do
#       alpha=${alphas[i]}
#       n_iter=${iterations[i]}
#       echo "Running: python main.py ansatz.alpha=${alpha} model=xxz task=${task} device='4' n_iter=${n_iter}"
#       python main.py ansatz.alpha=${alpha} model=xxz task=${task} device="4" n_iter=${n_iter}
#     done
#   done
# done


# python main.py ansatz=rnn model=heisenberg1d model.sign_rule=False model.L=14 task=analysis_dp device="3", diag_shift=4
python main.py sample_size=4
python main.py sample_size=8
python main.py sample_size=16
python main.py sample_size=32