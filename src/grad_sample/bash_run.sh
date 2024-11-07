# python main.py ansatz.alpha=2

# python main.py ansatz.alpha=1 model.L=4 task=train device="3" lr=0.01 n_iter=500
python main.py ansatz.alpha=2 model.L=4 task=train device="3" lr=0.01 n_iter=500
python main.py ansatz.alpha=3 model.L=4 task=train device="3" lr=0.01 n_iter=500
python main.py ansatz.alpha=4 model.L=4 task=train device="3" lr=0.01 n_iter=500
python main.py ansatz.alpha=5 model.L=4 task=train device="3" lr=0.01 n_iter=500
python main.py ansatz.alpha=6 model.L=4 task=train device="3" lr=0.01 n_iter=500
python main.py ansatz.alpha=7 model.L=4 task=train device="3" lr=0.01 n_iter=500

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

python main.py ansatz.alpha=1 model.L=4 lr=0.01 chunk_size_vmap=10