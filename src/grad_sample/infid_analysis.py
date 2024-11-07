from basic_test import RBM, _setup
import jax
import jax.numpy as jnp
from tqdm import tqdm
import netket as nk
# import netket_pro as nkp
import netket.jax as nkjax
import flax.linen as nn
import flax
import matplotlib.pyplot as plt
import os
import flax.serialization
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="4"

class SingleRunAnalysis:
    def __init__(self, config, its):
        # takes a yaml config file and instantiates everything necessary to perform an analysis of a run and save the results
        a = 0
        self.results_dict = {}

        self.vs = None
        self.H = None
    def compute_results_fixedit(self, it):
        FixedItAnalysis(self.vs, )
    def save_results(self):
        return None

class FixedItAnalysis:
    def __init__(self, vs, H, diag_shift):
        self.vs = vs
        self.H = H

class MultiRunAnalysis:
    def __init__(self, configs):

        for config in configs:
            srun = SingleRunAnalysis(config)