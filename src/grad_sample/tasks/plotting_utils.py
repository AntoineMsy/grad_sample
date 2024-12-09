"""
Script to hold functions for plotting full sum cutting results
takes in mainly a base_config file to load vstates
will find all computed ngs truncation in the folder and will return the relevant quantities for each of them
wavefunction, pdf, jacobian, H psi, jacobian
in a dict format/ json file ?
  {"run_cfg": dict with basic cfg to know what we're plotting,
    "its" : iterations for which ngd truncation have been computed
    "psi" : dense array rep of psi, (batched over numbers)
    "pdf" : pdf, batched
    "Hpsi" : H @ psi, batched
    "Jac" : non centered, non rescaled, jacobian of psi, batched
    "fid_ev" : list of overlap evolutions arrays, 
    "removed_indices": list of removed indices arrays
 }
"""

from basic_test import  _setup
import jax.numpy as jnp
import netket as nk
import numpy as np
# import netket_pro as nkp
import netket.jax as nkjax
import flax
import os
from tqdm import tqdm
import flax.serialization
import copy
import argparse
import json
import yaml
from grad_sample.utils import get_overlap_runs
from scipy.sparse.linalg import eigsh

def load_yaml_to_vars(yaml_path):
    # Open and parse the YAML file
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create variables dynamically
    for key, value in config.items():
        globals()[key] = value
        print(f"{key} = {globals()[key]}")  # Print each variable for confirmation

def main():

    parser = argparse.ArgumentParser(description="Parse YAML config to variables.")
    parser.add_argument("-y", "--y", required=False, type=str, help="Path to the YAML configuration file.", default="config/base_config.yaml")

    # Parse arguments
    args = vars(parser.parse_args())
    load_yaml_to_vars(args["y"])
    print(globals())
    os.environ["CUDA_VISIBLE_DEVICES"]= device

    is_complex=True
    
    output_prefix = f"/scratch/.amisery/optimization_run/state_{save_every}_{alpha}_{L}"
    if diag_shift != 1e-4:
        output_prefix = f"/scratch/.amisery/optimization_run/state_{save_every}_{alpha}_{L}_{int(-jnp.log10(diag_shift))}"

    H, opt, vstate = _setup(
        L=L,
        n_samples=n_samples,
        lr=lr,
        is_complex=is_complex,
        full_summation=full_summation,
        alpha=alpha
    )
    H_sp = H.to_sparse()
        
    eig_vals, eig_vecs = eigsh(
        H_sp, k=2, which="SA"
    )  # k is the number of eigenvalues desired,
    E_gs = eig_vals[0]  # "SA" selects the ones with smallest absolute value

    # load the optimization run energies
    log_opt = output_prefix + ".log"
    
    data=json.load(open(log_opt))
    E_err=  jnp.abs((data["Energy"]["Mean"]["real"] - E_gs)/E_gs)

    # Find the iteration numbers for which post proc has been done
    numbers = get_overlap_runs(output_prefix)

    # declare fields for batched dictionary
    out_dict = {"run_cfg": args["y"], "its": numbers, "E_gs": E_gs, "E_err": E_err}
    psi_arr = []
    pdf_arr = []
    H_psi_arr = []
    jac_arr = []
    overlap = []
    removed_indices = []

    for n in tqdm(numbers):
        variables=None
        vs = copy.copy(vstate)
        with open(output_prefix + "/%d.mpack"%n, 'rb') as file:
            vs.variables = flax.serialization.from_bytes(variables, file.read())

        # Compute jacobian in full summation using the new vstate
        pdf = vs.probability_distribution()
        H_sp = H.to_sparse()
        # to array returns the exponential of vs.model.apply (log wf), normalized
        Hloc = H_sp @ vs.to_array() / vs.to_array()

        chunk_size_jac = 2
        
        # jacobian not centered bc it is handled dynamically in the computation
        jacobian_orig = nkjax.jacobian(#only for FullSumState
            vs._apply_fun,
            vs.parameters,
            vs.hilbert.all_states(), #in MC state, this is vstate.samples
            vs.model_state,
            pdf=pdf,
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=chunk_size_jac,
            _sqrt_rescale=False, #rescale by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )

        psi_arr.append(vs.to_array())
        pdf_arr.append(pdf)
        H_psi_arr.append(H_sp @ vs.to_array())
        jac_arr.append(jacobian_orig)

        fid_ev = jnp.load(output_prefix + "/fid_ev%d_state.npz"%n , allow_pickle=True)["arr_0"]
        basis_idx = jnp.load(output_prefix + "/removed_indices%d_state.npz"%n , allow_pickle=True)["arr_0"]
        overlap.append(fid_ev)
        removed_indices.append(basis_idx)
    
    # make dict and save as json
    out_dict["psi"] = jnp.stack(psi_arr)
    out_dict["pdf"] = jnp.stack(pdf_arr)
    out_dict["hpsi"] = jnp.stack(H_psi_arr)
    out_dict["jac"] = jnp.stack(jac_arr)
    out_dict["overlap"] = overlap
    out_dict["removed_indices"] = removed_indices
    
    jnp.savez(f"out_rbm_{save_every}_{alpha}_{L}_{int(-jnp.log10(diag_shift))}_state.npz", out_dict)
    # with open("out_rbm_{save_every}_{alpha}_{L}.json", "w") as file:
    #     json.dump(out_dict, file, indent=4)  # indent=4 for pretty formatting

if __name__ == "__main__":
    main()