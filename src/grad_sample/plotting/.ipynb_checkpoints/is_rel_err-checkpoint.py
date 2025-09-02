import argparse
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import colors
from matplotlib.cm import viridis, seismic, coolwarm, Spectral

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Specify model and coupling parameters.")
parser.add_argument("-m", "--model", type=str, required=False, default="J1J2", help="Model type (e.g., J1J2)")
parser.add_argument("-c", "--coupling", type=str, required=False, default=0.5, help="Coupling value (e.g., 0.5)")
args = parser.parse_args()

# Use parsed arguments
model = args.model
coupling = args.coupling

# Define other parameters
width = 6
base_path = os.environ['SCRATCH'] + f"/vit_runs_random/{model}_{coupling}/L36/ViT2D/4_24_12"

# Example placeholder print statement
print(f"Model: {model}, Coupling: {coupling}, Base path: {base_path}")

# for larger models print either energy per site or rel err wrt litterature
mc_samples = [8, 9, 10, 11, 12, 13]
is_keys = ['0.1', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0']
is_keys = ['0.1', '0.4', '0.8', '1.0', '1.2', '1.6', '2.0', "isauto"]

if "is_auto" in is_keys:
    colors = viridis(jnp.linspace(0, 1,len(is_keys)))
else:
    colors = viridis(jnp.linspace(0, 1,len(is_keys)-1))

fig, ax = plt.subplots(figsize=(12,8))

for idx, key in enumerate(is_keys):
    m_list = []
    err_list = []
    for mc_sample in mc_samples:
        main_folder = base_path + f"/MC_{mc_sample}_{key}"
        
        m_list_per_samp = []
        m_rel = 100
        good_run = False
        # Iterate through subfolders
        try:
            for subfolder in os.listdir(main_folder):
                subfolder_path = os.path.join(main_folder, subfolder)
                num_nan_runs = 0
                num_cv = 0
                num_runs = 0
                # Check if the subfolder name is in the form a_b
                # Process run log files in the subfolder
                good_run = True
                if os.path.isdir(subfolder_path):
                    for file in os.listdir(subfolder_path):
                        if file.startswith("run_") and file.endswith(".log"):
                            file_path = os.path.join(subfolder_path, file)
                            try:
                                with open(file_path, 'r') as f:
                                    log_data = json.load(f)
                                    rel_err = jnp.array(log_data['rel_err']['value'])
                                        
                                    # m_rel = min(min(rel_err[rel_err>0]), m_rel)
                                    # print(len(rel_err))
                                    if len(rel_err) == 120:
                                        if key=="isauto":
                                            alpha_opt = jnp.array(log_data['info']['alpha']['value'])
                                            alpha_inf = jnp.mean(alpha_opt[-1000:])
                                        good_run = True
                                        m_list_per_samp.append(min(rel_err[rel_err>0]))
                                    # if jnp.nan in rel_err or min(rel_err) > 1e-2:
                                    #     num_nan_runs +=1
                                    #     print('nan found')
                            except (json.JSONDecodeError, KeyError):
                                print(f"Warning: Could not parse file {file_path}")
            # if good_run:
            #     m_list_per_samp.append(m_rel)
        except:
            pass 
        
        m_list.append(jnp.mean(jnp.array(m_list_per_samp)))
        err_list.append(jnp.std(jnp.array(m_list_per_samp)))
    # plt.plot(rel_err_it, mean_rel, color = colors[idx], label=r'$\alpha = %s$'%key + ' , failure ratio %.2f'%(1 - num_cv/num_runs))
    m_list = jnp.array(m_list)
    err_list = jnp.array(err_list)
    if key=="isauto":
        print(alpha_inf)
        ax.plot(2**jnp.array(mc_samples[:len(m_list)]), m_list, marker = 'o', linestyle = '--', color = 'black', label=r'adaptive, $\bar{\alpha}_{\infty} = %.2f$'%(alpha_inf))
        ax.fill_between(2**jnp.array(mc_samples[:len(m_list)]), m_list + err_list/2, m_list - err_list/2, color = 'black', alpha=0.3)
    else:
        ax.plot(2**jnp.array(mc_samples[:len(m_list)]), m_list, marker = 'o', linestyle = '--', color = colors[idx], label=r'$\alpha = %s$'%key)
        ax.fill_between(2**jnp.array(mc_samples[:len(m_list)]), m_list + err_list/2, m_list - err_list/2, color = colors[idx], alpha=0.3)
   
ax.set_xscale('log', base=2)
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('sample size')
ax.set_ylabel('Best relative error')
if model == "J1J2":
    ax.set_ylim(1e-3, 1e-1)
    ax.set_title(f'Scaling of relative error vs samples, same hparams, 6x6 {model}, J2 = {coupling}')
elif model == "Square_Heisenberg":
    ax.set_ylim(1e-6, 1e-1)
    ax.set_title('Scaling of relative error vs samples, same hparams, 6x6 Heisenberg model')

lgd = ax.legend(bbox_to_anchor=(1,1))
fig.tight_layout()
fig.savefig(f'rel_err_scaling_{model}_{coupling}.png')