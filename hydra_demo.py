import subprocess

import hydra
import numpy as np
import pickle
from omegaconf import OmegaConf

##Adjust to the location of your code...
REPO_ROOT = "/home/aleksey/code/python/starforge_mult_vis/"

def get_git_commit() -> str:
    """Return the current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "UNKNOWN"

def is_repo_dirty() -> bool:
    """Return True if there are uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"],
            text=True
        ).strip()
        return bool(out)  # non-empty means dirty
    except subprocess.CalledProcessError:
        return False

# Resolver for Hydra
def git_state():
    commit = get_git_commit()
    dirty = is_repo_dirty()
    return f"{commit}{'*' if dirty else ''}"  # append '*' if dirty

# Register the resolver *before* Hydra runs
OmegaConf.register_new_resolver("git_commit", lambda: git_state(), use_cache=True)

@hydra.main(version_base=None)
def main(params):
    params = OmegaConf.to_container(params, resolve=True, structured_config_mode=False)

    M_values = np.linspace(0, 2*np.pi, params["nm"], endpoint = False) #100 even spaces from 0 to 2pi
    #Defining inclination
    cos_inc_values = np.linspace(-1, 1, params["ni"], endpoint=True)
    inc_values = np.arccos(cos_inc_values) #100 even spaces from 0 to pi
    ecc_values = [1e-3]
    Omega_values = [0]
    omega_values = [0]

    if params["full_grid"]:
        ecc_values = np.linspace(0, 1, params["ne"], endpoint=False)
        Omega_values = np.linspace(0, 2*np.pi, params["nO"], endpoint=False)
        omega_values = np.linspace(0, 2*np.pi, params["nw"], endpoint=False)
    
    coords = np.meshgrid(M_values, inc_values, ecc_values, Omega_values, omega_values, indexing="ij")
    MS, IS, ES, OMS, oms = coords
    coords = np.transpose([MS.ravel(), IS.ravel(), ES.ravel(), OMS.ravel(), oms.ravel()])

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(output_dir + "/coords.p", "wb") as ff:
        pickle.dump(coords, ff)
    with open(output_dir + "/params.p", "wb") as ff:
        pickle.dump(params, ff)

if __name__=="__main__":
    main()

