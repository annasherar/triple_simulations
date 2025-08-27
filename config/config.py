# config/config.py
from omegaconf import OmegaConf
import os

def load_config(user_config_path="config/fig_config.yaml"):
    default_config = OmegaConf.create({
        "contig_suff": "",
        "flat_suff": "_flat",
        "smao": "False",
        "my_tides": False,
        "my_ft": 1.0,
        "seeds": [1, 2, 42],
        "base_path": "data/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
    })

    if os.path.exists(user_config_path):
        user_config = OmegaConf.load(user_config_path)
        return OmegaConf.merge(default_config, user_config)
    return default_config