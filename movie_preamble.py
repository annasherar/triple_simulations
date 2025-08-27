import pickle
from collections import defaultdict

import numpy as np

from config.config import load_config
from data_io.loaders import load_parquets, load_pickles, npz_stack

cfg = load_config()
suffix = f"/analyze_multiples_output__Tides{cfg.my_tides}_smao{cfg.smao}_mult4_ngrid1_hmTrue_ft{cfg.my_ft}_coFalse"
seed_paths = [f"{cfg.base_path}{seed}{suffix}" for seed in cfg.seeds]
npz_files = [f"{path}/dat_coll_mult.npz" for path in seed_paths]

my_data = npz_stack(npz_files)
high_df = load_parquets(seed_paths, cfg.flat_suff)

path_lookup = load_pickles(seed_paths, "path_lookup.p")
spin_lookup = load_pickles(seed_paths, "spin_lookup.p")
lookup_dict = load_pickles(seed_paths, "lookup_dict.p")

##Filtering high multiples table
high_df = high_df.loc[
    (high_df[f"frac_of_orbit{cfg.contig_suff}"] >= 1)
    & (high_df[f"nbound_snaps{cfg.contig_suff}"] > 1)
]
tval = high_df.index.get_level_values("t")
high_df["tval"] = tval

smao_suffix = ""
if cfg.smao=="False":
    smao_suffix = "_smaoFalse"
##Separate from the rest of the data hierarchy--because not spit by seeds?!
##Should at least have some marking for smao!!!
ex_time_max = np.load(
    f"data/pmult_before_bin_{cfg.my_ft}{cfg.flat_suff}{cfg.contig_suff}{smao_suffix}.npz"
)["ex_time_max"]
ex_time_max_end = np.load(
    f"data/pmult_before_bin_{cfg.my_ft}{cfg.flat_suff}{cfg.contig_suff}{smao_suffix}.npz"
)["ex_time_max_end"]
ex_filt = ~np.isinf(ex_time_max)
ex_filt = ex_filt & my_data[f"quasi_filter{cfg.contig_suff}"]
ex_index = np.where(ex_filt)[0]


print(cfg)
##Separate from the rest of the data hierarchy--because not spit by seeds?!
##Should at least have some marking for smao!!!
with open(f"data/companions{cfg.flat_suff}{cfg.contig_suff}{smao_suffix}.p", "rb") as ff:
    comps_dict = pickle.load(ff)
