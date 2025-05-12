# io/loaders.py
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

def npz_stack(npz_list):
    """
    Stack data from different seeds
    """
    # Dictionary to hold lists of arrays for each key
    data_dict = defaultdict(list)

    # Loop over .npz files in the directory
    for filename in npz_list:
        # Load the .npz file
        data = np.load(filename, allow_pickle=True)

        # Iterate over keys in the .npz file
        for key in data.keys():
            data_dict[key].append(data[key])  # Append data for this key
    concatenated_data_dict = {key: np.concatenate(arrays) for key, arrays in data_dict.items()}

    return concatenated_data_dict

def load_pickles(seed_paths, filename):
    merged = {}
    for path in seed_paths:
        with open(f"{path}/{filename}", "rb") as f:
            tmp = pickle.load(f)
            assert not any(k in merged for k in tmp), f"Overlapping keys in {filename}"
            merged.update(tmp)
    return merged

def load_parquets(seed_paths, flat_suff):
    return pd.concat([pd.read_parquet(f"{path}/mults{flat_suff}.pq") for path in seed_paths])

