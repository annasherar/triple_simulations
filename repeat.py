import pandas as pd
from collections import Counter

##End positions of binaries and initial companions.
df = pd.read_parquet("softened_trials.pq")

# stt = pd.read_parquet("data/end_state_all_time3.pq")

# df = pd.read_parquet("data/soft_trials.pq")

# ##Occurrences of all IDs
# id_counts = Counter(es["id"])


# ## --------------------------- Total number of exchanges that have repeats ----------------------------------------
# binaries_with_repeats = set()
# for id_val, count in id_counts.items():
#     if count > 1:
#         binaries = es[es["id"] == id_val].index.unique()
#         binaries_with_repeats.update(binaries)
#         #print(f"ID {id_val} appears {count} times in binaries: {list(binaries)}")
# print(f"\nTotal number of unique binaries with repeated IDs: {len(binaries_with_repeats)}")


## --------------------------------- Total number of IDs that repeat ----------------------------------------------
# num_repeating_ids = sum(1 for count in id_counts.values() if count > 1)
# print(f"Number of IDs that repeat more than once: {num_repeating_ids}")


## ------------------------------------ Exchanges with same IDs ---------------------------------------------------
# Storing each binary's ID set in a dictionary
# binary_id_sets = {
#     'A':set([3,2,1]),
#     'B':set([1,2,3]),
#     'C':set([4,5,6]),
#     'D':set([4,5,3]),
#     'E':set([3,1,2]),
#     'F':set([6,4,5]),
#         binary_id: set(group["id"])
#     for binary_id, group in es.groupby(es.index)
# }

# # Compare each group with all others
# checked = set()
# for id1, id_set1 in binary_id_sets.items():
#     for id2, id_set2 in binary_id_sets.items():
#         if id1 >= id2:
#             continue  # avoid duplicates and self-comparisons
#         if id_set1 == id_set2:
#             checked.add(id1)
#             checked.add(id2)
# print(checked)            
# print(f"Amount of unique binaries that have repeated IDs: {len(checked)}")
print(df)        



