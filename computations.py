import copy
import warnings

import numpy as np

sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
mcol = np.where(sink_cols == "m")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]


def get_com_winf(paths):
    ##If particles don't exist set all path variables to 0 (i.e. non-existent particles will not affect the com).
    paths2 = np.copy(paths)
    paths2[np.isinf(paths2)] = 0

    tmp_ms = paths2[:, :, mcol]
    ##Have to do this reshaping for the division...
    tmp_ms.shape = (len(tmp_ms), -1, 1)
    ##Time series of total mass of particles
    tot_mass = np.sum(tmp_ms, axis=0)
    ##If the total mass is 0 number of particles is 0, replace mass with infinity to flag these times should be filtered from the division
    tot_mass[tot_mass == 0] = np.inf
    com = path_divide_3d(
        np.sum((tmp_ms * paths2[:, :, pxcol : pzcol + 1]), axis=0), tot_mass
    )

    return com


def path_divide_3d(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:, 0])) & (~np.isinf(p2[:, 0]))
    print(p1[filt].shape, p2[filt].shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            diff[filt] = p1[filt] / p2[filt]
        except RuntimeWarning:
            breakpoint()

    return diff


def max_pairwise_dist(pos_list):
    max_dist = 0
    for ii in range(len(pos_list)):
        for jj in range(ii + 1, len(pos_list)):
            tmp_dist = np.linalg.norm(pos_list[ii] - pos_list[jj])
            if tmp_dist > max_dist:
                max_dist = tmp_dist

    return max_dist


def get_bound_snaps_adjust(bin_list, high_df):
    ##Use high_df table to get more stringent binary snapshots(!!!)
    curr_bin_list = copy.copy(bin_list)
    curr_bin_list.sort()
    bin_sel = high_df.loc[str(curr_bin_list)]

    return bin_sel
