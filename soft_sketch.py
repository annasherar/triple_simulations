import pandas as pd
import numpy as np
from soft_correction import corr_pipeline

from scipy.spatial.transform import Rotation as R


##Start positions of binaries and initial companions.
ic = pd.read_parquet("data/init_state_all.pq")
##End positions of binaries and initial companions.
es = pd.read_parquet("data/end_state_all.pq")
##Applying various filters: (1) Halo mass cannot change too much from start to end (2) No new companions can appear from start to end (3) No particle pairs are
##with softening length of each other.
filt1 = ic["frac_halo_max"]<0.01
filt2 = ic["nonew_comps"]>0
# filt_soft = (~ic["is_soft"].astype(bool)) & (~es["is_soft"].astype(bool))
# filt = filt1 & filt2 & filt_soft
filt = filt1 & filt2
ic = ic.loc[filt]
es = es.loc[filt]

##You can play around with the trials here...
trials = 20

for trial in range(trials):
    ##For storing auxilliary data
    aux_data1 = []
    aux_data2 = []
    ##Reset after every trial -- Have to make a copy of the original initial conditions file(!!)
    ic_for_newtonian = ic.copy()
    ic_for_softened = ic.copy()
    ##Apply correction for softened binaries.
    ##To keep things simple, we will only consider exchanges where there are a total of three stars,
    ##and only one pair is softened. We will ignore all other encounters.
    ##We only want to apply the correction pipeline if we are integrating WITHOUT the softened gravitational force in REBOUND!!!

    ##Same random rotation for all binaries(!)
    r = R.random()
    ##Initial conditions for Newtonian simulations(!)
    ##Maybe add an additional structure to store additional data.
    flags = np.array([corr_pipeline(ic_for_newtonian.loc[ex1], aux_data1, r=r) for ex1 in ic_for_newtonian.index.unique()])
    idx_select = ic_for_newtonian.index.unique()[flags]
    ic_for_newtonian = ic_for_newtonian.loc[idx_select]
    ##Auxiliary data--applying same filtering
    aux_data1 = np.array(aux_data1)[flags]
    ##We also want to apply the rotation (r) only! -- this has some unnecessary work and complexity: try to get rid. At least make sure it is not causing issues!
    ic_for_softened = ic_for_softened.loc[idx_select]
    [corr_pipeline(ic_for_softened.loc[ex1], aux_data2, r=r, rot_only=True) for ex1 in ic_for_softened.index.unique()]
    aux_data2 = np.array(aux_data2)
    ##Filtered original ics for validation
    ic_filt = ic.loc[idx_select]

    ##We just want to randomize phase(!) -- Not other elements(!)--so get rid of rotation
    ##Hagai suggestion randomize Keplerian phase and then translate to softened potential??
    ##This is the inverse of the translation we normally have (softened positions and velocities --> Newtonian positions and velocities)
    ##Not sure how to do this? 
    ##May be easier to to numerically...
    cleaned_index = ic_for_softened.index.unique().to_numpy()
    ##Validation for transformation -- Double check indexing here...
    for ii in range(len(cleaned_index)):        
        #VALIDATION: MAKE SURE ONLY THE APPROPRIATE ROWS CHANGED
        ic_row = ic_for_newtonian.loc[cleaned_index[ii]]
        aux_row = aux_data1[ii]
        unchanged_index = np.array(~np.isin(np.array([0, 1, 2]), aux_row[2:].astype(int)))
        assert np.all(ic_row.iloc[unchanged_index] == ic_filt.loc[cleaned_index[ii]].iloc[unchanged_index])
        assert np.any(ic_row.iloc[int(aux_row[2])] != ic_filt.loc[cleaned_index[ii]].iloc[int(aux_row[2])])
        assert np.any(ic_row.iloc[int(aux_row[3])] != ic_filt.loc[cleaned_index[ii]].iloc[int(aux_row[3])])
        xrel_newtonian = ic_row.iloc[int(aux_row[2])][["x", "y", "z"]] - ic_row.iloc[int(aux_row[3])][["x", "y", "z"]]
        vrel_newtonian = ic_row.iloc[int(aux_row[2])][["vx", "vy", "vz"]] - ic_row.iloc[int(aux_row[3])][["vx", "vy", "vz"]]

        #VALIDATION: MAKE SURE ONLY THE APPROPRIATE ROWS CHANGED
        ic_row = ic_for_softened.loc[cleaned_index[ii]]
        aux_row = aux_data2[ii]
        unchanged_index = np.array(~np.isin(np.array([0, 1, 2]), aux_row[2:].astype(int)))
        assert np.all(ic_row.iloc[unchanged_index] == ic_filt.loc[cleaned_index[ii]].iloc[unchanged_index])
        assert np.any(ic_row.iloc[int(aux_row[2])] != ic_filt.loc[cleaned_index[ii]].iloc[int(aux_row[2])])
        assert np.any(ic_row.iloc[int(aux_row[3])] != ic_filt.loc[cleaned_index[ii]].iloc[int(aux_row[3])])
        xrel_soft = ic_row.iloc[int(aux_row[2])][["x", "y", "z"]] - ic_row.iloc[int(aux_row[3])][["x", "y", "z"]]
        vrel_soft = ic_row.iloc[int(aux_row[2])][["vx", "vy", "vz"]] - ic_row.iloc[int(aux_row[3])][["vx", "vy", "vz"]]

        ##VALIDATION: MAKE SURE THAT THE ORBITAL ANGULAR MOMENTA ARE THE SAME IN NEWTONIAN AND SOFTENED INITIAL CONDITIONS
        ell_newtonian = np.cross(xrel_newtonian, vrel_newtonian) / np.linalg.norm(xrel_newtonian) / np.linalg.norm(vrel_newtonian)
        ell_soft = np.cross(xrel_soft, vrel_soft) / np.linalg.norm(xrel_soft) / np.linalg.norm(vrel_soft)
        assert np.isclose(np.dot(ell_newtonian, ell_soft), 1)


        ##VALIDATION: INTEGRATE THE BINARY ONLY IN SOFTENED GRAVITY AND CHECK THAT THE APOCENTER AND PERICENTER ARE AS EXPECTED -- FOR ANNA! 
        ##THIS IS A BIT TRICKY, AND YOU MIGHT HAVE TO EXPERIMENT WITH THE INTEGRATION TIMES...
        #VALIDATION: VISUAL SANITY CHECKS ON ORBITS




    ##It seems like this line is wrong! es should also be reset...
    # es = es.loc[idx_select]
    ###YOUR NORMAL INTEGRATION PIPELINE WOULD GO HERE!


