import pandas as pd
import numpy as np

##Start positions of binaries and initial companions.
ic = pd.read_parquet("data/init_state_all.pq")
##End positions of binaries and initial companions.
es = pd.read_parquet("data/end_state_all.pq")
##Applying various filters: (1) Halo mass cannot change too much from start to end (2) No new companions can appear from start to end (3) No particle pairs are
##with softening length of each other.
filt1 = ic["frac_halo_max"]<0.01
filt2 = ic["nonew_comps"]>0
filt_soft = (~ic["is_soft"].astype(bool)) & (~es["is_soft"].astype(bool))
filt = filt1 & filt2 & filt_soft
ic = ic.loc[filt]
es = es.loc[filt]

##Sketch of method for Nbody integration
## (1) Iterate through indices Pandas dataframe
# e.g. for index in ic.index:
## (2) For each index: Add all corresponding rows to a REBOUND (or other Nbody?) simulation (keep units the same), AND SET NEWTONS CONSTANT TO 4.301E3 IN REBOUND TO MATCH STARFORGE.
## use x, y, z, vx, vy, vz, and mcol for the position, velocity, and mass of each particle.
## (3) Move to COM
## (4) Integrate forward in time at least until the corresponding end state (that is the integration time show be the difference between t in ic and t in the es frame*).
## (5) Compute the orbital elements of first two particles--NB the particles may actually be unbound...
## (6) Compare to the orbital elements in end_state
## (7) Double check energy and angular momentum conservation.

##*You will have convert the simulation snapshot interval to a time in code units 1 snapshot = 2.47e4 yr ~ 2.526e-05 [CODE UNITS]


