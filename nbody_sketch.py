import pandas as pd
# import numpy as np
import rebound

##Start positions of binaries and initial companions.
ic = pd.read_parquet("data/init_state_all.pq")
##End positions of binaries and initial companions.
es = pd.read_parquet("data/end_state_all.pq")

es3 = pd.read_parquet("data/end_state_all_time3.pq"); end_orbits = es3.groupby(es3.index).first()[["sma", "ecc"]] 
breakpoint()
##Applying various filters: (1) Halo mass cannot change too much from start to end (2) No new companions can appear from start to end (3) No particle pairs are
##with softening length of each other.
<<<<<<< HEAD
# filt1 = ic["frac_halo_max"]<0.01
# filt2 = ic["nonew_comps"]>0
# filt_soft = (~ic["is_soft"].astype(bool)) & (~es["is_soft"].astype(bool))
# filt = filt1 & filt2 & filt_soft
# ic = ic.loc[filt]
# es = es.loc[filt]

for binary_id, group in ic.groupby(ic.index, sort=False):
    ##Start simulation
    sim = rebound.Simulation()
    ##Reassign Newton's Constant
    sim.G = 4.301e3
    ##Add variables       
    for ii in range(len(group)):
            sim.add(m=group.iloc[ii]["mcol"], x=group.iloc[ii]["x"], y=group.iloc[ii]["y"], z=group.iloc[ii]["z"], vx=group.iloc[ii]["vx"], vy=group.iloc[ii]["vy"], vz=group.iloc[ii]["vz"])
    ##Moves particles to center of momentum
    sim.move_to_com()
    #Time conversion
    time = (es.loc[binary_id].iloc[0]["t"]-ic.loc[binary_id].iloc[0]["t"])*2.526e-05
    ##Integration
    sim.integrate(time)

    orb = sim.particles[1].orbit(primary=sim.particles[0])
    #print(orb.a, orb.e)
    #o1 = sim.particles[0].orbit(primary=sim.particles[1]) 
    #o2 = sim.particles[1].orbit(primary=sim.particles[0]) 

    print("Expected Semimajor axis =",es.loc[binary_id].iloc[1]["sma"],"Rebound Semimajor axis =",orb.a)
    print("Expected Eccentricity = ",es.loc[binary_id].iloc[1]["ecc"], "Rebound Eccentricity =",orb.e)
    print("Expected Distance = ",es.loc[binary_id].iloc[1]["bdist"], "Rebound Distance =",orb.d)

    #print("Expected Semimajor axis =",es.loc[binary_id].iloc[1]["sma"],"Rebound Semimajor axis =",o1.a, "Rebound Semimajor axis =",o2.a)
    #print("Expected Eccentricity = ",es.loc[binary_id].iloc[1]["ecc"], "Rebound Eccentricity =",o1.e, "Rebound Semimajor axis =",o2.a)
    
    #Visualization
    sim.start_server(port=1234)
    input("Server running at http://localhost:1234. Press Enter to stop.")

>>>>>>> 840c005 (N-body integration pass 1)
###
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
