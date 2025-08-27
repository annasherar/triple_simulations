import pandas as pd
import rebound
import os
import rebound

##Start positions of binaries and initial companions.
ic = pd.read_parquet("data/init_state_all.pq")
##End positions of binaries and initial companions.
es = pd.read_parquet("data/end_state_all.pq")

es3 = pd.read_parquet("data/end_state_all_time3.pq"); end_orbits = es3.groupby(es3.index).first()[["sma", "ecc"]] 
breakpoint()
##Applying various filters: (1) Halo mass cannot change too much from start to end (2) No new companions can appear from start to end (3) No particle pairs are
##with softening length of each other.

filt1 = ic["frac_halo_max"]<0.01
filt2 = ic["nonew_comps"]>0
filt_soft = (~ic["is_soft"].astype(bool)) & (~es["is_soft"].astype(bool))
filt = filt1 & filt2 & filt_soft
ic = ic.loc[filt]
es = es.loc[filt]
##Make archive directory, if exists nothing
os.makedirs("archives", exist_ok=True)

##Initializing list rows
binary_ids = []
a_expected = []
a_rebound = []
e_expected = []
e_rebound = []

for binary_id, group in ic.groupby(ic.index, sort=False):
	##Start simulation
	sim = rebound.Simulation()
	##Reassign Newton's Constant
	sim.G = 4.301e3
	##Add variables       
	for ii in range(len(group)):
			sim.add(m=group.iloc[ii]["mcol"], x=group.iloc[ii]["x"], y=group.iloc[ii]["y"], z=group.iloc[ii]["z"], vx=group.iloc[ii]["vx"], vy=group.iloc[ii]["vy"], vz=group.iloc[ii]["vz"])
	##Moves particles to center of mass
	sim.move_to_com()
	##Time conversion
	t_end = (es.loc[binary_id].iloc[0]["t"]-ic.loc[binary_id].iloc[0]["t"])*2.526e-05*3
	##Creating archive directory for IDs
	nested_dir_path = f"archives/{binary_id}"
	os.makedirs(nested_dir_path, exist_ok=True)
	archive_name = os.path.join(nested_dir_path,f"{binary_id}.bin")
	sim.save_to_file(archive_name, interval=t_end / 40, delete_file=True)
	##Integration
	sim.integrate(t_end)
	#Initializing orbital elements
	orb = sim.particles[1].orbit(primary=sim.particles[0])
	##Selecting row
	row = es.loc[binary_id].iloc[1]
	##Appending values to table
	binary_ids.append(binary_id)
	a_expected.append(row["sma"])
	a_rebound.append(orb.a)
	e_expected.append(row["ecc"])
	e_rebound.append(orb.e)
   
##Creating dataframe
df = pd.DataFrame({
	"binary_id": binary_ids,
	"a_expected": a_expected,
	"a_rebound": a_rebound,
	"e_expected": e_expected,
	"e_rebound": e_rebound
})

##Convert dataframe to parquet file
df.to_parquet('C:/Users/Admin/starforge_mult_vis/data/triple_time.pq')

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
