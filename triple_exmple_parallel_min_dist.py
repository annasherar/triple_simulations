import numpy as np
import rebound
import time
import pandas as pd
import sys
from starforge_mult_search.code.find_multiples_new2 import cluster, system
import webbrowser
import pytreegrav
from multiprocessing import Pool
import starforge_functions

softening_length = 0

def run_simulation(coord):
    x = coord[0]
    i = coord[1]
    iecc = coord[2]
    o1 = coord[3]
    o2 = coord[4]
    sim = rebound.Simulation()

    tracker = starforge_functions.MinDistTracker()
    sim.collision = "line"
    sim.collision_resolve = tracker.callback
    ##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
    sim.add(m=1)
    ##Introduce slight non-zero eccentricity because otherwise mean anomaly is poorly defined.
    sim.add(m=1, a=1, M= x, inc=i, e=iecc, Omega=o1, omega=o2, r=0.9)
    sim.add(m=1, a=10, e=0.80, M=np.pi, r=0.9)
    ##Switch to com frame...
    sim.move_to_com()
    orbs = sim.orbits()
    loop_time = 0.1 * orbs[1].P 
    # sim.start_server(port=1236)
    ##Using simulation archive to save data -- will be used to generate a movie later(!)
    # sim.save_to_file("archive.bin", interval=loop_time, delete_file=True)
    # sim.save_to_file("archive.bin", delete_file=True)
    # webbrowser.open("file:////home/aleksey/code/python/starforge_mult_vis/rebound_recorder.html")
    # time.sleep(10)
    end_state = {
    "time": np.inf,
    "sma": np.inf,
    "ecc": np.inf,
    "i_id1": 0,
    "i_id2": 1,
    "id1": np.inf,
    "id2": np.inf,
    "min_dist": np.inf,
    "all_en0": np.inf,
    "all_enf": np.inf,
    "inner_enf": np.inf,
    "inner_en0": np.inf,
    "energy_ratio": np.inf,
    "i_M": x,
    "i_inc": i,
    "i_ecc": iecc,
    "i_Omega": o1,
    "i_omega": o2,
    "state": 2
    }
    calculation_start_time = time.time()
    max_calculation_time = 120
    max_simulation_time = 1000 * orbs[1].P
    calculation_time = time.time() - calculation_start_time
    dissociated = False
    current_systems = starforge_functions.get_state(sim, softening_length)
    outer = current_systems[-1]
    #Initial energy
    en0 = sim.energy()
    inner_en0 = starforge_functions.get_binary_energy(outer.sub_pos[:2], outer.sub_vel[:2], outer.sub_mass[:2], outer.sub_soft[:2])
    flag = -1
    
    while not dissociated and (calculation_time < max_calculation_time) and (sim.t < max_simulation_time):
        sim.integrate(max(sim.t * 1.005, sim.t + (0.01 * orbs[1].P)))
        calculation_time = time.time() - calculation_start_time
        end_state["time"] = sim.t
        current_systems = starforge_functions.get_state(sim, softening_length)
        dissociated = starforge_functions.is_dissociated(current_systems, thres=30)
        if dissociated or (sim.t > max_simulation_time and len(current_systems) == 1):
            flag = int(dissociated)
            current_systems = starforge_functions.get_state(sim, softening_length)
            outer = current_systems[-1]
            current_orbits_after = outer.orbits[0]
            inner_enf = starforge_functions.get_binary_energy(outer.sub_pos[:2], outer.sub_vel[:2], outer.sub_mass[:2], outer.sub_soft[:2])
            enf = sim.energy()
            end_state.update({
                "time": sim.t, 
                "sma": float(current_orbits_after[0]),
                "ecc": float(current_orbits_after[1]),
                "id1": float(current_orbits_after[-3]),
                "id2": float(current_orbits_after[-4]),
                "min_dist": float(tracker.min_dist),
                "all_en0": en0,
                "all_enf": enf,
                "inner_enf": inner_enf,
                "inner_en0": inner_en0,
                "energy_ratio": (enf-en0)/en0,
                "state": flag
                })
    return end_state

#Initializing random generator
rng = np.random.default_rng()
amt_trials = 10
M_values = rng.uniform(0, 2*np.pi, size=amt_trials) #100 even spaces from 0 to 2pi
#Defining inclination
cos_inc_values = rng.uniform(-1, 1, size=amt_trials)
inc_values = np.arccos(cos_inc_values) #100 even spaces from 0 to pi
ecc_values = np.full(shape=amt_trials, fill_value = [1e-3])
Omega_values = rng.uniform(0, 2*np.pi, size=amt_trials)
omega_values = rng.uniform(0, 2*np.pi, size=amt_trials)

#Trial indices
coords = list(zip(M_values, inc_values, ecc_values, Omega_values, omega_values))
with open("start", "w") as ff:
    ff.write(f"start pool {time.time()}")
with Pool(56) as p:
    end_state_all = p.map(run_simulation, coords)

##Creating dataframe
df = pd.DataFrame(end_state_all)
##Convert dataframe to parquet file -- YOU SHOULD CHANGE THIS NAME
df.to_parquet('test.pq')
