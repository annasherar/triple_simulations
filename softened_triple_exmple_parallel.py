import numpy as np
import rebound
import time
import pandas as pd
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from softened_orbit_analysis import get_apo_softened, get_peri_softened
import reboundx
from multiprocessing import Pool
import pytreegrav
import starforge_functions

def ForceKernel(r, h):
    """
    Returns the quantity equivalent to (fraction of mass enclosed)/ r^3 for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational force.
    Arguments:
    r - radius
    h - softening
    """
    if r > h:
        return 1.0 / (r * r * r)
    hinv = 1.0 / h
    q = r * hinv
    if q <= 0.5:
        return (10.666666666666666666 + q * q * (-38.4 + 32.0 * q)) * hinv * hinv * hinv
    else:
        return (
            (21.333333333333 - 48.0 * q + 38.4 * q * q - 10.666666666667 * q * q * q - 0.066666666667 / (q * q * q))
            * hinv
            * hinv
            * hinv
        )

softening_length = 0.9

def run_simulation(coord):
    x = coord[0]
    i = coord[1]
    iecc = coord[2]
    o1 = coord[3]
    o2 = coord[4]
    sim = rebound.Simulation()

    ## Adding collision tracker
    tracker = starforge_functions.MinDistTracker()
    sim.collision = "line"
    sim.collision_resolve = tracker.callback


    ##Numerical trick. This softening does not matter because the corresponding acceleration is subtacted anyways
    sim.softening = 1e-1
    ##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
    sim.add(m=1)
    ##Introduce slight non- eccentricity because otherwise mean anomaly is poorly defined.
    sim.add(m=1, a=1, M=x, inc=i, e=iecc, Omega=o1, omega=o2, r=0.9)
    sim.add(m=1, a=10, e=0.9, M=np.pi, r=0.9)
    ##Switch to com frame...
    sim.move_to_com()
    orbs = sim.orbits()
    ps = sim.particles
    ##SETTING UP SOFTENING
    rebx = reboundx.Extras(sim)
    softened_force = rebx.load_force("softened_force")
    rebx.add_force(softened_force)
    sim.particles[0].params["softening_length"] = 0.9
    sim.particles[1].params["softening_length"] = 0.9
    sim.particles[2].params["softening_length"] = 0.9

    sim.move_to_com()
    loop_time = 0.1 * orbs[1].P
    #sim.start_server(port=1236)
    #save_interval = loop_time
    ##Using simulation archive to save data -- will be used to generate a movie later(!)
    #sim.save_to_file("archive.bin", interval=save_interval, delete_file=True)
    # sim.save_to_file("archive.bin", delete_file=True)
    #webbrowser.open("file:////home/aleksey/code/python/starforge_mult_vis/rebound_recorder.html")
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
    ##Initial energy
    current_systems = starforge_functions.get_state(sim, softening_length)
    outer = current_systems[-1]
    en0 = starforge_functions.get_all_energy(sim, softening_length)
    inner_en0 = starforge_functions.get_binary_energy(outer.sub_pos[:2], outer.sub_vel[:2], outer.sub_mass[:2], outer.sub_soft[:2])
    ## Ran to max CPU time
    flag = -1

    while not dissociated and (calculation_time < max_calculation_time) and (sim.t < max_simulation_time):
        sim.integrate(max(sim.t * 1.005, sim.t + (0.01 * orbs[1].P)))
        calculation_time = time.time() - calculation_start_time
        end_state["time"] = sim.t
        current_systems = starforge_functions.get_state(sim, softening_length)
        dissociated = starforge_functions.is_dissociated(current_systems, thres=30)
        if dissociated or (sim.t > max_simulation_time and len(current_systems) == 1):
            ## Stable condition
            flag = int(dissociated)
            outer = current_systems[-1]
            current_orbits_after = outer.orbits[0]
            ##Getting orbits in softened potential the right way
            relative_position = np.diff(outer.sub_pos[:2], axis=0)[0]
            relative_velocity = np.diff(outer.sub_vel[:2], axis=0)[0]
            #Confirming first 2 entries in sub_pos are closer
            if len(current_systems) == 1:
                r01 = np.linalg.norm(outer.sub_pos[1]-outer.sub_pos[0])
                r02 = np.linalg.norm(outer.sub_pos[2]-outer.sub_pos[0])
                r12 = np.linalg.norm(outer.sub_pos[2]-outer.sub_pos[1])
                assert r01 < r02, "02 > 01"
                assert r01 < r12, "12 > 01"
            outer_mass = outer.sub_mass[0] + outer.sub_mass[1]
            rp = get_peri_softened(relative_position[0], relative_position[1], relative_position[2],
                            relative_velocity[0], relative_velocity[1], relative_velocity[2], outer_mass, softening_length)
            ra = get_apo_softened(relative_position[0], relative_position[1], relative_position[2],
                        relative_velocity[0], relative_velocity[1], relative_velocity[2], outer_mass, softening_length)
            sma = 0.5 * (rp + ra)
            # if np.isnan(sma):
            #     print(f"[rank {rank}] rp={rp}, sma={sma}")
            ecc = 1 - rp / sma
            ##Final energy
            enf = starforge_functions.get_all_energy(sim, softening_length)
            inner_enf = starforge_functions.get_binary_energy(outer.sub_pos[:2], outer.sub_vel[:2], outer.sub_mass[:2], outer.sub_soft[:2])
            ##Need to use our more accurate method to calculate the orbits here...
            end_state.update({
                "time": sim.t, 
                "sma": float(sma),
                "ecc": float(ecc),
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
            ##Update trial index
    return end_state


#Initializing random generator
rng = np.random.default_rng()
amt_trials = 10000
M_values = rng.uniform(0, 2*np.pi, size=amt_trials) #100 even spaces from 0 to 2pi
#Defining inclination
cos_inc_values = rng.uniform(-1, 1, size=amt_trials)
inc_values = np.arccos(cos_inc_values) #100 even spaces from 0 to pi
ecc_values = np.full(shape=amt_trials, fill_value = [1e-3])
Omega_values = rng.uniform(0, 2*np.pi, size=amt_trials)
omega_values = rng.uniform(0, 2*np.pi, size=amt_trials)

#Creating grid of values
coords = list(zip(M_values, inc_values, ecc_values, Omega_values, omega_values))
with Pool(56) as p:
    end_state_all = p.map(run_simulation, coords)


##Creating dataframe
df = pd.DataFrame(end_state_all)
##Convert dataframe to parquet file
df.to_parquet('soft_random.pq')