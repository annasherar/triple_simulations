import numpy as np
import rebound
import time
import pandas as pd
import sys
from starforge_mult_search.code.find_multiples_new2 import cluster, system
import webbrowser
import tqdm

from multiprocessing import Pool

def heartbeat(sim_pointer):
    # The function argument is a pointer to the simulation:
    # Here we get its contents:
    sim = sim_pointer.contents 
    d1, d2, d3 = (np.linalg.norm(np.array(sim.particles[1].xyz)-np.array(sim.particles[0].xyz)),
                np.linalg.norm(np.array(sim.particles[2].xyz)-np.array(sim.particles[0].xyz)),
                np.linalg.norm(np.array(sim.particles[2].xyz)-np.array(sim.particles[1].xyz)))
    d1 = min(d1, d2, d3)
    if d1 < 0.03: 
        print(d1)


class MinDistTracker:
    def __init__(self):
        self.min_dist = np.inf

    def callback(self, sim_pointer, collision):
        sim = sim_pointer.contents           # get simulation object from pointer
        pos1 = np.array(sim.particles[collision.p1].xyz) # position of particle 1
        pos2 = np.array(sim.particles[collision.p2].xyz) # position of particle 1
        sep = np.linalg.norm(pos1 - pos2)
        cur_min = getattr(sim, "min_dist", np.inf)
        # compute sep...
        if sep < self.min_dist:
            self.min_dist = sep
        return 0

def get_state(sim):
    ##Trick to synchronize the accelerations
    ##Is there a way to force the accelerations to be recomputed??
    sim.integrate(sim.t + 1e-15 * sim.t)
    accel1 = np.array((sim.particles[0].ax, sim.particles[0].ay, sim.particles[0].az))
    accel2 = np.array((sim.particles[1].ax, sim.particles[1].ay, sim.particles[1].az))
    accel3 = np.array((sim.particles[2].ax, sim.particles[2].ay, sim.particles[2].az))
    ps = np.array([pp.xyz for pp in sim.particles])
    vs = np.array([pp.vxyz for pp in sim.particles])
    ms = np.array([pp.m for pp in sim.particles])
    ##You will have to adjust the 4th argument when you have softening(!)
    ##Systems are only grouped if we have
    cl = cluster(ps, vs, ms, np.zeros(len(ps)), np.array([0, 1, 2]), np.array((accel1, accel2, accel3)), tides=True, sma_order=True, tides_factor=0.01)
    return cl.systems

def is_dissociated(current_systems, thres=30):
    ##There should be two systems identified at the end of the integration: a binary and a single.
    if len(current_systems) != 2:
        return False
    vel1 = current_systems[0].vel
    vel2 = current_systems[1].vel
    pos1 = current_systems[0].pos
    pos2 = current_systems[1].pos
    relative_velocity = vel1 - vel2
    relative_position = pos1 - pos2
    ##Binary and single have to be far apart and have to be moving away from each other
    condition1 = np.linalg.norm(relative_position) > thres
    condition2 = np.dot(relative_position, relative_velocity) > 0
    return condition1 and condition2

def run_simulation(coord):
    x = coord[0]
    i = coord[1]
    iecc = coord[2]
    o1 = coord[3]
    o2 = coord[4]
    sim = rebound.Simulation()

    tracker = MinDistTracker()
    sim.collision = "line"
    sim.collision_resolve = tracker.callback
    ##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
    sim.add(m=1)
    ##Introduce slight non-zero eccentricity because otherwise mean anomaly is poorly defined.
    sim.add(m=1, a=1, M= x, inc=i, e=iecc, Omega=o1, omega=o2, r=0.9)
    sim.add(m=1, a=10, e=0.9, M=np.pi, r=0.9)
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
    "energy_ratio": np.inf,
    "enf": np.inf,
    "M": x,
    "inc": i,
    "state": np.inf
    }
    calculation_start_time = time.time()
    max_calculation_time = 120
    max_simulation_time = 1000 * orbs[1].P
    calculation_time = time.time() - calculation_start_time
    dissociated = False
    current_systems = get_state(sim)
    #Initial energy
    en0 = sim.energy()
    flag = -1
    
    while not dissociated and (calculation_time < max_calculation_time) and (sim.t < max_simulation_time):
        sim.integrate(max(sim.t * 1.005, sim.t + (0.01 * orbs[1].P)))
        calculation_time = time.time() - calculation_start_time
        end_state["time"] = sim.t
        current_systems = get_state(sim)
        dissociated = is_dissociated(current_systems, thres=30)
        if dissociated or (sim.t > max_simulation_time and len(current_systems) == 1):
            flag = int(dissociated)
            outer = current_systems[-1]
            current_orbits_after = outer.orbits[0]
            current_systems = get_state(sim)
            ##Have to store this somewhere for each initial condition...
            enf = sim.energy()
            end_state.update({
                "time": sim.t, 
                "sma": float(current_orbits_after[0]),
                "ecc": float(current_orbits_after[1]),
                "id1": float(current_orbits_after[-3]),
                "id2": float(current_orbits_after[-4]),
                "min_dist": float(tracker.min_dist),
                "energy_ratio": (enf-en0)/en0,
                "enf": enf,
                "state": flag
                })
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

#Trial indices
coords = list(zip(M_values, inc_values, ecc_values, Omega_values, omega_values))
with open("start", "w") as ff:
    ff.write(f"start pool {time.time()}")
with Pool(56) as p:
    end_state_all = p.map(run_simulation, coords)

##We should have a placeholder in canse the integration is never finished...
##Creating dataframe
df = pd.DataFrame(end_state_all)
##Convert dataframe to parquet file -- YOU SHOULD CHANGE THIS NAME
df.to_parquet('newt_oldtime_ecc9.pq')
