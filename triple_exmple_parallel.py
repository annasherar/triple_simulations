import numpy as np
import rebound
import time
import pandas as pd
import sys
from starforge_mult_search.code.find_multiples_new2 import cluster, system
import webbrowser
import tqdm

from multiprocessing import Pool

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
def is_finished(current_systems, thres=30):
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
    ##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
    sim.add(m=1)
    ##Introduce slight non-zero eccentricity because otherwise mean anomaly is poorly defined.
    sim.add(m=1, a=1, M= x, inc=i, e=iecc, Omega=o1, omega=o2)
    sim.add(m=1, a=10, e=0.9, M=np.pi)
    ##Switch to com frame...
    sim.move_to_com()
    orbs = sim.orbits()
    #Looping 100 times
    integration_loops = 1000
    loop_time = 0.1 * orbs[1].P
    # sim.start_server(port=1236)
    ##Using simulation archive to save data -- will be used to generate a movie later(!)
    # sim.save_to_file("archive.bin", interval=loop_time, delete_file=True)
    # sim.save_to_file("archive.bin", delete_file=True)
    # webbrowser.open("file:////home/aleksey/code/python/starforge_mult_vis/rebound_recorder.html")
    # time.sleep(10)
    end_state = [np.inf for ii in range(6)]
    calculation_start_time = time.time()
    max_calculation_time = 60
    max_simulation_time = np.inf
    calculation_time = time.time() - calculation_start_time
    finished = False
    en0 = sim.energy()
    while not finished and (calculation_time < max_calculation_time) and (sim.t < max_simulation_time):
        sim.integrate(sim.t + loop_time)
        calculation_time = time.time() - calculation_start_time
        # time.sleep(0.05)
        end_state[0] = sim.t
        current_systems = get_state(sim)
        finished = is_finished(current_systems, thres=30)
        if finished:
            binary = current_systems[1]
            current_orbits = binary.orbits[0]
            current_systems = get_state(sim)
            finished = is_finished(current_systems, thres=30)
            ##Have to store this somewhere for each initial condition...
            enf = sim.energy()
            end_state = [sim.t, current_orbits[0], current_orbits[1], current_orbits[-3], current_orbits[-4], x, i, iecc, o1, o2, (enf-en0)/en0]
            ##Prevent float
            end_state = [float(x) for x in end_state]
            ##Update trial index
    return end_state

M_values = np.linspace(0, 2*np.pi, 10, endpoint = False) #100 even spaces from 0 to 2pi
#Defining inclination
cos_inc_values = np.linspace(-1, 1, 10, endpoint=True)
inc_values = np.arccos(cos_inc_values) #100 even spaces from 0 to pi
ecc_values = np.linspace(0, 1, 10, endpoint=False)
Omega_values = np.linspace(0, 2*np.pi, 10, endpoint=False)
omega_values = np.linspace(0, 2*np.pi, 10, endpoint=False)
#Progress bar
pbar = tqdm.tqdm(total= 100*100)
#Trial indices
coords = np.meshgrid(M_values, inc_values, ecc_values, Omega_values, omega_values, indexing="ij")
MS, IS, ES, OMS, oms = coords
coords = np.transpose([MS.ravel(), IS.ravel(), ES.ravel(), OMS.ravel(), oms.ravel()])
with open("start", "w") as ff:
    ff.write(f"start pool {time.time()}")
with Pool(56) as p:
    end_state_all = p.map(run_simulation, coords)

##We should have a placeholder in canse the integration is never finished...
##Creating dataframe
df = pd.DataFrame(end_state_all, columns = ("time", "sma", "ecc", "id1", "id2", "M", "inc", "iecc", "Omega", "omega", "energy"))
##Convert dataframe to parquet file -- YOU SHOULD CHANGE THIS NAME
df.to_parquet('newtonian_M_inc_o1_o2_trial2.pq')
