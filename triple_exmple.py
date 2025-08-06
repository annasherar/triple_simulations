import numpy as np

import rebound 
import time

from starforge_mult_search.code.find_multiples_new2 import cluster, system
import webbrowser

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


##Simple example triple
##We will vary the orbital properties later (eccentricities, inclinations)
##FOR NOW just randomize the mean anomaly (0 to 2 pi) in this example (NB mean anomaly is not well-defined for a circular orbit...)
##Note rebound uses Jacobi coordinates, so that the 
##orbit of the outer triple is defined with respect to the 
sim = rebound.Simulation()
##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
sim.add(m=1)
##Introduce slight non-zero eccentricity because otherwise mean anomaly is poorly defined.
sim.add(m=1, a=1, M=np.pi / 4, e=1e-3)
sim.add(m=1, a=10, e=0.9, M=np.pi)
##Switch to com frame...
sim.move_to_com()
orbs = sim.orbits()

#Looping 100 times
integration_loops = 1000
loop_time = 0.1 * orbs[1].P
sim.start_server(port=1236)

##Using simulation archive to save data -- will be used to generate a movie later(!)
sim.save_to_file("archive.bin", interval=loop_time, delete_file=True)
# sim.save_to_file("archive.bin", delete_file=True)
# webbrowser.open("file:////home/aleksey/code/python/starforge_mult_vis/rebound_recorder.html")
time.sleep(10)
for loop in range(integration_loops):
    sim.integrate(sim.t + loop_time)
    time.sleep(0.05)
    current_systems = get_state(sim)
    finished = is_finished(current_systems, thres=30)
    if finished:
        current_orbits = current_systems[1].orbits[0]
        ##Have to store this somewhere for each initial condition...
        end_state = [sim.t, current_orbits[0], current_orbits[1], current_orbits[-3], current_orbits[-4]]
        break
    # finished = is_finished(current_state)
    # finished == (i) binary and single (ii) moving away from each other (iii) separation of at least 30[?]
##We should have a placeholder in canse the integration is never finished...
print("end state", end_state)

# print(current_systems[1].orbits[:,0])
# print(current_systems[1].hierarchy)
# orbs = sim.orbits()
# print(orbs[0].a)
# breakpoint()

