import numpy as np
import rebound
import time
import pandas as pd
import tqdm
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from softened_orbit_analysis import get_apo_softened, get_peri_softened
import reboundx

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
    ##Systems are only grouped if we have the tidal force < 0.01 times the internal force.
    cl = cluster(ps, vs, ms, np.ones(len(ps)) * softening_length, np.array([0, 1, 2]), np.array((accel1, accel2, accel3)), tides=True, sma_order=True, tides_factor=0.01)
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
##Initializing list rows
end_state_all = []
##Defining mean anomaly
M_values = np.linspace(0, 2*np.pi, 100, endpoint = False) #100 even spaces from 0 to 2pi
for i, x in tqdm.tqdm(enumerate(M_values)):
    sim = rebound.Simulation()
    ##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
    sim.add(m=1)
    ##Introduce slight non-zero eccentricity because otherwise mean anomaly is poorly defined.
    sim.add(m=1, a=1, M=x, e=1e-3)
    sim.add(m=1, a=10, e=0.9, M=np.pi)
    ##Switch to com frame...
    sim.move_to_com()
    orbs = sim.orbits()
    ps = sim.particles
    # ##SETTING UP SOFTENING
    # def softened_force(reb_sim):
    #     ##ii will be the target and jj will be the source particle.
    #     for ii in range(len(ps)):
    #         for jj in range(len(ps)):
    #             ##If it is the same particle -- continue
    #             if ii == jj:
    #                 continue
    #             sep = np.sqrt((ps[ii].x - ps[jj].x)**2 + (ps[ii].y - ps[jj].y)**2 + (ps[ii].z - ps[jj].z)**2)
    #             sep_vector = ((ps[jj].x - ps[ii].x), (ps[jj].y - ps[ii].y), (ps[jj].z - ps[ii].z))
    #             if sep < softening_length:
    #                 accel_magnitude = reb_sim[0].G * ps[jj].m * ForceKernel(sep, softening_length)
    #                 ##Trick to get the Newtonian force
    #                 accel_newton = reb_sim[0].G * ps[jj].m * ForceKernel(sep, 0)
    #                 sim.particles[ii].ax += (accel_magnitude - accel_newton) * sep_vector[0]
    #                 sim.particles[ii].ay += (accel_magnitude - accel_newton) * sep_vector[1]
    #                 sim.particles[ii].az += (accel_magnitude - accel_newton) * sep_vector[2]
    # sim.additional_forces = softened_force
    rebx = reboundx.Extras(sim)
    softened_force = rebx.load_force("softened_force")
    rebx.add_force(softened_force)
    sim.particles[0].params["softening_length"] = 0.9
    sim.particles[1].params["softening_length"] = 0.9
    sim.particles[2].params["softening_length"] = 0.9

    sim.move_to_com()
    #Looping 100 times
    integration_loops = 1000
    loop_time = 0.1 * orbs[1].P
    sim.start_server(port=1236)
    ##Using simulation archive to save data -- will be used to generate a movie later(!)
    sim.save_to_file("archive.bin", interval=loop_time, delete_file=True)
    # sim.save_to_file("archive.bin", delete_file=True)
    #webbrowser.open("file:////home/aleksey/code/python/starforge_mult_vis/rebound_recorder.html")
    # time.sleep(10)
    end_state = [np.inf for ii in range(6)]
    for loop in range(integration_loops):
        sim.integrate(sim.t + loop_time)
        # time.sleep(0.05)
        current_systems = get_state(sim)
        finished = is_finished(current_systems, thres=30)
        if finished:
            binary = current_systems[1]
            current_orbits = binary.orbits[0]
            ##Getting orbits in softened potential the right way
            relative_position = np.diff(binary.sub_pos, axis=0)[0]
            relative_velocity = np.diff(binary.sub_vel, axis=0)[0]
            rp = get_peri_softened(relative_position[0], relative_position[1], relative_position[2],
                            relative_velocity[0], relative_velocity[1], relative_velocity[2], binary.mass, softening_length)
            ra = get_apo_softened(relative_position[0], relative_position[1], relative_position[2],
                        relative_velocity[0], relative_velocity[1], relative_velocity[2], binary.mass, softening_length)
            sma = 0.5 * (rp + ra)
            ecc = 1 - rp / sma
            ##Have to store this somewhere for each initial condition...
            ##Need to use our more accurate method to calculate the orbits here...
            end_state = [sim.t, sma, ecc, current_orbits[-3], current_orbits[-4], x]
            ##Prevent float
            end_state = [float(x) for x in end_state]
            break
    end_state_all.append(end_state)
    # finished = is_finished(current_state)
    # finished == (i) binary and single (ii) moving away from each other (iii) separation of at least 30[?]
##We should have a placeholder in canse the integration is never finished...
##Creating dataframe
# df = pd.DataFrame(end_state_all, columns = ("time", "sma", "ecc", "id1", "id2", "M"))
##Convert dataframe to parquet file
# df.to_parquet('softened_trials.pq')
# print(current_systems[1].orbits[:,0])
# print(current_systems[1].hierarchy)
# orbs = sim.orbits()
# print(orbs[0].a)
# breakpoint()
###Code used for validation -- not necessary(!)
# max_sep = 0
# min_sep = np.inf
# for ii in range(1000):
#     sim.integrate(sim.t + 0.001  * 10. * np.pi)
#     sep_bin = np.linalg.norm(np.array(sim.particles[int(end_state[-2])].xyz) - np.array(sim.particles[int(end_state[-1])].xyz))
#     if sep_bin < min_sep:
#         min_sep = sep_bin
#     if sep_bin > max_sep:
#         max_sep = sep_bin
# print((min_sep - rp) / rp, min_sep, rp)
# print((max_sep - ra)/ ra, max_sep, ra)