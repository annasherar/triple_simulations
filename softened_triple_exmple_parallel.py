import numpy as np
import rebound
import time
import pandas as pd
import tqdm
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from softened_orbit_analysis import get_apo_softened, get_peri_softened
import reboundx
from multiprocessing import Pool
import pytreegrav


class MinDistTracker:
    def __init__(self):
        self.min_dist = np.inf

    def callback(self, sim_pointer, collision):
        sim = sim_pointer.contents      # get simulation object from pointer
        pos1 = np.array(sim.particles[collision.p1].xyz)        # position of particle 1
        pos2 = np.array(sim.particles[collision.p2].xyz)        # position of particle 2
        sep = np.linalg.norm(pos1 - pos2)
        cur_min = getattr(sim, "min_dist", np.inf)
        # compute sep...
        if sep < self.min_dist:
            self.min_dist = sep
        return 0


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

def PE(xc, mc, hc):
    """ xc - array of positions
        mc - array of masses
        hc - array of smoothing lengths
        bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(xc, mc, hc, G=1, theta=0.5, method='bruteforce')  # G in code units
    return 0.5 * (phic * mc).sum()
# Calculate kinetic energy of a set of cells, include internal energy
def KE(xc, mc, vc, uc):
    """ xc - array of positions
        mc - array of masses
        vc - array of velocities
        uc - array of internal energies
    """
    ## velocity w.r.t. com velocity
    v_bulk = np.average(vc, weights=mc, axis=0)
    v_well = vc - v_bulk
    vSqr = np.sum(v_well ** 2, axis=1)
    return (mc * (vSqr / 2 + uc)).sum()
def get_energy(sim):
    ##Defining particle variables for energy
    ps = np.array([pp.xyz for pp in sim.particles])
    vs = np.array([pp.vxyz for pp in sim.particles])
    ms = np.array([pp.m for pp in sim.particles])
    uc = 0
    hc = [0.9,0.9,0.9]
    kin_en = KE(ps, ms, vs, uc)
    ##Potential energy
    pot_en = PE(ps, ms, hc)
    ##Initial energy
    en = kin_en + pot_en
    return en


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

def run_simulation(coord):
    x = coord[0]
    i = coord[1]
    iecc = coord[2]
    o1 = coord[3]
    o2 = coord[4]
    sim = rebound.Simulation()

    ## Adding collision tracker
    tracker = MinDistTracker()
    sim.collision = "line"
    sim.collision_resolve = tracker.callback


    ##Numerical trick. This softening does not matter because the corresponding acceleration is subtacted anyways
    sim.softening = 1e-1
    ##Note that for simplicity we take GN = 1 (not the starforge value so we will need GN_override.txt for the code to work properly)
    sim.add(m=1)
    ##Introduce slight non-zero eccentricity because otherwise mean anomaly is poorly defined.
    sim.add(m=1, a=1, M=x, inc=i, e=iecc, Omega=np.pi/2, omega=3*np.pi/2, r=0.9)
    sim.add(m=1, a=10, e=0.9, M=np.pi, r=0.9)
    ##Switch to com frame...
    sim.move_to_com()
    orbs = sim.orbits()
    ps = sim.particles
    # ##SETTING UP SOFTENING
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
    "id1": np.inf,
    "id2": np.inf,
    "min_dist": np.inf,
    "energy": np.inf,
    "M": x,
    "inc": i
    }
    calculation_start_time = time.time()
    max_calculation_time = 60
    max_simulation_time = np.inf
    calculation_time = time.time() - calculation_start_time 
    finished = False
    ##Initial energy
    en0 = get_energy(sim)

    while not finished and (calculation_time < max_calculation_time) and (sim.t < max_simulation_time):
        sim.integrate(sim.t + loop_time)
        calculation_time = time.time() - calculation_start_time 
        # time.sleep(0.05)
        end_state["time"] = sim.t
        current_systems = get_state(sim)
        finished = is_finished(current_systems, thres=30)
        ##Finding initial energy
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
            ##Final energy
            enf = get_energy(sim)
            ##Have to store this somewhere for each initial condition...
            ##Need to use our more accurate method to calculate the orbits here...
            end_state.update({"time": sim.t, "sma": float(sma),
                "ecc": float(ecc),
                "id1": float(current_orbits[-3]),
                "id2": float(current_orbits[-4]),
                "min_dist": float(tracker.min_dist),
                "energy": (enf-en0)/en0
            })
            ##Update trial index
    return end_state


M_values = np.linspace(0, 2*np.pi, 100, endpoint = False) #100 even spaces from 0 to 2pi
#Defining inclination
cos_inc_values = np.linspace(-1, 1, 100, endpoint=True)
inc_values = np.arccos(cos_inc_values) #100 even spaces from 0 to pi
ecc_values = [1e-3]
Omega_values = [0]
omega_values = [0]

#Creating grid of values
coords = np.meshgrid(M_values, inc_values, ecc_values, Omega_values, omega_values, indexing="ij")
MS, IS, ES, OMS, oms = coords
coords = np.column_stack([MS.ravel(), IS.ravel(), ES.ravel(), OMS.ravel(), oms.ravel()])
coords = [tuple(row) for row in coords]
with Pool(56) as p:
    end_state_all = p.map(run_simulation, coords)


##We should have a placeholder in canse the integration is never finished...
##Creating dataframe
df = pd.DataFrame(end_state_all)
##Convert dataframe to parquet file
df.to_parquet('softened_M_inc_o1_o2_o.pq')

