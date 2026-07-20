import numpy as np
import pytreegrav
from starforge_mult_search.code.find_multiples_new2 import cluster, system


## Minimum distance
class MinDistTracker:
    def __init__(self):
        self.min_dist = np.inf

    def callback(self, sim_pointer, collision):
        sim = sim_pointer.contents  # get simulation object from pointer
        pos1 = np.array(sim.particles[collision.p1].xyz)  # position of particle 1
        pos2 = np.array(sim.particles[collision.p2].xyz)  # position of particle 2
        sep = np.linalg.norm(pos1 - pos2)
        cur_min = getattr(sim, "min_dist", np.inf)
        # compute sep...
        if sep < self.min_dist:
            self.min_dist = sep
        return 0


## Energy
def PE(xc, mc, hc):
    """xc - array of positions
    mc - array of masses
    hc - array of smoothing lengths
    bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(
        xc, mc, hc, G=1, theta=0.5, method="bruteforce"
    )  # G in code units
    return 0.5 * (phic * mc).sum()


# Calculate kinetic energy of a set of cells, include internal energy
def KE(xc, mc, vc, uc):
    """xc - array of positions
    mc - array of masses
    vc - array of velocities
    uc - array of internal energies
    """
    ## velocity w.r.t. com velocity
    v_bulk = np.average(vc, weights=mc, axis=0)
    v_well = vc - v_bulk
    vSqr = np.sum(v_well**2, axis=1)
    return (mc * (vSqr / 2 + uc)).sum()


def get_all_energy(sim, softening_length):
    ##Defining particle variables for energy
    ps = np.array([pp.xyz for pp in sim.particles])
    vs = np.array([pp.vxyz for pp in sim.particles])
    ms = np.array([pp.m for pp in sim.particles])
    uc = 0
    hc = np.ones(len(ps)) * softening_length
    kin_en = KE(ps, ms, vs, uc)
    ##Potential energy
    pot_en = PE(ps, ms, hc)
    ##Initial energy
    en = kin_en + pot_en
    return en


def get_binary_energy(ps, vs, ms, hc):
    ##Defining particle variables for energy
    ps = np.array(ps)
    vs = np.array(vs)
    ms = np.array(ms)
    uc = 0
    kin_en = KE(ps, ms, vs, uc)
    ##Potential energy
    pot_en = PE(ps, ms, hc)
    ##Initial energy
    en = kin_en + pot_en
    return en


def get_state(sim, softening_length):
    ##Trick to synchronize the accelerations
    ##Is there a way to force the accelerations to be recomputed??
    sim.integrate(sim.t + 1e-15 * sim.t)
    accel1 = np.array((sim.particles[0].ax, sim.particles[0].ay, sim.particles[0].az))
    accel2 = np.array((sim.particles[1].ax, sim.particles[1].ay, sim.particles[1].az))
    accel3 = np.array((sim.particles[2].ax, sim.particles[2].ay, sim.particles[2].az))
    ps = np.array([pp.xyz for pp in sim.particles])
    vs = np.array([pp.vxyz for pp in sim.particles])
    ms = np.array([pp.m for pp in sim.particles])
    ##Systems are only grouped if we have the tidal force < 0.01 times the internal force.
    cl = cluster(
        ps,
        vs,
        ms,
        np.ones(len(ps)) * softening_length,
        np.array([0, 1, 2]),
        np.array((accel1, accel2, accel3)),
        tides=False,
        sma_order=True,
        tides_factor=0.01,
    )
    return cl.systems


def is_dissociated(current_systems, thres=50):
    ##There should be two systems identified at the end of the integration: a binary and a single.
    if len(current_systems) != 2:
        return False
    vel1 = current_systems[0].vel
    vel2 = current_systems[1].vel
    pos1 = current_systems[0].pos
    pos2 = current_systems[1].pos
    sep_bin = np.linalg.norm(
        current_systems[1].sub_pos[1] - current_systems[1].sub_pos[0]
    )
    relative_velocity = vel1 - vel2
    relative_position = pos1 - pos2
    ##Binary and single have to be far apart and have to be moving away from each other
    condition1 = np.linalg.norm(relative_position) / sep_bin > thres
    condition2 = np.dot(relative_position, relative_velocity) > 0
    return condition1 and condition2
