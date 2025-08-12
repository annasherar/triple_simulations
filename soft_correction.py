import numpy as np
import rebound
import pytreegrav
##Include the necessary code below to get rid of these imports!
from starforge_mult_search.analysis.analyze_stack import get_peri_softened_numba, get_apo_softened_numba
import cgs_const as cgs

GN = 4.301e3

def PE(xc, mc, hc):
    """ xc - array of positions
        mc - array of masses
        hc - array of smoothing lengths
        bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(xc, mc, hc, GN, theta=0.5, method='bruteforce') # G in code units
    return 0.5 * (phic*mc).sum()


def get_orbit(p1, p2, v1, v2, m1, m2, h1=0, h2=0):
    """
    Auxiliary function to get binary properties for two particles.

    :param Array-like p1: -- Array with 1st particle position (3D)
    :param Array-like p2: -- Array with 2nd particle position (3D)
    :param Array-like v1: -- Array with 1st particle velocity (3D)
    :param Array-like v2: -- Array with 2nd particle velocity (3D)
    :param float m1: -- 1st particle mass
    :param float m2: -- 2nd particle mass

    :return: The semimajor axis, eccentricity, inclination, the particle separation, com position, com velocity, m1, and m2.
    :rtype: tuple
    """
    dp = np.linalg.norm(p1 - p2)

    com = (m1*p1 + m2*p2)/(m1 + m2)
    com_vel = (m1*v1 + m2*v2)/(m1 + m2)
    ##Particle velocities in com frame
    p1_com = p1 - com
    p2_com = p2 - com
    v1_com = v1 - com_vel
    v2_com = v2 - com_vel

    v12 = (v1_com[0]**2. + v1_com[1]**2. + v1_com[2]**2.)
    v22 = (v2_com[0]**2. + v2_com[1]**2. + v2_com[2]**2.)

    ##Kinetic and potential energies
    ke = 0.5*m1*v12 + 0.5*m2*v22
    npe = -PE(np.array([p1_com, p2_com]), np.array([m1, m2]), np.array([h1, h2]))
    a_bin = GN*(m1*m2)/(2.*(npe-ke))
    ##Angular momentum in binary com
    j_bin = m1*np.cross(p1_com, v1_com) + m2*np.cross(p2_com, v2_com)
    ##Angular momentum of binary com
    j_com = (m1 + m2)*np.cross(com, com_vel)

    #Inclination
    i_bin = np.arccos(np.dot(j_bin, j_com)/np.linalg.norm(j_bin)/np.linalg.norm(j_com))*180./np.pi
    mu = m1*m2/(m1+m2)
    ##Eccentricity of the binary
    e_bin = np.sqrt(1.-np.linalg.norm(j_bin)**2./(GN*(m1+m2)*a_bin)/(mu**2.))


    return a_bin, e_bin, i_bin, dp, com[0], com[1], com[2], com_vel[0], com_vel[1], com_vel[2], m1, m2


def fix_binary(X1, X2, V1, V2, m1, m2, eps):
    Vrel = V1 - V2
    Xrel = X1 - X2
    sep = np.linalg.norm(Xrel)
    tot_mass =  m1 + m2

    ##Compute sma and eccentricity based on turning points of orbit in the softened potential
    rp = get_peri_softened_numba(Xrel[0], Xrel[1], Xrel[2], Vrel[0], Vrel[1], Vrel[2], tot_mass, eps) 
    ra = get_apo_softened_numba(Xrel[0], Xrel[1], Xrel[2], Vrel[0], Vrel[1], Vrel[2], tot_mass, eps) 
    sma = 0.5 * (rp + ra)
    e = 1 - rp / sma
    ##Energy and angular momentum for above elements
    en_new = -GN * tot_mass / 2 / sma    
    ell_new = (GN * tot_mass * sma * (1 - e**2.))**.5

    ##REFACTOR INTO ITS OWN FUNCTION!!!
    ##Original tangential and radial velocities
    Vrel_radial = np.dot(Xrel / sep, Vrel) * (Xrel / sep)
    Vrel_tangential = Vrel - Vrel_radial
    ##Computing the new tangential and radial relative velocities...
    Vrel_tangential_new_magnitude = ell_new / sep
    Vrel_tangential = Vrel_tangential / np.linalg.norm(Vrel_tangential) * Vrel_tangential_new_magnitude
    ##Radial component of relative velocity
    Vrel_new_magnitude = (2. * (en_new  + GN * tot_mass / sep))**.5
    Vrel_radial_new_magnitude = (Vrel_new_magnitude**2. - Vrel_tangential_new_magnitude**2.)**.5
    Vrel_radial = Vrel_radial / np.linalg.norm(Vrel_radial) * Vrel_radial_new_magnitude
    ##New total relative velocity
    Vrel = Vrel_tangential + Vrel_radial
    ##New individual stellar velocities
    Vcm = (V1 * m1 + V2 * m2)/ (m1 + m2)
    V1 = Vcm + m2*Vrel / tot_mass
    V2 = V1 - Vrel

    return X1, X2, V1, V2, sma, e

def from_keplerian_simplified(X1, X2, V1, V2, m1, m2, sma, e, eps):
    Vrel = V1 - V2
    Xrel = X1 - X2
    sep = np.linalg.norm(Xrel)
    tot_mass = m1 + m2

    def Phi(r, eps):
        return -GN * tot_mass / (r + eps)

    rp = sma * (1 - e)
    ra = sma * (1 + e)
    ell_new = np.sqrt((Phi(rp, eps) - Phi(ra, eps)) / ((1 / 2 / ra) - (1 / 2 / rp)))
    en_new = ell_new**2. / (2. * ra) + Phi(ra, eps)

    ##REFACTOR INTO ITS OWN FUNCTION!!!
    ##Original tangential and radial velocities
    Vrel_radial = np.dot(Xrel / sep, Vrel) * (Xrel / sep)
    Vrel_tangential = Vrel - Vrel_radial
    ##Computing the new tangential and radial relative velocities...
    Vrel_tangential_new_magnitude = ell_new / sep
    Vrel_tangential = Vrel_tangential / np.linalg.norm(Vrel_tangential) * Vrel_tangential_new_magnitude
    ##Radial component of relative velocity
    Vrel_new_magnitude = (2. * (en_new  + GN * tot_mass / sep))**.5
    Vrel_radial_new_magnitude = (Vrel_new_magnitude**2. - Vrel_tangential_new_magnitude**2.)**.5
    Vrel_radial = Vrel_radial / np.linalg.norm(Vrel_radial) * Vrel_radial_new_magnitude
    ##New total relative velocity
    Vrel = Vrel_tangential + Vrel_radial
    ##New individual stellar velocities
    Vcm = (V1 * m1 + V2 * m2)/ (m1 + m2)
    V1 = Vcm + m2*Vrel / tot_mass
    V2 = V1 - Vrel

    return X1, X2, V1, V2, sma, e


# def fix_binary(X1,X2,V1,V2,m1,m2,eps):
#     ##Calculate the apocenter and pericenter of the orbit, accounting for softening
#     Xrel = X1 - X2
#     Vrel = V1 - V2
#     separation = np.linalg.norm(Xrel)

#     ##We will preserve the peri and apocenter of the orbit, but not its angular momentum[?]
#     ##Not sure about the energy?
#     rp = get_peri_softened_numba(Xrel[0], Xrel[1], Xrel[2], Vrel[0], Vrel[1], Vrel[2], m1 + m2, eps) * cgs.pc / cgs.au
#     ra = get_apo_softened_numba(Xrel[0], Xrel[1], Xrel[2], Vrel[0], Vrel[1], Vrel[2], m1 + m2, eps) * cgs.pc / cgs.au
#     a = 0.5 * (rp + ra)
#     e = 1 - rp / a

#     ##TO DO: revisit this criterion...
#     if np.isnan(a) or (a < 0):
#         return X1, X2, V1, V2, a, e, np.inf, np.inf, np.inf
#     # rsep = np.sqrt(np.dot( X1-X2, X1-X2))
#     mtot = m1 + m2
#     Vcm = (V1*m1 + V2*m2)/mtot

#     Lnew = (GN * mtot * a * (1. - e**2))**.5
#     Enew = -GN * mtot / (2. * a)
#     ##What the new relative velocity should be in a Keplerian potential!
#     Vrel_new_magnitude = (2. * (Enew  + GN * mtot / separation))**.5

#     # Vrel = Vrel / np.linalg.norm(Vrel) * Vrel_new_magnitude
#     # V1 = Vcm + m2*Vrel/mtot
#     # V2 = V1 - Vrel

#     ##Leaving the positions unchanged(!)
#     return X1, X2, V1, V2, a, e


    # vc = np.sqrt((GN * (mtot) * (1.-e) / a / (1.+e)))
    # rapo = a * (1+e)

    # ##Random orbital angles -- Maybe separate this out, so we could also apply the same rotations without the other corection...
    # pi = np.random.uniform(high=2*np.pi)
    # omega = np.random.uniform(high=2.*np.pi)
    # ## Make this uniform in the cosine of the angle...
    # # zi = np.random.uniform(high=np.pi) 
    # zi = np.arccos(np.random.uniform(-1, 1))
    # ##############relative positions and velocities                 
    # cos,sin = np.cos, np.sin
    # px1 = cos(pi)*cos(omega) - sin(pi)*sin(omega)*cos(zi)           
    # qx1 =-sin(pi)*cos(omega) - cos(pi)*sin(omega)*cos(zi)           
    # px2 = cos(pi)*sin(omega) + sin(pi)*cos(omega)*cos(zi)           
    # qx2 =-sin(pi)*sin(omega) + cos(pi)*cos(omega)*cos(zi)           
    # px3 = sin(pi)*sin(zi)                                           
    # qx3 = cos(pi)*sin(zi)     
                                                                    
    # ####relative velocities                                         
    # Xrel = np.array([ px1*rapo, px2*rapo, px3*rapo])                                                   
    # Vrel = np.array([ qx1*vc, qx2*vc,  qx3*vc])

    # X1 = Xcm + m2*Xrel/mtot
    # X2 = X1 - Xrel
    # V1 = Vcm + m2*Vrel/mtot
    # ## Why the change in sign?? -- This will flip the relative velocity, which will shuffle the orbital angles around -- not a big deal.
    # # V2 = V1 + Vrel
    # V2 = V1 - Vrel

    # return X1, X2, V1, V2, a, e, pi, omega, zi

def corr_pipeline(ex_group, aux_data, r=None, rot_only=False):
    flag = False
    aux_row = (np.inf, np.inf, np.inf, np.inf)
    if len(ex_group) > 3:
        aux_data.append(aux_row)
        return flag
    n_soft = 0
    soft_idx1 = 0
    soft_idx2 = 0
    for ii in range(len(ex_group)):
        for jj in range(ii + 1, len(ex_group)):
            h = max(ex_group.iloc[ii]["hcol"], ex_group.iloc[jj]["hcol"])
            sep = np.linalg.norm(ex_group.iloc[ii][["x", "y", "z"]] - ex_group.iloc[jj][["x", "y", "z"]])
            if sep < h:
                n_soft += 1
                soft_idx1 = ii
                soft_idx2 = jj

    if n_soft == 1:
        ##Should we filter out cases where we have soft_idx1 = 0 and soft_idx2 = 1? These may already be filtered out by the semi-major axis filter.
        row1 = ex_group.iloc[soft_idx1]
        row2 = ex_group.iloc[soft_idx2]

        eps = max(row1["hcol"], row2["hcol"])
        ##Hack to only apply rotation
        if rot_only:
            eps = 0
        X1, X2, V1, V2, a, e = fix_binary(row1[["x", "y", "z"]].to_numpy(), row2[["x", "y", "z"]].to_numpy(), row1[["vx", "vy", "vz"]].to_numpy(), row2[["vx", "vy", "vz"]].to_numpy(), 
                                          row1["mcol"], row2["mcol"], eps)
        ##Numerical issue in calculating orbit -- to be filtered out!
        if np.isnan(a):
            aux_data.append(aux_row)
            return flag

        ##Add rotation to new coordinates here(!) -- This will give coordinates to use with the Newtonian simulation.
        ##Also compute coordinates just applying rotation -- This will give coordinates for integrations with softened gravity.
        ##Generate random rotation using scipy -- Should apply the same rotation to both the relative position and velocity. 
        if r:
            X1 = r.apply(X1)
            X2 = r.apply(X2)
            V1 = r.apply(V1)
            V2 = r.apply(V2)
        row1[["x", "y", "z"]] = X1
        row1[["vx", "vy", "vz"]] = V1
        row2[["x", "y", "z"]] = X2
        row2[["vx", "vy", "vz"]] = V2
        ##VALIDATION--make sure X1, X2, V1, and V2 indeed correspond to the orbital elements above.
        sim = rebound.Simulation()
        sim.G = GN
        sim.add(x=X1[0], y=X1[1], z=X1[2], vx=V1[0], vy=V1[1], vz=V1[2], m=row1["mcol"])
        sim.add(x=X2[0], y=X2[1], z=X2[2], vx=V2[0], vy=V2[1], vz=V2[2], m=row2["mcol"])
        orbs = sim.orbits(primary=sim.particles[0])
        try:
            assert np.isclose(orbs[0].a, a)
        except AssertionError:
            breakpoint
        try:
            assert np.isclose(orbs[0].e, e)
        except AssertionError:
            breakpoint()

        aux_row = (a, e, soft_idx1, soft_idx2)
        ex_group.iloc[soft_idx1] = row1
        ex_group.iloc[soft_idx2] = row2
        if a > 0:
            flag = True
    aux_data.append(aux_row)
    return flag


# def corr_pipeline_ck(ex_group):
#     flag = False
#     if len(ex_group) > 3:
#         return flag
#     n_soft = 0
#     soft_idx1 = 0
#     soft_idx2 = 0
#     for ii in range(len(ex_group)):
#         for jj in range(ii + 1, len(ex_group)):
#             h = max(ex_group.iloc[ii]["hcol"], ex_group.iloc[jj]["hcol"])
#             sep = np.linalg.norm(ex_group.iloc[ii][["x", "y", "z"]] - ex_group.iloc[jj][["x", "y", "z"]])
#             if sep < h:
#                 n_soft += 1
#             soft_idx1 = ii
#             soft_idx2 = jj
#     if n_soft == 1:
#         row1 = ex_group.iloc[soft_idx1]
#         row2 = ex_group.iloc[soft_idx2]
#         X1, X2, V1, V2, a, e, pi, omega, zi = fix_binary(row1[["x", "y", "z"]].to_numpy(), row2[["x", "y", "z"]].to_numpy(), row1[["vx", "vy", "vz"]].to_numpy(), row2[["vx", "vy", "vz"]].to_numpy(), row1["mcol"], row2["mcol"])
#         row1[["x", "y", "z"]] = X1 
#         row1[["vx", "vy", "vz"]] = V1
#         row2[["x", "y", "z"]] = X2
#         row2[["vx", "vy", "vz"]] = V2
#         ex_group.iloc[soft_idx1] = row1
#         ex_group.iloc[soft_idx2] = row2
#         if a > 0:
#             flag = True

#     return X1, X2, V1, V2, row1["mcol"], row2["mcol"], a, e, pi, omega, zi, flag

