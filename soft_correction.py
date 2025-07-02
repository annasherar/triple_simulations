import numpy as np
import pytreegrav

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


def fix_binary(X1,X2,V1,V2,m1,m2):
    # This method keeps current separation as the apocenter, and then gives
    # random properties for eccentricity and inclination.
    # 
    out = get_orbit(X1,X2,V1,V2,m1,m2)
    a = out[0]
    # Assign a new eccentricity
    e = np.random.uniform(0,1) 
    ##If a < 0: Do nothing -- pair appears unbound...
    if a < 0:
        return X1, X2, V1, V2, a, e, np.inf, np.inf, np.inf
    # rsep = np.sqrt(np.dot( X1-X2, X1-X2))
    mtot = m1 + m2
    Xcm = (X1*m1 + X2*m2)/mtot
    Vcm = (V1*m1 + V2*m2)/mtot
    vc = np.sqrt((GN * (mtot) * (1.-e) / a / (1.+e)))
    rapo = a * (1+e)

    ##Random orbital angles -- shouldn't we take uniform distribution in cos(i)
    pi = np.random.uniform(high=2*np.pi)
    omega = np.random.uniform(high=2.*np.pi)
    ## Make this uniform in the cosine of the angle...
    # zi = np.random.uniform(high=np.pi) 
    zi = np.arccos(np.random.uniform(-1, 1))
    ##############relative positions and velocities                 
    cos,sin = np.cos, np.sin
    px1 = cos(pi)*cos(omega) - sin(pi)*sin(omega)*cos(zi)           
    qx1 =-sin(pi)*cos(omega) - cos(pi)*sin(omega)*cos(zi)           
    px2 = cos(pi)*sin(omega) + sin(pi)*cos(omega)*cos(zi)           
    qx2 =-sin(pi)*sin(omega) + cos(pi)*cos(omega)*cos(zi)           
    px3 = sin(pi)*sin(zi)                                           
    qx3 = cos(pi)*sin(zi)     
                                                                    
    ####relative velocities                                         
    Xrel = np.array([ px1*rapo, px2*rapo, px3*rapo])                                                   
    Vrel = np.array([ qx1*vc, qx2*vc,  qx3*vc])

    X1 = Xcm + m2*Xrel/mtot
    X2 = X1 - Xrel
    V1 = Vcm + m2*Vrel/mtot
    ## Why the change in sign?? -- This will flip the relative velocity, which will shuffle the orbital angles around -- not a big deal.
    # V2 = V1 + Vrel
    V2 = V1 - Vrel

    return X1, X2, V1, V2, a, e, pi, omega, zi

def corr_pipeline(ex_group):
    flag = False
    if len(ex_group) > 3:
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
        row1 = ex_group.iloc[soft_idx1]
        row2 = ex_group.iloc[soft_idx2]
        X1, X2, V1, V2, a, e, pi, omega, zi = fix_binary(row1[["x", "y", "z"]].to_numpy(), row2[["x", "y", "z"]].to_numpy(), row1[["vx", "vy", "vz"]].to_numpy(), row2[["vx", "vy", "vz"]].to_numpy(), row1["mcol"], row2["mcol"])
        row1[["x", "y", "z"]] = X1 
        row1[["vx", "vy", "vz"]] = V1
        row2[["x", "y", "z"]] = X2
        row2[["vx", "vy", "vz"]] = V2
        ex_group.iloc[soft_idx1] = row1
        ex_group.iloc[soft_idx2] = row2
        if a > 0:
            flag = True

    return flag


def corr_pipeline_ck(ex_group):
    flag = False
    if len(ex_group) > 3:
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
        row1 = ex_group.iloc[soft_idx1]
        row2 = ex_group.iloc[soft_idx2]
        X1, X2, V1, V2, a, e, pi, omega, zi = fix_binary(row1[["x", "y", "z"]].to_numpy(), row2[["x", "y", "z"]].to_numpy(), row1[["vx", "vy", "vz"]].to_numpy(), row2[["vx", "vy", "vz"]].to_numpy(), row1["mcol"], row2["mcol"])
        row1[["x", "y", "z"]] = X1 
        row1[["vx", "vy", "vz"]] = V1
        row2[["x", "y", "z"]] = X2
        row2[["vx", "vy", "vz"]] = V2
        ex_group.iloc[soft_idx1] = row1
        ex_group.iloc[soft_idx2] = row2
        if a > 0:
            flag = True

    return X1, X2, V1, V2, row1["mcol"], row2["mcol"], a, e, pi, omega, zi, flag
