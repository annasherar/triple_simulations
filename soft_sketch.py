import pandas as pd
import numpy as np
from soft_correction import corr_pipeline, PE
import tqdm

from scipy.spatial.transform import Rotation as R

import rebound
import pytreegrav

GN = 4.301e3
def from_keplerian(X1, X2, V1, V2, m1, m2, h1, h2, sma, e):
    Vrel = V1 - V2
    Xrel = X1 - X2
    sep = np.linalg.norm(Xrel)
    tot_mass = m1 + m2

    rp = sma * (1 - e)
    ra = sma * (1 + e)
    ##WLOG can take any coordinates spaced by rp to calculate the potential there...
    potential_rp = PE(np.array([[0, 0, 0], [rp, 0, 0]]), np.array([m1, m2]), np.array([h1, h2]))
    potential_ra = PE(np.array([[0, 0, 0], [ra, 0, 0]]), np.array([m1, m2]), np.array([h1, h2]))
    potential_curr = PE(np.array([X1, X2]), np.array([m1, m2]), np.array([h1, h2]))

    ell_new = np.sqrt((potential_rp - potential_ra) / ((1 / 2 / ra**2.) - (1 / 2 / rp**2.)))
    en_new = ell_new**2. / (2. * ra**2.) + potential_ra
    ##REFACTOR INTO ITS OWN FUNCTION!!!
    ##Original tangential and radial velocities
    Vrel_radial = np.dot(Xrel / sep, Vrel) * (Xrel / sep)
    Vrel_tangential = Vrel - Vrel_radial
    ##Computing the new tangential and radial relative velocities...
    Vrel_tangential_new_magnitude = ell_new / sep
    Vrel_tangential = Vrel_tangential / np.linalg.norm(Vrel_tangential) * Vrel_tangential_new_magnitude
    ##Radial component of relative velocity
    Vrel_new_magnitude = (2. * (en_new  - potential_curr))**.5
    Vrel_radial_new_magnitude = (Vrel_new_magnitude**2. - Vrel_tangential_new_magnitude**2.)**.5
    Vrel_radial = Vrel_radial / np.linalg.norm(Vrel_radial) * Vrel_radial_new_magnitude

    assert(np.isclose(np.dot(Vrel_tangential, Vrel_radial), 0))
    assert(np.isclose(np.linalg.norm(Vrel_tangential), Vrel_tangential_new_magnitude))
    assert(np.isclose(np.linalg.norm(Vrel_radial), Vrel_radial_new_magnitude))
    ##New total relative velocity
    Vrel = Vrel_tangential + Vrel_radial
    assert np.isclose(np.linalg.norm(Vrel), Vrel_new_magnitude)
    ##New individual stellar velocities
    Vcm = (V1 * m1 + V2 * m2)/ (m1 + m2)
    V1 = Vcm + m2*Vrel / tot_mass
    V2 = V1 - Vrel

    assert np.isclose(0.5 * np.linalg.norm(Vrel)**2 + potential_curr, en_new)
    assert np.isclose(np.linalg.norm(np.cross(Xrel, Vrel)), ell_new)

    return X1, X2, V1, V2

def randomize_binary_phase(X1, X2, V1, V2, m1, m2, h1, h2, time=None, diag=False):
    """
    TO DO: ADJUST TO HANDLE INTEGRATIONS WITH SOFTENING
    Randomize binary phase 

    X1, X2--particle positions
    V1, V2--particle velocities
    m1, m2--particle masses
    h1, h2--particle softening lengths
    """
    sim = rebound.Simulation()
    sim.G = GN
    sim.add(m=m1, x=X1[0], y=X1[1], z=X1[2], 
                    vx=V1[0], vy=V1[1], vz=V1[2])
    sim.add(m=m2, x=X2[0], y=X2[1], z=X2[2], 
                    vx=V2[0], vy=V2[1], vz=V2[2])
    ##Add softening here(!)
    ##if h1 > 0 or h2 > 0:
    ##activate softening...
    
    orbs = sim.orbits(primary=sim.particles[0])
    com = sim.com()
    ##Need these defaults for the algorithm to work but output may be counterintuitive if we are not doing a diagnostic simulation.
    rmin = np.inf
    rmax = 0
    if time is None:
        time = np.random.uniform(0, 1) * orbs[0].P
    sim.move_to_com()

    if diag:
        for ii in range(1000):
            sim.integrate(sim.t + time)
            current_separation = np.array(sim.particles[1].xyz) - np.array(sim.particles[0].xyz)
            current_separation = np.linalg.norm(current_separation)
            if current_separation < rmin: 
                rmin = current_separation
            if current_separation > rmax:
                rmax = current_separation

    else:
        sim.integrate(sim.t + time)

    return {"X1": np.array(sim.particles[0].xyz) + np.array(com.xyz),
            "X2": np.array(sim.particles[1].xyz) + np.array(com.xyz),
            "V1": np.array(sim.particles[0].vxyz) + np.array(com.vxyz),
            "V2": np.array(sim.particles[1].vxyz) + np.array(com.vxyz),
            "COM": np.array(com.xyz), "COMV": np.array(com.vxyz),
            "RMIN": rmin, "RMAX": rmax}

##Start positions of binaries and initial companions.
ic = pd.read_parquet("data/init_state_all.pq")
##End positions of binaries and initial companions.
es = pd.read_parquet("data/end_state_all.pq")
##Applying various filters: (1) Halo mass cannot change too much from start to end (2) No new companions can appear from start to end (3) No particle pairs are
##with softening length of each other.
filt1 = ic["frac_halo_max"]<0.01
filt2 = ic["nonew_comps"]>0
# filt_soft = (~ic["is_soft"].astype(bool)) & (~es["is_soft"].astype(bool))
# filt = filt1 & filt2 & filt_soft
filt = filt1 & filt2
ic = ic.loc[filt]
es = es.loc[filt]

##You can play around with the trials here...
trials = 20

##TO DO: corr_pipeline and from_keplerian need additional validation via numerical integration...
for trial in tqdm.tqdm(range(trials)):
    ##For storing auxilliary data
    aux_data1 = []
    aux_data2 = []
    ##Reset after every trial -- Have to make a copy of the original initial conditions file(!!)
    ic_for_newtonian = ic.copy()
    ic_for_softened = ic.copy()
    ##Apply correction for softened binaries.
    ##To keep things simple, we will only consider exchanges where there are a total of three stars,
    ##and only one pair is softened. We will ignore all other encounters.
    ##We only want to apply the correction pipeline if we are integrating WITHOUT the softened gravitational force in REBOUND!!!

    ##Same random rotation for all binaries(!) ##TO DO REMOVE ROTATION.
    # r = R.random()
    ##Initial conditions for Newtonian simulations(!)
    flags = np.array([corr_pipeline(ic_for_newtonian.loc[ex1], aux_data1) for ex1 in ic_for_newtonian.index.unique()])
    idx_select = ic_for_newtonian.index.unique()[flags]
    ic_for_newtonian = ic_for_newtonian.loc[idx_select]
    ##Auxiliary data--applying same filtering
    aux_data1 = np.array(aux_data1)[flags]
    ##We also want to apply the rotation (r) only! -- this has some unnecessary work and complexity: try to get rid. At least make sure it is not causing issues!
    ic_for_softened = ic_for_softened.loc[idx_select]
    # [corr_pipeline(ic_for_softened.loc[ex1], aux_data2, r=r, rot_only=True) for ex1 in ic_for_softened.index.unique()]
    aux_data2 = np.array(aux_data2)
    ##Filtered original ics for validation
    ic_filt = ic.loc[idx_select]

    ##We just want to randomize phase(!) -- Not other elements(!)--so get rid of rotation
    cleaned_index = ic_for_softened.index.unique().to_numpy()
    ##Validation for transformation -- Double check indexing here...
    for ii in range(len(cleaned_index)):        
        #VALIDATION: MAKE SURE ONLY THE APPROPRIATE ROWS CHANGED
        ic_row = ic_for_newtonian.loc[cleaned_index[ii]]
        aux_row = aux_data1[ii]
        unchanged_index = np.array(~np.isin(np.array([0, 1, 2]), aux_row[2:].astype(int)))
        assert np.all(ic_row.iloc[unchanged_index] == ic_filt.loc[cleaned_index[ii]].iloc[unchanged_index])
        assert np.any(ic_row.iloc[int(aux_row[2])] != ic_filt.loc[cleaned_index[ii]].iloc[int(aux_row[2])])
        assert np.any(ic_row.iloc[int(aux_row[3])] != ic_filt.loc[cleaned_index[ii]].iloc[int(aux_row[3])])
        xrel_newtonian = ic_row.iloc[int(aux_row[2])][["x", "y", "z"]] - ic_row.iloc[int(aux_row[3])][["x", "y", "z"]]
        vrel_newtonian = ic_row.iloc[int(aux_row[2])][["vx", "vy", "vz"]] - ic_row.iloc[int(aux_row[3])][["vx", "vy", "vz"]]

        ##Tertiary star properties will be the same in both the softened and unsoftened simulations.
        tertiary_star_mass, tertiary_star_xyz, tertiary_star_vxyz =  ic_row.iloc[unchanged_index]["mcol"], ic_row.iloc[unchanged_index][["x", "y", "z"]].to_numpy().ravel(), ic_row.iloc[unchanged_index][["vx", "vy", "vz"]].to_numpy().ravel()
        ##Randomizing binary phase in the Newtonian simulation
        binary_star1_mass, binary_star1_xyz_newton, binary_star1_vxyz_newton =  ic_row.iloc[int(aux_row[2])]["mcol"], ic_row.iloc[int(aux_row[2])][["x", "y", "z"]].to_numpy().ravel(), ic_row.iloc[int(aux_row[2])][["vx", "vy", "vz"]].to_numpy().ravel()
        binary_star2_mass, binary_star2_xyz_newton, binary_star2_vxyz_newton =  ic_row.iloc[int(aux_row[3])]["mcol"], ic_row.iloc[int(aux_row[3])][["x", "y", "z"]].to_numpy().ravel(), ic_row.iloc[int(aux_row[3])][["vx", "vy", "vz"]].to_numpy().ravel()
        binary_star1_h = ic_row.iloc[int(aux_row[2])]["hcol"]
        binary_star2_h = ic_row.iloc[int(aux_row[3])]["hcol"]
        tertiary_star_h = ic_row.iloc[unchanged_index]["hcol"]

        # ##All the below functionality could be folded into its own function
            # binary_com = (binary_star1_mass * binary_star1_xyz_newton + binary_star2_mass * binary_star2_xyz_newton) / (binary_star1_mass + binary_star2_mass)
        # binary_vcom = (binary_star1_mass * binary_star1_vxyz_newton + binary_star2_mass * binary_star2_vxyz_newton) / (binary_star1_mass + binary_star2_mass)
        # sim_Newton = rebound.Simulation()
        # sim_Newton.G = GN
        # sim_Newton.add(m=binary_star1_mass, x=binary_star1_xyz_newton[0], y=binary_star1_xyz_newton[1], z=binary_star1_xyz_newton[2], 
        #                vx=binary_star1_vxyz_newton[0], vy=binary_star1_vxyz_newton[1], vz=binary_star1_vxyz_newton[2])
        # sim_Newton.add(m=binary_star2_mass, x=binary_star2_xyz_newton[0], y=binary_star2_xyz_newton[1], z=binary_star2_xyz_newton[2],
        #                 vx=binary_star2_vxyz_newton[0], vy=binary_star2_vxyz_newton[1], vz=binary_star2_vxyz_newton[2])
        # sim_Newton.move_com()
        # orbs = sim_Newton.calculate_orbit(primary=sim_Newton.particles[0])
        # sma, e = orbs[0].a, orbs[0].e
        # ##Randomizing phase 
        # sim_Newton.integrate(np.random.uniform(0,1) * orbs[0].P)
        # ##Adjusting binary star positions
        # ########################################
        # binary_star1_xyz_newton = binary_com + np.array(sim_Newton.particles[0].xyz)
        # binary_star1_vxyz_newton = binary_vcom + np.array(sim_Newton.particles[0].vxyz)
        # binary_star2_xyz_newton = binary_com + np.array(sim_Newton.particles[1].xyz)
        # binary_star2_vxyz_newton = binary_vcom + np.array(sim_Newton.particles[1].vxyz)

        ##Randomizing binary phase for Newtonian simulation so that softening can be set to 0(!)
        scrambled_positions = randomize_binary_phase(binary_star1_xyz_newton, binary_star2_xyz_newton, binary_star1_vxyz_newton, binary_star2_vxyz_newton, 
                               binary_star1_mass, binary_star2_mass, 0, 0)
        X1 = scrambled_positions["X1"]
        X2 = scrambled_positions["X2"]
        V1 = scrambled_positions["V1"]
        V2 = scrambled_positions["V2"]
        COM = scrambled_positions["COM"]
        COMV = scrambled_positions["COMV"]
        sim_Newton = rebound.Simulation()
        sim_Newton.G = GN
        sim_Newton.add(m=binary_star1_mass, x=X1[0], y=X1[1], z=X1[2], 
                       vx=V1[0], vy=V1[1], vz=V1[2])
        sim_Newton.add(m=binary_star2_mass, x=X2[0], y=X2[1], z=X2[2],
                        vx=V2[0], vy=V2[1], vz=V2[2])
        ##Finally adding the teriary phase...
        sim_Newton.add(m = tertiary_star_mass, x=tertiary_star_xyz[0], y=tertiary_star_xyz[1], z=tertiary_star_xyz[2], 
                       vx=tertiary_star_vxyz[0], vy=tertiary_star_vxyz[1], vz=tertiary_star_vxyz[2])
        orbs = sim_Newton.orbits(primary=sim_Newton.particles[0])
        sma = orbs[0].a
        e = orbs[0].e
        rp = sma * (1 - e)
        ra = sma * (1 + e)
        assert(np.isclose(sma, aux_row[0]))
        assert(np.isclose(e, aux_row[1]))
        ###################################################################################################################
        #TO DO: ADD SOFTENING IN THE SIMULATION SETUP
        #Randomization of phase in softened potential: Hagai's method--converting randomized Keplerian orbit back to softened potential
        ic_row = ic_for_softened.loc[cleaned_index[ii]]
        X1, X2, V1, V2 = from_keplerian(X1 - COM, X2 - COM, 
                       V1 - COMV, V2 - COMV, 
                       binary_star1_mass, binary_star2_mass, 
                       binary_star1_h, binary_star2_h,
                       sma, e)
    
        binary_star1_xyz_soft = X1 + COM
        binary_star1_vxyz_soft = V1 + COMV
        binary_star2_xyz_soft = X2 + COM
        binary_star2_vxyz_soft = V2 + COMV
        ##VALIDATION OF CONVERSIONS--CHECK THAT THEY PRESERVE APOCENTER AND PERICENTER AS EXPECTED.
        scrambled_positions = randomize_binary_phase(binary_star1_xyz_soft, binary_star2_xyz_soft, binary_star1_vxyz_soft, binary_star2_vxyz_soft,
                               binary_star1_mass, binary_star2_mass, binary_star1_h, binary_star2_h, time = 10 * orbs[0].P, diag=True)
        assert(np.abs((scrambled_positions["RMIN"] - rp) / rp) < 1e-3)
        assert(np.abs((scrambled_positions["RMAX"] - ra) / ra) < 1e-3)

        # sim_soft = rebound.Simulation()
        # sim_soft.G = GN
        # sim_soft.add(m=binary_star1_mass, x=binary_star1_xyz_soft[0], y=binary_star1_xyz_soft[1], z=binary_star1_xyz_soft[2], 
        #                vx=binary_star1_vxyz_soft[0], vy=binary_star1_vxyz_soft[1], vz=binary_star1_vxyz_soft[2])
        # sim_soft.add(m=binary_star2_mass, x=binary_star2_xyz_soft[0], y=binary_star2_xyz_soft[1], z=binary_star2_xyz_soft[2],
        #                 vx=binary_star2_vxyz_soft[0], vy=binary_star2_vxyz_soft[1], vz=binary_star2_vxyz_soft[2])
        # ##Finally adding the teriary phase...
        # sim_soft.add(m = tertiary_star_mass, x=tertiary_star_xyz[0], y=tertiary_star_xyz[1], z=tertiary_star_xyz[2], 
        #                vx=tertiary_star_vxyz[0], vy=tertiary_star_vxyz[1], vz=tertiary_star_vxyz[2])
        #Also have to initialize the softening here(!) And set it for all the particles
        #Then integrate sim_soft and sim_Newton forward in time (with same stopping condition as in the triple case?) -- Watch the capitalization here.
        ###################################################################################################################
        ##TO DO: ADD SOFTENING IN THE SIMULATION SETUPS
        ##Randomization of phase in softened potential: Numerical method? -- Not sure about the integration time?? To get a converged result, we may 
        ##have to integrate for more than the precession time of the binary...
        ic_row = ic_for_softened.loc[cleaned_index[ii]]
        binary_star1_mass, binary_star1_xyz_soft, binary_star1_vxyz_soft =  ic_row.iloc[int(aux_row[2])]["mcol"], ic_row.iloc[int(aux_row[2])][["x", "y", "z"]].to_numpy().ravel(), ic_row.iloc[int(aux_row[2])][["vx", "vy", "vz"]].to_numpy().ravel()
        binary_star2_mass, binary_star2_xyz_soft, binary_star2_vxyz_soft =  ic_row.iloc[int(aux_row[3])]["mcol"], ic_row.iloc[int(aux_row[3])][["x", "y", "z"]].to_numpy().ravel(), ic_row.iloc[int(aux_row[3])][["vx", "vy", "vz"]].to_numpy().ravel()
        ##VALIDATION OF CONVERSIONS--CHECK THAT THEY PRESERVE APOCENTER AND PERICENTER AS EXPECTED.
        scrambled_positions = randomize_binary_phase(binary_star1_xyz_soft, binary_star2_xyz_soft, binary_star1_vxyz_soft, binary_star2_vxyz_soft,
                        binary_star1_mass, binary_star2_mass, binary_star1_h, binary_star2_h, time = 10 * orbs[0].P, diag=True)
        assert(np.abs((scrambled_positions["RMIN"] - rp) / rp) < 1e-3)
        assert(np.abs((scrambled_positions["RMAX"] - ra) / ra) < 1e-3)
        ##Not sure what to use for the time here!!!
        scrambled_positions = randomize_binary_phase(binary_star1_xyz_soft, binary_star2_xyz_soft, binary_star1_vxyz_soft, binary_star2_vxyz_soft,
                binary_star1_mass, binary_star2_mass, binary_star1_h, binary_star2_h, time = 1e-4, diag=True)
        X1 = scrambled_positions["X1"]
        X2 = scrambled_positions["X2"]
        V1 = scrambled_positions["V1"]
        V2 = scrambled_positions["V2"]
        COM = scrambled_positions["COM"]
        COMV = scrambled_positions["COMV"]
        ##Need to add softening here...
        sim_soft = rebound.Simulation()
        sim_soft.G = GN
        sim_soft.add(m=binary_star1_mass, x=X1[0], y=X1[1], z=X1[2], 
                       vx=V1[0], vy=V1[1], vz=V1[2])
        sim_soft.add(m=binary_star2_mass, x=X2[0], y=X2[1], z=X2[2],
                        vx=V2[0], vy=V2[1], vz=V2[2])
        ##Finally adding the teriary phase...
        sim_soft.add(m = tertiary_star_mass, x=tertiary_star_xyz[0], y=tertiary_star_xyz[1], z=tertiary_star_xyz[2], 
                       vx=tertiary_star_vxyz[0], vy=tertiary_star_vxyz[1], vz=tertiary_star_vxyz[2])


        # binary_com = (binary_star1_mass * binary_star1_xyz_soft + binary_star2_mass * binary_star2_xyz_soft) / (binary_star1_mass + binary_star2_mass)
        # binary_vcom = (binary_star1_mass * binary_star1_vxyz_soft + binary_star2_mass * binary_star2_vxyz_soft) / (binary_star1_mass + binary_star2_mass)
        # sim_soft = rebound.Simulation()
        # sim_soft.G = GN
        # sim_soft.add(m=binary_star1_mass, x=binary_star1_xyz_soft[0], y=binary_star1_xyz_soft[1], z=binary_star1_xyz_soft[2], 
        #                vx=binary_star1_vxyz_soft[0], vy=binary_star1_vxyz_soft[1], vz=binary_star1_vxyz_soft[2])
        # sim_soft.add(m=binary_star2_mass, x=binary_star2_xyz_soft[0], y=binary_star2_xyz_soft[1], z=binary_star2_xyz_soft[2],
        #                 vx=binary_star2_vxyz_soft[0], vy=binary_star2_vxyz_soft[1], vz=binary_star2_vxyz_soft[2])
        # sim_soft.move_com()
        # ##Randomizing phase -- not sure of time to use here.
        # sim_soft.integrate(1e-4)
        # ##Adjusting binary star positions
        # ########################################
        # binary_star1_xyz_soft = binary_com + np.array(sim_soft.particles[0].xyz)
        # binary_star1_vxyz_soft = binary_vcom + np.array(sim_soft.particles[0].vxyz)
        # binary_star2_xyz_soft = binary_com + np.array(sim_soft.particles[1].xyz)
        # binary_star2_vxyz_soft = binary_vcom + np.array(sim_soft.particles[1].vxyz)
        # ##Setup softian simulation with randomized binary phase -- overwriting the old softened simulation(!)
        # ##TO DO: ADD SOFTENING HERE
        # sim_soft = rebound.Simulation()
        # sim_soft.G = GN
        # sim_soft.add(m=binary_star1_mass, x=binary_star1_xyz_soft[0], y=binary_star1_xyz_soft[1], z=binary_star1_xyz_soft[2], 
        #                vx=binary_star1_vxyz_soft[0], vy=binary_star1_vxyz_soft[1], vz=binary_star1_vxyz_soft[2])
        # sim_soft.add(m=binary_star2_mass, x=binary_star2_xyz_soft[0], y=binary_star2_xyz_soft[1], z=binary_star2_xyz_soft[2],
        #                 vx=binary_star2_vxyz_soft[0], vy=binary_star2_vxyz_soft[1], vz=binary_star2_vxyz_soft[2])
        # ##Finally adding the teriary phase...
        # sim_soft.add(m = tertiary_star_mass, x=tertiary_star_xyz[0], y=tertiary_star_xyz[1], z=tertiary_star_xyz[2], 
        #                vx=tertiary_star_vxyz[0], vy=tertiary_star_vxyz[1], vz=tertiary_star_vxyz[2])


        

        
