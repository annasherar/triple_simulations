import rebound
sim = rebound.Simulation()
sim.integrator = "whfast"
sim.add(m=1.)
sim.add(m=1e-6,a=1.)
sim.move_to_com()  # Moves to the center of momentum frame
ps = sim.particles
c = 0.01
def starkForce(reb_sim):
    ps[1].ax += c
sim.additional_forces = starkForce
import numpy as np
Nout = 1000
es = np.zeros(Nout)
times = np.linspace(0.,100.*2.*np.pi,Nout)
for i, time in enumerate(times):
    sim.integrate(time, exact_finish_time=0) # integrate to the nearest timestep so WHFast's timestep stays constant
    es[i] = sim.particles[1].e 
