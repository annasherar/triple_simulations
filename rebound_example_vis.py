import time
import rebound

sim = rebound.Simulation()
sim.add(m = 1)
sim.add(a = 1)
## The integration time will be far smaller in your case
t = 1
sim.save_to_file("archive.bin", interval=t / 20, delete_file=True)
sim.integrate(t)

