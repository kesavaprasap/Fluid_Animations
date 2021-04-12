import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging

root=logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger=logging.getLogger(__name__)
#Aspect ratio 1
Lx, Ly = (1., 1.)
nx, ny = (96, 96)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

betahat = 1.0
nu = 1e-7
# linear problem
problem = de.IVP(domain, variables=['psi', 'zeta'])

problem.parameters['beta'] = betahat
problem.parameters['nu'] = nu 

problem.substitutions['v'] = "dx(psi)"
problem.substitutions['u'] = "-dy(psi)"

problem.add_equation("dt(zeta) + beta*v - nu*(dx(dx(zeta)) + dy(dy(zeta))) = - u*dx(zeta) - v*dy(zeta)")
problem.add_equation("zeta - dx(v) + dy(u) = 0", condition=" (nx!=0) or (ny!=0)")
problem.add_equation("psi=12", condition= "(nx==0) and (ny==0)")
ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)

zeta = solver.state['zeta']
psi = solver.state['psi']



a = 0.1
sigx2 = 0.01
sigy2 = 0.01 
xo = 0
yo = 0 

# initilize a gaussian vorticity bump
zeta['g'] = a*np.exp(- (x-xo)**2/2/sigx2)*np.exp(-(y-yo)**2/2/sigy2)

solver.stop_sim_time = 200.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.02*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.1, max_writes=50)
analysis.add_task('psi')
analysis.add_task('zeta')


logger.info('Starting loop')
start_time = time.time()

while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 5 == 0:
        #zeta['g']
        #plot_bot_2d(zeta, axes = ax, figkw={"vmin":-0.05, "vmax":0.05});
        #display.clear_output()
        #display.display(plt.gcf())
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()


# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

post.merge_process_files("analysis_tasks",cleanup=True)
