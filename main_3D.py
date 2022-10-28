from decimal import ROUND_HALF_DOWN
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S:', level=logging.WARNING)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
#logger.debug("Starting Logger")

from pyclbr import Function
import numpy
from dolfinx import mesh, fem, plot, io
import ufl
import pyvista
from mpi4py import MPI
from petsc4py import PETSc
from visualization import plot_timeseries, plot_mesh
import numpy as np
import math
from datetime import datetime
import os
import shutil


def simulate_weld():
    print("simulating")
    clear_folder_contents(os.path.join(os.getcwd(), "result"))
    l, b, h = 500, 500, 10
    nx, ny, nz = 100, 100, 5

    t = 0 # Start time
    T = 10 # End time
    num_steps = 20 # Number of time steps
    dt = (T-t)/num_steps # Time step size

    domain = mesh.create_box(MPI.COMM_WORLD, [[0, -b/2, -h], [l, b/2, 0]], [nx, ny, nz], mesh.CellType.hexahedron)

    V = fem.FunctionSpace(domain, ("CG", 1))

    u_D = fem.Function(V)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))
    bcs = [bc]
    bcs = []

    u_n = fem.Function(V)


    solution = []
    uh_t = []
    file = io.VTKFile(MPI.COMM_WORLD, "result/u.vtk", "w")
    for n in range(num_steps):
        #f = fem.Constant(domain, beta - 2 - 2 * alpha)
        v = 50
        start = (0, 0, 0)
        q = goldak_heat(v=v, start=start)
        q.t = t
        f = fem.Function(V)   
        f.interpolate(q.eval)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        #http://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/webm/timedep.html
        # iron: rho: 7,86 	cp: 0,452 	lamda: 81 	-> a: 22,8 mm²/s
        lam = 81
        rho_cp = 7.86*0.452
        F = rho_cp*u*v*ufl.dx + lam*dt*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx - (rho_cp*u_n + dt*f)*v*ufl.dx
        a = fem.form(ufl.lhs(F))
        L = fem.form(ufl.rhs(F))

        A = fem.petsc.assemble_matrix(a, bcs=bcs)
        A.assemble()
        b = fem.petsc.create_vector(L)
        uh = fem.Function(V)


        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        # Update Diriclet boundary condition 

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, L)
        
        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        logger.info(f"Solving Timestep t={t}")
        # Solve linear problem
        solver.solve(b, uh.vector)
        uh.x.scatter_forward()
        logger.info(f"Maximum Temperature: {uh.x.array.max()}")
        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array
        uh_t.append(np.copy(uh.x.array.real))


        file.write_function(uh, t)
        t = t+dt    

    timesteps = [dt for i in range(num_steps)]

    plot_timeseries(V, timesteps, uh_t)

class goldak_heat:
    # always moves in x direction
    def __init__(self, v, af=10, c=3, start = (0,0,0)):
        self.t = 0.0
        self.v = v
        self.af = af
        self.c = c
        self.start = start
    def eval(self, x):
        # Added some spatial variation here. Expression is sin(t)*x
        af=self.af
        q = 1e5
        xs = self.start[0]
        ys = self.start[1]
        zs = self.start[2]
        xp = np.exp(-(x[0,:]-xs -self.v*self.t)**2/self.af**2)
        yp = np.exp(-(x[1,:]-ys)**2/self.af**2)
        zp = np.exp(-(x[2,:]-zs)**2/self.c**2)
        logger.debug(f"Maximum of q at {x[:, np.argmax(q*xp*yp*zp)]}")
        return q*xp*yp*zp



def clear_folder_contents(path_to_folder):
    logger.info(f"Clearing folder {path_to_folder} {shutil.which(path_to_folder)}")
    for root, dirs, files in os.walk(path_to_folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    return 0

if __name__ == "__main__":
    simulate_weld()
