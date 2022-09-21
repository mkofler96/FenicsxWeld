from pyclbr import Function
import numpy
from dolfinx import mesh, fem, plot, io
import ufl
import pyvista
from mpi4py import MPI
from petsc4py import PETSc
from visualization import plot_timeseries
import numpy as np
import math

t = 0 # Start time
T = 10 # End time
num_steps = 20 # Number of time steps
dt = (T-t)/num_steps # Time step size


nx, ny = 100, 100
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))


u_D = fem.Function(V)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

u_n = fem.Function(V)



class goldak_heat:
    def __init__(self, v, af=0.01):
        self.t = 0.0
        self.v = v
        self.af = af
    def eval(self, x):
        # Added some spatial variation here. Expression is sin(t)*x
        gold = []
        for i in range(x.shape[1]):
            af=self.af
            gold.append(1e6*math.exp(-(x[0,i]-0.5)**2/self.af**2)*math.exp(-(x[1,i]-self.v*t)**2/self.af**2))
        return np.array(gold)


solution = []
uh_t = []
file = io.VTKFile(MPI.COMM_WORLD, "result/u.vtk", "w")
for n in range(num_steps):
    #f = fem.Constant(domain, beta - 2 - 2 * alpha)
    v = 0.1
    q = goldak_heat(v=v)
    q.t = t
    f = fem.Function(V)   
    f.interpolate(q.eval)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    F = u*v*ufl.dx + dt*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx - (u_n + dt*f)*v*ufl.dx
    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    A = fem.petsc.assemble_matrix(a, bcs=[bc])
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

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    uh_t.append(np.copy(uh.x.array.real))



    file.write_function(u_n, t)
    t = t+dt    

timesteps = [dt for i in range(num_steps)]

plot_timeseries(V, timesteps, uh_t)


