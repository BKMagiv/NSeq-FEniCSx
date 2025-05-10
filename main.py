# 2D incompressible Navier{Stokes equations
import importlib.util
if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This program requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
else:
    print("This program requires petsc4py.")
    exit(0)

from mpi4py import MPI

from numpy import sqrt, linspace, vstack, sin, pi, zeros_like, ones_like
import matplotlib.pyplot as plt
from dolfinx import default_real_type
from dolfinx.mesh import create_unit_square, exterior_facet_indices
from dolfinx.fem import (
    functionspace, Function, Constant,
    assemble_scalar, form, locate_dofs_topological, dirichletbc
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from ufl import (
    TestFunctions, TrialFunctions, MixedFunctionSpace,
    inner, grad, div, dx,
    extract_blocks
)

try:
    from petsc4py import PETSc
    import dolfinx
    if not dolfinx.has_petsc:
        print("This program requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This program requires petsc4py.")
    exit(0)

π = pi
def initial_condition_u(x):
    return vstack(
        (
            8 * sin(π * x[0])**2 * (2 * x[1]) * (1 - x[1]) * (1 - 2 * x[1]),
            -8 * π * sin(2 * π * x[0]) * (x[1] * (1 - x[1]))**2
        )
    )


def boundary_values(x):
    # Define the boundary values
    return vstack((zeros_like(x[0]), zeros_like(x[1])))

def forsing_term(x):
    # Define the forcing term
    return vstack((10. * ones_like(x[0]), zeros_like(x[1])))

# Define some simulation parameters
Nx, Ny = 32, 32
T, M = 1, 200
Δt, t = T / M, linspace(0, T, M + 1)
ν = 1.
k = 1 # Polynomial degree

# Create a mesh and the required function spaces
domain = create_unit_square(MPI.COMM_WORLD, Nx, Ny)

# Function spaces for the velocity and for the pressure
V = functionspace(domain, ("Lagrange", k + 1, (domain.geometry.dim, )))
Q = functionspace(domain, ("Lagrange", k))
W = MixedFunctionSpace(V, Q)

# Boundary conditions
u_D = Function(V)
u_D.interpolate(boundary_values)

# Create facet to cell connectivity required to find boundary facets
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
boundary_facets = exterior_facet_indices(domain.topology)
boundary_vel_dofs = locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
boundary_conditions_u = dirichletbc(u_D, boundary_vel_dofs)

bcs = [boundary_conditions_u]

# Define trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Create the function to store solution and previous time step
u_n = Function(V)
u_n.interpolate(initial_condition_u)

# Define the variational forms
a = (
    inner(u / Δt, v) * dx
    + ν * inner(grad(u), grad(v)) * dx
    + inner(grad(u_n) * u, v) * dx
    - inner(p, div(v)) * dx
    - inner(q, div(u)) * dx
)
a_blocked = form(extract_blocks(a))

# Define the forcing term
f = Function(V)
f.interpolate(forsing_term)
L = (
    inner(u_n / Δt, v) * dx + inner(f, v) * dx
    + inner(Constant(domain, default_real_type(0.0)), q) * dx
)
L_blocked = form(extract_blocks(L))

A = assemble_matrix_block(a_blocked, bcs = bcs)
A.assemble()
b = assemble_vector_block(L_blocked, a_blocked, bcs = bcs)
x = A.createVecRight()

# Create and configure solver
ksp = PETSc.KSP().create(domain.comm) # type: ignore
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options() # type: ignore
opts["mat_mumps_icntl_14"] = 80 # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
for j in range(M):
    # Compute the solution
    ksp.solve(b, x)
    u_n.x.array[:offset] = x.array_r[:offset]
    u_n.x.scatter_forward()

    A.zeroEntries()
    assemble_matrix_block(A, a_blocked, bcs = bcs) # type: ignore
    A.assemble()

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector_block(b, L_blocked, a_blocked, bcs = bcs) # type: ignore

if domain.comm.rank == 0
  x = V.tabulate_dof_coordinates()[:, :2]
  u_values = u_h.x.array.reshape(-1, domain.geometry.dim)
  speed = sqrt(u_values[:,0]**2 + u_values[:,1]**2)


  fig, axs = plt.subplots(1, 2, figsize = (12, 6))
  # 启用 LaTeX 渲染
  plt.rcParams["text.usetex"] = True
  plt.rcParams["font.family"] = "serif"

  # 速度标量场
  pcm = axs[0].tripcolor(x[:,0], x[:,1], speed, shading = 'gouraud', cmap = 'plasma')
  axs[0].set_title(r"Speed field $u$ at t = {}".format(T), fontsize = 16)
  axs[0].set_xlabel(r"$x$", fontsize = 14)
  axs[0].set_ylabel(r"$y$", fontsize = 14)
  axs[0].axis('equal')
  axs[0].grid(True, linestyle = '--', alpha = 0.5)  # 添加网格线

  # # 速度向量场

  # 提取节点位置
  x_coords = x[:, 0]
  y_coords = x[:, 1]

  # 提取速度分量
  u_vals = u_h.x.array.reshape(-1, domain.geometry.dim)

  # 稀疏采样：选择每隔几个点画一个箭头
  skip = 4  # 调整密度
  x_sub = x_coords[::skip]
  y_sub = y_coords[::skip]
  u_sub = u_vals[::skip, 0]
  v_sub = u_vals[::skip, 1]

  axs[1].quiver(x_sub, y_sub, u_sub, v_sub, scale = 50, color = 'r')
  axs[1].set_title(r"Velocity vector field at $t = {}$".format(T), fontsize = 16)
  axs[1].set_xlabel(r"$x$")
  axs[1].set_ylabel(r"$y$")
  axs[1].axis('equal')
  axs[1].grid(True, linestyle = '--', alpha = 0.5)

  plt.tight_layout()
  plt.show()
