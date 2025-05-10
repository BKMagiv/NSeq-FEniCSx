# ---
# Copyright (C) 2025
# Author: C. Jin
# ---
# # 2D Lid-Driven Cavity Problem
# ν = 0.1
# 
#                       u = (1, 0)
#           (0,1) --------------------- (1,1)
#             |                           |
#             |                           |
#             |                           |
#  u = (0, 0) |                           | u = (0, 0)
#             |                           |
#             |                           |
#             |                           |
#           (0,0) --------------------- (1,0)
#                       u = (0, 0)
#
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

from numpy import sqrt, linspace, vstack, where, zeros_like, ones_like, isclose, max
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dolfinx import default_real_type
from dolfinx.mesh import create_unit_square, locate_entities_boundary, exterior_facet_indices
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

# --- 参数设置 ---
Nx, Ny = 32, 32  # 网格分辨率
T  = 1.0         # 总时间
M  = 100         # 时间步数
Δt = T / M       # 时间步长
ν  = 0.1         # 粘性系数
k  = 2           # 有限元阶数

# --- 网格构建 ---
domain = create_unit_square(MPI.COMM_WORLD, Nx, Ny)
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

# --- 有限元空间构建 ---
V = functionspace(domain, ("Lagrange", k + 1, (domain.geometry.dim, )))
Q = functionspace(domain, ("Lagrange", k))
W = MixedFunctionSpace(V, Q)

# --- 初始条件 ---
def initial_condition(x):
    condition = isclose(x[1], 1.0)
    return vstack(
        (
            where(condition, 1.0, 0.0),
            zeros_like(x[1])
        )
    )

# --- 边界条件 ---
def lid_boundary(x):
    return isclose(x[1], 1.0)   # y = 1 (顶部边界)

def walls_boundary(x):
    return (
        isclose(x[0], 0.0) | isclose(x[0], 1.0) | isclose(x[1], 0.0)
    )  # x = 0, x = 1, y = 0 (墙壁边界)

# --- 顶部边界(u = 1, v = 0) --- 
u_lid = Function(V)
u_lid.interpolate(lambda x: (ones_like(x[0]), zeros_like(x[1])))
lid_facets = locate_entities_boundary(domain, domain.topology.dim - 1, lid_boundary)
lid_dofs   = locate_dofs_topological(V, domain.topology.dim - 1, lid_facets)
bc_lid     = dirichletbc(u_lid, lid_dofs)

# --- 壁面边界(u = 0, v = 0) --- 
u_walls = Function(V)
u_walls.interpolate(lambda x: (zeros_like(x[0]), zeros_like(x[1])))
walls_facets = locate_entities_boundary(domain, domain.topology.dim - 1, walls_boundary)
walls_dofs   = locate_dofs_topological(V, domain.topology.dim - 1, walls_facets)
bc_walls     = dirichletbc(u_walls, walls_dofs)

bcs = [bc_lid, bc_walls]

# --- 初始条件 --- 
u_n = Function(V)
u_n.interpolate(initial_condition)

# --- 变分形式 ---
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Semi-Implicit Euler Methods
a = (
    inner(u / Δt, v) * dx
    + ν * inner(grad(u), grad(v)) * dx
    + inner(grad(u) * u_n, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
)
a_blocked = form(extract_blocks(a))

L = (
    inner(u_n / Δt, v) * dx
    + inner(Constant(domain, default_real_type(0.0)), q) * dx
)
L_blocked = form(extract_blocks(L))

# --- 组装矩阵和向量 ---
A = assemble_matrix_block(a_blocked, bcs = bcs)
A.assemble()
b = assemble_vector_block(L_blocked, a_blocked, bcs = bcs)
x = A.createVecRight() # 解向量

# --- 设置求解器 (PETSc) ---
ksp = PETSc.KSP().create(domain.comm)
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

# --- 时间循环 ---
if domain.comm.rank == 0:
    fig, ax = plt.subplots(figsize = (5, 5))
    # 启用 LaTeX 渲染
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    frames = []

    u_vals = u_n.x.array.reshape(-1, domain.geometry.dim)
    speed  = sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)

    vmin, vmax = 0, max(speed)
    levels = linspace(vmin, vmax, 21)

    triang = V.tabulate_dof_coordinates()[:, :2]
    im = ax.tripcolor(
        triang[:, 0], triang[:, 1], speed,
        shading = "gouraud", cmap = "plasma"
        )
    contour = ax.tricontour(
        triang[:, 0], triang[:, 1], speed,
        levels = levels, colors = "white", linewidths = 0.5
        )
    frames.append([im, contour])    

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
for j in range(M):
    ksp.solve(b, x)
    u_n.x.array[:offset] = x.array[:offset]

    # 更新矩阵和向量
    A.zeroEntries()
    assemble_matrix_block(A, a_blocked, bcs = bcs)
    A.assemble()

    with b.localForm() as b_local:
        b_local.set(0)
    assemble_vector_block(b, L_blocked, a_blocked, bcs = bcs)

    # --- 更新动画帧 ---
    if domain.comm.rank == 0:
        u_vals = u_n.x.array.reshape(-1, domain.geometry.dim)
        speed  = sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)

        vmin, vmax = 0, max(speed)
        levels = linspace(vmin, vmax, 21)

        im = ax.tripcolor(
            triang[:, 0], triang[:, 1], speed,
            shading = "gouraud", cmap = "plasma"
            )
        contour = ax.tricontour(
            triang[:, 0], triang[:, 1], speed,
            levels = levels, colors = "white", linewidths = 0.5
            )
        
        frames.append([im, contour])

if domain.comm.rank == 0:
    ani = animation.ArtistAnimation(fig, frames, interval = 50, blit = False)
    plt.tight_layout()
    ani.save("2D_lid_driven_cavity.gif", writer = "pillow", fps = 24, dpi = 240)
    plt.show()
