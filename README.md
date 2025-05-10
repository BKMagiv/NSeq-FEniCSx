# NSeq-FEniCSx
using FEniCSx to solve the 2D incompressible Naver-Stokes equations

## Example 1
We consider the 2D incompressible Navier-Stokes equations on a spatial domain $\Omega = [0, 1]^{2}$:

$$
\begin{align*}
\partial_{t} u - \nu\Delta u + (u\cdot\nabla)u + \nabla p &= f     \quad &\text{in}\quad\Omega\times[0,T],\\
                                           \nabla \cdot u &= 0     \quad &\text{in}\quad\Omega\times[0,T],\\
                                                   u(x,0) &= u_{0} \quad &\text{in}\quad\Omega,\\
                                                        u &= g     \quad &\text{on}\quad\partial\Omega\times[0,T].
\end{align*}
$$

## Example 2
We consider the 2D Lid-Driven Cavity Flow on a spatial domain $\Omega = [0, 1]^{2}$:

$$
\begin{align*}
\partial_{t} u - \nu\Delta u + (u\cdot\nabla)u + \nabla p &= f      \quad &\text{in}\quad\Omega\times[0,T],\\
                                           \nabla \cdot u &= 0      \quad &\text{in}\quad\Omega\times[0,T],\\
                                                   u(x,0) &= 0      \quad &\text{in}\quad\Omega,\\
                                              u(y = 1, t) &= (1, 0) \quad &\text{in}\quad\{y = 1\}\times[0,T],\\
                                                        u &= 0      \quad &\text{in}\quad\text{otherwise}\times[0,T].
\end{align*}
$$

<div align = "left">
  <img src = "https://github.com/BKMagiv/NSeq-FEniCSx/blob/71a3f5f9c24e48737ee8a9880cf0d5648b89922f/images/2D_lid_driven_cavity.gif" width = 300>
</div>
