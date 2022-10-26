#!/usr/bin/env python
import time
import numpy as np
import pathlib
from scipy.io import loadmat
from numba import jit, uint8, boolean, int64, float64

VELOCITY_OUT_DIR = "NS_velocity"
PREFIX = "NSTOKES_TEMP_"

MATLAB_data_anon_functions = None  # updated in load_anon_test_data()
MATLAB_data_functions = None  # updated in load_fn_test_data()

# jit def dxuu(a,b,dn)                                       # {{{
@jit(
    float64[:, :](float64[:, :], float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def dxuu(a, b, dn):
    # finite difference derivative of U*U w.r.t x  (staggered grid)
    # dxuu=@(a,b,i,j)                             ...
    #      (((a(i,j+1)     + a(i,j  )).*0.5).^2 - ...
    #       ((a(i,j)       + a(i,j-1)).*0.5).^2)./dn;
    return (
        ((a[1:-1, 2:] + a[1:-1, 1:-1]) * 0.5) ** 2
        - ((a[1:-1, 1:-1] + a[1:-1, :-2]) * 0.5) ** 2
    ) / dn


# }}}
# jit def dyuv(a,b,dn)                                       {{{
@jit(
    float64[:, :](float64[:, :], float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def dyuv(a, b, dn):
    # finite difference derivative of U*V w.r.t y  (staggered grid)
    return (
        (a[2:, 1:-1] + a[1:-1, 1:-1]) * (b[1:-1, 2:] + b[1:-1, 1:-1]) * 0.25
        - (a[1:-1, 1:-1] + a[:-2, 1:-1]) * (b[:-2, 2:] + b[:-2, 1:-1]) * 0.25
    ) / dn


# }}}
# jit def dyvv(a,b,dn)                                       # {{{
@jit(
    float64[:, :](float64[:, :], float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def dyvv(a, b, dn):
    # finite difference derivative of V*V w.r.t y  (staggered grid)
    return (
        ((a[2:, 1:-1] + a[1:-1, 1:-1]) * 0.5) ** 2
        - ((a[1:-1, 1:-1] + a[:-2, 1:-1]) * 0.5) ** 2
    ) / dn


# }}}
# jit def dxuv(a,b,dn)                                       # {{{
@jit(
    float64[:, :](float64[:, :], float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def dxuv(a, b, dn):
    # finite difference derivative of U*V w.r.t x  (staggered grid)
    return (
        (a[2:, 1:-1] + a[1:-1, 1:-1]) * (b[1:-1, 2:] + b[1:-1, 1:-1]) * 0.25
        - (a[1:-1, :-2] + a[2:, :-2]) * (b[1:-1, 1:-1] + b[1:-1, :-2]) * 0.25
    ) / dn


# }}}
# jit def DX2(a,dn)                                          # {{{
@jit(
    float64[:, :](float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def DX2(a, dn):
    # finite difference for laplace of operator in two dimensions (second deriv x + second deriv ystaggered grid)
    return (
        a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2] - 4 * a[1:-1, 1:-1]
    ) / (dn**2)


# }}}
# jit def dx0(a,dn)                                          # {{{
@jit(
    float64[:, :](float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def dx0(a, dn):
    # first order derivative finite difference used for pressure w.r.t x (un-staggered grid)
    return (a[1:-1, 2:] - a[1:-1, 1:-1]) / dn


# }}}
# jit def dy0(a,dn)                                          # {{{
@jit(
    float64[:, :](float64[:, :], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def dy0(a, dn):
    # first order derivative  finite difference used for pressure w.r.t x (un-staggered grid)
    return (a[2:, 1:-1] - a[1:-1, 1:-1]) / dn


# }}}
# jit def apply_bc_wall(U, V, BC, CVC)                       # {{{
@jit(
    uint8(float64[:, :], float64[:, :], uint8[:, :], boolean[:, :]),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def apply_bc_wall(U, V, BC, CVC):
    # U, V updated in-place
    # BC  : integer matrix
    # CVC : Boolean matrix
    U[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 1) * U[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 1) * (
        (CVC[:-2, 1:-1]) * 0.5 * (U[:-2, 1:-1] + U[2:, 1:-1])
    )  # velocity parallel with wall
    V[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 1) * V[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 1) * (
        (CVC[:-2, 1:-1]) * 0.5 * (V[:-2, 1:-1] + V[2:, 1:-1])
        + (~CVC[:-2, 1:-1]) * (-V[2:, 1:-1]) * 2 / 3
    )  # velocity tangent to wall
    # If node node i,j is "above" wall or const vel [BC[1:-1,1:-1] = 3]
    U[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 3) * U[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 3) * (
        (CVC[2:, 1:-1]) * 0.5 * (U[:-2, 1:-1] + U[2:, 1:-1])
    )  # velocity parallel with wall (zero if wall is non-moving solid)
    V[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 3) * V[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 3) * (
        (CVC[2:, 1:-1]) * 0.5 * (V[2:, 1:-1] + V[:-2, 1:-1])
        + (~CVC[2:, 1:-1]) * (-V[:-2, 1:-1]) * 2 / 3
    )  # velocity parallel to wall Boundary (zero if wall)
    # If node node i,j is "left" of wall or const vel [BC[1:-1,1:-1] = 2]
    V[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 2) * V[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 2) * (
        (CVC[1:-1, 2:]) * 0.5 * (V[1:-1, :-2] + V[1:-1, 2:])
    )  # velocity parallel with wall
    U[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 2) * U[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 2) * (
        (CVC[1:-1, 2:]) * 0.5 * (U[1:-1, :-2] + U[1:-1, 2:])
        + (~CVC[1:-1, 2:]) * (-U[1:-1, :-2]) * 2 / 3
    )  # velocity tangent to wall
    # If node node i,j is "right" of wall or const vel [BC[1:-1,1:-1] = 4]
    V[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 4) * V[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 4) * (
        (CVC[1:-1, :-2]) * 0.5 * (V[1:-1, :-2] + V[1:-1, 2:])
    )  # velocity parallel with wall
    U[1:-1, 1:-1] = (BC[1:-1, 1:-1] != 4) * U[1:-1, 1:-1] + (BC[1:-1, 1:-1] == 4) * (
        (CVC[1:-1, :-2]) * 0.5 * (U[1:-1, :-2] + U[1:-1, 2:])
        + (~CVC[1:-1, :-2]) * (-U[1:-1, 2:]) * 2 / 3
    )  # velocity tangent to wall
    return 0


# }}}
# jit def neumann_bc_for_outflow(U, V, OF)                   # {{{
@jit(
    uint8(float64[:, :], float64[:, :], boolean[:, :]),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def neumann_bc_for_outflow(U, V, OF):
    # U, V updated in-place
    # OF is a Boolean matrix
    # apply neumann boundary condition for outflow nodes
    # if node OF[1:-1,1:-1] = 1 , where i,j is exterior node, u[1:-1,1:-1]
    # and v[1:-1,1:-1] are equal to velocity at nearest nodes (which are not
    # outlow nodes) e.g- if OF[0,1:-2] = 1, then set U[0,1:-2]=U[1,1:-2]
    # and V[0,1:-2]=V[1,1:-2]

    U[:, -1] = (OF[:, -1]) * U[:, -2] + (~OF[:, -1]) * U[:, -1]  # right boundary
    U[:, 0] = (OF[:, 0]) * U[:, 1] + (~OF[:, 0]) * U[:, 0]  # left boundary
    U[0, :] = (OF[0, :]) * U[1, :] + (~OF[0, :]) * U[0, :]  # Top boundary
    U[-1, :] = (OF[-1, :]) * U[-2, :] + (~OF[-1, :]) * U[-1, :]  # Bottom boundary

    #  V velocity (handle exterior nodes)
    V[:, -1] = (OF[:, -1]) * V[:, -2] + (~OF[:, -1]) * V[:, -1]  # right boundary
    V[:, 0] = (OF[:, 0]) * V[:, 1] + (~OF[:, 0]) * V[:, 0]  # left boundary
    V[0, :] = (OF[0, :]) * V[1, :] + (~OF[0, :]) * V[0, :]  # Top boundary
    V[-1, :] = (OF[-1, :]) * V[-2, :] + (~OF[-1, :]) * V[-1, :]  # Bottom boundary

    #  V velocity (handle exterior (corner) nodes)
    V[0, 0] = (OF[0, 0]) * V[1, 1] + (~OF[0, 0]) * V[0, 0]  # Top left corner
    V[0, -1] = (OF[0, -1]) * V[1, -2] + (~OF[0, -1]) * V[0, -1]  # Top right corner
    V[-1, 0] = (OF[-1, 0]) * V[-2, 1] + (~OF[-1, 0]) * V[-1, 0]  # Bottom left corner
    V[-1, -1] = (OF[-1, -1]) * V[-2, -2] + (~OF[-1, -1]) * V[
        -1, -1
    ]  # Bottom right corner

    #  U velocity (handle exterior (corner) nodes)
    U[0, 0] = (OF[0, 0]) * U[1, 1] + (~OF[0, 0]) * U[0, 0]  # Top left corner
    U[0, -1] = (OF[0, -1]) * U[1, -2] + (~OF[0, -1]) * U[0, -1]  # Top right corner
    U[-1, 0] = (OF[-1, 0]) * U[-2, 1] + (~OF[-1, 0]) * U[-1, 0]  # Bottom left corner
    U[-1, -1] = (OF[-1, -1]) * U[-2, -2] + (~OF[-1, -1]) * U[
        -1, -1
    ]  # Bottom right corner

    return 0


# }}}
# jit def temporary_velocities(U, V, BC, CV, L, uc, vc, dt, dn) # {{{
@jit(
    uint8(
        float64[:, :],
        float64[:, :],
        uint8[:, :],  # U, V, BC
        boolean[:, :],
        float64,
        float64[:, :],  # CV, L, uc
        float64[:, :],
        float64,
        float64,
    ),  # vc, dt, dn
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def temporary_velocities(U, V, BC, CV, L, uc, vc, dt, dn):
    # U, V updated in-place
    # BC is an integer array, values 0, 1, 2, 3, 4
    # CV is a boolean array that defines interior solids: 1=empty, 0=solid
    # if node is not solid wall or constant velocity region (CV = 1),
    # update velocity and, also, if node is not next to boundary(i.e - BC == 0),
    # update velocity [nodes i,j for B[1:-1,1:-1] != 0 already had velocities
    # determined when apply B.C (above)]
    # update x-velocity
    U[1:-1, 1:-1] += (
        (BC[1:-1, 1:-1] == 0)
        * (CV[1:-1, 1:-1] == 1)
        * dt
        * (L * (DX2(uc, dn)) - (dxuu(uc, uc, dn) + dyuv(uc, vc, dn)))
    )
    # update y-velocity
    V[1:-1, 1:-1] += (
        (BC[1:-1, 1:-1] == 0)
        * (CV[1:-1, 1:-1] == 1)
        * dt
        * (L * (DX2(vc, dn)) - (dxuv(uc, vc, dn) + dyvv(vc, vc, dn)))
    )
    return 0


# }}}
# jit def fd_solve_poisson_pressure(P0, U, V, T,             # {{{
@jit(
    int64(
        float64[:, :],
        float64[:, :],
        float64[:, :],  # P0, U, V,
        int64,
        uint8[:, :],
        boolean[:, :],  # T, CP, OF
        boolean[:, :],
        float64[:, :],
        float64,  # OF2, CA, dn
        float64,
        float64,
        int64,
        int64,
    ),  # dt, error, MAXIT, MINIT
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def fd_solve_poisson_pressure(
    P0, U, V, T, CP, OF, OF2, CA, dn, dt, error, MAXIT, MINIT  # inputs and outputs
):
    # P0, U, V are updated in-place
    #   PIT = 0 # number of iterations during each time pressure poisson equation is solved
    f = V.copy()
    P1 = V.copy()
    # Finite difference solve for poisson pressure
    # continuity equation is used in RHS side of poisson equation (f) to enforce incompressible flow
    errach = T == 0  # parameter that determine pressure poisson solution has converged
    # or surpassed maximum allowed iterations (MAXIT)
    # also, calculation not calculated for first instance (errach = 1 if T=0)
    mite = 0  # counts number of iterations run for poisson equation
    cp = (T != 1) * 1 + (
        T == 1
    ) * 10  # multiply maximum allowed iterations, for poisson equation solve, by 10 to allow for smoothing at first step of calculation (T=1)
    f[1:-1, 1:-1] = (
        (OF2[1:-1, 1:-1] == 0)
        * (CA[1:-1, 1:-1] == 1)
        * (dn / dt)
        * (V[1:-1, 1:-1] + U[1:-1, 1:-1] - V[:-2, 1:-1] - (U[1:-1, :-2]))
    )  #  [right hand side of poisson equation]
    while errach == 0:  # while solution has not converged for poisson equation
        # Finite difference solve of poisson pressure equation (for pressure field)
        P1[1:-1, 1:-1] = (
            0.75
            * (
                P0[2:, 1:-1]
                + P0[:-2, 1:-1]
                + P0[1:-1, :-2]
                + P0[1:-1, 2:]
                - f[1:-1, 1:-1]
            )
            / CP[1:-1, 1:-1]
            + 0.25 * P0[1:-1, 1:-1]
        )

        #  Next line not allowed by numba
        #  P1[OF == 1] = 0 # if node is an outflow condition set pressure to zero
        #                  # using relaxtion method  A=0.75 and 1-A=0.25 (last term)
        P1 *= ~(OF == 1)  # equivalent to P1[OF == 1] = 0

        #  Next line not allowed by numba
        #  P1[CA == 0] = 0 # if node is solid wall node (still or moving wall) set
        #                  # pressure to zero
        P1 *= CA  # equivalent to P1[CA == 0] = 0
        # print(f'while loop mite={mite:3d} error={np.max(np.abs(P1-P0)): 18.12e}')

        # determine if max difference between next and current pressure
        errach = np.max(np.abs(P1 - P0)) <= error
        # field satifies allowed tolerance for error
        # [If yes, errach =  1 = solution converged (exit while loop)]
        mite += 1  # add to count of iterations of poisson equation solution

        # Next line not allowed by numba
        # np.copyto(P0, P1) # not P0 = P1.copy(), which makes a new variable
        P0[:] = P1[:]

        #  PA(:,:,size(PA,3)+1) = P0# temp use
        if mite >= (MAXIT * cp):
            # if iterations of poisson solver surpases max number of allow iterations
            errach = 1  # set errach =1 (exit while loop) regardless of
            # whether tolerance of error has not been reached
        if mite < MINIT:
            # if iterations of poisson solver is less than mininum required
            # number of iteration (MINIT) set errach =0 (don't exit while loop)
            # regardless of whether tolerance of error has been reached
            errach = 0

    return mite


# }}}
# jit def index_mask_equivalence(a, I, RHS)                  # {{{
@jit(
    float64[:, :](float64[:, :], boolean[:, :], float64[:]),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def index_mask_equivalence(a, I, RHS):
    nR, nC = a.shape
    n_true = np.sum(I)
    trial_A = a.copy()
    # things that don't work in numba:
    #   A[I] = RHS
    #   A.itemset(i_true, RHS[i_rhs])
    #   A.put(i_true, RHS)
    #   np.unravel_index()

    i_true = np.argwhere(I.flatten()).flatten()
    for i_rhs, (r, c) in enumerate(zip(i_true // nC, i_true % nC)):
        trial_A[r, c] = RHS[i_rhs]

    return trial_A


# }}}


def storevar(array, prefix, counter):  # {{{
    vel_file = f"{VELOCITY_OUT_DIR}/{prefix}{counter:03d}.npy"
    np.save(vel_file, array)
    print(f"wrote {vel_file}")


# }}}
def solve_flow(  # {{{
    velinit,
    bounds,
    outflow,
    XI,
    YI,
    ST,
    domainX,
    velxi,
    velyi,
    MI,
    dt,
    mu,
    dens,
    error,
    MINIT,
    MAXIT,
):

    tic = time.time()  # start timer for calculations
    frame_time = np.zeros((MI, 2))  # pairs of elapsed time and simulation time
    dn = domainX / XI  # dn=dx=dy (distance between nodes) [same result as domainY/YI]
    domainY = domainX * (YI / XI)  # domain length of y axis (domainY);
    #  this is scaled with respect to;
    #  -> specified domainX
    #  -> and, w.r.t to dimensions of input image (Scenario)
    # i = 2:YI+1 # column index for calculations
    # j = 2:XI+1 # row index for calculations

    ts = int(
        np.round(np.prod(ST) / (XI * YI))
    )  # ts = number of timesteps to calculate u and v before saving as external file

    # initialize variables pressure and velocity
    U = np.zeros((YI + 2, XI + 2))
    # initialise variable space for x and y velocity (staggered grid)
    V = U.copy()
    # initialise variable space for pressure (P0,P01,P1&f used at various stages in calculations)
    P0 = V.copy()
    # initialize variable space for variables that will help signal boundary conditions
    C = V.copy().astype(np.uint8)
    OF = V.copy().astype(np.bool_)
    U[1:-1, 1:-1] = velinit * velxi  #  set velocity of constant velocity nodes for u
    velx = 0.5 * (
        U[1:-1, 2:] + U[1:-1, 1:-1]
    )  # initialize matrix that will store unstaggered grid for u
    vely = 0.5 * (
        V[2:, 1:-1] + V[1:-1, 1:-1]
    )  # initialize matrix that will store unstaggered grid for v

    # _______________________________
    #  YI = 50   XI = 200
    # _______________________________

    #  Define variables that are used to signal particular boundary conditions
    #  C[1:-1,1:-1] = 1, if node i,j is not a solid wall node (=0 if otherwise)
    C[1:-1, 1:-1] = 1
    C[1:-1, 1:-1] = bounds == 0
    C[1:-1, 1:-1] = (velinit == 0) * C[1:-1, 1:-1] + (velinit == 1)
    #  CV[1:-1,1:-1] = 1, when i,j is not solid wall,constant velocity region or outlfow node (=0 if otherwise)
    CV = C.copy().astype(
        np.bool_
    )  # initial equal CV=C [CV=1 velocity time variant, CV=0 if velocity constant over time]
    CV[1:-1, 1:-1] = (
        velinit == 0
    )  #  set nodes of constant velocity C=0 (C=0=unchanged velocity)
    CV[1:-1, 1:-1] = (bounds == 0) * CV[1:-1, 1:-1]  # if bound
    #  CVC[1:-1,1:-1] = 1, only if node i,j is in region of constant velocity  (=0 if otherwise)
    CVC = V.copy().astype(
        np.bool_
    )  # CVC=1 only solely for when a constant velocity condition is set.
    # CVC[1:-1,1:-1] = velinit
    CVC[1:-1, 1:-1] = (velinit == 1) + (velinit == 0) * (outflow == 0) * CVC[1:-1, 1:-1]
    #  OF[1:-1,1:-1] = 1, if node is outflow node (=0 if otherwise)
    OF[1:-1, 1:-1] = outflow.astype(np.bool_)

    # Redefine Outflow boundary nodes (if exist in scenario)
    # This section is to reduce amount nodes which are outflow, keeping only exterior
    # nodes as outflow node (if defined): [only want exterior nodes to be outflow]
    # Set exterior nodes of OF to replicate closest interior node
    OF[:, -1] = OF[:, -2] == 1
    OF[:, 0] = OF[:, 1] == 1
    OF[-1, :] = OF[-2, :] == 1
    OF[0, :] = OF[1, :] == 1
    # Set any interior node,previously outflow node, to not be outflow node:
    BN = np.zeros(C.shape)
    BN[1:-1, 1:-1] = 1  # temporary variable interior nodes =1 , exterior equal zero
    OF[
        BN == 1
    ] = False  # set interior node OF to 0 (only exterior can be 1, if it is outflow node)
    # OF2[1:-1,1:-1] = 1 if it has a neighbouring cell that is an outflow node (OF[1:-1,1:-1]=1):
    OF2 = OF.copy()  # initialize OF2
    OF2[1:-1, 1:-1] = (
        OF[2:, 1:-1] + OF[:-2, 1:-1] + OF[1:-1, 2:] + OF[1:-1, :-2]
    ) != 0  # OF2[1:-1,1:-1]=1 if no OF=1 in neighbour cells

    # Define variable BC which determines orientation of Boundary with respect to node
    BC = np.zeros(C.shape, dtype=np.uint8)  # initialize boundary nodes
    # if BC[1:-1,1:-1] = 2 , node [1:-1,2:] is either solid wall or constant velocity node [node [1:-1,1:-1] is "left" of solid wall or const vel node]
    bc = (CV[1:-1, 1:-1]) * (
        CV[1:-1, 2:] == 0
    )  # if node i,j is a  free flow node but node i,j+1 is not (i.e- is a wall)
    BC[1:-1, 1:-1] = (BC[1:-1, 1:-1] == 0) * (
        (bc == 1) * 2 + (bc == 0) * BC[1:-1, 1:-1]
    ) + (BC[1:-1, 1:-1] != 0) * BC[1:-1, 1:-1]
    # if BC[1:-1,1:-1] = 4 , node [1:-1,:-2] is either solid wall or constant velocity node [node [1:-1,1:-1] is "right" of solid wall or const vel node]
    bc = (CV[1:-1, 1:-1]) * (
        CV[1:-1, :-2] == 0
    )  # if node i,j is a  free flow node but node i,j-1 is not
    BC[1:-1, 1:-1] = (BC[1:-1, 1:-1] == 0) * (
        (bc == 1) * 4 + (bc == 0) * BC[1:-1, 1:-1]
    ) + (BC[1:-1, 1:-1] != 0) * BC[1:-1, 1:-1]
    # if BC[1:-1,1:-1] = 1 , node [:-2,1:-1] is either solid wall or constant velocity node [node [1:-1,1:-1] is "above" solid wall or const vel node]
    bc = (CV[1:-1, 1:-1]) * (
        CV[:-2, 1:-1] == 0
    )  # if node i,j is a  free flow node but node i-1,j is not
    BC[1:-1, 1:-1] = (BC[1:-1, 1:-1] == 0) * (
        (bc == 1) * 1 + (bc == 0) * BC[1:-1, 1:-1]
    ) + (BC[1:-1, 1:-1] != 0) * BC[1:-1, 1:-1]
    # if BC[1:-1,1:-1] = 3 , node [2:,1:-1] is either solid wall or constant velocity node [node [1:-1,1:-1] is "below" solid wall or const vel node]
    bc = (CV[1:-1, 1:-1] == 1) * (
        CV[2:, 1:-1] == 0
    )  # if node i,j is a  free flow node but node i+1,j is not
    BC[1:-1, 1:-1] = (BC[1:-1, 1:-1] == 0) * (
        (bc == 1) * 3 + (bc == 0) * BC[1:-1, 1:-1]
    ) + (BC[1:-1, 1:-1] != 0) * BC[1:-1, 1:-1]
    BC[
        OF == 1
    ] = 0  # exclude outflow nodes as being handled as solid wall/const vel boundary
    BC[
        OF2 == 1
    ] = 0  # exclude outflow nodes as being handled as solid wall/const vel boundary
    BC = (CV == 1) * BC

    #  initialize matrix that will store unstaggered grid for u
    # velx = 0.5*(U(2:YI+1,3:XI+2]+U(2:YI+1,2:XI+1))
    velx = 0.5 * (U[1:-1, 2:] + U[1:-1, 1:-1])
    #  initialize matrix that will store unstaggered grid for u
    # vely = 0.5*(V(3:YI+2,2:XI+1)+V(2:YI+1,2:XI+1))
    vely = 0.5 * (V[2:, 1:-1] + V[1:-1, 1:-1])

    # determine divisor for pressure poisson equation
    CP = np.zeros(
        P0.shape, dtype=np.uint8
    )  # initial matrix to store divisor coefficients in poisson solver
    CP[1:-1, 1:-1] = (
        (C[2:, 1:-1] == 1).astype(np.uint8)
        + (C[:-2, 1:-1] == 1).astype(np.uint8)
        + (C[1:-1, :-2] == 1).astype(np.uint8)
        + (C[1:-1, 2:] == 1).astype(np.uint8)
    )
    # only solid walls are zeros  %<<<<<<<<<<<<<<<<<<<<<<<<<typically using
    # -->CP[1:-1,1:-1] = 4 if node i,j not next to solid wall (interior node)
    # -->CP[1:-1,1:-1] = 3 if node i,j next solid wall (above/down/left/right)
    # -->CP[1:-1,1:-1] = 2 if node i,j next corner of solid wall (corner node)
    # -->node i,j is surrounded by solid wall nodes below will be zero (but correct to 1)
    CP[
        CP <= 1
    ] = 4  #  if CP[1:-1,1:-1]==0, set to 4 to avoid division by zero in poisson equation (happens if node i,j is "inside" solid wall)
    CP[
        OF2 == 1
    ] = 4  #  if node i,j is neighbouring outflow node [OF2[1:-1,1:-1]=1] set divisor to 4

    #  DETERMINE MEMORY USAGE
    dirr = pathlib.Path(VELOCITY_OUT_DIR)
    if not dirr.is_dir():  # if directory does not exist (external save of u and v)
        dirr.mkdir()  # create directory to store u and v

    # 1 kilobyte  =  2^10 bytes , 1 megabyte = 2^10 kilobytes , 1 gigabyte = 2^10 megabytes
    # ans = 1 (variable) , ans=8 bytes   ans=ones(2,2,3) ,ans=2*2*3 bytes
    # matrix of size a x b x c  =  a*b*c*8 bytes = a*b*c*8/(C^3) gigabytes  C=2^10
    # FILES THAT WILL BE SAVE velx and vely so if mesh  =  200 x 200 and MI=1000
    # total saved on hardrive space  =  8*2*200*200*MI/(C^3)
    spaceused = 1.0889 * (
        8 * XI * YI * MI / (10**9)
    )  # determine approximate space used
    print(f"Approximately {spaceused:.4f} GB of files will be written")
    print(f"to {str(dirr)}. For each velocity (x and y) there will be ")
    print(f"{np.ceil(MI/ts):.0f} files")
    # print(f'therefor %.0f files with average size of %.6f gigabytes \n',2*ceil(MI/ts),1.0889*(8*XI*YI*ts)/(2*10^9))
    # if spaceused>spacelim: # if estimated space used is more than limitations set by user (flash warning)
    #    print(f'\nWarning!!!!\n')
    #    print(f'The above estimated hard drive space usage (%.4fgigabytes)\n',spaceused)
    #    print(f'is more than the limit set by user of %.4fgigabytes.\n',spacelim)
    #    userpromp = input('<Press enter to continue or cntrl+c to cancel calculations>','s')
    #    commandwindow %bring command window to foreground

    # ts was already computed
    # ts = int(np.round(np.prod(ST)/(XI*YI))) # parameter that determines number of timesteps of calculating
    #  u and v before they are externally saved to a file (on hardrive)

    #  Ininitialise parameters and variables used in calculations
    x, y = np.meshgrid(
        np.linspace(0, domainX, XI), np.linspace(0, domainY, YI)
    )  #  x and y coordinates [used in visualisation sections]
    # i = 2:YI+1 # column index for calculations
    # j = 2:XI+1 # row index for calculations
    L = mu / dens  # kinematic viscosity (
    TIME = 0  #  paramer which keeps track of simulated time
    docalc = 1  # parameter that determines if calculations should continue (1=yes)
    T = 0  # parameter which keeps track of total number of time steps calculated
    T2 = 0  # parameter that determines when set of u and v calculations should be save externally
    nsave = (
        0  # parameter that keeps track of file number for external saving of u and v
    )

    # CP = 4*ones(size(CP))
    #  cv2 from last happy!!!!!!!!!!!!!!
    CA = np.zeros(CV.shape)

    CA[1:-1, 1:-1] = (CVC[1:-1, 1:-1] == 1) * (
        (CV[2:, 1:-1] + CV[:-2, 1:-1] + CV[1:-1, 2:] + CV[1:-1, :-2]) != 0
    )
    CA = CA + CV
    #  THE ACTUAL CALCULATIONS (Calculating TIME-DEPENDENT VELOCITY AND PRESSURE)

    # redefine velx, vely
    i_frame = 0
    Frames_per_save_block = ts  # compute this as a function of available memory
    velx = np.zeros((YI, XI, Frames_per_save_block))
    vely = np.zeros((YI, XI, Frames_per_save_block))
    solution_converged = True
    while docalc == 1:  # WHILE CALCULATIONS ARE BEING RUN
        if not (T % 300):
            print(f"completed {T*100/MI:5.1f} %\r")

        #  Specify velocities on constant velocity region
        #  (\[ CVC[1:-1,1:-1] = 1 if node i,j is in constant velocity region]
        U[1:-1, 1:-1] = (~CVC[1:-1, 1:-1]) * U[1:-1, 1:-1] + (
            CVC[1:-1, 1:-1]
        ) * velxi  # set x-velocity to velxi if node is in contant vel B.C
        V[1:-1, 1:-1] = (~CVC[1:-1, 1:-1]) * V[1:-1, 1:-1] + (
            CVC[1:-1, 1:-1]
        ) * velyi  # set y-velocity to velyi if node is in contant vel B.C

        if T:  # if not instant of calculation
            #  Apply boundary conditions for nodes next to solid wall or next to constant velocity
            #  Non-slip boundary condition e.g - B[1:-1,1:-1] = 1 & CVC[:-2,1:-1]=0  [wall exist above i,j and is non-moving solid parallel vel is zero and tangent is reflect vel from i+1,j]
            #  free-slip boundary condition e.g - B[1:-1,1:-1] = 2 & CVC[1:-1,2:]=1  [wall exist right of i,j and is moving wall (interpolate velocity at i,j) parallel is mean of vel for nodes i,j+1 and i,j-1]
            # If node node i,j is "below" wall or const vel [BC[1:-1,1:-1] = 1]

            apply_bc_wall(U, V, BC, CVC)
            neumann_bc_for_outflow(U, V, OF)

        uc = (
            U.copy()
        )  # record current x-vel (for use later on to enforce constant x vel conditions)
        vc = (
            V.copy()
        )  # record current x-vel (for use later on to enforce constant y vel conditions)

        #  Solve temporary velocities
        # calculate temporary velocity using convective and diffusive terms of momentum equations
        if T:  # if not first instant of calculation
            temporary_velocities(U, V, BC, CV, L, uc, vc, dt, dn)
            mite = fd_solve_poisson_pressure(
                P0, U, V, T, CP, OF, OF2, CA, dn, dt, error, MAXIT, MINIT
            )

        # Prompt to warn user that pressure solution has diverged (and also forcibly stop calculations [since no good will come from further running code])
        if np.any(np.isnan(P0)) or np.any(
            np.isinf(P0)
        ):  # if Pressure has infinite or nan value
            print("Error! The solution for pressure has diverged (going to inf or nan)")
            print("Try modifying dt,mu,xinc,domainX,velxi/velyi and try again!")
            solution_converged = False
            break
        #        force_exit_code = intentional_error #  this line intentionally causes error to stop code (remove this line if code won't run at all)

        #  CORRECT Velocities BY ADDING PRESURE GRADIENT
        if T:  # if not first instant being recorded (where time=0 seconds)
            # As with calculating temporary velocity (a previous section)
            #    If node is not solid wall or constant velocity region (CV = 1) , update velocity
            #    And, also, if node is not next to boundary(i.e - BC  == 0), update velocity [nodes i,j for B[1:-1,1:-1] != 0 already had velocities determined when apply B.C (above)]
            mask = (~OF[1:-1, 1:-1]) * (BC[1:-1, 1:-1] == 0) * CV[1:-1, 1:-1]
            U[1:-1, 1:-1] += mask * dt * (-(dx0(P0, dn) / dens))  # update x-velocity
            V[1:-1, 1:-1] += mask * dt * (-(dy0(P0, dn) / dens))  # update y-velocity

        #  Obtaining velocities on unstaggered grid and saving temporary files

        #  obtain velocities for unstaggered grid
        #     u1 = 0.5*(U(2:YI+1,3:XI+2)+U(2:YI+1,2:XI+1)) # x-comp velocity of unstaggerd grid (velx): velx[1:-1,1:-1]=(U[1:-1,2:]+U[1:-1,1:-1])/2 [U=staggered grid x vel]
        u1 = 0.5 * (
            U[1:-1, 2:] + U[1:-1, 1:-1]
        )  # x-comp velocity of unstaggerd grid (velx): velx[1:-1,1:-1]=(U[1:-1,2:]+U[1:-1,1:-1])/2 [U=staggered grid x vel]
        #     v1 = 0.5*(V(3:YI+2,2:XI+1)+V(2:YI+1,2:XI+1)) # y-comp velocity of unstaggerd grid (vely): vely[1:-1,1:-1]=(V[2:,1:-1]+V[1:-1,1:-1])/2 [V=staggered grid y vel]
        v1 = 0.5 * (
            V[2:, 1:-1] + V[1:-1, 1:-1]
        )  # y-comp velocity of unstaggerd grid (vely): vely[1:-1,1:-1]=(V[2:,1:-1]+V[1:-1,1:-1])/2 [V=staggered grid y vel]
        velx[:, :, T2] = (CV[1:-1, 1:-1]) * u1 + (~CV[1:-1, 1:-1]) * uc[
            1:-1, 1:-1
        ]  #  record unstaggered velocity to list velx
        vely[:, :, T2] = (CV[1:-1, 1:-1]) * v1 + (~CV[1:-1, 1:-1]) * vc[
            1:-1, 1:-1
        ]  #  record unstaggered velocity to list vely
        TIME = TIME + dt  # update time (simulation time)
        T2 += 1  # update parameter which determines when to externally save set calculations (x,y velocity)
        #  determine if current set of x and y velocity should be saved
        if T2 >= Frames_per_save_block:
            # save velx, vely
            nsave += 1  # update file save number
            storevar(
                velx, f"{PREFIX}vx_", nsave
            )  # save x velocity as file number nsave
            storevar(
                vely, f"{PREFIX}vy_", nsave
            )  # save y velocity as file number nsave
            T2 = 0
        T += 1  # update parameter which is total number of time steps

        # determine if calcuations are finished
        if T >= MI:  # if maximum allowed number of timesteps is reached
            docalc = (
                0  # set parameter to finish calculations (while loop does not continue)
            )
            if T2:  # if last set of calculations was not already saved
                # then save final set of calculations (keeping only vely and velx calculated in last set (1 to T2)
                nsave += 1  # update file save number
                storevar(
                    velx[:, :, : T2 - 1], f"{PREFIX}vx_", nsave
                )  # save x velocity as file number nsave
                storevar(
                    vely[:, :, : T2 - 1], f"{PREFIX}vy_", nsave
                )  # save y velocity as file number nsave
        toc = time.time() - tic
        frame_time[i_frame, :] = [toc, TIME]
        i_frame += 1
        if not i_frame % 10:
            print(f"frame {i_frame:6d}:  elapsed= {toc:10.6f}  sim= {TIME:10.6f}")

    toc = time.time() - tic

    np.save(f"{VELOCITY_OUT_DIR}/{PREFIX}ft.npy", frame_time)
    if solution_converged:
        print(
            "Calculations are 100%% complete! at %.4f seconds "
            "[%.0f of %.0f time-steps]" % (toc, T, MI)
        )
    else:
        print("failed at %.4f seconds " "[%.0f of %.0f time-steps]" % (toc, T, MI))
    print(f"End at simulation time {T*dt:.3f}")
    print("---------------------------------------------------------------------------")


# }}}


def load_anon_test_data():  # {{{
    global MATLAB_data_anon_functions
    # reads dxuu_data.mat
    """
    % create_dxuu_data_mat.m
    clear
    dn = 0.01;
    dxuu=@(a,b,i,j) (((a(i,j+1)+a(i,j)).*0.5).^2-((a(i,j)+a(i,j-1)).*0.5).^2)./dn;
    dyuv=@(a,b,i,j) ((a(i+1,j)+a(i,j)).*(b(i,j+1)+b(i,j)).*0.25-(a(i,j)+a(i-1,j)).*(b(i-1,j+1)+b(i-1,j)).*0.25)./dn;
    dyvv=@(a,b,i,j) (((a(i+1,j)+a(i,j)).*0.5).^2-((a(i,j)+a(i-1,j)).*0.5).^2)./dn;
    dxuv=@(a,b,i,j) ((a(i+1,j)+a(i,j)).*(b(i,j+1)+b(i,j)).*0.25-(a(i,j-1)+a(i+1,j-1)).*(b(i,j)+b(i,j-1)).*0.25)./dn;
    DX2=@(a,i,j) (a(i-1,j)+a(i+1,j)+a(i,j+1)+a(i,j-1)-4.*a(i,j))./(dn.^2);
    dx0=@(a,i,j) (a(i,j+1)-a(i,j))./dn;
    dy0=@(a,i,j) (a(i+1,j)-a(i,j))./dn;

    nX = 3000; nY = 1000;
    i=2:nY-1;
    j=2:nX-1;
    a = 2*(0.5 - rand(nY, nX));
    b = 2*(0.5 - rand(nY, nX));
    dxuu_matlab = dxuu(a,b,i,j);
    dyuv_matlab = dyuv(a,b,i,j);
    dyvv_matlab = dyvv(a,b,i,j);
    dxuv_matlab = dxuv(a,b,i,j);
    DX2_matlab  = DX2(a,i,j);
    dx0_matlab  = dx0(a,i,j);
    dy0_matlab  = dy0(a,i,j);
    save('dxuu_data.mat', 'a', 'b', 'i', 'j', 'dn', 'dxuu_matlab', ...
         'dyuv_matlab', 'dyvv_matlab', 'dxuv_matlab', ...
         'DX2_matlab', 'dx0_matlab', 'dy0_matlab')
    """

    if MATLAB_data_anon_functions is not None:
        return MATLAB_data_anon_functions
    F = "dxuu_data.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_anon_functions = {
        "a": np.ascontiguousarray(mat["a"]),
        "b": np.ascontiguousarray(mat["b"]),
        "i": mat["i"] - 1,  # unused by Python version
        "j": mat["j"] - 1,  # unused by Python version
        "dn": mat["dn"],
        "dxuu_matlab": mat["dxuu_matlab"],
        "dyuv_matlab": mat["dyuv_matlab"],
        "dyvv_matlab": mat["dyvv_matlab"],
        "dxuv_matlab": mat["dxuv_matlab"],
        "DX2_matlab": mat["DX2_matlab"],
        "dx0_matlab": mat["dx0_matlab"],
        "dy0_matlab": mat["dy0_matlab"],
    }

    return MATLAB_data_anon_functions


# }}}
def load_fn_test_data():  # {{{
    # reads:
    #   fd_solve_poisson_inputs.mat
    #   fd_solve_poisson_outputs.mat
    #   neumann_bc_inputs.mat
    #   neumann_bc_outputs.mat
    #   temporary_velocities_inputs.mat
    #   temporary_velocities_outputs.mat

    global MATLAB_data_functions

    if MATLAB_data_functions is not None:
        return MATLAB_data_functions

    MATLAB_data_functions = {}

    F = "apply_bc_wall_inputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["apply_bc_wall_in"] = {
        "BC": np.ascontiguousarray(mat["BC"]),
        "CVC": np.ascontiguousarray(mat["CVC"]),
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
    }

    F = "apply_bc_wall_outputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["apply_bc_wall_out"] = {
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
    }

    F = "fd_solve_poisson_inputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["fd_solve_poisson_in"] = {
        "P0": np.ascontiguousarray(mat["P0"]),
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
        "CP": np.ascontiguousarray(mat["CP"]).astype(np.uint8),
        "OF": np.ascontiguousarray(mat["OF"]).astype(np.uint8),
        "OF2": np.ascontiguousarray(mat["OF2"]).astype(np.uint8),
        "CA": np.ascontiguousarray(mat["CA"]).astype(np.bool_),
        "T": int(mat["T"]),
        "dn": mat["dn"],
        "dt": mat["dt"],
        "error": mat["error"],
        "MAXIT": int(mat["MAXIT"]),
        "MINIT": int(mat["MINIT"]),
    }

    F = "fd_solve_poisson_outputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["fd_solve_poisson_out"] = {
        "P0": np.ascontiguousarray(mat["P0"]),
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
    }

    F = "neumann_bc_inputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["neumann_bc_in"] = {
        "OF": np.ascontiguousarray(mat["OF"]),
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
    }

    F = "neumann_bc_outputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["neumann_bc_out"] = {
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
    }

    F = "temporary_velocities_inputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["temporary_vel_in"] = {
        "BC": np.ascontiguousarray(mat["BC"]),
        "CV": np.ascontiguousarray(mat["CV"]),
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
        "uc": np.ascontiguousarray(mat["uc"]),
        "vc": np.ascontiguousarray(mat["vc"]),
        "dn": mat["dn"],
        "dt": mat["dt"],
        "L": mat["L"],
    }

    F = "temporary_velocities_outputs.mat"
    mat = loadmat(F, squeeze_me=True)
    print(f'loadmat("{F}")')
    MATLAB_data_functions["temporary_vel_out"] = {
        "U": np.ascontiguousarray(mat["U"]),
        "V": np.ascontiguousarray(mat["V"]),
    }

    return MATLAB_data_functions


# }}}
def test_dxuu():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    b = M_data["b"]
    dn = M_data["dn"]
    dxuu_matlab = M_data["dxuu_matlab"]

    T_s = time.time()
    dxuu_python = dxuu(a, b, dn)
    error = np.max(np.abs(dxuu_python - dxuu_matlab))
    print(f"dxuu error= {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_dyuv():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    b = M_data["b"]
    dn = M_data["dn"]
    dyuv_matlab = M_data["dyuv_matlab"]

    T_s = time.time()
    dyuv_python = dyuv(a, b, dn)
    error = np.max(np.abs(dyuv_python - dyuv_matlab))
    print(f"dyuv error= {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_dyvv():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    b = M_data["b"]
    dn = M_data["dn"]
    dyvv_matlab = M_data["dyvv_matlab"]

    T_s = time.time()
    dyvv_python = dyvv(a, b, dn)
    error = np.max(np.abs(dyvv_python - dyvv_matlab))
    print(f"dyvv error= {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_dxuv():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    b = M_data["b"]
    dn = M_data["dn"]
    dxuv_matlab = M_data["dxuv_matlab"]

    T_s = time.time()
    dxuv_python = dxuv(a, b, dn)
    error = np.max(np.abs(dxuv_python - dxuv_matlab))
    print(f"dxuv error= {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_DX2():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    dn = M_data["dn"]
    DX2_matlab = M_data["DX2_matlab"]

    T_s = time.time()
    DX2_python = DX2(a, dn)
    error = np.max(np.abs(DX2_python - DX2_matlab))
    print(f"DX2 error = {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_dx0():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    dn = M_data["dn"]
    dx0_matlab = M_data["dx0_matlab"]

    T_s = time.time()
    dx0_python = dx0(a, dn)
    error = np.max(np.abs(dx0_python - dx0_matlab))
    print(f"dx0 error = {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_dy0():  # {{{
    M_data = load_anon_test_data()

    a = M_data["a"]
    dn = M_data["dn"]
    dy0_matlab = M_data["dy0_matlab"]

    T_s = time.time()
    dy0_python = dy0(a, dn)
    error = np.max(np.abs(dy0_python - dy0_matlab))
    print(f"dy0 error = {error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    assert error < 1.0e-10


# }}}
def test_index_mask_equivalence(nR=2000, nC=2000):  # {{{
    I = np.random.choice([True, False], (nR, nC))
    a = 11 + np.arange(nR * nC).reshape(nR, nC).astype(np.float64)

    # masked items = zero                    A[I] = 0
    baseline_A = a.copy()
    trial_A = a.copy()
    baseline_A[I] = 0
    T_s = time.time()
    trial_A *= ~I
    print(f"A[I] = 0    elapsed= {time.time() - T_s:.6f}")
    assert np.all(baseline_A - trial_A) == 0

    # masked items = non-zero constant       A[I] = -33
    baseline_A = a.copy()
    trial_A = a.copy()
    baseline_A[I] = -33
    T_s = time.time()
    trial_A = trial_A * (~I) + (-33) * I
    print(f"A[I] = -33  elapsed= {time.time() - T_s:.6f}")
    assert np.all(baseline_A - trial_A) == 0

    # masked items = array
    n_true = np.sum(I)
    RHS = (-20000 + np.arange(n_true)).astype(np.float64)
    baseline_A = a.copy()
    trial_A = a.copy()
    baseline_A[I] = RHS
    T_s = time.time()
    i_true = np.argwhere(I.flatten()).flatten()

    # numba does not support .put()
    # trial_A.put(i_true, RHS)

    # numba gives error on using .itemset() with linear index
    # for i_rhs, i_true in enumerate(np.argwhere(I.flatten()).flatten()):
    #    trial_A.itemset(i_true, RHS[i_rhs])

    for i_rhs, (r, c) in enumerate(zip(i_true // nC, i_true % nC)):
        trial_A[r, c] = RHS[i_rhs]
    print(f"A[I] = RHS       took {time.time() - T_s:.6f}")
    assert np.all(baseline_A - trial_A) == 0

    T_s = time.time()
    trial_A = index_mask_equivalence(a, I, RHS)
    print(f"A[I] = RHS numba took {time.time() - T_s:.6f}")
    assert np.all(baseline_A - trial_A) == 0
    return 3


# }}}
def test_apply_bc_wall():  # {{{
    M_data = load_fn_test_data()

    in_BC = M_data["apply_bc_wall_in"]["BC"].astype(np.uint8)
    in_CVC = M_data["apply_bc_wall_in"]["CVC"].astype(np.bool_)
    in_U = M_data["apply_bc_wall_in"]["U"]
    in_V = M_data["apply_bc_wall_in"]["V"]

    matlab_out_U = M_data["apply_bc_wall_out"]["U"]
    matlab_out_V = M_data["apply_bc_wall_out"]["V"]

    T_s = time.time()
    py_out_U, py_out_V = in_U.copy(), in_V.copy()
    apply_bc_wall(py_out_U, py_out_V, in_BC, in_CVC)
    U_error = np.max(np.abs(py_out_U - matlab_out_U))
    V_error = np.max(np.abs(py_out_V - matlab_out_V))
    print(f"apply_bc_wall U error = {U_error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    print(f"apply_bc_wall V error = {V_error:12.6e}")
    assert U_error < 1.0e-10
    assert V_error < 1.0e-10


# }}}
def test_neumann_bc_for_outflow():  # {{{
    M_data = load_fn_test_data()

    in_U = M_data["neumann_bc_in"]["U"]
    in_V = M_data["neumann_bc_in"]["V"]
    OF = M_data["neumann_bc_in"]["OF"].astype(np.bool_)

    matlab_out_U = M_data["neumann_bc_out"]["U"]
    matlab_out_V = M_data["neumann_bc_out"]["V"]

    T_s = time.time()
    py_out_U, py_out_V = in_U.copy(), in_V.copy()
    neumann_bc_for_outflow(py_out_U, py_out_V, OF)
    U_error = np.max(np.abs(py_out_U - matlab_out_U))
    V_error = np.max(np.abs(py_out_V - matlab_out_V))
    print(f"neumann_bc U error    = {U_error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    print(f"neumann_bc V error    = {V_error:12.6e}")
    assert U_error < 1.0e-10
    assert V_error < 1.0e-10


# }}}
def test_temporary_velocities():  # {{{
    M_data = load_fn_test_data()

    in_U = M_data["temporary_vel_in"]["U"]
    in_V = M_data["temporary_vel_in"]["V"]
    BC = M_data["temporary_vel_in"]["BC"].astype(np.uint8)
    CV = M_data["temporary_vel_in"]["CV"].astype(np.bool_)
    L = M_data["temporary_vel_in"]["L"]
    uc = M_data["temporary_vel_in"]["uc"]
    vc = M_data["temporary_vel_in"]["vc"]
    dt = M_data["temporary_vel_in"]["dt"]
    dn = M_data["temporary_vel_in"]["dn"]

    matlab_out_U = M_data["temporary_vel_out"]["U"]
    matlab_out_V = M_data["temporary_vel_out"]["V"]

    T_s = time.time()
    py_out_U, py_out_V = in_U.copy(), in_V.copy()
    temporary_velocities(py_out_U, py_out_V, BC, CV, L, uc, vc, dt, dn)
    U_error = np.max(np.abs(py_out_U - matlab_out_U))
    V_error = np.max(np.abs(py_out_V - matlab_out_V))
    print(f"temporary_vel U error = {U_error:12.6e}  elapsed= {time.time() - T_s:.6f}")
    print(f"temporary_vel V error = {V_error:12.6e}")
    assert U_error < 1.0e-10
    assert V_error < 1.0e-10


# }}}
def test_fd_solve_poisson_pressure():  # {{{
    M_data = load_fn_test_data()

    in_P0 = M_data["fd_solve_poisson_in"]["P0"]
    in_U = M_data["fd_solve_poisson_in"]["U"]
    in_V = M_data["fd_solve_poisson_in"]["V"]
    CP = M_data["fd_solve_poisson_in"]["CP"]
    OF = M_data["fd_solve_poisson_in"]["OF"]
    OF2 = M_data["fd_solve_poisson_in"]["OF2"]
    CA = M_data["fd_solve_poisson_in"]["CA"]
    T = M_data["fd_solve_poisson_in"]["T"]
    dn = M_data["fd_solve_poisson_in"]["dn"]
    dt = M_data["fd_solve_poisson_in"]["dt"]
    error = M_data["fd_solve_poisson_in"]["error"]
    MAXIT = M_data["fd_solve_poisson_in"]["MAXIT"]
    MINIT = M_data["fd_solve_poisson_in"]["MINIT"]

    matlab_out_P0 = M_data["fd_solve_poisson_out"]["P0"]
    matlab_out_U = M_data["fd_solve_poisson_out"]["U"]
    matlab_out_V = M_data["fd_solve_poisson_out"]["V"]

    py_out_P0 = in_P0.copy()
    py_out_U = in_U.copy()
    py_out_V = in_V.copy()
    T_s = time.time()
    mite = fd_solve_poisson_pressure(
        py_out_P0, py_out_U, py_out_V, T, CP, OF, OF2, CA, dn, dt, error, MAXIT, MINIT
    )
    P0_error = np.max(np.abs(py_out_P0 - matlab_out_P0))
    U_error = np.max(np.abs(py_out_U - matlab_out_U))
    V_error = np.max(np.abs(py_out_V - matlab_out_V))
    print(
        f"fd_solve_poisson_P0 error = {P0_error:12.6e}  elapsed= {time.time() - T_s:.6f}"
    )
    print(f"fd_solve_poisson_U error  = {U_error:12.6e}")
    print(f"fd_solve_poisson_V error  = {V_error:12.6e}")
    assert U_error < 1.0e-10
    assert V_error < 1.0e-10


# }}}


def main():  # {{{

    load_fn_test_data()

    test_dxuu()
    test_dyuv()
    test_dyvv()
    test_dxuv()
    test_DX2()
    test_dx0()
    test_dy0()
    # test_index_mask_equivalence(3000,5000)
    test_index_mask_equivalence(300, 500)

    test_apply_bc_wall()
    test_temporary_velocities()
    test_neumann_bc_for_outflow()
    test_fd_solve_poisson_pressure()


# }}}

if __name__ == "__main__":
    main()
