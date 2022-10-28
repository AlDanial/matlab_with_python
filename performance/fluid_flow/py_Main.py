#!/usr/bin/env python
# Al Danial, October 2022

# https://github.com/AlDanial/matlab_with_python/
# Python translation by Al Danial of MATLAB code by
# Jamie Johns.  The original MATLAB code is at
# https://github.com/JamieMJohns/Navier-stokes-2D-numerical-solve-incompressible-flow-with-custom-scenarios-MATLAB-
# The Python version is covered by the MIT license.

import numpy as np
import time

T_start = time.time()
import os
from scipy.io import loadmat, savemat
from navier_stokes_functions import (
    dx0,
    dy0,
    apply_bc_wall,
    neumann_bc_for_outflow,
    temporary_velocities,
    fd_solve_poisson_pressure,
    solve_flow,
    VELOCITY_OUT_DIR,
)

def from_imagegrab():  # {{{
    F = "imagegrab_results.mat-50x200"
    F = "imagegrab_maxell_BC.mat"
    m = loadmat(F, squeeze_me=True)
    return (
        m["velinit"],
        m["bounds"],
        m["outflow"],
        m["BLUE"],
        int(m["XI"]),
        int(m["YI"]),
    )


# }}}
def saveall(BC, CA, CP, CV, CVC, OF, OF2, P0, U, uc, V, vc, velxi, velyi):  # {{{
    msave = {
        "BC": BC,
        "CA": CA,
        "CP": CP,
        "CV": CV,
        "CVC": CVC,
        "OF": OF,
        "OF2": OF2,
        "P0": P0,
        "U": U,
        "uc": uc,
        "V": V,
        "vc": vc,
        "velxi": velxi,
        "velyi": velyi,
    }
    savemat("dump_py.mat", msave)


# }}}

def main():
    # Parameters for scenario (Modify these)
    # set information about domain of simulation
    SCENARIO = "scenario_sphere2.png"  # <--- file (image) that is scenario for navier stokes fluid simulation

    domainX = 2  # length of domain (x-axis) [unit: meters]
    domainX = 15  # length of domain (x-axis) [unit: meters]

    xinc = 200  # number of nodes across x-component of domain (number of nodes from x=0 to x=domainX); where dx=domainX/xinc (=dy=dn)
    xinc = 400  # number of nodes across x-component of domain (number of nodes from x=0 to x=domainX); where dx=domainX/xinc (=dy=dn)

    # Note xinc needed by matlab for imagegrab(), but not used here
    dt = 0.0030  # set set delta time [unit: seconds]

    MI = 12900  # number of time steps to perform calculations [time(at time step)=MI*dt]
    )
    velyi = 0  # y-component velocity of region with constant velocity (regions coloured red in scenario image)  [unit: meters/second]
    # [velyi>0,velocity has vector -y with mag abs(velyi) and velyi<0, vel has vector +y with mag of abs(velyi)]
    velxi = 0.45  # x-component velocity of region with constant velocity (regions coloured red in SCENARIO)   [unit: meters/second]
    # [velxi>0,velocity has vector +x with mag abs(velxi) and velxi<0, vel has vector -x with mag of abs(velxi)]
    dens = 1  # density  [unit: kg/m^3] , water(pure)=1000 blood=1025 air~1
    mu = 1 / 1000  # dynamic viscosity [kg/(m*s)]

    # Poisson Pressure solver parameters
    error_tol = 0.001  # set tolerance of error for convergence poisson solver of Pressure field (good value to start with is 0.001; which is for most incompressible applications)
    MAXIT = 1000  # maximum number of iterations allowed for poisson solver (increasing this will allow for further convergence of p solver)
    MINIT = 1  # mininum number of iterations allowed for poisson solver (increasing this will allow for further convergence of p solver)
    #  Note that: MINIT should be less than MAXIT

    # save parameters
    ST = [100, 100, 500]  # FOR variables of dimensions of ST(1)xST(2)
    #  save variable data for x and y component velocities
    #  in chuncks of files , each with ST(1)xST(2)xST(3)
    #  size matrix.........this reduces memory as only one file is openend at a time.
    # (increasing ST(3) will reduce number of externally saved files [still using same amount of space)
    # (decreasing ST(3) will reduce number of externally saved files [still using same amount of space)
    # [Files; openvar.m and savevar.m are used]

    # [velinit, bounds, outflow, BLUE, XI, YI]=imagegrab(SCENARIO,xinc);
    velinit, bounds, outflow, BLUE, XI, YI = from_imagegrab()

    print(f"Start-up time penalty = {time.time() - T_start:.3} seconds")
    solve_flow(
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
        error_tol,
        MINIT,
        MAXIT,
    )

if __name__ == "__main__":
    main()
