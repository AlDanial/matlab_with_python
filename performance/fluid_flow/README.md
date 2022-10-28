# 2D Navier-Stokes solver for incompressible fluid flow

MATLAB, Python, and MATLAB calling Python implementations of code
originally written by 
[Jamie Johns](https://github.com/JamieMJohns/Navier-stokes-2D-numerical-solve-incompressible-flow-with-custom-scenarios-MATLAB-)
to compute 2D incompressible flows.
This example is built on techniques described in Chapter 14 of
[Python for MATLAB Development](https://github.com/Apress/python-for-matlab-development), Apress, 2022.
See [my article describing details of this example](https://al.danial.org/posts/accelerate_matlab_with_python_and_numba/).

### ``Main.m``
The MATLAB and MATLAB-calling-Python implementation.
To switch between modes, edit the file and set the value of 
`Run_Python_Solver` to `false` (pure MATLAB) or `true`
(MATLAB-calling-Python) as desired.

### ``imagegrab.m``
Original code by Jamie Johns that reads an image file (such
as a ``.png``) and creates boundary condition information
based on pixel colors.

### ``storevar.m``
Original code by Jamie Johns to save a batch of results
to a ``.mat`` file.

### ``animate_navier_stokes.py``
A Python program that can animate or store PNG images of the
flow field produced by earlier runs of ``Main.m`` or ``py_Main.py``.
