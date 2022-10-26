# Mandelbrot set

### ``MB_main.m``
The plain MATLAB implementation.

### ``MB.py``
The plain Python implementation.  It is pretty slow
compared to the plain MATLAB version.

### ``MB_numba.py``
The Python version enhanced by Numba.  It is crazy fast.

### ``MB_python_numba.m``
A MATLAB main program calling the Python+Numba version.
It is as fast as ``MB_numba.py``.
Include [``py2mat.m``](https://github.com/Apress/python-for-matlab-development/code/matlab_py/py2mat.m)
in your path.

### ``MB_view.m`` and ``MB_view.py``
MATLAB and Python sample codes that display an image
of the 2000 x 2000 pixel Mandelbrot set computed
by the above programs.
