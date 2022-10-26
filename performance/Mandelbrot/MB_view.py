#!/usr/bin/env python3
# file: MB_view.py
import numpy as np
from MB_numba import MB
import matplotlib.pyplot as plt

imax = 255
nR, nC = 2000, 2000
Re = np.linspace(-0.7440, -0.7433, nC)
Im = np.linspace( 0.1315,  0.1322, nR)
img = MB(Re, Im, imax)
plt.imshow(img)
plt.show()
