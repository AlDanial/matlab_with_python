#!/usr/bin/env python3
# file: MB_numba.py
import numpy as np
import time
from numba import jit, prange, uint8, int64, float64, complex128
@jit(uint8(complex128,int64), nopython=True, fastmath=True)
def nIter(c, imax):
  z = complex(0, 0)
  for i in range(imax):
    z = z*z + c
    if z.real*z.real + z.imag*z.imag > 4:
      break
  return i

@jit(uint8[:,:](float64[:], float64[:],int64), nopython=True,
                fastmath=True, parallel=True)
def MB(Re, Im, imax):
  nR = len(Im)
  nC = len(Re)
  img = np.zeros((nR, nC), dtype=np.uint8)
  for i in prange(nR):
    for j in range(nC):
      c = complex(Re[j], Im[i])
      img[i,j] = nIter(c,imax)
  return img

def main():
  imax = 255
  for N in [500, 1000, 2000, 5000]:
    T_s = time.time()
    nR, nC = N, N
    Re = np.linspace(-0.7440, -0.7433, nC)
    Im = np.linspace( 0.1315,  0.1322, nR)
    img = MB(Re, Im, imax)
    print(N, time.time() - T_s)

if __name__ == '__main__': main()
