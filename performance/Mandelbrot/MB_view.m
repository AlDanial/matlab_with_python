% file: MB_view.m
np = py.importlib.import_module('numpy');
MB_numba = py.importlib.import_module('MB_numba');
imax = int64(255);
nR = 2000; nC = 2000;
Re = np.array(linspace(-0.7440, -0.7433, nC));
Im = np.array(linspace( 0.1315,  0.1322, nR));
img = py2mat(MB_numba.MB(Re, Im, imax));
imagesc(img)
