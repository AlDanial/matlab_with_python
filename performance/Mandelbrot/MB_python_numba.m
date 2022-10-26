% file: MB_python_numba.m
np = py.importlib.import_module('numpy');
MB_numba = py.importlib.import_module('MB_numba');
imax = int64(255);
for N = [500 1000 2000 5000]
    tic
    nR = N; nC = N;
    Re = np.array(linspace(-0.7440, -0.7433, nC));
    Im = np.array(linspace( 0.1315,  0.1322, nR));
    img = py2mat(MB_numba.MB(Re, Im, imax));
    fprintf('%5d %.3f\n',N,toc);
end
