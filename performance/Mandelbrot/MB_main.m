% file: MB_main.m
main()

function [i]=nIter(c, imax)
  z = complex(0,0);
  for i = 0:imax-1
    z = z^2 + c;
    if (real(z)^2 + imag(z)^2) > 4
        break
    end
  end
end

function [img]=MB(Re,Im,imax)
  nR = size(Im,2);
  nC = size(Re,2);
  img = zeros(nR, nC, 'uint8');
% parfor i = 1:nR % gives worse performance
  for i = 1:nR
    for j = 1:nC
      c = complex(Re(j),Im(i));
      img(i,j) = nIter(c,imax);
    end
  end
end

function [] = main()
  imax = 255;
  for N = [500 1000 2000 5000]
    tic
    nR = N; nC = N;
    Re = linspace(-0.7440, -0.7433, nC);
    Im = linspace( 0.1315,  0.1322, nR);
    img = MB(Re, Im, imax);
    fprintf('%5d %.3f\n',N,toc);
  end
end
