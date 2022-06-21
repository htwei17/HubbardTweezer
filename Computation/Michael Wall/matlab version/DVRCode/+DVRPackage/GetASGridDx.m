function [xvals,Nx] = GetASGridDx(ax,dx)
%Returns a grid of (2N+1) equally spaced values between -ax and ax, where N=ax/dx, rounded up.  Used for non-parity-symmetric problems.
Nx=ceil(ax/dx);
xvals=dx*[-Nx:Nx];
end
