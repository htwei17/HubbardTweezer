function [xvals, Nx] = GetGridDx(ax, dx)
    %Returns a grid of (N+1) equally spaced values between 0 and ax, where N=ax/dx, rounded up.  Used for parity-symmetric problems.
    Nx = ceil(ax / dx);
    xvals = dx * [0:Nx];
end
