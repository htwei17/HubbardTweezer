function xvals = GetGrid(ax, Nx)
    %Returns a grid of (N+1) equally spaced values between 0 and ax.  Used for parity symmetric problems.
    dx = ax / (Nx * 1.0);
    xvals = dx * [0:Nx];
end
