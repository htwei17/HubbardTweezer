function y = ApplyDVR3D(Tx, Ty, Tz, V, x)
    %Apply the sparse 3D DVR Hamiltonian to a state x in vectorized format and return in state y
    %This routine is used as the matrix-vector multiply operation within eigs
    %

    Nx = size(Tx, 1); Ny = size(Ty, 1); Nz = size(Tz, 1);
    %Unpack state from vectorized to tensor format
    xr = reshape(x, [Nx, Ny, Nz]);
    %Apply (diagonal) V operator
    y = V .* xr;
    %Kinetic energy operators, e.g., T_x, are sparse in the sense that they have the form T_x\otimes I_y\otimes I_z
    %This means that they can be applied in O(N_x^2 N_y N_z) time
    %Here, we account for this sparsity by reshaping the 3-tensor \psi into a matrix and using fast matrix-matrix multiplies
    %Tx
    y = y + reshape(Tx * reshape(xr, [Nx, Ny * Nz]), [Nx, Ny, Nz]);
    %Ty
    y = y + permute(reshape(Ty * reshape(permute(xr, [2, 1, 3]), Ny, Nx * Nz), Ny, Nx, Nz), [2, 1, 3]);
    %Tz
    y = y + permute(reshape(Tz * reshape(permute(xr, [3, 1, 2]), Nz, Nx * Ny), Nz, Nx, Ny), [2, 3, 1]);
    %put output state in vectorized format
    y = reshape(y, [Nx * Ny * Nz, 1]);
end
