function [outvecs, S] = LocalizeState(psis, Nx, Ny, Nz, dx, dy, dz)
    %Compute the eigenvectors of the localization matrix in order to define maximally localized Wannier functions.
    %psis is a 1XN cell of unpacked 3D state tensors whose elements are the states to be localized.

    %Get matrix of sinc overlaps for NormOnLeft
    %This matrix has elements I_{i,j}=\int_{-\infty}^0 dz \sinc(\pi(z-i))\sinc(\pi(z-j))
    %with \sinc(x)=\sin(x)/x.  It can be shown that the diagonal elements are
    %I_{i,i}=(1/2)-Si(2\pi i) and the upper triangle (note I is symmetric) is
    %I_{i,i+p}=(-1)^p (Cin(2\pi (i+p))-Cin(2\pi i))/(2\pi^2 p), where Cin(x)=\gamma+\log x-Ci(x).
    sincover = diag(0.5 - sinint(2.0 * pi * ([1:2 * Nx + 1] - (Nx + 1))) / pi);
    Cvec = zeros([2 * Nx + 1, 1]);

    for i = 1:2 * Nx + 1
        Cvec(i) = DVRPackage.Cin(2.0 * pi * (i - (Nx + 1)));
    end

    for i = 1:2 * Nx + 1

        for p = 1:2 * Nx + 1 - i
            sincover(i, i + p) = ((-1.0)^p) * (Cvec(i + p) - Cvec(i)) / (2.0 * pi * pi * p);
            sincover(i + p, i) = sincover(i, i + p);
        end

    end

    Nstates = size(psis, 2);
    %Form the localization matrix of overlaps G_{i,j}=\mathcal{L}(\psi_i,\psi_j) with \mathcal{L} the localization functional
    GramMat = zeros(Nstates);

    for i = 1:Nstates
        GramMat(i, i) = DVRPackage.NormOnLeft(psis{i}, psis{i}, Nx, Ny, Nz, dx, dy, dz, sincover);

        for j = i + 1:Nstates
            GramMat(i, j) = DVRPackage.NormOnLeft(psis{i}, psis{j}, Nx, Ny, Nz, dx, dy, dz, sincover);
            GramMat(j, i) = GramMat(i, j);
        end

    end

    %The eigenvectors of the localization matrix are expansion coefficients of the optimal Wannier functions
    %in the basis of psis, and the eigenvalues are the norm of these Wannier functions on the left half.
    [U, D] = eig(GramMat);
    S = diag(D);
    outvecs = U'; %don't impose any other constraints on eigenvectors...
end
