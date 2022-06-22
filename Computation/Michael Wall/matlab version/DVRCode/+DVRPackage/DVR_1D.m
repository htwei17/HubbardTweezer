function [vecs, vals] = DVR_1D(vals, BareScal, par, Nbands, potential)
    %Set up and diagonalize the DVR representation of a 1D problem with potential given by potential(x).
    %vals is a vector of gridpoints, BareScal is the coefficient of the (d^2/dx^2) operator, and Nbands are the number of eigenstates desired.
    %This routine uses parity symmetry, with par='p' being even parity and par='m' being odd parity.

    %Note that N in this routine is equivalent to using the non-symmetric routines with N=2*N+1
    N = length(vals) - 1;
    dq = vals(2) - vals(1);
    scal = BareScal / dq^2;

    T = DVRPackage.SetupT(N, scal, par);

    if (par == 'p')
        Np = N + 1;
        pvec = potential(vals);
        V = diag(pvec);
    else
        Np = N;
        pvec = potential(vals(2:end));
        V = diag(pvec);
    end

    H = T + V;
    [U, D] = eig(H);
    vecs = zeros(Nbands, Np);
    vals = diag(D(1:Nbands, 1:Nbands));
    %Set overall (real) phase consistent with harmonic oscillator convention
    % that \psi_n(0^+)>0 for n=0,1,4,5,8,9,... and \psi_n(0^+)<0 for n=2,3,6,7, [n accounting for parity symmetry]
    for i = 1:Nbands
        val = mod(i - 1, 2);
        phs = (-1.0)^val;

        if (phs * U(1, i) > 0)
            vecs(i, :) = U(:, i);
        else
            vecs(i, :) = -U(:, i);
        end

    end

end
