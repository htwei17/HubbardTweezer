function unpackedpsi = Unpack3DState(psi, Nx, Ny, Nz, xpar, ypar, zpar)
    %Converts a state in the parity-adpated representation into the non-parity adapted representation
    xweight = 1.0 * [1:2 * Nx + 1]; xind = 1 * [1:2 * Nx + 1];
    yweight = 1.0 * [1:2 * Ny + 1]; yind = 1 * [1:2 * Ny + 1];
    zweight = 1.0 * [1:2 * Nz + 1]; zind = 1 * [1:2 * Nz + 1];

    if (xpar == 'p')
        Nxp = Nx + 1;
        xweight(Nx + 1) = 1.0;
        xind(Nx + 1) = 1;

        for j = 1:Nx
            xind(Nx + j + 1) = j + 1;
            xweight(Nx + j + 1) = 1.0 / sqrt(2.0);
            xind(Nx + 1 - j) = j + 1;
            xweight(Nx + 1 - j) = 1.0 / sqrt(2.0);
        end

    else
        Nxp = Nx;
        xweight(Nx + 1) = 0.0;
        xind(Nx + 1) = 1;

        for j = 1:Nx
            xind(Nx + j + 1) = j;
            xweight(Nx + j + 1) = 1.0 / sqrt(2.0);
            xind(Nx + 1 - j) = j;
            xweight(Nx + 1 - j) = -1.0 / sqrt(2.0);
        end

    end

    if (ypar == 'p')
        Nyp = Ny + 1;
        yweight(Ny + 1) = 1.0;
        yind(Ny + 1) = 1;

        for j = 1:Ny
            yind(Ny + j + 1) = j + 1;
            yweight(Ny + j + 1) = 1.0 / sqrt(2.0);
            yind(Ny + 1 - j) = j + 1;
            yweight(Ny + 1 - j) = 1.0 / sqrt(2.0);
        end

    else
        Nyp = Ny;
        yweight(Ny + 1) = 0.0;
        yind(Ny + 1) = 1;

        for j = 1:Ny
            yind(Ny + j + 1) = j;
            yweight(Ny + j + 1) = 1.0 / sqrt(2.0);
            yind(Ny + 1 - j) = j;
            yweight(Ny + 1 - j) = -1.0 / sqrt(2.0);
        end

    end

    if (zpar == 'p')
        Nzp = Nz + 1;
        zweight(Nz + 1) = 1.0;
        zind(Nz + 1) = 1;

        for j = 1:Nz
            zind(Nz + j + 1) = j + 1;
            zweight(Nz + j + 1) = 1.0 / sqrt(2.0);
            zind(Nz + 1 - j) = j + 1;
            zweight(Nz + 1 - j) = 1.0 / sqrt(2.0);
        end

    else
        Nzp = Nz;
        zweight(Nz + 1) = 0.0;
        zind(Nz + 1) = 1;

        for j = 1:Nz
            zind(Nz + j + 1) = j;
            zweight(Nz + j + 1) = 1.0 / sqrt(2.0);
            zind(Nz + 1 - j) = j;
            zweight(Nz + 1 - j) = -1.0 / sqrt(2.0);
        end

    end

    ptmp = reshape(psi, [Nxp, Nyp, Nzp]);
    unpackedpsi = zeros(2 * Nx + 1, 2 * Ny + 1, 2 * Nz + 1);

    for i = 1:2 * Nx + 1

        for j = 1:2 * Ny + 1

            for k = 1:2 * Nz + 1
                unpackedpsi(i, j, k) = xweight(i) * yweight(j) * zweight(k) * ptmp(xind(i), yind(j), zind(k));
            end

        end

    end

end
