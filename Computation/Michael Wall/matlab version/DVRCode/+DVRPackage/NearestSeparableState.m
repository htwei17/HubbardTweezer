function [X, Y, Z] = NearestSeparableState(psi)
    %Find the nearest separable state to psi in the 2-norm, i.e. solve the minimization problem \min_{X,Y,Z} |\psi(x,y,z)-X(x)Y(y)Z(z)|^2
    %This routine first does so by converting the state to a matrix product state to provide a good initial guess.

    SVDtol = 1e-10; %tolerance of 2-norm discarded weight per Schmidt decomposition
    %convert state to an MPS
    Nxsiz = size(psi, 1); Nysiz = size(psi, 2); Nzsiz = size(psi, 3);
    %Split psi[x,y,z]-> \sum_{\nu} U_{\nu}([x,y]) Z_{\nu}(z)
    A = zeros(Nxsiz * Nysiz, Nzsiz);
    %maybe can be vectorized with a reshape...fast enough for now
    for ix = 1:Nxsiz

        for iy = 1:Nysiz
            A((ix - 1) * Nysiz + iy, :) = psi(ix, iy, :);
        end

    end

    [U, S, V] = svd(A, 'econ');
    V = V';
    S = diag(S);
    Snrm = norm(S);
    S = S / Snrm; %normalize state for easy renormalization (remove factors of dx, dy,dz, but save norm for later)

    %truncate number of singular values to meet desired tolerance
    Scum = 0;
    nS = size(S, 1);

    for chiyz = 1:nS
        Scum = Scum + S(chiyz)^2;

        if 1.0 - Scum <= SVDtol
            break
        end

    end

    chiyz = min([chiyz, nS]);
    %Renormalize
    S = S / norm(S(1:chiyz));

    vnyz = 0.0;

    for iz = 1:chiyz
        vnyz = vnyz - (S(iz)^2) * log(S(iz)^2);
    end

    Z = zeros(chiyz, Nzsiz);
    Z = V([1:chiyz], [1:Nzsiz]);

    %Split U_{\nu} ([xy])->\sum_{\mu} X_{\mu}(x) Y_{mu,nu}(y)
    A = zeros(Nxsiz, Nysiz * chiyz);

    for ix = 1:Nxsiz

        for iy = 1:Nysiz

            for iz = 1:chiyz
                A(ix, (iy - 1) * chiyz + iz) = U((ix - 1) * Nysiz + iy, iz) * S(iz); %factor of S makes this a unit-normalized "density matrix" with z traced out
            end

        end

    end

    [U, S, V] = svd(A, 'econ');
    S = diag(S);
    V = V';
    %truncate number of singular values to meet desired tolerance
    Scum = 0.0;
    nS = size(S, 1);

    for chixy = 1:nS
        Scum = Scum + S(chixy)^2;

        if 1.0 - Scum <= SVDtol
            break
        end

    end

    chixy = min([chixy, nS]);
    S = S / norm(S(1:chixy));
    vnxy = 0.0;

    for ix = 1:chixy
        vnxy = vnxy - (S(ix)^2) * log(S(ix)^2);
    end

    X = zeros(Nxsiz, chixy);
    Y = zeros(chixy, Nysiz, chiyz);

    for ix = 1:chixy
        X(:, ix) = U(:, ix);

        for iz = 1:chiyz

            for iy = 1:Nysiz
                Y(ix, iy, iz) = Snrm * S(ix) * V(ix, (iy - 1) * chiyz + iz);
            end

        end

    end

    %Final output is an MPS \psi_{MPS}(x,y,z)=\sum_{\mu \nu} X_{\mu}(x) Y_{\mu \nu}(y)Z_{\nu} (z)

    %set if(1) to see the distance between this MPS representation and the actual non-separable state
    if (0)
        dist = 0.0; d1 = 0.0; d2 = 0.0;

        for ix = 1:Nxsiz

            for iy = 1:Nysiz

                for iz = 1:Nzsiz
                    tmp = 0.0;

                    for mu = 1:chixy

                        for nu = 1:chiyz
                            tmp = tmp + X(ix, mu) * Y(mu, iy, nu) * Z(nu, iz);
                        end

                    end

                    d1 = d1 + (psi(ix, iy, iz))^2;
                    d2 = d2 + (tmp)^2;
                    dist = dist + (psi(ix, iy, iz) - tmp)^2;
                end

            end

        end

        disp('distance')
        dist
        d1
        d2
    end

    %round-robin optimization scheme to find nearest separable state from MPS
    %Initialize states with zeroth-order "Schmidt" guess
    Xt = X(:, 1); Xt = Xt / norm(Xt);
    Yt = Y(1, :, 1); Yt = Yt / norm(Yt);
    Zt = Z(1, :); Zt = Zt / norm(Zt);

    %overlaps of the actual state with current guess
    xov = zeros(1, chixy); yov = zeros(chixy, chiyz); zov = zeros(chiyz, 1);

    for mu = 1:chixy
        xov(mu) = dot(Xt, X(:, mu));

        for nu = 1:chiyz
            yov(mu, nu) = dot(Y(mu, :, nu), Yt);
        end

    end

    for nu = 1:chiyz
        zov(nu) = dot(Z(nu, :), Zt);
    end

    prnm = norm(reshape(psi, [1, size(psi, 1) * size(psi, 2) * size(psi, 3)]));
    overlap = sum(xov * (yov * zov)) / (prnm * norm(Xt) * norm(Yt) * norm(Zt));

    nsweeps = 10; %arbitrary-just set to a constant value in lieu of an exit tolerance.
    xnrm = norm(Xt); ynrm = norm(Yt); znrm = norm(Zt);

    for sweep = 1:nsweeps
        %X update
        Xt = 0.0; xtmp = yov * zov;

        for mu = 1:chixy
            Xt = Xt + xtmp(mu) * X(:, mu);
        end

        xnrm = norm(Xt); Xt = Xt / xnrm; xnrm = norm(Xt);

        for mu = 1:chixy
            xov(mu) = dot(X(:, mu), Xt);
        end

        %Y update
        for mu = 1:chixy
            ytmp(mu, :) = xov(mu) * zov;
        end

        Yt = 0.0;

        for mu = 1:chixy

            for nu = 1:chiyz
                Yt = Yt + ytmp(mu, nu) * Y(mu, :, nu);
            end

        end

        ynrm = norm(Yt); Yt = Yt / ynrm; ynrm = norm(Yt);

        for mu = 1:chixy

            for nu = 1:chiyz
                yov(mu, nu) = dot(Y(mu, :, nu), Yt);
            end

        end

        %Z update
        Zt = 0.0; ztmp = xov * yov;

        for nu = 1:chiyz
            Zt = Zt + ztmp(nu) * Z(nu, :);
        end

        znrm = norm(Zt); Zt = Zt / znrm; znrm = norm(Zt);

        for nu = 1:chiyz
            zov(nu) = dot(Z(nu, :), Zt);
        end

        %Y update
        for mu = 1:chixy
            ytmp(mu, :) = xov(mu) * zov;
        end

        Yt = 0.0;

        for mu = 1:chixy

            for nu = 1:chiyz
                Yt = Yt + ytmp(mu, nu) * Y(mu, :, nu);
            end

        end

        ynrm = norm(Yt); Yt = Yt / ynrm; ynrm = norm(Yt);

        for mu = 1:chixy

            for nu = 1:chiyz
                yov(mu, nu) = dot(Y(mu, :, nu), Yt);
            end

        end

        overlap = sum(xov * (yov * zov)) / (prnm * norm(Xt) * norm(Yt) * norm(Zt));
    end

    %Note that X and Z are unit-normalized vectors satisfying, e.g., dot(X,X)=1
    %In order to be unit-normalized wavefunctions in the DVR basis, need to divide through by, e.g., sqrt(dx)
    X = Xt; Y = Yt; Z = Zt;
    overlap

end
