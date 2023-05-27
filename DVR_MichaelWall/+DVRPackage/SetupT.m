function T = SetupT(N, Scal, par)
    %Returns the DVR second-derivative matrix scaled by the parameter Scal (=\hbar^2/(2m*lengthscale**2)).  par='p','m' specifies even or odd parity.

    pisqo3 = (pi^2) / 3.0;
    sqrtTwo = sqrt(2.0);

    if par == 'p'
        Np = N + 1;
        T = zeros(Np);
        T(1, 1) = pisqo3;

        for i = 1:N
            T(i + 1, i + 1) = pisqo3 + 1.0 / (2.0 * i^2);
        end

        for r = 1:N
            T(1, r + 1) = 2.0 * sqrtTwo * ((-1.0)^(r)) / (1.0 * (r^2));
            T(r + 1, 1) = T(1, r + 1);
        end

        for r = 1:N - 1
            lphase = (-1.0)^r;
            invr = 2.0 / (1.0 * (r^2));

            for i = 1:N - r
                invl = 2.0 / (1.0 * ((2 * i + r)^2));
                pval = lphase * invr + lphase * invl;
                T(i + 1, i + 1 + r) = pval;
                T(i + 1 + r, i + 1) = pval;
            end

        end

    else
        Np = N;
        T = zeros(Np);

        for i = 1:N
            T(i, i) = pisqo3 - 1.0 / (2.0 * i^2);
        end

        %Upper triangle
        for r = 1:N - 1
            lphase = (-1.0)^r;
            invr = 2.0 / (1.0 * (r^2));

            for i = 1:N - r
                invl = 2.0 / (1.0 * ((2 * i + r)^2));
                mval = lphase * 2.0 * invr * invl * i * (i + r);
                T(i, i + r) = mval;
                T(i + r, i) = mval;
            end

        end

    end

    T = Scal * T;
end
