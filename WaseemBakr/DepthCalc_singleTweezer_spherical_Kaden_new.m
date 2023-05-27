%This code computes the energy levels for a single tweezer.
tic;
% close all
%Fundamental constants
% constants;
% c = 3 * 1e8;
h = 6.63 * 1e-34;
hbar = h / (2 * pi);
kb = 1.38 * 1e-23;

% Lithium-6
mLi = 6 * 1.67 * 1e-27;
gamma = 2 * pi * 6 * 1e6;
lambda0 = 671 * 1e-9;
c = 3e8;
f0 = c / lambda0;
omega0 = 2 * pi * f0;

% Laser Params
lambda = 780 * 1e-9; %wavelength (m)
omega = 2 * pi * c / lambda;
k = 2 * pi / lambda;
Er = hbar^2 * k^2 / (2 * mLi);

P = .5 * 55 * 1e-6; %power per tweezer (W)
asymm = 0; %asymmetry in units of depth
wx = 1000 * 1e-9; %waist perp direction (m) 1350
wy = wx; %waist tunneling direction (m) 1000% 810e-9

for wi = 1:length(wx)
    zRx = pi * wx(wi)^2 / lambda;
    zRy = pi * wy(wi)^2 / lambda;
    %
    I0 = 2 * P / (pi * wx(wi) * wy(wi));
    V0 = 3 * pi * c^2 / (2 * omega0^3) * (gamma / (omega0 - omega) + gamma / (omega0 + omega)) * I0;

    % Tweezer Params
    Mx = 2 * wx(wi);
    My = 2 * wy(wi);
    Mz = 4 * max(zRx, zRy);
    Nx = 15;
    Ny = 15;
    Nz = 31;

    V_tweezers = @(x, y, z) -V0 ./ (1 + z.^2 .* (1 ./ zRx.^2 + 1 ./ zRy.^2) / 2) .* exp(-2 * x.^2 ./ (wx(wi).^2 .* (1 + z.^2 ./ zRx.^2))) ...
        .* exp(-2 * y.^2 ./ (wy(wi).^2 .* (1 + z.^2 ./ zRy.^2)));

    % Gravity
    g = 9.80665;
    V_gravity = @(x, y, z) mLi * g * z;
    % B Gradient, 1.4MHz/G is bohr magneton, state |1> has -1/2, |3> has -1 at
    % low field.  At high field they're equal.
    % 20G/cm is 20G/0.01m = 2000/m
    gradGaussM = 0;
    V_gradient = @(x, y, z) 1.4e6 * h * gradGaussM * z;

    V = @(x, y, z) V_tweezers(x, y, z) + V_gravity(x, y, z);

    % Discretization params
    x = linspace(-Mx / 2, Mx / 2, Nx);
    y = linspace(-My / 2, My / 2, Ny);
    z = linspace(-Mz / 2, Mz / 2, Nz);
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    dz = z(2) - z(1);
    [xx, yy, zz] = meshgrid(x, y, z);
    fprintf('Initialized parameters. Elapsed Time %.3f s\n', toc)

    %get potential
    VV = V(xx, yy, zz);
    VV = spdiags(VV(:), 0, Nx * Ny * Nz, Nx * Ny * Nz);
    fprintf('Set up potential. Elapsed Time %.3f s\n', toc)

    %get kinetic DVR term
    TTx = spdiags(ones(numel(xx), 1) * pi^2/3, 0, Nx * Ny * Nz, Nx * Ny * Nz);

    for Dx = 1:((max(x) - min(x)) / dx)
        M = [ones(1, (Nx - Dx) * Ny), zeros(1, Ny * Dx)];
        M = repmat(M, 1, Nz)';
        TTx = spdiags((-1)^Dx * 2 / Dx^2 * M(end:-1:1), +Dx * Ny, TTx);
        TTx = spdiags((-1)^Dx * 2 / Dx^2 * M, -Dx * Ny, TTx);
    end

    TTy = spdiags(ones(numel(yy), 1) * pi^2/3, 0, Nx * Ny * Nz, Nx * Ny * Nz);

    for Dy = 1:((max(y) - min(y)) / dy)
        M = [ones(1, Ny - Dy), zeros(1, Dy)];
        M = repmat(M, 1, Nx * Nz)';
        TTy = spdiags((-1)^Dy * 2 / Dy^2 * M(end:-1:1), +Dy, TTy);
        TTy = spdiags((-1)^Dy * 2 / Dy^2 * M, -Dy, TTy);
    end

    TTz = spdiags(ones(numel(zz), 1) * pi^2/3, 0, Nx * Ny * Nz, Nx * Ny * Nz);

    for Dz = 1:((max(z) - min(z)) / dz)
        M = ones(1, Nx * Ny * Nz)';
        TTz = spdiags((-1)^Dz * 2 / Dz^2 * M, +Dz * Nx * Ny, TTz);
        TTz = spdiags((-1)^Dz * 2 / Dz^2 * M, -Dz * Nx * Ny, TTz);
    end

    TT = (hbar^2/2 / mLi * (TTx / dx^2 + TTy / dy^2 + TTz / dz^2));
    % TT = (TTx + TTy + TTz);
    fprintf('Kinetic Energy Term. Elapsed Time %.3f s\n', toc)

    %diagonalize
    NN = 16;
    H = TT + VV - spdiags(ones(numel(xx), 1) * min(VV(:)), 0, Nx * Ny * Nz, Nx * Ny * Nz);
    [eigvects, eigenergies] = eigs(H, NN, 'SM');
    fprintf('Diagonalize. Elapsed Time %.3f s\n', toc)

    % extract energies and sort
    E = diag(eigenergies + min(VV(:))) / h * 1e-3;
    [E, II] = sort(E);
    PSI = eigvects(:, II);

    % rearange psi and normalize
    psi = zeros(Nx, Ny, Nz, NN);

    for ii = 1:NN
        psi_temp = reshape(PSI(:, ii), Ny, Nx, Nz);
        Norm = sum(psi_temp(:).^2) * dx * dy * dz;
        PSI(:, ii) = PSI(:, ii) / sqrt(Norm);
        psi_temp = psi_temp / sqrt(Norm) * sign(max(psi_temp(:)));
        psi(:, :, :, ii) = permute(psi_temp, [2 1 3]);
    end

    % close all
    figure(45); clf;
    colors = lines(NN);
    subplot(1, 3, 1)
    hold on
    plot(x * 1e6, V(x, 0, 0) / h * 1e-3, 'k')
    % plot(x*1e6, V(x, a/2, 0)/h*1e-3, 'k')
    for ii = 1:NN
        plot([min(x), max(x)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
    end

    xlabel('X (\mum)')
    ylabel('Potential (kHz)')
    title('X axis')
    grid on
    xlim([min(x), max(x)] * 1e6)
    ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
    subplot(1, 3, 2)
    hold on
    plot(y * 1e6, V(0, y, 0) / h * 1e-3, 'k')

    for ii = 1:NN
        plot([min(y), max(y)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
    end

    xlabel('Y (\mum)')
    title('Y axis')
    grid on
    xlim([min(y), max(y)] * 1e6)
    ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
    subplot(1, 3, 3)
    hold on
    plot(z * 1e6, V(0, 0, z) / h * 1e-3, 'k')
    % plot(z*1e6, V(0, a/2, z)/h*1e-3, 'k')
    for ii = 1:NN
        plot([min(z), max(z)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
    end

    xlabel('Z (\mum)')
    title('Z axis')
    grid on
    xlim([min(z), max(z)] * 1e6)
    ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
    sgtitle(sprintf('Potential and bound state energies\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW', wx(wi) * 1e9, wy(wi) * 1e9, P * 1e3))

    figure(8); clf;

    for ii = 1:NN
        subplot(NN, 3, 3 * ii - 2)

        if ii == 1
            title('X axis')
        end

        ylabel(sprintf('Psi_%d', ii))

        if ii == NN
            xlabel('X (\mum)')
        end

        hold on

        for jj = 1:round(Ny / 5):Ny

            for kk = 1:round(Nz / 5):Nz
                plot(x * 1e6, squeeze(psi(:, jj, kk, ii)))
            end

        end

        xlim([min(x), max(x)] * 1e6)
        grid on
        subplot(NN, 3, 3 * ii - 1)

        if ii == 1
            title('Y axis')
        end

        if ii == NN
            xlabel('Y (\mum)')
        end

        hold on

        for jj = 1:round(Nx / 5):Nx

            for kk = 1:round(Nz / 5):Nz
                plot(y * 1e6, squeeze(psi(jj, :, kk, ii)))
            end

        end

        xlim([min(y), max(y)] * 1e6)
        grid on
        subplot(NN, 3, 3 * ii - 0)

        if ii == 1
            title('Z axis')
        end

        if ii == NN
            xlabel('Z (\mum)')
        end

        hold on

        for jj = 1:round(Nx / 5):Nx

            for kk = 1:round(Ny / 5):Ny
                plot(z * 1e6, squeeze(psi(jj, kk, :, ii)))
            end

        end

        xlim([min(z), max(z)] * 1e6)
        grid on
    end

    sgtitle(sprintf('Wavefunctions\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW', wx(wi) * 1e9, wy(wi) * 1e9, P * 1e3))
    %

    disp('________________________');
    %     disp(wx(wi));
    %     disp((E(2)-E(1))/3.67)
    %     disp((E(7:9)-E(1))/24.6 )
    figure(102); clf;
    plot(1:numel(E), E - E(1), 'o');
    ylabel('E (kHz)'); xlabel('Eigenlevels');
    legend(num2str(E - E(1)), 'location', 'best');
    %     waitforbuttonpress;
end
