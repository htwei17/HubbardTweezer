%This code computes the energy levels for a single tweezer.

%Fundamental constants
% constants;
h = 6.62607004 * 1e-34;
hbar = h / (2 * pi);
kb = 1.38064852 * 1e-23;

% Lithium-6
mLi = 6.015122 * 1.6605339 * 1e-27;
gamma = 2 * pi * 6 * 1e6;
lambda0 = 671 * 1e-9;
c = 299792458;
f0 = c / lambda0;
omega0 = 2 * pi * f0;

% Laser Params
lambda = 780 * 1e-9; %wavelength (m)
omega = 2 * pi * c / lambda;
k = 2 * pi / lambda;
Er = hbar^2 * k^2 / (2 * mLi);

P = 55 * 1e-6; %power per tweezer (W)
asymm = 0; %asymmetry in units of depth
wx = 1000 * 1e-9; %waist perp direction (m) 1350
wy = 1000 * 1e-9; %waist tunneling direction (m) 1000% 810e-9
zRx = pi * wx^2 / lambda;
zRy = pi * wy^2 / lambda;
%
I0 = 2 * P / (pi * wx * wy);
V0 = 3 * pi * c^2 / (2 * omega0^3) * (gamma / (omega0 - omega) + gamma / (omega0 + omega)) * I0;

V_tweezers = @(x, y, z) -V0 ./ (1 + z.^2 .* (1 ./ zRx.^2 + 1 ./ zRy.^2) / 2) ...
    .* exp(-2 * x.^2 ./ (wx.^2 .* (1 + z.^2 ./ zRx.^2))) ...
    .* exp(-2 * y.^2 ./ (wy.^2 .* (1 + z.^2 ./ zRy.^2)));

% Gravity
g = 9.80665;
V_gravity = @(x, y, z) mLi * g * z;

% V = @(x, y, z) V_tweezers(x, y, z) + V_gravity(x, y, z);
V = @(x, y, z) V_tweezers(x, y, z);

% Tweezer Params
M = [2 * wx 2 * wy 4 * max(zRx, zRy)];
% N = [15 15 31];
Nmax = 8;
E = [];

for index = 1:Nmax
    Ni = 2 * (index * [1, 1, 2] + [0 0 1]) + 1;
    E(end + 1, :) = DVR3d_run(M, Ni, hbar, mLi, V)'/V0;
end

dE = abs(diff(E));
close all

for index = 1:5
    plot((2:Nmax)', dE(:, index))
    hold on
end

xlabel("N")
ylabel("\Delta E/V_0")
set(gca, "YScale", 'log')
title(['Convergence of energies vs N @ Tweezer, fixing R = ', num2str(M(1) / 2 / wx), 'w'])

function E = DVR3d_run(M, N, hbar, mLi, V)
    %DVR3d - Description
    %
    % Syntax: E = DVR3d_run(M, N, hbar, mLi, V)
    %
    % Long description

    tic;
    % Tweezer Params
    M = num2cell(M);
    N = num2cell(N);
    [Mx, My, Mz] = deal(M{:});
    [Nx, Ny, Nz] = deal(N{:});

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
    TTx = spdiags(ones(Nx * Ny * Nz, 1) * pi^2/3, 0, Nx * Ny * Nz, Nx * Ny * Nz); % diagonal term

    % build off-diagonals in matrix form
    for Dx = 1:(Nx - 1)
        M = [ones(1, (Nx - Dx) * Ny), zeros(1, Ny * Dx)]; % [1 1 1 ... 1 0 0 ...]
        M = repmat(M, 1, Nz)'; % repeat by Nz times
        TTx = spdiags((-1)^Dx * 2 / Dx^2 * M(end:-1:1), +Dx * Ny, TTx); % replace Dx * Ny-diagonals of TTx by (-1)^Dx * 2 / Dx^2 * M(end:-1:1)
        TTx = spdiags((-1)^Dx * 2 / Dx^2 * M, -Dx * Ny, TTx); % the symmetric entries
    end

    TTy = spdiags(ones(numel(yy), 1) * pi^2/3, 0, Nx * Ny * Nz, Nx * Ny * Nz);

    for Dy = 1:(Ny - 1)
        M = [ones(1, Ny - Dy), zeros(1, Dy)];
        M = repmat(M, 1, Nx * Nz)';
        TTy = spdiags((-1)^Dy * 2 / Dy^2 * M(end:-1:1), +Dy, TTy);
        TTy = spdiags((-1)^Dy * 2 / Dy^2 * M, -Dy, TTy);
    end

    TTz = spdiags(ones(numel(zz), 1) * pi^2/3, 0, Nx * Ny * Nz, Nx * Ny * Nz);

    for Dz = 1:(Nz - 1)
        M = ones(1, Nx * Ny * Nz)';
        TTz = spdiags((-1)^Dz * 2 / Dz^2 * M, +Dz * Nx * Ny, TTz);
        TTz = spdiags((-1)^Dz * 2 / Dz^2 * M, -Dz * Nx * Ny, TTz);
    end

    TT = hbar^2 / (2 * mLi) * (TTx / dx^2 + TTy / dy^2 + TTz / dz^2);
    % TT = (TTx + TTy + TTz);
    fprintf('Kinetic Energy Term. Elapsed Time %.3f s\n', toc)

    %diagonalize
    NN = 16;
    H = TT + VV - spdiags(ones(numel(xx), 1) * min(VV(:)), 0, Nx * Ny * Nz, Nx * Ny * Nz); % subtract zero poin energy
    [eigvects, eigenergies] = eigs(H, NN, 'SM');
    fprintf('Diagonalize. Elapsed Time %.3f s\n', toc)

    % extract energies and sort
    E = diag(eigenergies + min(VV(:)));
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

end

% % close all
% figure(45); clf;
% colors = lines(NN);
% subplot(1, 3, 1)
% hold on
% plot(x * 1e6, V(x, 0, 0) / h * 1e-3, 'k')
% % plot(x*1e6, V(x, a/2, 0)/h*1e-3, 'k')
% for ii = 1:NN
%     plot([min(x), max(x)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
% end

% xlabel('X (\mum)')
% ylabel('Potential (kHz)')
% title('X axis')
% grid on
% xlim([min(x), max(x)] * 1e6)
% ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
% subplot(1, 3, 2)
% hold on
% plot(y * 1e6, V(0, y, 0) / h * 1e-3, 'k')

% for ii = 1:NN
%     plot([min(y), max(y)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
% end

% xlabel('Y (\mum)')
% title('Y axis')
% grid on
% xlim([min(y), max(y)] * 1e6)
% ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
% subplot(1, 3, 3)
% hold on
% plot(z * 1e6, V(0, 0, z) / h * 1e-3, 'k')
% % plot(z*1e6, V(0, a/2, z)/h*1e-3, 'k')
% for ii = 1:NN
%     plot([min(z), max(z)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
% end

% xlabel('Z (\mum)')
% title('Z axis')
% grid on
% xlim([min(z), max(z)] * 1e6)
% ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
% sgtitle(sprintf('Potential and bound state energies\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW', wx * 1e9, wy * 1e9, P * 1e3))

% figure(8); clf;

% for ii = 1:NN
%     subplot(NN, 3, 3 * ii - 2)

%     if ii == 1
%         title('X axis')
%     end

%     ylabel(sprintf('Psi_%d', ii))

%     if ii == NN
%         xlabel('X (\mum)')
%     end

%     hold on

%     for jj = 1:round(Ny / 5):Ny

%         for kk = 1:round(Nz / 5):Nz
%             plot(x * 1e6, squeeze(psi(:, jj, kk, ii)))
%         end

%     end

%     xlim([min(x), max(x)] * 1e6)
%     grid on
%     subplot(NN, 3, 3 * ii - 1)

%     if ii == 1
%         title('Y axis')
%     end

%     if ii == NN
%         xlabel('Y (\mum)')
%     end

%     hold on

%     for jj = 1:round(Nx / 5):Nx

%         for kk = 1:round(Nz / 5):Nz
%             plot(y * 1e6, squeeze(psi(jj, :, kk, ii)))
%         end

%     end

%     xlim([min(y), max(y)] * 1e6)
%     grid on
%     subplot(NN, 3, 3 * ii - 0)

%     if ii == 1
%         title('Z axis')
%     end

%     if ii == NN
%         xlabel('Z (\mum)')
%     end

%     hold on

%     for jj = 1:round(Nx / 5):Nx

%         for kk = 1:round(Ny / 5):Ny
%             plot(z * 1e6, squeeze(psi(jj, kk, :, ii)))
%         end

%     end

%     xlim([min(z), max(z)] * 1e6)
%     grid on
% end

% sgtitle(sprintf('Wavefunctions\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW', wx * 1e9, wy * 1e9, P * 1e3))
% %
% figure(102); clf;
% plot(1:numel(E), E - E(1), 'o');
% ylabel('E (kHz)'); xlabel('Eigenlevels');
% legend(num2str(E - E(1)), 'location', 'best');
