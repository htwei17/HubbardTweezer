% This is a general code to calculate the tunneling of ground states atoms
% in a double-well potential generated through two tweezers close to
% eachother.
tic;
%Fundamental constants
% constants;
c = 3 * 1e8;
h = 6.63 * 1e-34;
hbar = h / (2 * pi);
kb = 1.38 * 1e-23;

% Lithium-6
mLi = 6 * 1.67 * 1e-27;
gamma = 2 * pi * 6 * 1e6;
lambda0 = 671 * 1e-9;
f0 = c / lambda0;
omega0 = 2 * pi * f0;

% Laser Params
lambda = 780 * 1e-9; %wavelength (m) 830
omega = 2 * pi * c / lambda;
k = 2 * pi / lambda;
Er = hbar^2 * k^2 / (2 * mLi);

P = 35 * 1.1e-6; %power per tweezer (W) 92.6
asymm = 0; %asymmetry in units of depth
wx = 1000 * 1e-9; %waist perp direction (m) 1350
wy = wx; %waist tunneling direction (m) 1000% 810e-9
zRx = pi * wx^2 / lambda;
zRy = pi * wy^2 / lambda;
%
I0 = 2 * P / (pi * wx * wy);
V0 = 3 * pi * c^2 / (2 * omega0^3) * (gamma / (omega0 - omega) + gamma / (omega0 + omega)) * I0;

% Tweezer Params
a = 1450 * 1e-9; %distance between tweezers (m) is 1.51 (830) and 1.32 um (for 770)
Mx = 1.75 * max(a, wx);
My = 2.75 * max(a, wy);
Mz = 0.75 * max(zRx, zRy);
Nx = 15;
Ny = 31;
Nz = 15;

V_tweezers = @(x, y, z) -V0 ./ (1 + z.^2 .* (1 ./ zRx.^2 + 1 ./ zRy.^2) / 2) .* exp(-2 * x.^2 ./ (wx.^2 .* (1 + z.^2 ./ zRx.^2))) ...
    .* ((1 + asymm / 2) * exp(-2 * (y - a ./ 2).^2 ./ (wy.^2 .* (1 + z.^2 ./ zRy.^2))) ...
    + (1 - asymm / 2) * exp(-2 * (y + a ./ 2).^2 ./ (wy.^2 .* (1 + z.^2 ./ zRy.^2))));

% Gravity
g = 9.80665;
V_gravity = @(x, y, z) mLi * g * z;

% axial Trapping
omega_z = 2 * pi * 20e3; %2pi x kHz
omega_z = 2 * pi * 5e3;
V_axial = @(x, y, z) 0 * 1/2 * mLi * omega_z.^2 .* z.^2;

gradient = 0 * 2.7e2; %gauss/m
splitting = 0.7e6; %Hz/Gauss
V_magnetic = @(x, y, z) splitting * gradient * h * x;

V = @(x, y, z) V_tweezers(x, y, z) + V_gravity(x, y, z) + V_axial(x, y, z) + V_magnetic(x, y, z);

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
NN = 20;
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

close all
figure
colors = lines(NN);
subplot(1, 3, 1)
hold on
[~, II_min] = min(V(0, y, 0));
II_min = II_min(1);
y_min = y(II_min);
plot(x * 1e6, V(x, 0, 0) / h * 1e-3, 'k')
plot(x * 1e6, V(x, y_min, 0) / h * 1e-3, 'k')

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
plot(a / 2 * 1e6 * [1 1], V0 / h * 1e-3 * [-1.1 -1], 'k--')
plot(-a / 2 * 1e6 * [1 1], V0 / h * 1e-3 * [-1.1 -1], 'k--')

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
plot(z * 1e6, V(0, y_min, z) / h * 1e-3, 'k')

for ii = 1:NN
    plot([min(z), max(z)] * 1e6, [E(ii), E(ii)], '--', 'color', colors(ii, :));
end

xlabel('Z (\mum)')
title('Z axis')
grid on
xlim([min(z), max(z)] * 1e6)
ylim([1.1 * min(VV(:)) / h * 1e-3, 0])
sgtitle(sprintf('Potential and bound state energies\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW, a = %.0f nm, Delta = %.3f kHz', wx * 1e9, wy * 1e9, P * 1e3, a * 1e9, asymm * V0 / h * 1e-3))

figure(87);

for ii = 1:NN
    subplot(NN, 3, 3 * ii - 2)

    if ii == 1
        title('X axis')
    end

    ylabel(sprintf('Psi_{%d}', ii))

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

sgtitle(sprintf('Wavefunctions\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW, a = %.0f nm, Delta = %.3f kHz', wx * 1e9, wy * 1e9, P * 1e3, a * 1e9, asymm * V0 / h * 1e-3))

% print tunneling and interaction

tunneling_simple = abs(E(1) - E(2)) / 2;

PxP = zeros(2, 2);

for ii = [1 2]

    for jj = [1 2]
        PxP(ii, jj) = (PSI(:, ii)' * (yy(:) .* PSI(:, jj))); %*dx*dy*dz;
    end

end

[aa, ee] = eig(PxP);
[~, II] = sort(diag(ee));
aa = aa(:, II);

PSI_L = aa(2, 2) * PSI(:, 1) + aa(2, 1) * PSI(:, 2);
PSI_R = aa(1, 2) * PSI(:, 1) + aa(1, 1) * PSI(:, 2);
psi_L = aa(2, 2) * psi(:, :, :, 1) + aa(2, 1) * psi(:, :, :, 2);
psi_R = aa(1, 2) * psi(:, :, :, 1) + aa(1, 1) * psi(:, :, :, 2);
% PSI_L = 1 / sqrt(2) * (PSI(:, 1) - PSI(:, 2));
% PSI_R = 1 / sqrt(2) * (PSI(:, 1) + PSI(:, 2));
% psi_L = 1 / sqrt(2) * (psi(:, :, :, 1) - psi(:, :, :, 2));
% psi_R = 1 / sqrt(2) * (psi(:, :, :, 1) + psi(:, :, :, 2));
tunneling_wannier = abs(PSI_L' * H * PSI_R * dx * dy * dz / h * 1e-3)
fields = 0:1:1000;
a_s = a3D(fields, 2);
interaction_L = 4 * pi * hbar^2 * a_s / mLi * sum(PSI_L(:).^4) * dx * dy * dz / h * 1e-3;
interaction_R = 4 * pi * hbar^2 * a_s / mLi * sum(PSI_R(:).^4) * dx * dy * dz / h * 1e-3;

fprintf('Tunneling_{simple} = %.3f kHz ; Tunneling_{wannier} = %.3f kHz \n', tunneling_simple, tunneling_wannier)
figure(98);
subplot(3, 1, 1)
hold on
plot(fields, interaction_L, 'o')
plot(fields, interaction_R, 'o')
legend('Left', 'Right')
ylim([-5, 5])
grid on
xlabel('Magnetic Field (G)')
ylabel('Interaction (kHz)')
title(sprintf('Interaction vs B-field ; t_{simple} = %.3f kHz ; t_{wannier} = %.3f kHz\n wx = %.0f nm, wy = %.0f nm, power = %.3f mW, a = %.0f nm, Delta = %.3f kHz', tunneling_simple, tunneling_wannier, wx * 1e9, wy * 1e9, P * 1e3, a * 1e9, asymm * V0 / h * 1e-3))

for ii = 1:2

    if ii == 1
        psi_i = psi_L;
    else
        psi_i = psi_R;
    end

    subplot(3, 3, 3 + 3 * ii - 2)

    if ii == 1
        title('X axis')
    end

    if ii == 1
        ylabel('Psi_L');
    else
        ylabel('Psi_R');
    end

    if ii == NN
        xlabel('X (\mum)')
    end

    hold on

    for jj = 1:round(Ny / 5):Ny

        for kk = 1:round(Nz / 5):Nz
            plot(x * 1e6, squeeze(psi_i(:, jj, kk)))
        end

    end

    xlim([min(x), max(x)] * 1e6)
    grid on
    subplot(3, 3, 3 + 3 * ii - 1)

    if ii == 1
        title('Y axis')
    end

    if ii == NN
        xlabel('Y (\mum)')
    end

    hold on

    for jj = 1:round(Nx / 5):Nx

        for kk = 1:round(Nz / 5):Nz
            plot(y * 1e6, squeeze(psi_i(jj, :, kk)))
        end

    end

    xlim([min(y), max(y)] * 1e6)
    grid on
    subplot(3, 3, 3 + 3 * ii - 0)

    if ii == 1
        title('Z axis')
    end

    if ii == NN
        xlabel('Z (\mum)')
    end

    hold on

    for jj = 1:round(Nx / 5):Nx

        for kk = 1:round(Ny / 5):Ny
            plot(z * 1e6, squeeze(psi_i(jj, kk, :)))
        end

    end

    xlim([min(z), max(z)] * 1e6)
    grid on
end

%% interaction in kHz
interaction_1000 = 4 * pi * hbar^2 * 1000 * abohr / mLi * sum(PSI_L(:).^4) * dx * dy * dz / h * 1e-3;
interaction_1770 = 4 * pi * hbar^2 * 1770 * abohr / mLi * sum(PSI_L(:).^4) * dx * dy * dz / h * 1e-3;
interaction_1650 = 4 * pi * hbar^2 * 1650 * abohr / mLi * sum(PSI_L(:).^4) * dx * dy * dz / h * 1e-3

U_t = interaction_1770 / tunneling_wannier
