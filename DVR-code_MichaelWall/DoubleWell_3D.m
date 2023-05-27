import DVRPackage.*

%Set trap parameters
mass = UnitsConstants.mRb87;
w0 = 707 * UnitsConstants.nm;
asep = 808 * UnitsConstants.nm / w0;
V0 = 100 * UnitsConstants.kHz;
zR = 2170 * UnitsConstants.nm;

effOmega = sqrt(UnitsConstants.h * V0 * 4.0 / (mass * w0^2));
holen = sqrt(UnitsConstants.hbar / (mass * effOmega));

Nbands = 4;
%%%%%%%%Double-well Gaussian potential
%range of DVR spaces (units of w_0 for x,y, units of zR for z)
ax = 3.5; dx = 0.04; [xvals, Nx] = GetGridDx(ax, dx);
ay = 2.0; dy = 0.04; [yvals, Ny] = GetGridDx(ay, dy);
az = 2.36; dz = 0.022; [zvals, Nz] = GetGridDx(az, dz);
BarewScal = EnergyFromLengthScale(w0, mass); %Scaling of second derivative in waist units
BarezScal = EnergyFromLengthScale(zR, mass); %Scaling of second derivative in Rayleigh-range units

invzf = @(z) 1.0 / (1.0 + z.^2);
Gaussf = @(x, y, z) -V0 * invzf(z) * exp(-2.0 * invzf(z) * y.^2) * (exp(-2.0 * invzf(z) * (x - 0.5 * asep).^2) + exp(-2.0 * invzf(z) * (x + 0.5 * asep).^2)); %Double-Gaussian potential with x,y in waist units, z in Rayleigh-range units
%Get even and odd parity harmonic oscillator states/energies
t = cputime;
[pppvecs, pppvals] = DVR_3D(xvals, yvals, zvals, BarewScal, BarewScal, BarezScal, 'p', 'p', 'p', Nbands, Gaussf);
disp('Time for even')
e1 = cputime - t
t = cputime;
[mppvecs, mppvals] = DVR_3D(xvals, yvals, zvals, BarewScal, BarewScal, BarezScal, 'm', 'p', 'p', Nbands, Gaussf);
disp('Time for odd')
e2 = cputime - t

disp('Even Double-well energies')
pppvals / UnitsConstants.kHz
disp('Odd Double-well energies')
mppvals / UnitsConstants.kHz

%Unpack states from the parity-adapted representation
G_gs = Unpack3DState(pppvecs(1, :), Nx, Ny, Nz, 'p', 'p', 'p') / sqrt(dx * dy * dz);
G_es = Unpack3DState(mppvecs(1, :), Nx, Ny, Nz, 'm', 'p', 'p') / sqrt(dx * dy * dz);
G_gs2 = Unpack3DState(pppvecs(2, :), Nx, Ny, Nz, 'p', 'p', 'p') / sqrt(dx * dy * dz);
G_es2 = Unpack3DState(mppvecs(2, :), Nx, Ny, Nz, 'm', 'p', 'p') / sqrt(dx * dy * dz);
long_G_xvals = dx * [-Nx:Nx]; long_G_yvals = dy * [-Ny:Ny]; long_G_zvals = dz * [-Nz:Nz];

if (1)
    [Xg, Yg, Zg] = NearestSeparableState(G_gs);
    [Xe, Ye, Ze] = NearestSeparableState(G_es);

    figure
    subplot(3, 1, 1)
    semilogy(long_G_xvals, abs(Xg), 'r')
    hold
    semilogy(long_G_xvals, abs(Xe), 'b')
    subplot(3, 1, 2)
    semilogy(long_G_yvals, abs(Yg), 'r')
    hold
    semilogy(long_G_yvals, abs(Ye), 'b')
    subplot(3, 1, 3)
    semilogy(long_G_zvals, abs(Zg), 'r')
    hold
    semilogy(long_G_zvals, abs(Ze), 'b')
end

%Define states to localize (along x)
psis = cell(1, 2);
psis{1} = G_gs;
psis{2} = G_es;
t = cputime;
[outvecs, S] = LocalizeState(psis, Nx, Ny, Nz, dx, dy, dz);
disp('Time to localize')
e2 = cputime - t
S
outvecs
wannier = zeros(size(psis{1}));

for i = 1:size(outvecs, 1)
    wannier = wannier + outvecs(2, i) * psis{i};
end

%Hubbard model
%Define Hamiltonian in basis of states 1,2
Hami = diag([pppvals(1), mppvals(1)]) / UnitsConstants.kHz;
%Rotate Hamiltonian to this basis to obtain (non-interacting) Hubbard model
HHubb = (outvecs') * Hami * outvecs;
disp('Hubbard parameters E1, E2, J')
disp([HHubb(1, 1), HHubb(2, 2), HHubb(1, 2)])

%Interaction energy
U = InteractionFromLengthScale(w0, w0, zR, mass) * swaveintegral(wannier, wannier, dx, dy, dz) / UnitsConstants.kHz;
disp('Hubbard interaction U')
disp(U)

%Nearest separable
[X, Y, Z] = NearestSeparableState(wannier);

figure
subplot(3, 1, 1)
plot(long_G_xvals, X, 'r')
subplot(3, 1, 2)
plot(long_G_yvals, Y, 'r')
subplot(3, 1, 3)
plot(long_G_zvals, Z, 'r')
