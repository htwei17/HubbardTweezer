import DVRPackage.*

Nbands = 4;
ax = 5.0; dx = 0.4; [xvals, Nx] = GetGridDx(ax, dx);
ay = 5.0; dy = 0.4; [yvals, Ny] = GetGridDx(ay, dy);
az = 5.0; dz = 0.4; [zvals, Nz] = GetGridDx(az, dz);

BarewScal = 0.5;
BarezScal = 0.5;
Gaussf = @(x, y, z) 0.5 * (x.^2 + y.^2 + z.^2);
%Get even and odd parity harmonic oscillator states/energies along x direction
t = cputime;
[pppvecs, pppvals] = DVR_3D(xvals, yvals, zvals, BarewScal, BarewScal, BarezScal, 'p', 'p', 'p', Nbands, Gaussf);
disp('Time for even')
e1 = cputime - t
t = cputime;
[mppvecs, mppvals] = DVR_3D(xvals, yvals, zvals, BarewScal, BarewScal, BarezScal, 'm', 'p', 'p', Nbands, Gaussf);
disp('Time for odd')
e2 = cputime - t

disp('Even energies')
pppvals
disp('Odd energies')
mppvals

%Unpack states from the parity-adapted representation
G_gs = Unpack3DState(pppvecs(1, :), Nx, Ny, Nz, 'p', 'p', 'p') / sqrt(dx * dy * dz);
G_es = Unpack3DState(mppvecs(1, :), Nx, Ny, Nz, 'm', 'p', 'p') / sqrt(dx * dy * dz);
G_gs2 = Unpack3DState(pppvecs(2, :), Nx, Ny, Nz, 'p', 'p', 'p') / sqrt(dx * dy * dz);
G_es2 = Unpack3DState(mppvecs(2, :), Nx, Ny, Nz, 'm', 'p', 'p') / sqrt(dx * dy * dz);
long_G_xvals = dx * [-Nx:Nx]; long_G_yvals = dy * [-Ny:Ny]; long_G_zvals = dz * [-Nz:Nz];

%If you want to see what the eigenstates look like
if (1)
    [Xg, Yg, Zg] = NearestSeparableState(G_gs);
    [Xe, Ye, Ze] = NearestSeparableState(G_es);

    figure
    subplot(3, 1, 1)
    plot(long_G_xvals, Xg, 'r')
    hold
    plot(long_G_xvals, Xe, 'b')
    subplot(3, 1, 2)
    plot(long_G_yvals, Yg, 'r')
    hold
    plot(long_G_yvals, Ye, 'b')
    subplot(3, 1, 3)
    plot(long_G_zvals, Zg, 'r')
    hold
    plot(long_G_zvals, Ze, 'b')
end

%Define states to localize (along x)-here defined to be the states (0,0,0) and (1,0,0)
psis = cell(1, 2);
psis{1} = G_gs;
psis{2} = G_es;
t = cputime;
[outvecs, S] = LocalizeState(psis, Nx, Ny, Nz, dx, dy, dz);
disp('Time to localize')
e2 = cputime - t
%Analytical values are (1/2)\mp 1/\sqrt{2\pi}
disp('Norms of optimal Wannier functions on the left')
S
disp('Expansions of optimal Wannier functions')
outvecs
%construct optimal wannier function localized on the left
wannier = zeros(size(psis{1}));

for i = 1:size(outvecs, 1)
    wannier = wannier + outvecs(2, i) * psis{i};
end

%Define Hubbard model by rotating Hamiltonian to Wannier basis
%Hamiltonian in basis of eigenstates
Hami = diag([pppvals(1), mppvals(1)]);
%Rotate Hamiltonian to Wannier basis to obtain (non-interacting) Hubbard model
HHubb = (outvecs') * Hami * outvecs;
disp('Hubbard parameters E1, E2, J')
disp([HHubb(1, 1), HHubb(2, 2), HHubb(1, 2)])

%Interaction energy (analytical value is (19/16)(2\pi)^{-3/2}
U = swaveintegral(wannier, wannier, dx, dy, dz);
disp('Hubbard interaction U')
disp(U)

%Nearest separable state to Wannier function
[X, Y, Z] = NearestSeparableState(wannier);

figure
subplot(3, 1, 1)
plot(long_G_xvals, X, 'r')
subplot(3, 1, 2)
plot(long_G_yvals, Y, 'r')
subplot(3, 1, 3)
plot(long_G_zvals, Z, 'r')
