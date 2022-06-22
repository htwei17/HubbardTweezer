import DVRPackage.*

%Set trap parameters
mass = UnitsConstants.mRb87;
w0 = 707 * UnitsConstants.nm;
asep = 808 * UnitsConstants.nm / w0;
V0 = 100 * UnitsConstants.kHz;
Nbands = 4;

%%%%%%%%Double-well Gaussian potential
%range of DVR space (units of w_0)
ax = 2.26;
dx = 0.05;
[xvals, Nx] = GetGridDx(ax, dx);

BareScal = EnergyFromLengthScale(w0, mass); %Scaling of second derivative
Gaussf = @(x) -V0 * (exp(-2.0 * (x - 0.5 * asep).^2) + exp(-2.0 * (x + 0.5 * asep).^2)); %Double-Gaussian potential with x in waist units
%Get states/energies
[evecs, G_evals] = DVR_1D(xvals, BareScal, 'p', Nbands, Gaussf);
[ovecs, G_ovals] = DVR_1D(xvals, BareScal, 'm', Nbands, Gaussf);

disp('Even Double-well energies')
G_evals / UnitsConstants.kHz
disp('Odd Double-well energies')
G_ovals / UnitsConstants.kHz

G_gs = Unpack1DState(evecs(1, :), 'p') / sqrt(dx);
G_es = Unpack1DState(ovecs(1, :), 'm') / sqrt(dx);
G_gs2 = Unpack1DState(evecs(2, :), 'p') / sqrt(dx);
G_es2 = Unpack1DState(ovecs(2, :), 'm') / sqrt(dx);
long_G_xvals = dx * [-Nx:Nx];

figure
subplot(2, 1, 1)
plot(long_G_xvals, G_gs, 'r')
hold
plot(long_G_xvals, G_es, 'b')
subplot(2, 1, 2)
plot(long_G_xvals, G_gs2, 'r')
hold
plot(long_G_xvals, G_es2, 'b')
