import DVRPackage.*

%Set trap parameters
mass = UnitsConstants.mRb87;
w0 = 707 * UnitsConstants.nm;
asep = 808 * UnitsConstants.nm / w0;
V0 = 100 * UnitsConstants.kHz;
Delta = 2 * UnitsConstants.kHz;
Nbands = 8;

%%%%%%%%Biased Double-well Gaussian potential
%range of DVR space (units of w_0)
ax = 2.26;
dx = 0.05;
[xvals, Nx] = GetASGridDx(ax, dx);

BareScal = EnergyFromLengthScale(w0, mass); %Scaling of second derivative
Gaussf = @(x) -(V0 + 0.5 * Delta) * exp(-2.0 * (x - 0.5 * asep).^2) - (V0 - 0.5 * Delta) * exp(-2.0 * (x + 0.5 * asep).^2); %Gaussian potential with x in waist units
%Get states/energies
[vecs, G_vals] = DVR_AS1D(xvals, BareScal, Nbands, Gaussf);

disp('Double-well energies')
G_vals / UnitsConstants.kHz

G_gs = vecs(1, :) / sqrt(dx);
G_es = vecs(2, :) / sqrt(dx);
G_gs2 = vecs(3, :) / sqrt(dx);
G_es2 = vecs(4, :) / sqrt(dx);
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
