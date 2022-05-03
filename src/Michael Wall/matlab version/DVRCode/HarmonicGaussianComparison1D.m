import DVRPackage.*

%Set trap parameters
mass = UnitsConstants.mRb87;
w0 = 707*UnitsConstants.nm;
V0 = 100*UnitsConstants.kHz;

%%%%%%%%Harmonic oscillator expansion of Gaussian potential
%Expanding Gaussian potential -V*exp(-2 x^2/w_0^2) to harmonic order, find
%-V+0.5*m \omega^2 x^2, where \hbar \omega=sqrt(V \hbar^2 4/(m w_0^2))
effOmega=sqrt(UnitsConstants.h*V0*4.0/(mass*w0^2));
holen=sqrt(UnitsConstants.hbar/(mass*effOmega));

%range of DVR space (harmonic oscillator units)
ax=5.0;
dx=0.4;
[xvals,Nx] = GetGridDx(ax,dx);
Nbands=4;
BareScal=0.5; %Scaling of second derivative in ho units
hof=@(x) 0.5*x.^2; %Harmonic oscillator potential
%Get even and odd parity harmonic oscillator states/energies
[evecs,ho_evals] = DVR_1D(xvals,BareScal,'p',Nbands,hof);
[ovecs,ho_ovals] = DVR_1D(xvals,BareScal,'m',Nbands,hof);

ho_gs = Unpack1DState(evecs(1,:),'p')/sqrt(dx);
ho_es = Unpack1DState(ovecs(1,:),'m')/sqrt(dx);
ho_es2 = Unpack1DState(evecs(2,:),'p')/sqrt(dx);
long_ho_xvals=dx*[-Nx:Nx];

%%%%%%%%Actual Gaussian potential
%range of DVR space (units of w_0-set to be same as harmonic approx)
ax=5*holen/w0;
dx=0.4*holen/w0;
[xvals,Nx] = GetGridDx(ax,dx);

BareScal=EnergyFromLengthScale(w0,mass); %Scaling of second derivative in waist units
Gaussf=@(x) -V0*exp(-2.0*x.^2); %Gaussian potential with x in waist units
%Get even and odd parity harmonic oscillator states/energies
[evecs,G_evals] = DVR_1D(xvals,BareScal,'p',Nbands,Gaussf);
[ovecs,G_ovals] = DVR_1D(xvals,BareScal,'m',Nbands,Gaussf);

disp('Even Harmonic oscillator energies (ho units)')
ho_evals
disp('Even Gaussian energies (ho units)')
(G_evals+V0)/(effOmega/(2.0*pi))

disp('Odd Harmonic oscillator energies (ho units)')
ho_ovals
disp('Odd Gaussian energies (ho units)')
(G_ovals+V0)/(effOmega/(2.0*pi))

G_gs = Unpack1DState(evecs(1,:),'p')/sqrt(dx*w0/holen);
G_es = Unpack1DState(ovecs(1,:),'m')/sqrt(dx*w0/holen);
G_es2 = Unpack1DState(evecs(2,:),'p')/sqrt(dx*w0/holen);
long_G_xvals=dx*[-Nx:Nx]*w0/holen;

figure
subplot(3,1,1)    
plot(long_G_xvals,G_gs,'r')
hold
plot(long_ho_xvals,ho_gs,'b')
subplot(3,1,2)    
plot(long_G_xvals,G_es,'r')
hold
plot(long_ho_xvals,ho_es,'b')
subplot(3,1,3)    
plot(long_G_xvals,G_es2,'r')
hold
plot(long_ho_xvals,ho_es2,'b')
