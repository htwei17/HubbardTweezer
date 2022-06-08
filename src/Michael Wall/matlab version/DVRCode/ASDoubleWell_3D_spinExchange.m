import DVRPackage.*

%Set trap parameters
mass = UnitsConstants.mRb87;
w0 = 710*UnitsConstants.nm;
asep = 854*UnitsConstants.nm/w0;
V0 = 91*UnitsConstants.kHz;
zR=2170*UnitsConstants.nm;
Delta= 6.4*UnitsConstants.kHz;

Nbands=8;
%%%%%%%%Double-well Gaussian potential
%range of DVR spaces (units of w_0 for x,y, units of zR for z)
ax=3.5; dx=0.04; [xvals,Nx] = GetASGridDx(ax,dx);
ay=2.0; dy=0.04; [yvals,Ny] = GetGridDx(ay,dy);
az=2.36; dz=0.022; [zvals,Nz] = GetGridDx(az,dz);
BarewScal=EnergyFromLengthScale(w0,mass); %Scaling of second derivative in waist units
BarezScal=EnergyFromLengthScale(zR,mass); %Scaling of second derivative in Rayleigh-range units

invzf=@(z) 1.0/(1.0+z.^2);
Gaussf=@(x,y,z) -invzf(z)*exp(-2.0*invzf(z)*y.^2)*((V0+0.5*Delta)*exp(-2.0*invzf(z)*(x-0.5*asep).^2)+(V0-0.5*Delta)*exp(-2.0*invzf(z)*(x+0.5*asep).^2)); %Biased Double-Gaussian potential with x,y in waist units, z in Rayleigh-range units
%Get even and odd parity harmonic oscillator states/energies
t = cputime;
[vecs,vals] = DVR_AS3D(xvals, yvals,zvals,BarewScal,BarewScal, BarezScal,'p','p',Nbands,Gaussf);
disp('Time for DVR')
e1 = cputime-t

disp('Double-well energies')
vals/UnitsConstants.kHz

%Unpack states from the parity-adapted representation
G_gs = UnpackAS3DState(vecs(1,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es = UnpackAS3DState(vecs(2,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es2 = UnpackAS3DState(vecs(3,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es3 = UnpackAS3DState(vecs(4,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es4 = UnpackAS3DState(vecs(5,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es5 = UnpackAS3DState(vecs(6,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es6 = UnpackAS3DState(vecs(7,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
G_es7 = UnpackAS3DState(vecs(8,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
long_G_xvals=dx*[-Nx:Nx]; long_G_yvals=dy*[-Ny:Ny]; long_G_zvals=dz*[-Nz:Nz];

[Xg,Yg,Zg] = NearestSeparableState(G_gs);
[Xe,Ye,Ze] = NearestSeparableState(G_es);
[Xe2,Ye2,Ze2] = NearestSeparableState(G_es2);
[Xe3,Ye3,Ze3] = NearestSeparableState(G_es3);
[Xe4,Ye4,Ze4] = NearestSeparableState(G_es4);
[Xe5,Ye5,Ze5] = NearestSeparableState(G_es5);
[Xe6,Ye6,Ze6] = NearestSeparableState(G_es6);
[Xe7,Ye7,Ze7] = NearestSeparableState(G_es7);

figure
subplot(8,1,1)    
plot(long_G_xvals,abs(Xg),'b')
subplot(8,1,2)    
plot(long_G_xvals,abs(Xe),'b')
subplot(8,1,3)    
plot(long_G_xvals,abs(Xe2),'b')
subplot(8,1,4)    
plot(long_G_xvals,abs(Xe3),'b')
subplot(8,1,5)    
plot(long_G_xvals,abs(Xe4),'b')
subplot(8,1,6)    
plot(long_G_xvals,abs(Xe5),'b')
subplot(8,1,7)    
plot(long_G_xvals,abs(Xe6),'b')
subplot(8,1,8)    
plot(long_G_xvals,abs(Xe7),'b')

figure
subplot(8,1,1)    
plot(long_G_yvals,abs(Yg),'r')
subplot(8,1,2)    
plot(long_G_yvals,abs(Ye),'b')
subplot(8,1,3)    
plot(long_G_yvals,abs(Ye2),'b')
subplot(8,1,4)    
plot(long_G_yvals,abs(Ye3),'b')
subplot(8,1,5)    
plot(long_G_yvals,abs(Ye4),'b')
subplot(8,1,6)    
plot(long_G_yvals,abs(Ye5),'b')
subplot(8,1,7)    
plot(long_G_yvals,abs(Ye6),'b')
subplot(8,1,8)    
plot(long_G_yvals,abs(Ye7),'b')

figure
subplot(8,1,1)    
plot(long_G_zvals,abs(Zg),'b')
subplot(8,1,2)    
plot(long_G_zvals,abs(Ze),'b')
subplot(8,1,3)    
plot(long_G_zvals,abs(Ze2),'b')
subplot(8,1,4)    
plot(long_G_zvals,abs(Ze3),'b')
subplot(8,1,5)    
plot(long_G_zvals,abs(Ze4),'b')
subplot(8,1,6)    
plot(long_G_zvals,abs(Ze5),'b')
subplot(8,1,7)    
plot(long_G_zvals,abs(Ze6),'b')
subplot(8,1,8)    
plot(long_G_zvals,abs(Ze7),'b')


%Define states to localize (along x)
psis=cell(1,2);
psis{1}=G_es2;
psis{2}=G_es3;
t = cputime;
[outvecs,S] = LocalizeState(psis,Nx,Ny,Nz,dx,dy,dz);
disp('Time to localize')
e2 = cputime-t
S
outvecs
wannierL=zeros(size(psis{1}));
for i=1:size(outvecs,1)
    wannierL=wannierL+outvecs(2,i)*psis{i};
end
wannierR=zeros(size(psis{1}));
for i=1:size(outvecs,1)
    wannierR=wannierR+outvecs(1,i)*psis{i};
end

%Hubbard model
%Define Hamiltonian in basis of states 1,2
Hami=diag([vals(3), vals(4)])/UnitsConstants.kHz;
%Rotate Hamiltonian to this basis to obtain (non-interacting) Hubbard model
HHubb=(outvecs')*Hami*outvecs;
disp('Hubbard parameters E1, E2, J')
disp([ HHubb(1,1) ,HHubb(2,2), HHubb(1,2)])


%Interaction energy
U=InteractionFromLengthScale(w0,w0,zR,mass)*swaveintegral(wannierR,wannierR,dx,dy,dz)/UnitsConstants.kHz;
disp('Hubbard interaction U' )
disp(U)

%Nearest separable
[XL,YL,ZL] = NearestSeparableState(wannierL);
[XR,YR,ZR] = NearestSeparableState(wannierR);

figure
subplot(3,1,1)    
plot(long_G_xvals,XL,'r')
hold
plot(long_G_xvals,XR,'b')
subplot(3,1,2)    
plot(long_G_yvals,YL,'r')
hold
plot(long_G_yvals,YR,'b')
subplot(3,1,3)    
plot(long_G_zvals,ZL,'r')
hold
plot(long_G_zvals,ZR,'b')


