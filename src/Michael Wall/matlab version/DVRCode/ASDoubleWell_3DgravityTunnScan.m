import DVRPackage.*

%Set trap parameters
mass = UnitsConstants.mRb87;
w0 = 710*UnitsConstants.nm;
asep = 900*UnitsConstants.nm/w0;
V0 = 16.4*UnitsConstants.kHz;
zR=2170*UnitsConstants.nm;
Deltamin= 0.75*UnitsConstants.kHz;
Deltamax= 3.25*UnitsConstants.kHz;
ndels=20;
GravityCoef=(mass*9.81*w0/(UnitsConstants.h)); %mgz in Hz/waist


Nbands=4;
%%%%%%%%Double-well Gaussian potential
%range of DVR spaces (units of w_0 for x,y, units of zR for z)
ax=3.5; dx=0.04; [xvals,Nx] = GetASGridDx(ax,dx);
ay=2.0; dy=0.04; [yvals,Ny] = GetGridDx(ay,dy);
az=2.36; dz=0.022; [zvals,Nz] = GetGridDx(az,dz);
BarewScal=EnergyFromLengthScale(w0,mass); %Scaling of second derivative in waist units
BarezScal=EnergyFromLengthScale(zR,mass); %Scaling of second derivative in Rayleigh-range units
long_G_xvals=dx*[-Nx:Nx]; long_G_yvals=dy*[-Ny:Ny]; long_G_zvals=dz*[-Nz:Nz];

gsE=zeros(1,ndels);
esE=zeros(1,ndels);
dels=Deltamin+[0:ndels-1]*(Deltamax-Deltamin)/(1.0*(ndels-1))
HubbDels=zeros(1,ndels);
HubbJs=zeros(1,ndels);

for di=1:ndels
    Delta=Deltamin+(di-1)*(Deltamax-Deltamin)/(1.0*(ndels-1))

    invzf=@(z) 1.0/(1.0+z.^2);
    Gaussf=@(x,y,z) -invzf(z)*exp(-2.0*invzf(z)*y.^2)*((V0+0.5*Delta)*exp(-2.0*invzf(z)*(x-0.5*asep).^2)+(V0-0.5*Delta)*exp(-2.0*invzf(z)*(x+0.5*asep).^2))+GravityCoef*x; %Biased Double-Gaussian potential with x,y in waist units, z in Rayleigh-range units
    t = cputime;
    [vecs,vals] = DVR_AS3D(xvals, yvals,zvals,BarewScal,BarewScal, BarezScal,'p','p',Nbands,Gaussf);
    disp('Time for DVR')
    e1 = cputime-t

    disp('Double-well energies')
    vals/UnitsConstants.kHz
    G_gs = UnpackAS3DState(vecs(1,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
    gsE(di)=vals(1);
    [Xg,Yg,Zg] = NearestSeparableState(G_gs);
    testvec=zeros(1,20);
    for i=1:20
        testvec(i)=Zg(Nz+i)*Zg(Nz+i+1);
    end
    testvec
    esi=2;
    G_es = UnpackAS3DState(vecs(2,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
    esE(di)=vals(2);
    [Xe,Ye,Ze] = NearestSeparableState(G_es);
    testvec=zeros(1,20);
    for i=1:20
        testvec(i)=Ze(Nz+i)*Ze(Nz+i+1);
    end
    testvec
    if ~all(testvec>0)
        esi=3;
        G_es = UnpackAS3DState(vecs(3,:),Nx,Ny,Nz,'p','p')/sqrt(dx*dy*dz);
        esE(di)=vals(3);
        [Xe,Ye,Ze] = NearestSeparableState(G_es);
        testvec=zeros(1,20);
        for i=1:20
            testvec(i)=Ze(Nz+i)*Ze(Nz+i+1);
        end
        disp('second try')
        testvec
    end
    if ~all(testvec>0)
        disp('FAIL')
        stop
    end

    if(0)
        figure
        subplot(2,1,1)    
        plot(long_G_xvals,abs(Xg),'b')
        subplot(2,1,2)    
        plot(long_G_xvals,abs(Xe),'b')

        figure
        subplot(2,1,1)    
        plot(long_G_zvals,abs(Zg),'b')
        subplot(2,1,2)    
        plot(long_G_zvals,abs(Ze),'b')
    end

    %Define states to localize (along x)
    psis=cell(1,2);
    psis{1}=G_gs;
    psis{2}=G_es;
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
    Hami=diag([vals(1), vals(esi)])/UnitsConstants.kHz;
    %Rotate Hamiltonian to this basis to obtain (non-interacting) Hubbard model
    HHubb=(outvecs')*Hami*outvecs;
    disp('Hubbard parameters E1, E2, J')
    disp([ HHubb(1,1) ,HHubb(2,2), HHubb(1,2)])
    HubbDels(di)=HHubb(2,2)-HHubb(1,1);
    HubbJs(di)=HHubb(1,2);
end
HubbDels
HubbJs


