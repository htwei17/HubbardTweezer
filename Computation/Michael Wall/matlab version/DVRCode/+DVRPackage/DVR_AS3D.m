function [vecs,vals] = DVR_3D(xvals,yvals,zvals,BarexScal, BareyScal, BarezScal, pary, parz,Nbands,potential)
%Set up and diagonalize the DVR representation of a 3D problem with potential given by potential(x,y,z).
%*vals is a vector of gridpoints along direction *, and Bare*Scal is the coefficient of (d^2/d*^2).  Nbands is the desired number of eigenstates.
%This routine uses parity symmetries for y and z, with par='p' being even parity and par='m' being odd parity, but no parity symmetry along x.

Nx=floor(0.5*(length(xvals)-1));
dx=xvals(2)-xvals(1);
xscal=BarexScal/dx^2;
Ny=length(yvals)-1;
dy=yvals(2)-yvals(1);
yscal=BareyScal/dy^2;
Nz=length(zvals)-1;
dz=zvals(2)-zvals(1);
zscal=BarezScal/dz^2;

Tx=DVRPackage.SetupAST(Nx,xscal);
Ty=DVRPackage.SetupT(Ny,yscal,pary);
Tz=DVRPackage.SetupT(Nz,zscal,parz);

Nxp=2*Nx+1;lxvals=xvals;
if(pary=='p') 
    Nyp=Ny+1;lyvals=yvals;
else
    Nyp=Ny; lyvals=yvals(2:end);
end
if(parz=='p') 
    Nzp=Nz+1;lzvals=zvals;
else
    Nzp=Nz; lzvals=zvals(2:end);
end
N=Nxp*Nyp*Nzp

V=zeros( Nxp,Nyp,Nzp);
for ix=1:Nxp
    for iy=1:Nyp
        for iz=1:Nzp
            V(ix,iy,iz)=potential(lxvals(ix),lyvals(iy),lzvals(iz));
        end 
    end
end

%Define function handle to H*psi multiply with kinetic and potential operators specified
multfunc = @(x) DVRPackage.ApplyDVR3D(Tx,Ty,Tz,V,x);
%specify symmetric matrix
opts.issym=1;
%Sparse diagonalization routine
[V,D,flag]=eigs(multfunc,N,Nbands,'SA',opts);
if flag ~=0
    disp('Nonzero info output from eigs!')
    flag
end
vals=diag(D([1:Nbands],[1:Nbands]));
vecs=V';
end
