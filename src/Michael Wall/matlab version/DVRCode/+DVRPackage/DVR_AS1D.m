function [vecs,vals] = DVR_AS1D(vals,BareScal,Nbands,potential)
%Set up and diagonalize the DVR representation of a 1D problem with potential given by potential(x).
%vals is a vector of gridpoints, BareScal is the coefficient of the (d^2/dx^2) operator, and Nbands are the number of eigenstates desired.
%This routine does not use parity symmetry.

N=floor(0.5*(length(vals)-1));
dq=vals(2)-vals(1);
scal=BareScal/dq^2;
T=DVRPackage.SetupAST(N,scal);
pvec=potential(vals);
V=diag(pvec);

H=T+V;       
[U, D]=eig(H);
vecs=zeros(Nbands,2*N+1);
vals=diag(D([1:Nbands],[1:Nbands]));
%Do not fix any arbitrary phases...
for i=1:Nbands
    vecs(i,:)=U(:,i);
end
end
