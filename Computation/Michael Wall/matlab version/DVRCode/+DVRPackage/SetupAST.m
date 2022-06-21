function T = SetupAST(N,Scal)
%Returns the DVR second-derivative matrix scaled by the parameter Scal (=\hbar^2/(2m*lengthscale**2)), non-parity-adapted case

pisqo3=(pi^2)/3.0;
sqrtTwo=sqrt(2.0);

T=zeros(2*N+1);
for i=1:2*N+1
    T(i,i)=Scal*pisqo3;
end
for j=1:2*N
    invl=Scal*((-1.0)^(j))*2.0/((j^2)*1.0);
    for i=1:2*N+1-j
        T(i,i+j)=invl;
        T(i+j,i)=invl;
    end
end
end
