function unpackedpsi = Unpack1DState(psi,par)
%Converts a state in the parity-adpated representation into the non-parity adapted representation
if (par=='p')
    N=length(psi)-1;
    unpackedpsi=0*[1:2*N+1];
    unpackedpsi(N+1)=psi(1);
    unpackedpsi(N+2:end)=psi(2:end)/sqrt(2.0);
    unpackedpsi(1:N)=psi(end:-1:2)/sqrt(2.0);
else
    N=length(psi);
    unpackedpsi=0*[1:2*N+1];
    unpackedpsi(N+1)=0.0;
    unpackedpsi(N+2:end)=psi(1:end)/sqrt(2.0);
    unpackedpsi(1:N)=-psi(end:-1:1)/sqrt(2.0);
end
end
