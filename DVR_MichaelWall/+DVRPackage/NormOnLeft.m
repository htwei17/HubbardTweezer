function y = NormOnLeft(psi1, psi2, Nx, Ny, Nz, dx, dy, dz, sincover)
    %Integrate two states psi1 and psi2 over the left half of the system using an analytic representation of
    %the sinc DVR basis functions over [-\infty,0]

    %Multiply \psitmp(i,j,k)=I_{i,i'} \psi(i',j,k), where I_{i,i'} is matrix of half-domain sinc DVR overlaps
    ptmp = reshape(sincover * reshape(psi2, [size(psi2, 1), size(psi2, 2) * size(psi2, 3)]), size(psi2));

    %Perform remaining integrations over full domain of y,z with DVR quadrature
    y = sum(sum(sum(psi1 .* ptmp))) * dx * dy * dz
end
