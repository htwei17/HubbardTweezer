function y = swaveintegral(psi1, psi2, dx, dy, dz)
    %Obtain the s-wave pseudopotential integral \int dr |\psi_1|^2 |\psi_2|^2 in units of the volume element [dx*dy*dz].
    %As we integrate over the full domain, DVR quadrature (trapezoid rule) is exponentially convergent.
    %Multiply by InteractionFromLengthScale to get the interaction matrix element in energy units
    t1 = psi1.^2; t2 = psi2.^2;
    y = sum(sum(sum(t1 .* t2))) * dx * dy * dz;
end
