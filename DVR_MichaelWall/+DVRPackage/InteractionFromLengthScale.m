function y = InteractionFromLengthScale(lx, ly, lz, m)
    %Obtain the scattering energy scale 4\pi\hbar^2 a_s/(m lx ly lz) associated with x,y,z length scales lx,ly,lz and mass m, in h Hz
    y = 2.0 * DVRPackage.UnitsConstants.hbar * DVRPackage.UnitsConstants.as / (m * lx * ly * lz);
end
