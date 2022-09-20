function y = EnergyFromLengthScale(l, m)
    %Obtain the "recoil energy" \hbar^2/(2ml^2) associated with length scale l and mass m, in h Hz
    y = DVRPackage.UnitsConstants.hbar / (4 * pi * m * l^2);
end
