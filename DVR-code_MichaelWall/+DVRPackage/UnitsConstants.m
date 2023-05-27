classdef UnitsConstants
    %define relevant constants
    properties (Constant)
        kg = 1; m = 1; K = 1; s = 1;
        hbar = 1.05457173 * 10^(-34) * DVRPackage.UnitsConstants.m^2 * DVRPackage.UnitsConstants.kg / DVRPackage.UnitsConstants.s;
        h = 2.0 * pi * DVRPackage.UnitsConstants.hbar;
        kBoltzmann = 1.3806488 * 10^(-23) * DVRPackage.UnitsConstants.m^2 * DVRPackage.UnitsConstants.kg / DVRPackage.UnitsConstants.s^2 / DVRPackage.UnitsConstants.K;
        ns = 10^(-9); mus = 10^(-6); ms = 10^(-3);
        Hz = 1 / DVRPackage.UnitsConstants.s; kHz = 10^3 * DVRPackage.UnitsConstants.Hz; MHz = 10^6 * DVRPackage.UnitsConstants.Hz; GHz = 10^9 * DVRPackage.UnitsConstants.Hz; THz = 10^(12) * DVRPackage.UnitsConstants.Hz;
        nm = 10^(-9); mum = 10^(-6); mm = 10^(-3); cm = 10^(-2);
        mW = 10^(-3); muW = 10^(-6); nW = 10^(-9);
        muF = 10^(-6); nF = 10^(-9);
        muH = 10^(-6); nH = 10^(-9);
        au = 1.660538921 * 10^(-27) * DVRPackage.UnitsConstants.kg;
        mMg = 24 * DVRPackage.UnitsConstants.au; mBe = 9 * DVRPackage.UnitsConstants.au; mCa = 40 * DVRPackage.UnitsConstants.au; me = 9.10938291 * 10^(-31) * DVRPackage.UnitsConstants.kg; mRb87 = 87 * DVRPackage.UnitsConstants.au; mRb85 = 85 * DVRPackage.UnitsConstants.au;
        lambdaMg = 280 * DVRPackage.UnitsConstants.nm; lambdaBe = 313 * DVRPackage.UnitsConstants.nm;
        gammaBe = 19.4 * DVRPackage.UnitsConstants.MHz;
        mK = 10^(-3) * DVRPackage.UnitsConstants.K; muK = 10^(-6) * DVRPackage.UnitsConstants.K; nK = 10^(-9) * DVRPackage.UnitsConstants.K;
        epsilon0 = 8.85418781762 * 10^(-12); %*vacuum permittivity
        c0 = 299792458 * DVRPackage.UnitsConstants.m / DVRPackage.UnitsConstants.s; %*speed of light
        mu0 = 1.2566370614 * 10^(-6); %*vacuum*permeability

        e0 = 1.60217657 * 10^(-19); %Coulombs
        muB = (DVRPackage.UnitsConstants.e0 * DVRPackage.UnitsConstants.hbar) / (2 * DVRPackage.UnitsConstants.me); %*Bohr magneton
        ge = -2.00231930436153; %*electron g-factor
        gp = 5.585694713; %*proton g-factor
        alpha0 = DVRPackage.UnitsConstants.e0^2 / (4 * pi * DVRPackage.UnitsConstants.epsilon0 * DVRPackage.UnitsConstants.hbar * DVRPackage.UnitsConstants.c0);
        a0 = DVRPackage.UnitsConstants.hbar / (DVRPackage.UnitsConstants.alpha0 * DVRPackage.UnitsConstants.me * DVRPackage.UnitsConstants.c0);

        G0 = 6.67384 * 10^(-11);
        Gauss = 10^(-4);

        as = 5.45 * DVRPackage.UnitsConstants.nm; %s-wave scattering length of Rb87

    end

end
