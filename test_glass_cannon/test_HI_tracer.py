import numpy as np
import glass
import camb

from glass_cannon.HI_tracer import b_HI, T_HI_bar, convert_DM_to_HI

def test_b_HI():

    z = np.array([0.047207579689725554, 0.09549701085050732, 0.14497037793633272, 0.19573704554756666, 0.24791426348217693, 0.30162784506518125, 0.3570129270679938, 0.4142148213585366, 0.473389969923508, 0.5347070166563821, 0.598348011354985, 0.6645097637765192, 0.7334053684248278, 0.8052659240755213, 0.8803424759815801, 0.9589082133665099])
    expected_b = 0.667 + 0.178*z + 0.050*z**2 

    b = b_HI(z)

    assert all(np.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(expected_b, b))

def test_T_HI_bar():

    z = np.array([0.047207579689725554, 0.09549701085050732, 0.14497037793633272, 0.19573704554756666, 0.24791426348217693, 0.30162784506518125, 0.3570129270679938, 0.4142148213585366, 0.473389969923508, 0.5347070166563821, 0.598348011354985, 0.6645097637765192, 0.7334053684248278, 0.8052659240755213, 0.8803424759815801, 0.9589082133665099])
    expected_t = 0.0559 + 0.2324*z - 0.0241*z**2

    t = T_HI_bar(z)

    assert all(np.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(expected_t, t))

def test_convert_DM_to_HI():

    z = np.array([0.047207579689725554, 0.09549701085050732, 0.14497037793633272, 0.19573704554756666, 0.24791426348217693, 0.30162784506518125, 0.3570129270679938, 0.4142148213585366, 0.473389969923508, 0.5347070166563821, 0.598348011354985, 0.6645097637765192, 0.7334053684248278, 0.8052659240755213, 0.8803424759815801, 0.9589082133665099])
    shells=glass.linear_windows(z)
    nside = lmax = 128

    h = 0.7
    Oc = 0.25
    Ob = 0.05

    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both,
    )
    prev_cls = glass.ext.camb.matter_cls(pars, lmax, shells)

    cls = glass.discretized_cls(prev_cls, nside=nside, lmax=lmax, ncorr=3)
    fields = glass.lognormal_fields(shells)
    gls = glass.solve_gaussian_spectra(fields, cls)
    rng = np.random.default_rng(seed=42)

    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)

    expected_hi_temperature_fields = []
    for shell, delta_m in zip(shells, matter):
        z = shell.zeff
        #need mean not boundary
        
        # Apply bias to overdensity
        delta_HI = b_HI(z) * delta_m
        
        # Map overdensity to brightness temperature
        #  - this is the observed temperature, not just fluctuations around the mean
        #  - so the 1 + fluctuation factor is included rather than multiplying just be the fluctuation 
        #       as is the case in equation 3 in Cunnington et al. 2019
        T_HI = T_HI_bar(z) * (1 + delta_HI)
        
        expected_hi_temperature_fields.append(T_HI)

    hi_temperature_fields = convert_DM_to_HI(shells, matter)

    assert all(np.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(expected_hi_temperature_fields, hi_temperature_fields))