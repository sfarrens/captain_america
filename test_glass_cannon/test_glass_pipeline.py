import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

from glass_cannon.glass_pipeline import simulator, galaxy_bias, convert_DM_to_galaxy_overdensity
from glass_cannon.HI_tracer import b_HI, T_HI_bar, convert_DM_to_HI


def test_galaxy_bias():

    z = np.array([0.047207579689725554, 0.09549701085050732, 0.14497037793633272, 0.19573704554756666, 0.24791426348217693, 0.30162784506518125, 0.3570129270679938, 0.4142148213585366, 0.473389969923508, 0.5347070166563821, 0.598348011354985, 0.6645097637765192, 0.7334053684248278, 0.8052659240755213, 0.8803424759815801, 0.9589082133665099])
    expected_b = 0.7 * (1 + z)

    b = galaxy_bias(z)

    assert all(np.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(expected_b, b))

def test_convert_DM_to_galaxy_overdensity():
    """
    """
        # cosmology for the simulation
    h = 0.7
    Oc = 0.25
    Ob = 0.05

    rng = np.random.default_rng(seed=42)
    # basic parameters of the simulation
    nside = lmax = 128

    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)

    # get the cosmology from CAMB
    cosmo = Cosmology(results)
    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)

    # linear radial window functions
    shells = glass.linear_windows(zb)

    cls = glass.ext.camb.matter_cls(pars, lmax, shells)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)
    fields = glass.lognormal_fields(shells)
    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    # generator for lognormal matter fields
    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)

    expected_galaxy_overdensities = []
    for shell, delta_m in zip(shells, matter):
        z = shell.zeff
        #need mean not boundary
        
        # Apply bias to overdensity
        delta_HI = galaxy_bias(z) * delta_m
        
        expected_galaxy_overdensities.append(delta_HI)

    galaxy_overdensities = convert_DM_to_galaxy_overdensity(shells, matter)

    assert all(np.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(expected_galaxy_overdensities, galaxy_overdensities))

def test_simulator():
    # creating a numpy random number generator for sampling
    rng = np.random.default_rng(seed=42)

    # cosmology for the simulation
    h = 0.7
    Oc = 0.25
    Ob = 0.05

    # basic parameters of the simulation
    nside = lmax = 128

    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)

    # get the cosmology from CAMB
    cosmo = Cosmology(results)

    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)

    # linear radial window functions
    shells = glass.linear_windows(zb)

    # compute the angular matter power spectra of the shells with CAMB
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)
    
    # set up lognormal matter fields for simulation
    fields = glass.lognormal_fields(shells)

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)

    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    # generator for lognormal matter fields
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

    expected_galaxy_overdensities = []
    for shell, delta_m in zip(shells, matter):
        z = shell.zeff
        #need mean not boundary
        
        # Apply bias to overdensity
        delta_HI = galaxy_bias(z) * delta_m
        
        expected_galaxy_overdensities.append(delta_HI)

    sim_galaxy_overdensities, sim_hi_fields, all_cls = simulator(h=0.7, OmegaC=0.25, OmegaB=0.05)


    

    for expected, sim in zip(expected_galaxy_overdensities, sim_galaxy_overdensities):
        assert np.allclose(expected, sim, rtol=1e-6, atol=1e-8)

    for expected, sim in zip(expected_hi_temperature_fields, sim_hi_fields):
        assert np.allclose(expected, sim, rtol=1e-6, atol=1e-8)

    


