# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

# import function to test
from glass_cannon.Cls import angular_power_spectrum

def test_angular_power_spectrum():

    # set up CAMB parameters for matter angular power spectrum
    h = 0.7
    pars = camb.set_params(
        H0=100 * h,
        omch2=0.25 * h**2,
        ombh2=0.05 * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)
    
    # get the cosmology from CAMB
    cosmo = Cosmology(results)

    # basic parameters of the simulation
    lmax = 128
    
    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)
    
    # linear radial window functions
    shells = glass.linear_windows(zb)
    
    # compute the angular matter power spectra of the shells with CAMB
    test_cls, test_shells = glass.ext.camb.matter_cls(pars, lmax, shells)

    assert angular_power_spectrum(pars,lmax,zb) == test_cls





