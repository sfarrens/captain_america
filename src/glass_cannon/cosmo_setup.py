import pytest
# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

def set_cosmo(h, Oc, Ob):
    """A function to set the cosmological parameters

    Args:
        h (float): hubble parameter 
        Oc (float): critical density parameter
        Ob (float): baryon density parameter

    Returns:
        class: cosmology class giving all the parameter
    """
    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both    )
    
    return pars

def make_bkg(cosmo):
    results = camb.get_background(cosmo)
    return results

def make_cosmo(bkg):
    return Cosmology(bkg)

def make_cosmology_class(h, Oc, Ob):
    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both)
    
    results = camb.get_background(pars)
    
    return Cosmology(results), pars