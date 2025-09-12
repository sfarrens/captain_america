"""Cosmology setup helpers for CAMB and GLASS compatibility.

This module provides small helpers to construct CAMB parameter objects,
retrieve background quantities, and adapt them to the GLASS-compatible
`Cosmology` wrapper used elsewhere in the package.
"""

import pytest
# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb


def set_cosmo(h, Oc, Ob):
    """Build a CAMB parameter object from basic densities.

    Parameters
    ----------
    h : float
        Hubble parameter in units of 100 km s^-1 Mpc^-1.
    Oc : float
        Cold dark matter density parameter, Omega_c.
    Ob : float
        Baryon density parameter, Omega_b.

    Returns
    -------
    camb.model.CAMBparams
        The configured CAMB parameter object.
    """
    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both    )
    
    return pars


def make_bkg(cosmo):
    """Compute CAMB background for a given parameter object.

    Parameters
    ----------
    cosmo : camb.model.CAMBparams
        CAMB parameter object.

    Returns
    -------
    camb.results.CAMBdata
        Background results from CAMB.
    """
    results = camb.get_background(cosmo)
    return results


def make_cosmo(bkg):
    """Wrap CAMB background results into a GLASS-compatible Cosmology.

    Parameters
    ----------
    bkg : camb.results.CAMBdata
        CAMB background results.

    Returns
    -------
    cosmology.compat.camb.Cosmology
        Wrapper providing the API that GLASS expects.
    """
    return Cosmology(bkg)


def make_cosmology_class(h, Oc, Ob):
    """Construct both GLASS-compatible cosmology and CAMB params.

    Parameters
    ----------
    h : float
        Hubble parameter in units of 100 km s^-1 Mpc^-1.
    Oc : float
        Cold dark matter density parameter, Omega_c.
    Ob : float
        Baryon density parameter, Omega_b.

    Returns
    -------
    tuple
        A pair ``(cosmo, pars)`` with a GLASS-compatible ``Cosmology`` and
        the corresponding CAMB parameters ``pars``.
    """
    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both)
    
    results = camb.get_background(pars)
    
    return Cosmology(results), pars