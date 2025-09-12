"""Computation of angular matter power spectra with CAMB via GLASS.

This module exposes a helper to compute the angular power spectra for a set
of linear redshift shells using the CAMB-backed GLASS extension.
"""

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb


def angular_power_spectrum(pars,lmax,zb):
    """Compute angular matter power spectra for linear redshift shells.

    Parameters
    ----------
    pars : camb.model.CAMBparams
        CAMB parameter object describing the background cosmology.
    lmax : int
        Maximum angular multipole to compute.
    zb : ndarray
        Grid in comoving distance or redshift boundaries passed to
        ``glass.linear_windows`` to build shells.

    Returns
    -------
    tuple
        A pair ``(cls, shells)`` where ``cls`` are the CAMB-computed angular
        matter spectra in the GLASS format and ``shells`` are the associated
        linear radial window functions.
    """
    
    # linear radial window functions
    shells = glass.tophat_windows(zb, dz=0.001)
    
    # compute the angular matter power spectra of the shells with CAMB
    return glass.ext.camb.matter_cls(pars, lmax, shells), shells