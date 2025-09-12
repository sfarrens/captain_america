"""HI 21 cm tracer utility functions.

This module provides convenience functions to model neutral hydrogen (HI)
bias and mean brightness temperature as a function of redshift, and a
helper to convert simulated matter overdensity fields into HI brightness
temperature fields.

References
----------
Cunnington, S. et al. (2019), doi:10.1093/mnras/stz1916
Sarkar, T. G. et al. (2021), `2107.14057`.
"""

import numpy as np
import glass


def b_HI(z):
    """HI bias as a function of redshift.

    The functional form follows Sarkar et al. (2021) `2107.14057`.

    Parameters
    ----------
    z : float or ndarray
        Redshift value(s).

    Returns
    -------
    float or ndarray
        HI bias evaluated at ``z``.
    """
    return 0.667 + 0.178*z + 0.050*z**2 


def T_HI_bar(z):
    """Mean 21 cm brightness temperature as a function of redshift.

    The polynomial approximation follows Sarkar et al. (2021) `2107.14057`.

    Parameters
    ----------
    z : float or ndarray
        Redshift value(s).

    Returns
    -------
    float or ndarray
        Mean HI brightness temperature at ``z`` (arbitrary units consistent
        with the simulation setup).
    """
    return 0.0559 + 0.2324*z - 0.0241*z**2


def convert_DM_to_HI(shells, matter):
    """Map matter overdensity shells to HI brightness temperature fields.

    For each redshift shell, the matter overdensity field ``delta_m`` is
    biased using the HI bias, and mapped to an observed brightness
    temperature field ``T_HI = T_HI_bar(z) * (1 + delta_HI)`` following the
    convention in Cunnington et al. (2019).

    Parameters
    ----------
    shells : Sequence
        Sequence of GLASS radial window shell objects. Each must provide a
        ``zeff`` attribute giving the effective redshift of the shell.
    matter : Sequence of ndarray
        Sequence of matter overdensity fields corresponding one-to-one to
        ``shells``. Each element is an array on the HEALPix grid or similar
        discretisation used by GLASS.

    Returns
    -------
    list of ndarray
        List of HI brightness temperature fields, one per input shell, with
        the same shape as the input matter fields.
    """
    hi_temperature_fields = []
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
        
        hi_temperature_fields.append(T_HI) #is this the correct way to combine?

    return hi_temperature_fields