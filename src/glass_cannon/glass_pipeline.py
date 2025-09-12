"""High-level pipeline to simulate galaxy overdensity shells with GLASS.

This module wires together GLASS and CAMB utilities defined elsewhere in the
package to produce lognormal matter fields and biased galaxy overdensity
fields across redshift shells. Optionally, pseudo-3D visualisation code is
provided (commented-out) to render a volumetric view.
"""

import glass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys

from glass_cannon.cosmo_setup import make_cosmology_class
from glass_cannon.Cls import angular_power_spectrum
import glass_cannon.matter as ma 
from glass_cannon.galaxies import add_galaxies
from glass_cannon.HI_tracer import b_HI, T_HI_bar, convert_DM_to_HI

def galaxy_bias(z):
     """Linear galaxy bias model.

     Parameters
     ----------
     z : float or ndarray
         Redshift value(s).

     Returns
     -------
     float or ndarray
         Galaxy bias at ``z`` defined as ``0.7 * (1 + z)``.
     """
     return 0.7 * (1 + z)


def convert_DM_to_galaxy_overdensity(shells, matter):
    """

    """
    galaxy_overdensities = []
    for shell, delta_m in zip(shells, matter):
        z = shell.zeff
        #need mean not boundary
        
        # Apply bias to overdensity
        delta_g = galaxy_bias(z) * delta_m
        
        galaxy_overdensities.append(delta_g)

    return galaxy_overdensities

def simulator(h, OmegaB, OmegaC, length = 128, seed = np.random.default_rng(seed=42), PLOT=False):
        """Run the GLASS-based simulation pipeline.

        This constructs a cosmology, builds redshift shells, computes angular
        power spectra, generates lognormal matter fields, and returns the
        corresponding biased galaxy overdensity fields per shell.

        Parameters
        ----------
        h : float
            Hubble parameter in units of 100 km s^-1 Mpc^-1.
        OmegaB : float
            Baryon density parameter.
        OmegaC : float
            Cold dark matter density parameter.
        length : int, optional
            Resolution parameter (e.g., HEALPix nside or similar) used by
            GLASS when discretising spectra and generating fields. Default is
            128.
        seed : numpy.random.Generator, optional
            Random number generator used for reproducible sampling. Default is
            ``np.random.default_rng(seed=42)``.
        PLOT : bool, optional
            If True, execute the plotting code path (currently commented-out)
            to visualise a pseudo-3D galaxy distribution. Default is False.

        Returns
        -------
        list of ndarray
            A list of galaxy overdensity fields, one per redshift shell.
        """
        cosmo, pars = make_cosmology_class(h=h, Oc=OmegaC, Ob=OmegaB)
        
        # shells of 200 Mpc in comoving distance spacing
        zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)

        cls, shells = angular_power_spectrum(pars,length,zb)

        fields = ma.run_ln_fields(shells)

        cls = ma.run_discretized_cls(cls, length, length)
        
        # compute Gaussian spectra for lognormal fields from discretised spectra
        gls = ma.run_solve_gauss_spectra(fields, cls)

        # generator for lognormal matter fields
        matter = ma.run_generate(fields, gls, length, seed)

        galaxy_overdensities = convert_DM_to_galaxy_overdensity(shells, matter)
        hi_temperature_fields = convert_DM_to_HI(shells, matter)
 
        #if PLOT==True:

        return galaxy_overdensities, hi_temperature_fields










       

