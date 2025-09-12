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
     """
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
        """
        --------------------------
        Define cosmological parameters:

        h (float): hubble parameter 
        Omegac (float): critical density parameter
        Omegab (float): baryon density parameter
        length (int): size of box
        seed (int): random seed to use for simulation
        PLOT (boolean): set to True if you want to plot the 3D galaxy distribution
        ----------------------------
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









       

