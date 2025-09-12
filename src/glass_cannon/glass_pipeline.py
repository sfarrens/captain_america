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
import healpy as hp

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

def simulator(h, OmegaB, OmegaC, length = 128, PLOT=False):
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
        rng = np.random.default_rng(seed=42)
        cosmo, pars = make_cosmology_class(h=h, Oc=OmegaC, Ob=OmegaB)
        lmax = 128
        
        # Create n_shells redshift bins
        zb = glass.redshift_grid(zmin=0., zmax=1.2, dz=0.4)
        cls, shells = angular_power_spectrum(pars, length, zb)
        print(len(shells))
        
        # Lognormal matter fields
        fields = ma.run_ln_fields(shells)
        cls = ma.run_discretized_cls(cls, length, length)
        gls = ma.run_solve_gauss_spectra(fields, cls)
        matter = ma.run_generate(fields, gls, length, rng)
        
        # Convert to galaxy and HI fields
        galaxy_overdensities = convert_DM_to_galaxy_overdensity(shells, matter)
        matter = ma.run_generate(fields, gls, length, rng)
        hi_temperature_fields = convert_DM_to_HI(shells, matter)
        print(np.array(hi_temperature_fields).shape)
        print(np.array(galaxy_overdensities).shape)
        
        # Upper-triangular indices
        num_shells = len(zb)-1
        print(num_shells)
        
        # Prepare Cl lists for each type
        cls_hi_hi = []
        cls_gg = []
        cls_hi_g = []
        
        # Loop over upper-triangular combinations
        for i in range(num_shells):
            for j in range(i+1):
                print(i,j)
                # HI-HI
                cls_hi_hi.append(hp.anafast(hi_temperature_fields[i], hi_temperature_fields[j], pol=False, lmax=lmax))
                # Galaxy-Galaxy
                cls_gg.append(hp.anafast(galaxy_overdensities[i], galaxy_overdensities[j], pol=False, lmax=lmax))
                # HI-Galaxy cross
                cls_hi_g.append(hp.anafast(hi_temperature_fields[i], galaxy_overdensities[j], pol=False, lmax=lmax))
        
        # Optional: plot
        if PLOT:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(np.array(galaxy_overdensities), aspect='auto')
            axes[0].set_title("Galaxy overdensity fields")
            axes[1].imshow(np.array(hi_temperature_fields), aspect='auto')
            axes[1].set_title("HI temperature fields")
            plt.show()

            i = len(hi_temperature_fields) // 2
            hp.mollview(
                hi_temperature_fields[i],
                title=f"HI Brightness Temperature (z ~ {shells[i].zeff:.2f})",
                unit="mK",
                cmap="inferno"
            )
            plt.show()
            
        all_cls = {
        "hi_hi": np.array(cls_hi_hi),
        "g_g": np.array(cls_gg),
        "hi_g": np.array(cls_hi_g)
        }

        return galaxy_overdensities, hi_temperature_fields, all_cls

galaxy_overdensities, hi_temperature_fields, all_cls = simulator(h=0.7, OmegaC=0.25, OmegaB=0.05, length = 128, PLOT=False, n_shells=3)


