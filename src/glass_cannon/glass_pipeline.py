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


"""def make_3D_galaxy_cube(zb, matter, ngal, rng, shells):
    # make a cube for galaxy number in redshift
    zcub = np.linspace(-zb[-1], zb[-1], 21)
    cube = np.zeros((zcub.size - 1,) * 3)

    # simulate and add galaxies in each matter shell to cube
    for i, delta_i in enumerate(matter):
        # simulate positions from matter density
        for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
            ngal[i],
            delta_i,
            rng=rng,
        ):
            # sample redshifts uniformly in shell
            gal_z = glass.redshifts(gal_count, shells[i], rng=rng)

            # add counts to cube
            z1 = gal_z * np.cos(np.deg2rad(gal_lon)) * np.cos(np.deg2rad(gal_lat))
            z2 = gal_z * np.sin(np.deg2rad(gal_lon)) * np.cos(np.deg2rad(gal_lat))
            z3 = gal_z * np.sin(np.deg2rad(gal_lat))
            indices, count = np.unique(
                np.searchsorted(zcub[1:], [z1, z2, z3]),
                axis=1,
                return_counts=True,
            )
            cube[*indices] += count
            
    return cube, zcub"""

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

        galaxy_overdensities = []
        for shell, delta_m in zip(shells, matter):
            z = shell.zeff
            #need mean not boundary
            
            # Apply bias to overdensity
            delta_g = galaxy_bias(z) * delta_m
            
            galaxy_overdensities.append(delta_g)
        # constant galaxy density distribution
        #z = np.linspace(0.0, 1.0, 100)
        #dndz = np.full_like(z, 0.01)
        
        #ngal = add_galaxies(z,dndz, shells)

        #cube, zcub = make_3D_galaxy_cube(zb, matter, ngal, seed, shells)
        
        if PLOT==True:
            """ PLOT THE FIGURE  
            # positions of grid cells of the cube
            z = (zcub[:-1] + zcub[1:]) / 2
            z1, z2, z3 = np.meshgrid(z, z, z)

            # plot the galaxy distribution in pseudo-3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
            norm = LogNorm(vmin=np.min(cube[cube > 0]), vmax=np.max(cube), clip=True)
            for i in range(len(zcub) - 1):
                v = norm(cube[..., i])
                c = plt.cm.inferno(v)
                c[..., -1] = 0.2 * v
                ax.plot_surface(
                    z1[..., i],
                    z2[..., i],
                    z3[..., i],
                    rstride=1,
                    cstride=1,
                    facecolors=c,
                    linewidth=0,
                    shade=False,
                    antialiased=False,
                )
            fig.tight_layout()
            plt.show()
        
        return cube""" 
        
        return galaxy_overdensities









       

