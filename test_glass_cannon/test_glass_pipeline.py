import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

from glass_cannon.glass_pipeline import simulator

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
    
    # constant galaxy density distribution
    z = np.linspace(0.0, 1.0, 100)
    dndz = np.full_like(z, 0.01)

    # distribute the dN/dz over the linear window functions
    ngal = glass.partition(z, dndz, shells)
    
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
            
    assert np.allclose(cube, simulator(h=0.7, OmegaC = 0.25, OmegaB = 0.05), rtol=1e-6, atol=1e-8)


