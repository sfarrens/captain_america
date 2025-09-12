"""Utilities for generating lognormal matter fields with GLASS.

This module wraps common GLASS calls to build lognormal fields, discretise
angular power spectra, solve for Gaussian spectra, and generate correlated
matter shells.
"""

import glass
import camb


def run_ln_fields(shells):
    """Create lognormal field generator for the given shells.

    Parameters
    ----------
    shells : Sequence
        Sequence of GLASS radial window shells.

    Returns
    -------
    object
        Lognormal field descriptor as returned by ``glass.lognormal_fields``.
    """

    fields = glass.lognormal_fields(shells)

    return fields


def run_discretized_cls(prev_cls, nside, lmax):
    """Apply discretisation to a set of angular power spectra.

    The discretisation includes the HEALPix pixel window, a maximum angular
    multipole, and a fixed number of correlated shells.

    Parameters
    ----------
    prev_cls : object
        Input angular power spectra (e.g., from CAMB) in the GLASS format.
    nside : int
        Resolution parameter used for the HEALPix pixel window function.
    lmax : int
        Maximum multipole used for the discretisation.

    Returns
    -------
    object
        Discretised spectra suitable for GLASS field generation.
    """

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(prev_cls, nside=nside, lmax=lmax, ncorr=3)

    return cls


def run_solve_gauss_spectra(fields, cls):
    """Solve Gaussian spectra for lognormal fields.

    Parameters
    ----------
    fields : object
        Lognormal fields descriptor returned by ``glass.lognormal_fields``.
    cls : object
        Discretised spectra from ``run_discretized_cls``.

    Returns
    -------
    object
        Gaussian spectra compatible with GLASS lognormal field generation.
    """

    gls = glass.solve_gaussian_spectra(fields, cls)

    return gls


def run_generate(fields, gls, nside, rng):
    """Generate correlated lognormal matter fields.

    Parameters
    ----------
    fields : object
        Lognormal fields descriptor.
    gls : object
        Gaussian spectra from ``run_solve_gauss_spectra``.
    nside : int
        Resolution parameter used by GLASS when generating fields.
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    generator of ndarray
        Generator yielding matter overdensity fields per shell.
    """

    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)

    return matter

