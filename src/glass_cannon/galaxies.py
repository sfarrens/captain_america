"""Galaxy sampling and partitioning utilities.

This module contains helpers to partition a redshift distribution function
across GLASS radial windows and related galaxy utilities.
"""

import glass


# create 10 redshift shells between z=0 and z=1

def add_galaxies(z,dndz, shells):
    """Partition a galaxy redshift distribution over a sequence of windows.

    Parameters
    ----------
    z : ndarray
        Redshift bin centers or edges. If ``dndz`` is multi-dimensional, its
        last axis must match the size of ``z``.
    dndz : ndarray
        Galaxy redshift distribution to be partitioned. Can be multi-dimensional
        with the last axis corresponding to ``z``.
    shells : Sequence
        Ordered sequence of GLASS radial window functions.

    Returns
    -------
    ndarray
        Galaxy number counts per shell obtained by integrating ``dndz`` over
        the window functions in ``shells``.
    """
    # constant galaxy density distribution
    ngal = glass.partition(z, dndz, shells)
    return ngal 
