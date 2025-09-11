import glass


# create 10 redshift shells between z=0 and z=1

def add_galaxies(z,dndz, shells):
    """Partition the galaxy density function by a sequence of windows.

    Parameters
    ----------
    z : numpy.ndarray
        Array of redshift bins. If ``dndz`` is multi-dimensional, its last axis 
        must agree with ``z``.
    dndz : numpy.ndarray
        The galaxy density function to be partitioned.
    shells : Sequence[RadialWindow]
        Ordered sequence of window functions. Defines the redshift shells between 
        ``z = 0`` and ``z = 1``.

    Returns
    -------
    numpy.ndarray
        The partitioned galaxy number density, with shape matching ``dndz`` except 
        along the axis corresponding to ``z``.
    """
    # constant galaxy density distribution
    ngal = glass.partition(z, dndz, shells)
    return ngal 
