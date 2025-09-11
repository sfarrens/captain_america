import glass


# create 10 redshift shells between z=0 and z=1

def add_galaxies(z,dndz, shells):
    """Partition a the galaxy density function by a sequence of windows.

    Parameters
    ----------
        z (ndarray[tuple[int, ...], dtype[float64]]): List of redshifts bins. If dndz is multi-dimensional, its last axis must agree with z.
        dndz (ndarray[tuple[int, ...], dtype[float64]]): The function to be partitioned. 
        shells  (Sequence[RadialWindow]): Ordered sequence of window functions to be combined. The redshift shells between z=0 and z=1.

    Returns
    -------
        ngal (ndarray[tuple[int, ...], dtype[float64]]): _description_
    """
    # constant galaxy density distribution
    ngal = glass.partition(z, dndz, shells)
    return ngal 
