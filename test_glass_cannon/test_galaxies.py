from glass_cannon.galaxies import add_galaxies 
import glass
import numpy as np

""" Test the galaxy creation function """


def test_galaxy_creation():
    """
    Test the galaxy creation function : add_galaxies(z,dndz, shells).
    Parameters
    ----------
    No input parameters.
    Returns 
    ----------
    No return parameters.
    
    """
    #something simple to test
    zb = [0. ,     0.04720758, 0.09549701, 0.14497038 ,0.19573705 ,0.24791426,
    0.30162785, 0.35701293, 0.41421482, 0.47338997, 0.53470702, 0.59834801,
    0.66450976 ,0.73340537, 0.80526592 ,0.88034248, 0.95890821,1.04126096]

    shells = glass.linear_windows(zb) #-> 10 redshift shells between z=0 and z=1
    # constant galaxy density distribution
    z = np.linspace(0.0, 1.0, 100)
    dndz = np.full_like(z, 0.01)
    ngal = glass.partition(z, dndz, shells) # replace this 

    assert all(ngal  == add_galaxies(z,dndz, shells)), 'AAAAAAH NOOOOO! WRONG ! (but not sure why tho)'
