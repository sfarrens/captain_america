import glass
import numpy as np 

# create 10 redshift shells between z=0 and z=1

def add_galaxies(z,dndz, shells, zb):
    # constant galaxy density distribution
    ngal = glass.partition(z, dndz, shells)
    return ngal 
