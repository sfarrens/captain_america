import glass
import numpy as np 

# create 10 redshift shells between z=0 and z=1

def add_galaxies(z,dndz):
    zb = [0. ,     0.04720758, 0.09549701, 0.14497038 ,0.19573705 ,0.24791426,
    0.30162785, 0.35701293, 0.41421482, 0.47338997, 0.53470702, 0.59834801,
    0.66450976 ,0.73340537, 0.80526592 ,0.88034248, 0.95890821,1.04126096]

    shells = glass.linear_windows(zb)
    # constant galaxy density distribution
    ngal = glass.partition(z, dndz, shells)
    return ngal 