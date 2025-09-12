# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

def angular_power_spectrum(pars,lmax,zb):
    
    # linear radial window functions
    shells = glass.tophat_windows(zb, dz=0.001)
    
    # compute the angular matter power spectra of the shells with CAMB
    return glass.ext.camb.matter_cls(pars, lmax, shells), shells