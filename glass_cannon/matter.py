import glass
import camb

def run_ln_fields(shells):

    fields = glass.lognormal_fields(shells)

    return fields

def run_discretized_cls(cls, nside, lmax):

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)

    return cls



