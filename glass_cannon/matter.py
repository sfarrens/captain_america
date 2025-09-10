import glass
import camb

def run_ln_fields(shells):

    fields = glass.lognormal_fields(shells)

    return fields

def run_discretized_cls(prev_cls, nside, lmax):

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(prev_cls, nside=nside, lmax=lmax, ncorr=3)

    return cls

def run_solve_gauss_spectra(fields, cls):

    gls = glass.solve_gaussian_spectra(fields, cls)

    return gls

def run_generate(fields, gls, nside, rng):

    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)

    return matter