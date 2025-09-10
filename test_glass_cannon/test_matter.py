import glass
import glass.ext.camb
import numpy as np
import camb

from glass_cannon.matter import run_ln_fields
from glass_cannon.matter import run_discretized_cls

def test_run_ln_fields():

    shells=glass.linear_windows([0., 0.04720758, 0.09549701, 0.14497038, 0.19573705, 0.24791426,
    0.30162785, 0.35701293, 0.41421482, 0.47338997, 0.53470702, 0.59834801,
    0.66450976, 0.73340537, 0.80526592, 0.88034248, 0.95890821, 1.04126096])

    #expected output
    expected_fields = glass.lognormal_fields(shells)

    fields = run_ln_fields(shells)

    #eg assert if x = x or if x is true or false
    assert expected_fields == fields

def test_run_discretized_cls():

    nside = lmax = 128

    h = 0.7
    Oc = 0.25
    Ob = 0.05

    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both,
    )

    shells=glass.linear_windows([0., 0.04720758, 0.09549701, 0.14497038, 0.19573705, 0.24791426,
    0.30162785, 0.35701293, 0.41421482, 0.47338997, 0.53470702, 0.59834801,
    0.66450976, 0.73340537, 0.80526592, 0.88034248, 0.95890821, 1.04126096])

    prev_cls = glass.ext.camb.matter_cls(pars, lmax, shells)

    expected_cls = glass.discretized_cls(prev_cls, nside=nside, lmax=lmax, ncorr=3)

    cls = run_discretized_cls(prev_cls, nside, lmax)

    assert all(np.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(expected_cls, cls))



