import glass
import numpy as np

#doesnt work - not a package
from glass_cannon.matter import run_ln_fields

def test_lognormal_fields(shells=glass.linear_windows([0., 0.04720758, 0.09549701, 0.14497038, 0.19573705, 0.24791426,
    0.30162785, 0.35701293, 0.41421482, 0.47338997, 0.53470702, 0.59834801,
    0.66450976, 0.73340537, 0.80526592, 0.88034248, 0.95890821, 1.04126096])):

    #expected output
    expected_fields = glass.lognormal_fields(glass.linear_windows([0., 0.04720758, 0.09549701, 0.14497038, 0.19573705, 0.24791426,
    0.30162785, 0.35701293, 0.41421482, 0.47338997, 0.53470702, 0.59834801,
    0.66450976, 0.73340537, 0.80526592, 0.88034248, 0.95890821, 1.04126096]))

    fields = run_ln_fields(shells)

    #eg assert if x = x or if x is true or false
    assert expected_fields == fields

    



