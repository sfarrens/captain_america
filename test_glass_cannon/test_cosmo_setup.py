import sys
sys.path.append("/home/ppxjf3/repo/captain_america/glass_cannon/")
from cosmo_setup import set_cosmo
from cosmo_setup import make_bkg
from cosmo_setup import make_cosmo
from cosmo_setup import make_cosmology_class

"""
testing function to set up the cosmology for Glass simulation    
"""

def test_H0(h=0.7, Oc = 0.25, Ob = 0.05):
    assert set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05).H0 == 70.0

def test_Oc(h=0.7, Oc = 0.25, Ob = 0.05):
    assert set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05).omegac == 0.25
    
def test_Ob(h=0.7, Oc = 0.25, Ob = 0.05):
    assert set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05).omegab == 0.05
    
def test_background_is_made(h=0.7, Oc = 0.25, Ob = 0.05):
    assert make_bkg(set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05)).Params.H0 == 70.0

def test_background_is_made(h=0.7, Oc = 0.25, Ob = 0.05):
    assert make_bkg(set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05)).Params.omegac == 0.25
    
def test_background_is_made(h=0.7, Oc = 0.25, Ob = 0.05):
    assert make_bkg(set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05)).Params.omegab == 0.05

def test_cosmo_object(h=0.7, Oc = 0.25, Ob = 0.05):
    assert make_cosmo(make_bkg(set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05))).params.H0 == 70.0

def test_cosmo_object(h=0.7, Oc = 0.25, Ob = 0.05):
    assert make_cosmo(make_bkg(set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05))).params.omegac == 0.25
    
def test_cosmo_object(h=0.7, Oc = 0.25, Ob = 0.05):
    assert make_cosmo(make_bkg(set_cosmo(h=0.7, Oc = 0.25, Ob = 0.05))).params.omegab == 0.05

def test_all(h=0.7, Oc = 0.25, Ob = 0.05):
    result, extra = make_cosmology_class(h=0.7, Oc = 0.25, Ob = 0.05)
    assert result.params.H0 == 70.0

def test_all(h=0.7, Oc = 0.25, Ob = 0.05):
    result, extra = make_cosmology_class(h=0.7, Oc = 0.25, Ob = 0.05)
    assert result.params.omegac == 0.25
    
def test_all(h=0.7, Oc = 0.25, Ob = 0.05):
    result, extra = make_cosmology_class(h=0.7, Oc = 0.25, Ob = 0.05)
    assert result.params.omegab == 0.05
