import glass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
sys.path.append("/home/ppxjf3/repo/captain_america/glass_cannon/")

from cosmo_setup import make_cosmology_class
from Cls import angular_power_spectrum
import matter 
from galaxies import add_galaxies

# creating a numpy random number generator for sampling
rng = np.random.default_rng(seed=42)

def make_3D_galaxy_cube(zb, matter, ngal, rng, shells):
    # make a cube for galaxy number in redshift
    zcub = np.linspace(-zb[-1], zb[-1], 21)
    cube = np.zeros((zcub.size - 1,) * 3)

    # simulate and add galaxies in each matter shell to cube
    for i, delta_i in enumerate(matter):
        # simulate positions from matter density
        for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
            ngal[i],
            delta_i,
            rng=rng,
        ):
            # sample redshifts uniformly in shell
            gal_z = glass.redshifts(gal_count, shells[i], rng=rng)

            # add counts to cube
            z1 = gal_z * np.cos(np.deg2rad(gal_lon)) * np.cos(np.deg2rad(gal_lat))
            z2 = gal_z * np.sin(np.deg2rad(gal_lon)) * np.cos(np.deg2rad(gal_lat))
            z3 = gal_z * np.sin(np.deg2rad(gal_lat))
            indices, count = np.unique(
                np.searchsorted(zcub[1:], [z1, z2, z3]),
                axis=1,
                return_counts=True,
            )
            cube[*indices] += count
            
    return cube, zcub
"""
--------------------------
Define cosmological parameters:

h (float): hubble parameter 
Oc (float): critical density parameter
Ob (float): baryon density parameter
nside/lmax (int): size of box
----------------------------
"""

h = 0.7
Oc = 0.25
Ob = 0.05
nside = lmax = 128


cosmo, pars = make_cosmology_class(h=h, Oc=Oc, Ob=Ob)

# shells of 200 Mpc in comoving distance spacing
zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)

cls, shells = angular_power_spectrum(pars,lmax,zb)

fields = matter.run_ln_fields(shells)

cls = matter.run_discretized_cls(cls, nside, lmax)

# compute Gaussian spectra for lognormal fields from discretised spectra
gls = matter.run_solve_gauss_spectra(fields, cls)

# generator for lognormal matter fields
matter_ = matter.run_generate(fields, gls, nside, rng)

# constant galaxy density distribution
z = np.linspace(0.0, 1.0, 100)
dndz = np.full_like(z, 0.01)

ngal = add_galaxies(z,dndz, shells, zb)

cube, zcub = make_3D_galaxy_cube(zb, matter_, ngal, rng, shells)
       
""" PLOT THE FIGURE  """ 
# positions of grid cells of the cube
z = (zcub[:-1] + zcub[1:]) / 2
z1, z2, z3 = np.meshgrid(z, z, z)

# plot the galaxy distribution in pseudo-3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
norm = LogNorm(vmin=np.min(cube[cube > 0]), vmax=np.max(cube), clip=True)
for i in range(len(zcub) - 1):
    v = norm(cube[..., i])
    c = plt.cm.inferno(v)
    c[..., -1] = 0.2 * v
    ax.plot_surface(
        z1[..., i],
        z2[..., i],
        z3[..., i],
        rstride=1,
        cstride=1,
        facecolors=c,
        linewidth=0,
        shade=False,
        antialiased=False,
    )
fig.tight_layout()
plt.show()
