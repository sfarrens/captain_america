# Welcome Avengers!🦸

## Meet the team

### The boss: Sam Farrens
### Sidekick 1: Gavin
### Sidekick 2: Jen
### Sidekick 3: Tobias
### Sidekick 4: Jade

Fighting against code mal-practise! 💪

## Project: Large-scale structure with radio and cluster abundance

### Motivation: 
Extracting precise cosmological information from next-generation surveys requires inference techniques that can robustly handle complex systematics and intractable likelihoods.

### Scope idea:
This project would aim to develop and compare state-of-the-art inference methods by applying them to distinct, simulated cosmological datasets. The primary goal is to quantify the performance of Simulation-Based Inference (SBI) against traditional Hierarchical Bayesian Modelling (using MCMC) for cosmological parameter estimation.
To achieve this, the project could be structured around one or two parallel analysis pipelines feeding into a unified inference framework:

1. 21cm Radio Cosmology: Leveraging existing expertise in radio data simulation, one could generate mock 21cm power spectra. This dataset will incorporate realistic challenges, including instrumental beam effects and residuals from foreground removal techniques (e.g., GPR, FastICA). This allows one to test how each inference method handles highly correlated, non-Gaussian systematics characteristic of radio cosmology.

2. Weak Lensing & Galaxy Clusters: Drawing on expertise in large-scale structure, one could use simulated dark matter halo catalogs to produce mock observables such as galaxy cluster counts. This probe could implement a hierarchical model that explicitly accounts for selection effects and mass-observable uncertainties, providing a different set of statistical challenges.

The core of the project can be the implementation of our comparative inference engine, accelerated with JAX. Incorporating the use SBI, i.e. neural density estimation, to learn the posterior directly from forward simulations could allow to compare the resulting parameter constraints ($\Omega_m$​, $\sigma_8$​, etc.) against those derived from a hierarchical Bayesian MCMC analysis.
