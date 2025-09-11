import numpy as np
import emcee
from scipy.stats import multivariate_normal

def ln_posterior(x, likelihood, prior):
    """Compute the log posterior."""
    x = np.atleast_2d(x)
    ln_posterior = likelihood + prior.logpdf(x)
    return ln_posterior

def run_MCMC(likelihood, nchains, prior, samples_per_chain, nburn, nparams=3):
    
    pos = prior.rvs(nchains)

    sampler = emcee.EnsembleSampler(
        nwalkers=nchains,
        ndim=nparams,
        log_prob_fn=ln_posterior,
        vectorize=True,
        args=[likelihood, prior],
    )

    sampler.run_mcmc(pos, samples_per_chain, progress=True)

    samples = np.ascontiguousarray(sampler.chain[:, nburn:, :])
    log_prob = np.ascontiguousarray(sampler.lnprobability[:, nburn:])
    flat_samples, flat_weights = np.unique(
        sampler.get_chain(flat=True), axis=0, return_counts=True
    )
    flat_log_prob, _ = np.unique(
        sampler.get_log_prob(flat=True), axis=0, return_counts=True
    )

    samples = np.ascontiguousarray(sampler.chain[:, nburn:, :])
    log_prob = np.ascontiguousarray(sampler.lnprobability[:, nburn:])

    return flat_samples, flat_weights, flat_log_prob, samples, log_prob