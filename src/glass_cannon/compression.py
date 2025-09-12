import numpy as np
from scipy.linalg import eigh

def do_CCA(sim_samples, data_to_compress, n_params=3):
    """function to compress the data using CCA

    Args:
        sim_samples (_type_): np.random.multivariate_normal(means, prior_cov, size=10000)
        data_to_compress (numpy array): data to do CCA on
        n_params (int, optional): number of parameters which are varied in simulation. Defaults to 3.
    """
    cca_cov = np.cov(sim_samples.T, data_to_compress.T)
    cp = cca_cov[:n_params,:n_params]
    cd = cca_cov[n_params:,n_params:]
    cpd = cca_cov[:n_params,n_params:]

    # This 'cl' can be understood as the projection of 'cp' to data vector space
    cl = cpd.T@np.linalg.inv(cp)@cpd

    # As seen in the paper, this generalized eigenvalue problem is equivalent to CCA
    # but is more numerical stable as 'cd' and 'cd-cl' are both invertible.
    # This problem is motivated as mutual information maximization under Gaussian linear model assumptions
    evals, evecs = eigh(cd, cd - cl)

    # In the context of the CCA, only min( dim(param), dim(data vector) ) components are real and the rest are noise. 
    evals = evals[::-1][:n_params]
    evecs = evecs[:,::-1][:,:n_params]
    
    return evals, evecs