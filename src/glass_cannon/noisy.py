import numpy as np

def init_cov(ndim):
    """Initialise random non-diagonal covariance matrix.

    Args:

        ndim: Dimension of Gaussian.

    Returns:

        cov: Covariance matrix of shape (ndim,ndim).

    """

    cov = np.zeros((ndim, ndim))
    diag_cov = np.ones(ndim) + np.random.randn(ndim) * 0.1
    np.fill_diagonal(cov, diag_cov)
    off_diag_size = 0.5
    for i in range(ndim - 1):
        cov[i, i + 1] = (
            (-1) ** i * off_diag_size * np.sqrt(cov[i, i] * cov[i + 1, i + 1])
        )
        cov[i + 1, i] = cov[i, i + 1]

    return cov

def add_noise(sim_data):
    """Add Gaussian noise to simulated data

    Args:
        sim_data (numpy array): simulated data you want to add noise to (3x3x129)
    """
    cov = init_cov(sim_data.shape[2])
    noisy_sim_data = np.empty_like(sim_data)
    
    for ii in range(sim_data.shape[0]):
        for jj in range(sim_data.shape[1]):
            noisy_sim_data[ii,jj] = np.random.multivariate_normal(sim_data[ii, jj], cov)
            
    return noisy_sim_data


    