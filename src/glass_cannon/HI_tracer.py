import numpy as np
import glass


def b_HI(z):
    """
    https://arxiv.org/pdf/2107.14057

    """
    return 0.667 + 0.178*z + 0.050*z**2 

def T_HI_bar(z):
    """ 
    https://arxiv.org/pdf/2107.14057
    """
    return 0.0559 + 0.2324*z - 0.0241*z**2

def convert_DM_to_HI(shells, matter):

    """

    HI temperature fluctuation field - doi:10.1093/mnras/stz1916 - Cunnington et al. 2019
    """
    hi_temperature_fields = []
    for shell, delta_m in zip(shells, matter):
        z = shell.zeff
        #need mean not boundary
        
        # Apply bias to overdensity
        delta_HI = b_HI(z) * delta_m
        
        # Map overdensity to brightness temperature
        #  - this is the observed temperature, not just fluctuations around the mean
        #  - so the 1 + fluctuation factor is included rather than multiplying just be the fluctuation 
        #       as is the case in equation 3 in Cunnington et al. 2019
        T_HI = T_HI_bar(z) * (1 + delta_HI)
        
        hi_temperature_fields.append(T_HI) #is this the correct way to combine?

    return hi_temperature_fields