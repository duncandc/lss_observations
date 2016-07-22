# Duncan Campbell
# Yale University
# April, 2016

"""
projected two point correlation function measurements from Yang et al. 2012
"""

from __future__ import print_function, division
from astropy.table import Table
from astropy.io import ascii
import os
import numpy as np

__all__ = ['yang_2012_wp']
__author__=['Duncan Campbell']

def yang_2012_wp(min_mstar=10**9.0, max_mstar=10**9.5, sample='Volume1'):
    """
    projected two point correlation function measurements from Yang et al. 2012
    
    http://arxiv.org/abs/1110.1420
    
    Parameters
    ----------
    min_mstar : float
         minimum stellar mass of bin in :math:`h^{-2}M_{\odot}`
    
    max_mstar : float
        maximum stellar mass of bin in :math:`h^{-2}M_{\odot}`
    
    sample : string
        string indicating sample used in the wp calculation:
        e.g. 'Volume1', 'Volume2', 'Mass-limit'.
    
    Returns
    -------
    measurement : numpy.ndarray
        array of shape (2,14), where the first row is rp in :math:`h^-1` Mpc, and 
        the second row is wp in :math:`h^-1` Mpc.
    
    covariance : numpy.matrix
        matrix of shape (14,14) of covariances between the ith and jth rp
        measurments of wp 
    """
    
    #get files for specified sample
    if sample == 'Volume1':
        filenames = ['xi01.dat','xi02.dat','xi03.dat','xi04.dat','xi05.dat']
    elif sample == 'Volume2':
        filenames = ['xi06.dat','xi07.dat','xi08.dat','xi09.dat','xi10.dat']
    elif sample == 'Mass-limit':
        filenames = ['xi11.dat','xi12.dat','xi13.dat','xi14.dat','xi15.dat']
    else:
        msg = ("sample not recognized.")
        raise ValueError(msg)
    
    min_mstar = np.log10(min_mstar)
    max_mstar = np.log10(max_mstar)
    mass_bin = (min_mstar,max_mstar)
    
    #get files for specified stellar mass bin
    if mass_bin==(9.0,9.5):
        filename = filenames[0]
    elif mass_bin==(9.5,10.0):
        filename = filenames[1]
    elif mass_bin==(10.0,10.5):
        filename = filenames[2]
    elif mass_bin==(10.5,11.0):
        filename = filenames[3]
    elif mass_bin==(11.0,11.5):
        filename = filenames[4]
    else:
        msg = ("requested mass bin not available.")
        raise ValueError(msg)
    
    #read in data
    filepath = os.path.dirname(__file__)
    filepath = os.path.join(filepath,'wp_measurements/yang_2012_data/')
    wp_data = ascii.read(filepath+filename, data_start=1, data_end = 15)
    cov_data = ascii.read(filepath+filename, data_start=15, data_end = 29)
    
    rp = np.array(wp_data.columns[0])
    wp = np.array(wp_data.columns[1])
    sigma = np.array(wp_data.columns[2])
    
    measurement = np.vstack((rp,wp))
    
    #create covariance matrix
    cov = np.array([cov_data[c] for c in cov_data.columns])
    
    N = len(wp)
    for i in range(0,N):
        for j in range(0,N):
            cov[i,j] = wp[i]*wp[j]*cov[i,j]
    cov = np.matrix(cov)
    
    return measurement, cov
    
    