# Duncan Campbell
# Yale University
# April, 2016

"""
projected two point correlation function measurements from Campbell 2016 
"""

from __future__ import print_function, division
from astropy.table import Table
from astropy.io import ascii
import os
import numpy as np

__all__ = ['campbell_2016_wp']
__author__=['Duncan Campbell']

def campbell_2016_wp(min_mstar=10**9.5, max_mstar=10**10.0,
                     method='theta_weights', sample='all'):
    """
    projected two point correlation function measurements from Campbell et al. 2016
    
    Parameters
    ----------
    min_mstar : float
         minimum stellar mass of bin in :math:`h^{-2}M_{\odot}`
    
    max_mstar : float
        maximum stellar mass of bin in :math:`h^{-2}M_{\odot}`
    
    method : string
        string indicating method used compensate for fiber collisions in 
        the wp calculation: 'nearest_neighbor', 'theta_weights'
    
    sample : string
        all, red, blue
    
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
    if method == 'nearest_neighbor':
        filenames = ['wp_data_nearest_neighbor_all_11.0_11.5_0.npy',
                     'wp_data_nearest_neighbor_all_10.0_10.5_0.npy',
                     'wp_data_nearest_neighbor_all_10.5_11.0_0.npy',
                     'wp_data_nearest_neighbor_all_9.5_10.0_0.npy',
                     'wp_data_nearest_neighbor_blue_11.0_11.5_0.npy',
                     'wp_data_nearest_neighbor_blue_10.0_10.5_0.npy',
                     'wp_data_nearest_neighbor_blue_10.5_11.0_0.npy',
                     'wp_data_nearest_neighbor_blue_9.5_10.0_0.npy',
                     'wp_data_nearest_neighbor_red_11.0_11.5_0.npy',
                     'wp_data_nearest_neighbor_red_10.0_10.5_0.npy',
                     'wp_data_nearest_neighbor_red_10.5_11.0_0.npy',
                     'wp_data_nearest_neighbor_red_9.5_10.0_0.npy']
    elif method == 'theta_weights':
        filenames = ['wp_data_no_collisions_all_11.0_11.5_0.npy',
                     'wp_data_no_collisions_all_10.0_10.5_0.npy',
                     'wp_data_no_collisions_all_10.5_11.0_0.npy',
                     'wp_data_no_collisions_all_9.5_10.0_0.npy',
                     'wp_data_no_collisions_blue_11.0_11.5_0.npy',
                     'wp_data_no_collisions_blue_10.0_10.5_0.npy',
                     'wp_data_no_collisions_blue_10.5_11.0_0.npy',
                     'wp_data_no_collisions_blue_9.5_10.0_0.npy',
                     'wp_data_no_collisions_red_11.0_11.5_0.npy',
                     'wp_data_no_collisions_red_10.0_10.5_0.npy',
                     'wp_data_no_collisions_red_10.5_11.0_0.npy',
                     'wp_data_no_collisions_red_9.5_10.0_0.npy']
    else:
        msg = ("method not recognized.")
        raise ValueError(msg)
    
    if sample=='all':
        filenames = filenames[:4]
    elif sample=='blue':
        filenames = filenames[4:8]
    elif sample=='red':
        filenames = filenames[8:]
    else:
        msg = ("sample not recognized.")
        raise ValueError(msg)
    
    min_mstar = np.log10(min_mstar)
    max_mstar = np.log10(max_mstar)
    mass_bin = (min_mstar,max_mstar)
    
    #get files for specified stellar mass bin
    if mass_bin==(9.5,10.0):
        filename = filenames[3]
    elif mass_bin==(10.0,10.5):
        filename = filenames[2]
    elif mass_bin==(10.5,11.0):
        filename = filenames[1]
    elif mass_bin==(11.0,11.5):
        filename = filenames[0]
    else:
        msg = ("requested mass bin not available.")
        raise ValueError(msg)
    
    #read in data
    filepath = os.path.dirname(__file__)
    filepath = os.path.join(filepath,'wp_measurements/campbell_2016_data/')
    wp_data = np.load(filepath+filename)
    
    
    rp = np.array(wp_data[0])
    wp = np.array(wp_data[1])
    
    measurement = np.vstack((rp,wp))
    
    return measurement
    