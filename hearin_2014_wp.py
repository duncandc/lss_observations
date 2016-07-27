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

__all__ = ['hearin_2014_wp']
__author__=['Duncan Campbell']

def hearin_2014_wp(mstar_thresh=10**9.49, sample='all'):
    """
    projected two point correlation function measurements from Hearin et al. 2014
    
    http://arxiv.org/abs/1310.6747
    
    Parameters
    ----------
    mstar_thresh : float
         minimum stellar mass of the threshold sample in :math:`h^{-2}M_{\odot}`
         e.g. 9.49, 9.89, 10.29 (converted to h=1).
    
    sample : string
        string indicating sample used in the wp calculation:
        e.g. 'all', 'red', 'blue'.
    
    Returns
    -------
    measurement : numpy.ndarray
        array of shape (2,15), where the first row is rp in :math:`h^-1` Mpc, and 
        the second row is wp in :math:`h^-1` Mpc.
    
    err : numpy.array
        error on the wp measurement
    """
    
    littleh=0.7
    
    #get files for specified sample
    if sample == 'all':
        filename = 'table_1.dat'
    elif sample == 'red':
        filename = 'table_3.dat'
    elif sample == 'blue':
        filename = 'table_2.dat'
    else:
        msg = ("sample not recognized.")
        raise ValueError(msg)
    
    #what are the mass thresholds in h=1?
    mstar_thresholds = np.array([10.0**9.8, 10.0**10.2, 10.0**10.6]) * littleh**2.0
    mstar_thresholds = np.log10(mstar_thresholds)
    
    #find nearest to the input value
    mstar_thresh = np.log10(mstar_thresh)
    mask = np.isclose(mstar_thresh, mstar_thresholds, atol=0.01)
    
    if np.any(mask):
        mstar_thresh = mstar_thresholds[mask]
    else:
        msg = ("mass threshold not with 0.01 dex of an available threshold.")
        raise ValueError(msg)
    
    #get files for specified stellar mass bin
    if mask[0]:
        column = 1
    elif mask[1]:
        column = 3
    elif mask[2]:
        column = 5
    else:
        msg = ("requested mass threshold not available.")
        raise ValueError(msg)
    
    #read in data
    filepath = os.path.dirname(__file__)
    filepath = os.path.join(filepath,'wp_measurements/hearin_2014_data/')
    data = ascii.read(filepath+filename)
    
    rp = np.array(data.columns[0])
    wp = np.array(data.columns[column])
    sigma = np.array(data.columns[column+1])
    
    measurement = np.vstack((rp,wp))
    
    #convert to h=1
    measurement[0] = measurement[0,:]*littleh
    measurement[0] = measurement[1,:]*littleh
    sigma = sigma*littleh
    
    return measurement, sigma
    
    