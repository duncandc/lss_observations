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

def hearin_2014_wp(mstar_thresh=10**9.8, sample='all'):
    """
    projected two point correlation function measurements from Hearin et al. 2014
    
    http://arxiv.org/abs/1310.6747
    
    Parameters
    ----------
    mstar_thresh : float
         minimum stellar mass of the threshold sample in :math:`h^{-2}M_{\odot}`
    
    sample : string
        string indicating sample used in the wp calculation:
        e.g. 'all', 'red', 'blue'.
    
    Returns
    -------
    measurement : numpy.ndarray
        array of shape (2,14), where the first row is rp in :math:`h^-1` Mpc, and 
        the second row is wp in :math:`h^-1` Mpc.
    
    err : numpy.matrix
        err on measurement
    """
    
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
    
    mstar_thresh = np.log10(mstar_thresh)
    
    #get files for specified stellar mass bin
    if mstar_thresh==9.8:
        column = 1
    elif mstar_thresh==10.2:
        column = 3
    elif mstar_thresh==10.6:
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
    
    return measurement, sigma
    
    