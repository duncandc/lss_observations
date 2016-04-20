# Duncan Campbell
# Yale University
# April, 2016

"""
projected two point correlation function measurements from Zehavi et al. 2011
"""

from __future__ import print_function, division
from astropy.table import Table
from astropy.io import ascii
import os
import numpy as np

from observables_template import *

__all__ = ['zehavi_2011_wp']
__author__=['Duncan Campbell']


def zehavi_2011_wp(Mr_min = -18.0, Mr_max = -17.0, sample='all'):
    """
    projected two point correlation function measurements from Zehavi et al. 2011
    
    http://arxiv.org/abs/1005.2413
    
    covariance matrices available at: http://astroweb.cwru.edu/izehavi/dr7_covar/
    
    Parameters
    ----------
    Mr_min : float
         minimum magnitude of bin.  To retrieve threshold samples set to None.
    
    Mr_max : float
        maximum magnitude of bin
    
    sample : string
        'all', 'red', 'blue'
    
    Returns
    -------
    measurement : numpy.ndarray
        array of shape (2,13), where the first row is rp in :math:`h^-1` Mpc, and 
        the second row is wp in :math:`h^-1` Mpc.
    
    covariance : numpy.matrix
        matrix of shape (13,13) of covariances between the ith and jth rp
        measurments of wp 
    """
    
    filepath = os.path.dirname(__file__)
    filepath = os.path.join(filepath,'wp_measurements/zehavi_2011_data/')
    
    #is this a threshold sample
    if Mr_min == None: 
        threshold=True
        Mr_max = float(Mr_max)
        Mbin = (None,Mr_max)
    else: 
        Mr_max = float(Mr_max)
        Mr_min = float(Mr_min)
        Mbin = (float(Mr_min),float(Mr_max))
    threshold=False
    
    #get files for specified sample
    if (sample == 'all') & (not threshold):
        filepath = filepath = os.path.join(filepath,'table7/')
        cov_filenames = ['wp_covar_23.0_22.0.dat',
                         'wp_covar_22.0_21.0.dat',
                         'wp_covar_21.0_20.0.dat',
                         'wp_covar_20.0_19.0.dat',
                         'wp_covar_19.0_18.0.dat',
                         'wp_covar_18.0_17.0.dat']
        wp_filename = 'table7.dat'
    elif (sample == 'all') & (threshold):
        filepath = filepath = os.path.join(filepath,'table8/')
        cov_filenames = ['wp_covar_22.0.dat',
                         'wp_covar_21.5.dat',
                         'wp_covar_21.0.dat',
                         'wp_covar_20.5.dat',
                         'wp_covar_20.0.dat',
                         'wp_covar_19.5.dat',
                         'wp_covar_19.0.dat',
                         'wp_covar_18.5.dat',
                         'wp_covar_18.0.dat']
        wp_filename = 'table8.dat'
    elif (sample == 'blue') & (not threshold):
        filepath = filepath = os.path.join(filepath,'table9/')
        cov_filenames = ['wp_covar_23.0_22.0_mblue.dat',
                         'wp_covar_22.0_21.0_mblue.dat',
                         'wp_covar_21.0_20.0_mblue.dat',
                         'wp_covar_20.0_19.0_mblue.dat',
                         'wp_covar_19.0_18.0_mblue.dat',
                         'wp_covar_18.0_17.0_mblue.dat']
        wp_filename = 'table9.dat'
    elif (sample == 'red') & (not threshold):
        filepath = filepath = os.path.join(filepath,'table10/')
        cov_filenames = ['wp_covar_23.0_22.0_mred.dat',
                         'wp_covar_22.0_21.0_mred.dat',
                         'wp_covar_21.0_20.0_mred.dat',
                         'wp_covar_20.0_19.0_mred.dat',
                         'wp_covar_19.0_18.0_mred.dat',
                         'wp_covar_18.0_17.0_mred.dat']
        wp_filename = 'table10.dat'
    else:
        msg = ("sample and/or threshold combination not available.")
        raise ValueError(msg)
    
    if Mbin == (-23.0,-22.0): #binned samples
        cov_filename = cov_filenames[0]
        wp_col = 1
    elif Mbin == (-22.0,-21.0):
        cov_filename = cov_filenames[1]
        wp_col = 3
    elif Mbin == (-21.0,-20.0):
        cov_filename = cov_filenames[2]
        wp_col = 5
    elif Mbin == (-20.0,-19.0):
        cov_filename = cov_filenames[3]
        wp_col = 7
    elif Mbin == (-19.0,-18.0):
        cov_filename = cov_filenames[4]
        wp_col = 9
    elif Mbin == (-18.0,-17.0):
        cov_filename = cov_filenames[5]
        wp_col = 11
    elif Mbin == (None,-22.0): #threshold samples
        cov_filename = cov_filenames[0]
        wp_col = 1
    elif Mbin == (None,-21.5):
        cov_filename = cov_filenames[1]
        wp_col = 3
    elif Mbin == (None,-21.0):
        cov_filename = cov_filenames[2]
        wp_col = 5
    elif Mbin == (None,-20.5):
        cov_filename = cov_filenames[3]
        wp_col = 7
    elif Mbin == (None,-20.0):
        cov_filename = cov_filenames[4]
        wp_col = 9
    elif Mbin == (None,-19.5):
        cov_filename = cov_filenames[5]
        wp_col = 11
    elif Mbin == (None,-19.0):
        cov_filename = cov_filenames[6]
        wp_col = 13
    elif Mbin == (None,-18.5):
        cov_filename = cov_filenames[7]
        wp_col = 15
    elif Mbin == (None,-18.0):
        cov_filename = cov_filenames[8]
        wp_col = 17
    else:
        msg = ("sample bin is not available")
        raise ValueError(msg)
    
    #open relavent files
    #read in data wp data
    wp_data = ascii.read(filepath+wp_filename, delimiter  = '\s')
    rp = wp_data.columns[0]
    wp = wp_data.columns[wp_col]
    measurement = np.vstack((rp,wp))
    
    #read in covariance matrix
    cov_file= open(filepath+cov_filename)
    cov_data = cov_file.read()
    cov_data = cov_data.split()
    
    N = len(wp_data)
    cov = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            cov[i,j] = float(cov_data[int(i*N+j)])
    
    cov = np.matrix(cov)
    
    return measurement, cov
    