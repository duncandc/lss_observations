# -*- coding: utf-8 -*-

"""
callable stellar mass functions from the literature
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np

from astropy.table import Table
from astropy.modeling.models import custom_model
#from scipy.special import gammaincc, gammaincinv
from mpmath import gammainc

__all__ = ['Blanton_2003_phi']

class Blanton_2003_phi(object):
    """
    stellar mass function from Blanton et al. (2003)
    """
    def __init__(self, band='r', **kwargs):
        """
        """

        self.littleh = 1.0

        # parameters from table #2
        if band == 'u':
            self.phi0   = 3.05 * 10**(-2)
            self.x0     = -17.93
            self.alpha0 = -0.92
        elif band == 'g':
            self.phi0   = 2.18 * 10**(-2)
            self.x0     = -19.39
            self.alpha0 = -0.89
        elif band == 'r':
            self.phi0   = 1.49 * 10**(-2)
            self.x0     = -20.44
            self.alpha0 = -1.05
        elif band == 'i':
            self.phi0   = 1.47 * 10**(-2)
            self.x0     = -20.82
            self.alpha0 = -1.00
        elif band == 'z':
            self.phi0   = 1.35 * 10**(-2)
            self.x0     = -21.18
            self.alpha0 = -1.08
        else:
            msg = ('band not recognized.  `band` must be one of [u,g,r,i,z].')
            raise ValueError(msg)

        # define components of double Schechter function
        s = Mag_Schechter(phi0=self.phi0, x0=self.x0, alpha=self.alpha0)

        # create model
        self.s = s


    def __call__(self, mag):
        """
        stellar mass function from Blanton et al. (2003).

        Parameters
        ----------
        mag : array_like
            Absolute magnitude in units, Mag = Mag - 5log(h)

        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """

        return self.s(mag)


@custom_model
def Mag_Schechter(x, phi0=0.001, x0=-20.0, alpha=-1.0):
    """
    log schecter x function
    """
    x = np.asarray(x)
    x = x.astype(float)
    norm = (2.0/5.0)*phi0*np.log(10.0)
    val = norm*(10.0**(0.4*(x0-x)))**(alpha+1.0)*np.exp(-10.0**(0.4*(x0-x)))
    return val
