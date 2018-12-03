# -*- coding: utf-8 -*-

"""
callable stellar mass functions from the literature
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np

from astropy.table import Table
from astropy.io import ascii
from astropy.modeling.models import custom_model
#from scipy.special import gammaincc, gammaincinv
from mpmath import gammainc
from astro_utils.schechter_functions import MagSchechter

# set location of tabvulated data
import os
filepath = os.path.dirname(__file__)
filepath = os.path.join(filepath,'phi_measurements/')

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
            filename = 'lumfunc-u.sample10ubright15.dat'
            col_names = ['absolute_magnitude', 'phi', 'sigma_phi']
            self.data = ascii.read(filepath+filename, format='ascii', names=col_names)
            self.phi0   = 3.05 * 10**(-2)
            self.x0     = -17.93
            self.alpha0 = -0.92
        elif band == 'g':
            filename = 'lumfunc-g.sample10gbright15.dat'
            col_names = ['absolute_magnitude', 'phi', 'sigma_phi']
            self.data = ascii.read(filepath+filename, format='ascii', names=col_names)
            self.phi0   = 2.18 * 10**(-2)
            self.x0     = -19.39
            self.alpha0 = -0.89
        elif band == 'r':
            filename = 'lumfunc-r.sample10bbright15.dat'
            col_names = ['absolute_magnitude', 'phi', 'sigma_phi']
            self.data = ascii.read(filepath+filename, names=col_names)
            self.phi0   = 1.49 * 10**(-2)
            self.x0     = -20.44
            self.alpha0 = -1.05
        elif band == 'i':
            filename = 'lumfunc-i.sample10ibright15.dat'
            col_names = ['absolute_magnitude', 'phi', 'sigma_phi']
            self.data = ascii.read(filepath+filename, format='ascii', names=col_names)
            self.phi0   = 1.47 * 10**(-2)
            self.x0     = -20.82
            self.alpha0 = -1.00
        elif band == 'z':
            filename = 'lumfunc-z.sample10zbright15.dat'
            col_names = ['absolute_magnitude', 'phi', 'sigma_phi']
            self.data = ascii.read(filepath+filename, format='ascii', names=col_names)
            self.phi0   = 1.35 * 10**(-2)
            self.x0     = -21.18
            self.alpha0 = -1.08
        else:
            msg = ('band not recognized.  `band` must be one of [u,g,r,i,z].')
            raise ValueError(msg)

        # define components of double Schechter function
        s = MagSchechter(phi0=self.phi0, M0=self.x0, alpha=self.alpha0)

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

    def number_density(self, a, b):
        """
        """
        return self.s.number_density(a,b)

