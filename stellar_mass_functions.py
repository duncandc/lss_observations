# -*- coding: utf-8 -*-

"""
callable stellar mass functions from the literature
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np

from astropy.table import Table
from astropy.modeling.models import custom_model

__all__ = ['LiWhite_2009_phi', 'Baldry_2011_phi', 'Yang_2012_phi','Tomczak_2014_phi']

class LiWhite_2009_phi(object):
    """
    stellar mass function from Li & White 2009, arXiv:0901.0706
    """
    def __init__(self, **kwargs):
        """
        """
        
        self.publication = ['arXiv:0901.0706']
        
        self.littleh = 1.0
        
        #parameters from table #1
        self.min_mstar1 = 8.0
        self.phi1 = 0.01465
        self.x1 = 9.6124
        self.alpha1 = -1.1309
        self.max_mstar1 = 9.33
        
        self.min_mstar2 = 9.33
        self.phi2 = 0.01327
        self.x2 = 10.3702
        self.alpha2 = -0.9004
        self.max_mstar2 = 10.67
        
        self.min_mstar3 = 10.67
        self.phi3 = 0.0044
        self.x3 = 10.7104
        self.alpha3 = -1.9918
        self.max_mstar3 = 12.0
        
        #used to build piecewise function
        @custom_model
        def interval(x,x1=0.0,x2=1.0):
            """
            return 1 if x is in the range (x1,x2] and 0 otherwise
            """
            x = np.array(x)
            mask = ((x<=x2) & (x>x1))
            result = np.zeros(len(x))
            result[mask]=1.0
            return result
        
        #define components of double Schechter function
        s1 = Log_Schechter(phi0=self.phi1, x0=self.x1, alpha=self.alpha1)*interval(x1=-np.inf,x2=self.max_mstar1)
        s2 = Log_Schechter(phi0=self.phi2, x0=self.x2, alpha=self.alpha2)*interval(x1=self.min_mstar2,x2=self.max_mstar2)
        s3 = Log_Schechter(phi0=self.phi3, x0=self.x3, alpha=self.alpha3)*interval(x1=self.min_mstar3,x2=np.inf)
        
        #create piecewise model
        self.s = s1 + s2 + s3
        
    
    def __call__(self, mstar):
        """
        stellar mass function from Li & White 2009, arXiv:0901.0706
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        #take log of stellar masses
        mstar = np.log10(mstar)
        
        return self.s(mstar)


class Baldry_2011_phi(object):
    """
    stellar mass function from Baldry et al. 2011, arXiv:1111.5707
    """
    
    def __init__(self, **kwargs):
        """
        """
        
        self.littleh = 0.7
        
        #parameters from figure 13
        self.phi1 = 3.96*10**(-3)
        self.x1 = 10.66
        self.alpha1 = -0.35
        
        self.phi2 = 0.79*10**(-3)
        self.x2 = self.x1
        self.alpha2 = -1.47
        
        #define components of double Schechter function
        s1 = Log_Schechter(phi0=self.phi1, x0=self.x1, alpha=self.alpha1)
        s2 = Log_Schechter(phi0=self.phi2, x0=self.x2, alpha=self.alpha2)
        
        #create piecewise model
        self.s = s1 + s2
        
        #data from table #1
        data_rows = [(6.25, 0.50,31.1*10**(-3), 21.6*10**(-3),9),
                     (6.75, 0.50,18.1*10**(-3), 6.6*10**(-3), 19),
                     (7.10, 0.20,17.9*10**(-3), 5.7*10**(-3), 18),
                     (7.30, 0.20,43.1*10**(-3), 8.7*10**(-3), 46),
                     (7.50, 0.20,31.6*10**(-3), 9.0*10**(-3), 51),
                     (7.70, 0.20,34.8*10**(-3), 8.4*10**(-3), 88),
                     (7.90, 0.20,27.3*10**(-3), 4.2*10**(-3), 140),
                     (8.10, 0.20,28.3*10**(-3), 2.8*10**(-3), 243),
                     (8.30, 0.20,23.5*10**(-3), 3.0*10**(-3), 282),
                     (8.50, 0.20,19.2*10**(-3), 1.2*10**(-3), 399),
                     (8.70, 0.20,18.0*10**(-3), 2.6*10**(-3), 494),
                     (8.90, 0.20,14.3*10**(-3), 1.7*10**(-3), 505),
                     (9.10, 0.20,10.2*10**(-3), 0.6*10**(-3), 449),
                     (9.30, 0.20,9.59*10**(-3), 0.55*10**(-3), 423),
                     (9.50, 0.20,7.42*10**(-3), 0.41*10**(-3), 340),
                     (9.70, 0.20,6.21*10**(-3), 0.37*10**(-3), 290),
                     (9.90, 0.20,5.71*10**(-3), 0.35*10**(-3), 268),
                     (10.10,0.20,5.51*10**(-3), 0.34*10**(-3), 260),
                     (10.30,0.20,5.48*10**(-3), 0.34*10**(-3), 259),
                     (10.50,0.20,5.12*10**(-3), 0.33*10**(-3), 242),
                     (10.70,0.20,3.55*10**(-3), 0.27*10**(-3), 168),
                     (10.90,0.20,2.41*10**(-3), 0.23*10**(-3), 114),
                     (11.10,0.20,1.27*10**(-3), 0.16*10**(-3), 60),
                     (11.30,0.20,0.338*10**(-3),0.085*10**(-3),16),
                     (11.50,0.20,0.042*10**(-3),0.030*10**(-3),2),
                     (11.70,0.20,0.021*10**(-3),0.021*10**(-3),1),
                     (11.90,0.20,0.042*10**(-3),0.030*10**(-3),2)]
        self.data_table = Table(rows=data_rows,
            names=('bin_center', 'bin_width', 'phi', 'err', 'N'),
            dtype=('f4', 'f4', 'f4', 'f4', 'i4'))
        
        self.data_table['bin_center'] = 10**self.data_table['bin_center']
        
        self.data_table['bin_center'] = self.data_table['bin_center']*self.littleh**2
        self.data_table['phi'] = self.data_table['phi']/self.littleh**3
        self.data_table['err'] = self.data_table['err']/self.littleh**3
    
    def __call__(self, mstar):
        """
        stellar mass function from Li & White 2009, arXiv:0901.0706
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        #convert from h=1 to h=0.7
        mstar = mstar / self.littleh**2
        
        #take log of stellar masses
        mstar = np.log10(mstar)
        
        #convert from h=0.7 to h=1.0
        return self.s(mstar) / self.littleh**3


class Yang_2012_phi(object):
    """
    stellar mass function from Yang et al. 2012, arXiv:1110.1420
    """
    
    def __init__(self, **kwargs):
        """
        """
        
        self.publication = ['arXiv:1110.1420']
        
        self.littleh = 1.0
        
        #parameters from appendix B
        self.phi1 = 0.0083635
        self.x1 = 10.673
        self.alpha1 = -1.117
        
        #define components of double Schechter function
        s1 = Log_Schechter(phi0=self.phi1, x0=self.x1, alpha=self.alpha1)
        
        #create piecewise model
        self.s = s1
        
        #data from table #6
        data_rows  = [(8.2, 3.7705, 1.5258, 0.9436, 0.7870, 2.8269, 1.2665, 3.0870, 1.6328, 0.9436, 0.7870, 2.1434, 1.3832, 0.6835, 0.9345, 0.0000, 0.0000, 0.6835, 0.9345),
                     (8.3, 3.4598, 0.7363, 1.2416, 0.4523, 2.2182, 0.5867, 2.1801, 0.5884, 0.6520, 0.3011, 1.5281, 0.4684, 1.2796, 0.5566, 0.5896, 0.3436, 0.6900, 0.4418),
                     (8.4, 4.1293, 0.5891, 1.1804, 0.2965, 2.9489, 0.4627, 2.7961, 0.4415, 0.5128, 0.2023, 2.2833, 0.3748, 1.3332, 0.4736, 0.6676, 0.2879, 0.6656, 0.2905),
                     (8.5, 3.6421, 0.5547, 0.9305, 0.2886, 2.7116, 0.3771, 2.4913, 0.3387, 0.4905, 0.1597, 2.0008, 0.2727, 1.1508, 0.3368, 0.4400, 0.1997, 0.7108, 0.2176),
                     (8.6, 3.3055, 0.4245, 0.8003, 0.2345, 2.5052, 0.2674, 2.2182, 0.2612, 0.3709, 0.1511, 1.8474, 0.2058, 1.0873, 0.2604, 0.4294, 0.1365, 0.6578, 0.1831),
                     (8.7, 3.1321, 0.3100, 0.8215, 0.1561, 2.3106, 0.2224, 2.1598, 0.2294, 0.3756, 0.1010, 1.7842, 0.1819, 0.9723, 0.1686, 0.4459, 0.1020, 0.5264, 0.1109),
                     (8.8, 3.0391, 0.2499, 0.8716, 0.1181, 2.1675, 0.1865, 1.8100, 0.1428, 0.3005, 0.0669, 1.5095, 0.1253, 1.2291, 0.1729, 0.5711, 0.0936, 0.6580, 0.1253),
                     (8.9, 2.7949, 0.2433, 0.8404, 0.1442, 1.9545, 0.1538, 1.7266, 0.1265, 0.2997, 0.0582, 1.4269, 0.1027, 1.0683, 0.1745, 0.5407, 0.1141, 0.5276, 0.0929),
                     (9.0, 3.1430, 0.1822, 0.9815, 0.1210, 2.1614, 0.1161, 1.9179, 0.0875, 0.3476, 0.0524, 1.5702, 0.0968, 1.2251, 0.1483, 0.6339, 0.0941, 0.5912, 0.0871),
                     (9.1, 3.1047, 0.2357, 1.0438, 0.1518, 2.0609, 0.1199, 1.8162, 0.1129, 0.3595, 0.0608, 1.4568, 0.0765, 1.2884, 0.1579, 0.6843, 0.1131, 0.6042, 0.0723),
                     (9.2, 2.9365, 0.1816, 1.0557, 0.1398, 1.8808, 0.0760, 1.6895, 0.1021, 0.3411, 0.0622, 1.3484, 0.0606, 1.2470, 0.1151, 0.7146, 0.0969, 0.5324, 0.0409),
                     (9.3, 2.8092, 0.1786, 1.0230, 0.1329, 1.7861, 0.0766, 1.5992, 0.0730, 0.3624, 0.0458, 1.2368, 0.0476, 1.2100, 0.1358, 0.6607, 0.1043, 0.5493, 0.0518),
                     (9.4, 2.8013, 0.0925, 1.0764, 0.0703, 1.7249, 0.0477, 1.6116, 0.0621, 0.4012, 0.0381, 1.2104, 0.0420, 1.1897, 0.0549, 0.6753, 0.0479, 0.5145, 0.0292),
                     (9.5, 2.5093, 0.1140, 1.0816, 0.0917, 1.4277, 0.0418, 1.4360, 0.0522, 0.4290, 0.0420, 1.0070, 0.0354, 1.0733, 0.0804, 0.6526, 0.0637, 0.4208, 0.0290),
                     (9.6, 2.3481, 0.1002, 1.0112, 0.0787, 1.3369, 0.0362, 1.3756, 0.0508, 0.4339, 0.0302, 0.9416, 0.0307, 0.9725, 0.0653, 0.5772, 0.0598, 0.3953, 0.0173),
                     (9.7, 2.0970, 0.0640, 1.0132, 0.0488, 1.0837, 0.0286, 1.2562, 0.0420, 0.4760, 0.0288, 0.7802, 0.0231, 0.8408, 0.0368, 0.5373, 0.0304, 0.3035, 0.0153),
                     (9.8, 1.9927, 0.0653, 1.0453, 0.0526, 0.9473, 0.0239, 1.2189, 0.0254, 0.5060, 0.0210, 0.7129, 0.0171, 0.7738, 0.0509, 0.5393, 0.0407, 0.2345, 0.0168),
                     (9.9, 1.8551, 0.0555, 1.0426, 0.0446, 0.8125, 0.0210, 1.1423, 0.0240, 0.5284, 0.0176, 0.6139, 0.0143, 0.7128, 0.0398, 0.5142, 0.0337, 0.1986, 0.0119),
                     (10.0, 1.7485, 0.0555, 1.0329, 0.0448, 0.7156, 0.0184, 1.1068, 0.0241, 0.5713, 0.0197, 0.5355, 0.0110, 0.6417, 0.0393, 0.4616, 0.0320, 0.1801, 0.0119),
                     (10.1, 1.6715, 0.0430, 1.0297, 0.0343, 0.6418, 0.0156, 1.0844, 0.0161, 0.5863, 0.0100, 0.4981, 0.0121, 0.5871, 0.0326, 0.4434, 0.0293, 0.1436, 0.0066),
                     (10.2, 1.6340, 0.0417, 1.0560, 0.0331, 0.5780, 0.0142, 1.0963, 0.0122, 0.6465, 0.0088, 0.4498, 0.0082, 0.5377, 0.0346, 0.4095, 0.0289, 0.1282, 0.0087),
                     (10.3, 1.5273, 0.0419, 1.0368, 0.0355, 0.4905, 0.0113, 1.0326, 0.0104, 0.6491, 0.0078, 0.3835, 0.0069, 0.4947, 0.0356, 0.3877, 0.0318, 0.1071, 0.0064),
                     (10.4, 1.3308, 0.0339, 0.9331, 0.0266, 0.3978, 0.0105, 0.9275, 0.0104, 0.6129, 0.0081, 0.3146, 0.0067, 0.4033, 0.0273, 0.3202, 0.0232, 0.0831, 0.0057),
                     (10.5, 1.0870, 0.0292, 0.7817, 0.0237, 0.3052, 0.0084, 0.7882, 0.0095, 0.5383, 0.0066, 0.2499, 0.0051, 0.2988, 0.0225, 0.2435, 0.0193, 0.0553, 0.0045),
                     (10.6, 0.8692, 0.0265, 0.6337, 0.0210, 0.2354, 0.0077, 0.6445, 0.0097, 0.4523, 0.0069, 0.1922, 0.0047, 0.2247, 0.0194, 0.1814, 0.0164, 0.0432, 0.0040),
                     (10.7, 0.6629, 0.0208, 0.4876, 0.0164, 0.1753, 0.0061, 0.5122, 0.0089, 0.3643, 0.0064, 0.1479, 0.0039, 0.1507, 0.0138, 0.1232, 0.0116, 0.0274, 0.0029),
                     (10.8, 0.4749, 0.0168, 0.3555, 0.0133, 0.1194, 0.0045, 0.3796, 0.0078, 0.2776, 0.0056, 0.1020, 0.0031, 0.0953, 0.0101, 0.0779, 0.0088, 0.0173, 0.0019),
                     (10.9, 0.3130, 0.0133, 0.2368, 0.0104, 0.0762, 0.0038, 0.2598, 0.0083, 0.1923, 0.0063, 0.0675, 0.0029, 0.0532, 0.0057, 0.0445, 0.0047, 0.0087, 0.0013),
                     (11.0, 0.1913, 0.0086, 0.1491, 0.0066, 0.0422, 0.0026, 0.1636, 0.0059, 0.1260, 0.0044, 0.0376, 0.0021, 0.0277, 0.0032, 0.0231, 0.0027, 0.0046, 0.0007),
                     (11.1, 0.1055, 0.0056, 0.0840, 0.0041, 0.0215, 0.0019, 0.0943, 0.0044, 0.0745, 0.0031, 0.0198, 0.0015, 0.0112, 0.0015, 0.0095, 0.0012, 0.0016, 0.0004),
                     (11.2, 0.0540, 0.0028, 0.0447, 0.0021, 0.0092, 0.0009, 0.0495, 0.0023, 0.0408, 0.0017, 0.0087, 0.0008, 0.0045, 0.0006, 0.0039, 0.0006, 0.0006, 0.0001),
                     (11.3, 0.0245, 0.0015, 0.0207, 0.0013, 0.0039, 0.0004, 0.0231, 0.0014, 0.0194, 0.0011, 0.0037, 0.0004, 0.0015, 0.0002, 0.0013, 0.0002, 0.0001, 0.0001),
                     (11.4, 0.0104, 0.0007, 0.0086, 0.0006, 0.0019, 0.0002, 0.0101, 0.0006, 0.0083, 0.0005, 0.0018, 0.0002, 0.0003, 0.0001, 0.0003, 0.0001, 0.0000, 0.0000),
                     (11.5, 0.0042, 0.0003, 0.0034, 0.0003, 0.0008, 0.0001, 0.0041, 0.0003, 0.0033, 0.0003, 0.0008, 0.0001, 0.0001, 0.0000, 0.0001, 0.0000, 0.0000, 0.0000),
                     (11.6, 0.0013, 0.0001, 0.0010, 0.0001, 0.0003, 0.0001, 0.0013, 0.0001, 0.0010, 0.0001, 0.0003, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000),
                     (11.7, 0.0003, 0.0001, 0.0002, 0.0001, 0.0001, 0.0000, 0.0003, 0.0001, 0.0002, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000)]
        self.data_table = Table(rows=data_rows,
            names=('bin_center', 'all', 'all_err', 'red', 'red_err', 'blue', 'blue_err',
                   'cen_all', 'cen_all_err', 'cen_red', 'cen_red_err', 'cen_blue', 'cen_blue_err',
                   'sat_all', 'sat_all_err', 'sat_red', 'sat_red_err', 'sat_blue', 'sat_blue_err'),
            dtype=('f4', 'f4', 'f4', 'f4', 'f4','f4','f4','f4', 'f4', 'f4', 'f4', 'f4','f4','f4','f4', 'f4', 'f4', 'f4', 'f4'))
        
        for name in self.data_table.colnames[1:]:
            self.data_table[name] = self.data_table[name]*0.01
        
    def __call__(self, mstar):
        """
        stellar mass function from Yang et al. 2012, arXiv:1110.1420
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        #take log of stellar masses
        mstar = np.log10(mstar)
        
        return self.s(mstar)


class Tomczak_2014_phi(object):
    """
    stellar mass function from Tomczak et al. 2014, arXiv:1309.5972
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        redshift : float
            default is 1.0
        
        type : string
            default is 'all'
        """
        
        self.publication = ['arXiv:1309.5972']
        
        self.littleh = 0.7
        
        if 'redshift' in kwargs:
            self.z = kwargs['redshift']
        else:
            self.z = 1.0
        
        if 'type' in kwargs:
            self.type=kwargs['type']
        else:
            self.type = 'all'
        
        #parameters table 2 all
        self.z_bins = np.array([0.2,0.5,0.75,1.0,1.25,1.5,2.0,2.5,2.5,3.0])
        self.phi1_all = 10**np.array([-2.54,-2.55,-2.56,-2.72,-2.78,-3.05,-3.80,-4.54])
        self.x1_all = np.array([10.78,10.70,10.66,10.54,10.61,10.74,10.69,10.74])
        self.alpha1_all = np.array([-0.98,-0.39,-0.37,0.30,-0.12,0.04,1.03,1.62])
        self.phi2_all = 10**np.array([-4.29,-3.15,-3.39,-3.17,-3.43,-3.38,-3.26,-3.69])
        self.x2_all = self.x1_all
        self.alpha2_all = np.array([-1.90,-1.53,-1.61,-1.45,-1.56,-1.49,-1.33,-1.57])
        
        #parameters table 2 star-forming
        self.phi1_sf = 10**np.array([-2.67,-2.97,-2.81,-2.98,-3.04,-3.37,-4.30,-4.95])
        self.x1_sf = np.array([10.59,10.65,10.56,10.44,10.69,10.59,10.58,10.61])
        self.alpha1_sf = np.array([-1.08,-0.97,-0.46,0.53,-0.55,0.75,2.06,2.36])
        self.phi2_sf = 10**np.array([-4.46,-3.34,-3.36,-3.11,-3.59,-3.28,-3.28,-3.71])
        self.x2_sf = self.x1_all
        self.alpha2_sf = np.array([-2.00,-1.58,-1.61,-1.44,-1.62,-1.47,-1.38,-1.67])
        
        #parameters table 2 quiscent
        self.phi1_q = 10**np.array([-2.76,-2.67,-2.81,-3.03,-3.36,-3.41,-3.59,-4.22])
        self.x1_q = np.array([10.75,10.68,10.63,10.63,10.49,10.77,10.69,9.95])
        self.alpha1_q = np.array([0.47,0.10,0.04,0.11,0.85,-0.19,0.37,0.62])
        self.phi2_q = 10**np.array([-5.21,-4.29,-4.40,-4.80,-3.72,-3.91,-6.95,-4.51])
        self.x2_q = self.x1_all
        self.alpha2_q = np.array([-1.97,-1.69,-1.51,-1.57,-0.54,-0.18,-3.07,-2.51])
        
        self.s_all = np.empty((8,), dtype=object)
        self.s_sf = np.empty((8,), dtype=object)
        self.s_q = np.empty((8,), dtype=object)
        
        for i in range(0,8):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_all[i], x0=self.x1_all[i], alpha=self.alpha1_all[i])
            s2 = Log_Schechter(phi0=self.phi2_all[i], x0=self.x2_all[i], alpha=self.alpha2_all[i])
            #create piecewise model
            self.s_all[i] = s1 + s2
        
        for i in range(0,8):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_sf[i], x0=self.x1_sf[i], alpha=self.alpha1_sf[i])
            s2 = Log_Schechter(phi0=self.phi2_sf[i], x0=self.x2_sf[i], alpha=self.alpha2_sf[i])
            #create piecewise model
            self.s_sf[i] = s1 + s2
        
        for i in range(0,8):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_q[i], x0=self.x1_q[i], alpha=self.alpha1_q[i])
            s2 = Log_Schechter(phi0=self.phi2_q[i], x0=self.x2_q[i], alpha=self.alpha2_q[i])
            #create piecewise model
            self.s_q[i] = s1 + s2
    
    def __call__(self, mstar):
        """
        stellar mass function from Tomczak et al. 2014, arXiv:1309.5972
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        #convert from h=1 to h=0.7
        mstar = mstar / self.littleh**2
        
        #take log of stellar masses
        mstar = np.log10(mstar)
        
        i = np.searchsorted(self.z_bins,self.z)
        
        #convert from h=0.7 to h=1.0
        if self.type=='all':
            return self.s_all[i](mstar) / self.littleh**3
        elif self.type=='star-forming':
            return self.s_sf[i](mstar) / self.littleh**3
        elif self.type=='quiescent':
            return self.s_q[i](mstar) / self.littleh**3
        else:
            print('type not available')
        
        

@custom_model
def Log_Schechter(x, phi0=0.001, x0=10.5, alpha=-1.0):
    """
    log schecter x function
    """
    x = np.asarray(x)
    x = x.astype(float)
    norm = np.log(10.0)*phi0
    val = norm*(10.0**((x-x0)*(1.0+alpha)))*np.exp(-10.0**(x-x0))
    return val
