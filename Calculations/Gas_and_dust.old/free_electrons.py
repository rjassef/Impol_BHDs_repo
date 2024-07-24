import numpy as np
import astropy.units as u
from astropy.constants import eps0, c, m_e, e

class free_electrons(object):

    def __init__(self):
        #Classical electron radius
        self.r0 = ((1/(4*np.pi*eps0)) * e.si**2/(m_e*c**2))
        self.dsigma_norm = (self.r0**2).to(u.cm**2)
        return

    def pfrac(self, cos_th):
        x2 = cos_th**2
        return (1-x2)/(1+x2)

    def diff_cross_section(self, cos_th):
        return 0.5*(1+cos_th**2)

    