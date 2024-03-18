import numpy as np
import astropy.units as u
from scipy.integrate import dblquad

class torus_model(object):

    def __init__(self, inclination_angle, particle_object):
        if inclination_angle is not None:
            self.cos_b = np.cos(inclination_angle)
            self.sin_b = np.sin(inclination_angle)
        self.pobj = particle_object
        return

    def prime_to_og(self, cos_th_p, phi_p):
        cos_phi_p = np.cos(phi_p)
        sin_phi_p = np.sin(phi_p)
        sin_th_p  = (1-cos_th_p**2)**0.5

        x_p = cos_phi_p * sin_th_p
        y_p = sin_phi_p * sin_th_p
        z_p = cos_th_p

        x = self.cos_b*x_p + self.sin_b*z_p
        y = y_p
        z = -self.sin_b*x_p + self.cos_b*z_p

        cos_th   = z
        cos_2phi = (x**2-y**2)/(1-z**2)
        sin_2phi = 2*x*y/(1-z**2)
        return cos_th, cos_2phi, sin_2phi


    def S1_func(self, phi_p, cos_th_p):
        cos_th, _, _ = self.prime_to_og(cos_th_p, phi_p)
        return self.pobj.diff_cross_section(cos_th)

    def S2_func(self, phi_p, cos_th_p):
        cos_th, cos_2phi, _ = self.prime_to_og(cos_th_p, phi_p)
        return -self.pobj.pfrac(cos_th) * self.pobj.diff_cross_section(cos_th) * cos_2phi
    
    def S3_func(self, phi_p, cos_th_p):
        cos_th, _, sin_2phi = self.prime_to_og(cos_th_p, phi_p)
        return -self.pobj.pfrac(cos_th) * self.pobj.diff_cross_section(cos_th) * sin_2phi

    def get_integrals(self, psi_angles, forward_scattering=True, backward_scattering=True):
        S1 = np.zeros(len(psi_angles))
        S2 = np.zeros(len(psi_angles))
        S3 = np.zeros(len(psi_angles))
        for k, psi in enumerate(psi_angles):
            phi_p_min = 0
            phi_p_max = 2*np.pi
            if forward_scattering:
                cos_th_p_min = np.cos(psi)
                cos_th_p_max = 1.0
                S1[k] =  dblquad(self.S1_func, cos_th_p_min, cos_th_p_max, phi_p_min, phi_p_max)[0]
                S2[k] =  dblquad(self.S2_func, cos_th_p_min, cos_th_p_max, phi_p_min, phi_p_max)[0]
                S3[k] =  dblquad(self.S3_func, cos_th_p_min, cos_th_p_max, phi_p_min, phi_p_max)[0]
            if backward_scattering:
                cos_th_p_min = -1.0
                cos_th_p_max = -np.cos(psi)
                S1[k] += dblquad(self.S1_func, cos_th_p_min, cos_th_p_max, phi_p_min, phi_p_max)[0]
                S2[k] += dblquad(self.S2_func, cos_th_p_min, cos_th_p_max, phi_p_min, phi_p_max)[0]
                S3[k] += dblquad(self.S3_func, cos_th_p_min, cos_th_p_max, phi_p_min, phi_p_max)[0]

        self.S1 = S1 * self.pobj.dsigma_norm
        self.S2 = S2 * self.pobj.dsigma_norm
        self.S3 = S3 * self.pobj.dsigma_norm
        return

    #Polarization for psi=0.
    def p_psi0(self, costh, forward_scattering=True, backward_scattering=True):
        dp1 = self.pobj.pfrac(costh)
        dp2 = self.pobj.pfrac(-costh)
        ds1 = self.pobj.diff_cross_section(costh)
        ds2 = self.pobj.diff_cross_section(-costh)
        num = 0.
        dem = 0.
        if forward_scattering:
            num += dp1*ds1
            dem += ds1
        if backward_scattering:
            num += dp2*ds2
            dem += ds2
        return num/dem

    #Find the minimum theta that's able to produce the polarization.
    def find_th_min(self, p_targ):
        th_min =  0.*u.deg
        th_max = 90.*u.deg
        for i in range(50):
            th = 0.5*(th_max+th_min)
            p_test = self.p_psi0(np.cos(th))
            if np.abs(p_test-p_targ)/p_targ < 1e-3:
                break
            if p_test>p_targ:
                th_max = th
            else:
                th_min = th
        return th, self.p_psi0(np.cos(th))