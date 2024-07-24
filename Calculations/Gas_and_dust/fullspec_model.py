import numpy as np
import astropy.units as u
import multiprocessing as mp
from functools import partial
import subprocess
import os
import re

import sys
sys.path.append("../")

from torus_model import torus_model
from draine_dust_2D import draine_dust

###
#The code is written to generate a draine_dust model that only operates at a single wavelength. The draine_dust_2D model interpolates on wavelength. So here we will make a simple obejct to transform one into the other. 
class draine_dust_1D(object):

    def __init__(self, lam_targ, dust_type):
        self.dd2D_obj = draine_dust(dust_type)
        self.lam_targ = lam_targ
        return
    
    def pfrac(self, costh):
        return dd2D_obj.pfrac((lam_targ, costh))
    
    def diff_cross_section(self, costh):
        return dd2D_obj.diff_cross_section((lam_targ, costh))


###

def get_S(psi_angles, pobj, forward, backward, theta_scattering_angles):
    #This is the array that will hold the S values. 
    S = np.zeros((len(theta_scattering_angles), len(psi_angles), 3))*pobj.dsigma_norm.unit
    for k, theta in enumerate(theta_scattering_angles):
        tmod = torus_model(theta, pobj)
        tmod.get_integrals(psi_angles, forward_scattering=forward, backward_scattering=backward)
        S[k,:,0] = tmod.S1
        S[k,:,1] = tmod.S2
        S[k,:,2] = tmod.S3
    return S


###

#Set the wavelength grid.
waves = np.arange(1000., 3100., 20.)*u.AA

#This is the range of torus opening angles for which we'll calculate the integrals. 
psi_angles = np.arange(1.0, 90.1, 2.5)*u.deg

#This is the range of inclination angles we'll consider. 
theta_scattering_angles = np.arange(1.0, 90.1, 2.5)*u.deg

dust_types = ["SMC", "LMC", "MW"]

#Go through each model. 
for dust_type in dust_types:

    #Do forwards and backwards scattering separately. 
    for j in range(3):
        if j==0:
            forward=True
            backward=False
            suffix = "FW"
        elif j==1:
            forward=False
            backward=True
            suffix  = "BW"
        elif j==2:
            forward=True
            backward=True
            suffix = "FWBW"

        fname = "fullspec_dust_models/{}.hires.{}.txt".format(dust_type, suffix)
        if os.path.exists(fname):
            continue
        cato = open(fname,"w")
        s1_cato = open(re.sub(".txt",".S1.txt",fname),"w")
        for cato_aux in [cato, s1_cato]:
            for lam in waves:
                cato_aux.write("{:.1f} ".format(lam.value))
            cato_aux.write("\n")
            for th in theta_scattering_angles:
                cato_aux.write("{:.1f} ".format(th.value))
            cato_aux.write("\n")
            for psi in psi_angles:
                cato_aux.write("{:.1f} ".format(psi.value))
            cato_aux.write("\n")            

        for l, lam_targ in enumerate(waves):
            pobj = draine_dust_1D(lam_targ, dust_type)

            print(lam_targ, dust_type)

            #Start the multiprocessing.
            if __name__ == '__main__':
                mp.freeze_support()
                Ncpu = mp.cpu_count()
                Pool = mp.get_context("fork").Pool(Ncpu)
                func = partial(get_S, psi_angles, pobj, forward, backward)

                #Produce the data chunks.
                th_s_split = np.array_split(theta_scattering_angles, Ncpu)
                S = Pool.map(func, th_s_split)
                S = np.vstack(S)
                Pool.close()

                #Save S.
                S = S.to(u.cm**2).value
                p_array = -S[:,:,1]/S[:,:,0]

                np.savetxt(cato, p_array)
                np.savetxt(s1_cato, S[:,:,0])
        cato.close()
        s1_cato.close()
