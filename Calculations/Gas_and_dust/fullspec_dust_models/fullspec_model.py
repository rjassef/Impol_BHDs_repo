import numpy as np
import astropy.units as u
import multiprocessing as mp
from functools import partial
import subprocess
import os

import sys
sys.path.append("../")

from torus_model import torus_model
from free_electrons import free_electrons
from draine_dust import draine_dust

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

#wids = ['W0116-0505'] #, 'W0019-1046', 'W0220+0137', 'W0204-0506', 'W0831+0140']
#zs = [3.173] #, 1.641, 3.122, 2.100, 3.913]
#bands = ["v","R","I"]
#bands = ["I"]
#band_centers = np.array([5500., 6500., 8000.])*u.AA
#waves = np.arange(1000., 3000., 100.)*u.AA
#waves = np.array([1000.])*u.AA
#waves = np.array([1000., 2000., 3000.])*u.AA
#waves = np.arange(500., 3000., 100.)*u.AA
waves = np.arange(1000., 3100., 20.)*u.AA

#This is the range of torus opening angles for which we'll calculate the integrals. 
#psi_angles = np.arange(1.0,90.1,5.0)*u.deg
psi_angles = np.arange(1.0, 90.1, 2.5)*u.deg
#psi_angles = np.array([50.])*u.deg

#This is the range of inclination angles we'll consider. 
#theta_scattering_angles = np.arange(1.0, 90.1, 5.0)*u.deg
theta_scattering_angles = np.arange(1.0, 90.1, 2.5)*u.deg
#theta_scattering_angles = np.array([50.])*u.deg

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

        fname = "{}.hires.{}.txt".format(dust_type, suffix)
        if os.path.exists(fname):
            continue
        cato = open(fname,"w")
        for lam in waves:
            cato.write("{:.1f} ".format(lam.value))
        cato.write("\n")
        for th in theta_scattering_angles:
            cato.write("{:.1f} ".format(th.value))
        cato.write("\n")
        for psi in psi_angles:
            cato.write("{:.1f} ".format(psi.value))
        cato.write("\n")            

        for l, lam_targ in enumerate(waves):
            pobj = draine_dust(lam_targ, dust_type)


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
        cato.close()
