import numpy as np
import astropy.units as u
import multiprocessing as mp
from functools import partial
import subprocess
import os

from torus_model import torus_model
from free_electrons import free_electrons

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

#This is the range of angles for which we'll calculate the integrals. 
psi_angles = np.arange(1.0,90.1,1.0)*u.deg

#This is the range of inclination angles we'll consider. 
theta_scattering_angles = np.arange(1.0, 90.1, 1.0)*u.deg

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

    #Output filename. 
    fname = "gas_models/gas_S.hires.{}.txt".format(suffix)

    #Set the objects.
    pobj = free_electrons()

    #Start the multiprocessing.
    Ncpu = mp.cpu_count()
    Pool = mp.Pool(Ncpu)
    func = partial(get_S, psi_angles, pobj, forward, backward)

    #Produce the data chunks.
    th_s_split = np.array_split(theta_scattering_angles, Ncpu)
    S = Pool.map(func, th_s_split)
    S = np.vstack(S)
    Pool.close()

    #Save S.
    S = S.to(u.cm**2).value
    cat = open(fname,"w")
    for l,theta in enumerate(theta_scattering_angles):
        for k,psi in enumerate(psi_angles):
            cat.write("{0:10.1f} {1:10.2f} {2:15.6e} {3:15.6e} {4:15.6e}\n".format(
                theta.to(u.deg).value, psi.to(u.deg).value, S[l,k,0], S[l,k,1], S[l,k,2]))
    cat.close()
