import numpy as np
import astropy.units as u
import multiprocessing as mp
from functools import partial
import subprocess
import os

from torus_model import torus_model
from free_electrons import free_electrons
from draine_dust import draine_dust

###

def get_S(psi_angles, pobj, theta_scattering_angles):
    #This is the array that will hold the S values. 
    S = np.zeros((len(theta_scattering_angles), len(psi_angles), 3))*pobj.dsigma_norm.unit
    for k, theta in enumerate(theta_scattering_angles):
        tmod = torus_model(theta, pobj)
        tmod.get_integrals(psi_angles)
        S[k,:,0] = tmod.S1
        S[k,:,1] = tmod.S2
        S[k,:,2] = tmod.S3
    return S


###

lam_targ = 6500.*u.AA / (1+3.173)
pobj = draine_dust(lam_targ, "SMC")
#pobj = draine_dust("SMC")
psi = np.array([1.0])*u.deg
theta = np.array([1.0])*u.deg
S = get_S(psi, pobj, theta)
S = S.to(u.cm**2).value
print(S)
