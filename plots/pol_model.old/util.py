import numpy as np
import astropy.units as u
from scipy.interpolate import interp2d, RectBivariateSpline

import sys 
import os
sys.path.append("{}/Gas_and_dust/".format(os.getcwd()))
from torus_model import torus_model

#Find the opening for a given theta_S and p
def find_psi(model, th_targ, p_targ):

    #Interpolate the model.
    th_calc, psi_calc, p_interp = get_p_interp(model)

    #Start the bisection
    psi_min = 0*u.deg
    psi_max = 90*u.deg

    for i in range(50):
        #psi = np.mean([psi_min, psi_max])
        psi = 0.5*(psi_max+psi_min)
        p_test = p_interp(th_targ, psi)
        if np.abs(p_test-p_targ)/p_targ < 1e-3:
            break
        if p_test>p_targ:
            psi_min = psi
        else:
            psi_max = psi
    return psi, p_interp(th_targ, psi)[0,0]


def get_p_interp(model):
    
    #These are the calculated polarization fractrion values.
    th_calc = np.unique(model[:,0])*u.deg
    psi_calc = model[model[:,0]==th_calc[0].to(u.deg).value,1]*u.deg
    S1 = model[:,2].reshape(len(th_calc), len(psi_calc))
    S2 = model[:,3].reshape(len(th_calc), len(psi_calc))
    S3 = model[:,4].reshape(len(th_calc), len(psi_calc))
    #p_model = (S2**2+S3**2)**0.5 / S1
    p_model = -S2/S1

    #Create the interp 2D objects to get p for any psi and theta.
    p_interp = RectBivariateSpline(th_calc, psi_calc, p_model)

    return th_calc, psi_calc, p_interp


def read_model(fname, min_theta_filter=False, pobj=None, p_targ=None):

    #Read the model file. 
    model = np.loadtxt(fname)

    #Changes nans for 0s. Only happens for psi=90 degrees.
    model[np.isnan(model)] = 0

    #If requested, remove angles<=theta_min. 
    if min_theta_filter:
        tmod_aux = torus_model(None, pobj)
        th_min = tmod_aux.find_th_min(p_targ)[0]
        model = model[model[:,0]>th_min.to(u.deg).value]
    
    return model