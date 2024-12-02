import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

import sys
import os
base_folder = "{}/Impol_Blue_HotDOGs/Impol_BHDs_repo/Calculations/Gas_and_dust/".format(os.path.expanduser('~'))
sys.path.append(base_folder)
from free_electrons import free_electrons
from torus_model import torus_model

class gas_torus(object):

    def __init__(self):

        #Read the model. 
        self.theta_grid, self.psi_grid, self.p_grid = self.read_model()

        #Interpolate the model.
        self.pfrac = RegularGridInterpolator((self.theta_grid, self.psi_grid), self.p_grid, bounds_error=False, fill_value=None)

        return


    #Read the array.
    def read_model(self):

        fname = "{}/W0116-0505/gas_S.hires.FWBW.txt".format(base_folder)

        theta_grid = np.unique(np.loadtxt(fname, usecols=[0]))*u.deg
        psi_aux = np.unique(np.loadtxt(fname, usecols=[1]))*u.deg
        data_grid = np.loadtxt(fname, usecols=[2,3])
        p_aux = -data_grid[:,1]/data_grid[:,0]
        p_aux = p_aux.reshape([len(theta_grid),len(psi_aux)])

        #Now, we need to add for each wavelength an extra value for psi=0.
        psi_grid = np.zeros(len(psi_aux)+1)
        psi_grid[1:] = psi_aux
        p_grid = np.zeros((p_aux.shape[0], p_aux.shape[1]+1))

        gas_obj = free_electrons()
        for j, theta in enumerate(theta_grid):
            tmod = torus_model(theta, gas_obj)
            p_grid[j,0] = tmod.p_psi0(np.cos(theta), forward_scattering=True, backward_scattering=True)
            p_grid[j,1:] = p_aux[j]

        return theta_grid, psi_grid, p_grid

    