import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

import sys
sys.path.append("Gas_and_dust/")
from free_electrons import free_electrons
from torus_model import torus_model

class PolWave_gas(object):

    def __init__(self, fw=True, bw=True, folder="."):

        #Save the input parameters.
        self.fw = fw
        self.bw = bw
        self.folder = folder

        #Read the model. 
        self.theta_grid, self.psi_grid, self.p_grid = self.read_model()

        #Interpolate the model.
        self.p = RegularGridInterpolator((self.theta_grid, self.psi_grid), self.p_grid, bounds_error=False, fill_value=None)

        return


    #Read the array.
    def read_model(self):
        suffix = ""
        if self.fw:
            suffix += "FW"
        if self.bw:
            suffix += "BW"
        fname = "{}/gas_S.hires.{}.txt".format(self.folder, suffix)


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
            p_grid[j,0] = tmod.p_psi0(np.cos(theta), forward_scattering=self.fw, backward_scattering=self.bw)
            p_grid[j,1:] = p_aux[j]

        return theta_grid, psi_grid, p_grid
    