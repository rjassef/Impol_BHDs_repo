import numpy as np
import os
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

from free_electrons import free_electrons
from torus_model import torus_model

class PolWaveGas(object):

    def __init__(self, fw=True, bw=True, folder=None):

        #Save the input parameters.
        self.fw = fw
        self.bw = bw
        if folder is None:
            self.folder = os.path.dirname(os.path.realpath(__file__))+"/gas_models/"
        else:
            self.folder = folder

        #Read the model. 
        self.theta_grid, self.psi_grid, self.p_grid, self.s1_grid = self.read_model()

        #Interpolate the model.
        self.p = RegularGridInterpolator((self.theta_grid, self.psi_grid), self.p_grid, bounds_error=False, fill_value=None)
        self.s1 = RegularGridInterpolator((self.theta_grid, self.psi_grid), self.s1_grid, bounds_error=False, fill_value=None)

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
        s1_aux = data_grid[:,0]
        s1_aux = s1_aux.reshape([len(theta_grid),len(psi_aux)])

        #Now, we need to add for each wavelength an extra value for psi=0.
        psi_grid = np.zeros(len(psi_aux)+1)
        psi_grid[1:] = psi_aux
        p_grid = np.zeros((p_aux.shape[0], p_aux.shape[1]+1))
        s1_grid = np.zeros((p_aux.shape[0], p_aux.shape[1]+1))

        gas_obj = free_electrons()
        for j, theta in enumerate(theta_grid):
            tmod = torus_model(theta, gas_obj)
            p_grid[j,0] = tmod.p_psi0(np.cos(theta), forward_scattering=self.fw, backward_scattering=self.bw)
            p_grid[j,1:] = p_aux[j]
            s1_grid[j,0] = 0.
            s1_grid[j,1:] = s1_aux[j]


        return theta_grid, psi_grid, p_grid, s1_grid
    