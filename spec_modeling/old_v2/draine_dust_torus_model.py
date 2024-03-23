import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

import sys
import os
base_folder = "{}/Impol_Blue_HotDOGs/Impol_BHDs_repo/Calculations/Gas_and_dust".format(os.path.expanduser('~'))
sys.path.append(base_folder)
from draine_dust import draine_dust
from torus_model import torus_model

class ddust_torus(object):

    def __init__(self, dust_type, fw=True, bw=True, folder="fullspec_dust_models"):

        #Save the input parameters.
        self.dust_type = dust_type
        self.fw = fw
        self.bw = bw
        self.folder = folder

        #Read the model. 
        self.wave_grid, self.theta_grid, self.psi_grid, self.p_grid = self.read_model()

        #Interpolate the model.
        self.p = RegularGridInterpolator((self.wave_grid, self.theta_grid, self.psi_grid), self.p_grid, bounds_error=False, fill_value=None)

        return


    #Read the array.
    def read_model(self):
        suffix = ""
        if self.fw:
            suffix += "FW"
        if self.bw:
            suffix += "BW"
        fname = "{}/{}/{}.hires.{}.txt".format(base_folder, self.folder, self.dust_type, suffix)

        wave_grid  = np.loadtxt(fname, max_rows=1)*u.AA
        theta_grid = np.loadtxt(fname, max_rows=1, skiprows=1)*u.deg
        psi_aux    = np.loadtxt(fname, max_rows=1, skiprows=2)*u.deg
        data = np.loadtxt(fname, skiprows=3)
        p_aux = data.reshape([len(wave_grid), len(theta_grid), len(psi_aux)])

        #Now, we need to add for each wavelength an extra value for psi=0.
        psi_grid = np.zeros(len(psi_aux)+1)
        psi_grid[1:] = psi_aux
        p_grid = np.zeros((p_aux.shape[0], p_aux.shape[1], p_aux.shape[2]+1))
        for k, wave in enumerate(wave_grid):
            dust = draine_dust(wave, self.dust_type)
            for j, theta in enumerate(theta_grid):
                tmod = torus_model(theta, dust)
                p_grid[k,j,0] = tmod.p_psi0(np.cos(theta), forward_scattering=self.fw, backward_scattering=self.bw)
                p_grid[k,j,1:] = p_aux[k,j]

        return wave_grid, theta_grid, psi_grid, p_grid
    