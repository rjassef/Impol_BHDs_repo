import numpy as np
import os
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from synphot import SpectralElement, Observation, SourceSpectrum
from synphot.models import Empirical1D

from draine_dust_2D import draine_dust
from torus_model import torus_model

###
#The code is written to generate a draine_dust model that only operates at a single wavelength. The draine_dust_2D model interpolates on wavelength. So here we will make a simple obejct to transform one into the other. 
class draine_dust_1D(object):

    def __init__(self, lam_targ, dust_type):
        self.dd2D_obj = draine_dust(dust_type)
        self.dsigma_norm = self.dd2D_obj.dsigma_norm
        self.type = dust_type
        self.lam_targ = lam_targ
        return
    
    def pfrac(self, costh):
        return self.dd2D_obj.pfrac(self.lam_targ, costh)
    
    def diff_cross_section(self, costh):
        return self.dd2D_obj.diff_cross_section(self.lam_targ, costh)
    

class PolWaveDust(object):

    def __init__(self, dust_type, fw=True, bw=True, folder=None, interp_method='linear'):

        #Save the input parameters.
        self.dust_type = dust_type
        self.medium_type = dust_type
        self.fw = fw
        self.bw = bw
        if folder is None:
            self.folder = os.path.dirname(os.path.realpath(__file__))+"/fullspec_dust_models/"
        else:
            self.folder = folder

        #Read the model. 
        self.wave_grid, self.theta_grid, self.psi_grid, self.p_grid, self.s1_grid = self.read_model()

        #Interpolate the model.
        self.p = RegularGridInterpolator((self.wave_grid, self.theta_grid, self.psi_grid), self.p_grid, bounds_error=False, fill_value=None, method=interp_method)

        self.s1 = RegularGridInterpolator((self.wave_grid, self.theta_grid, self.psi_grid), self.s1_grid, bounds_error=False, fill_value=None, method=interp_method)

        return


    #Read the array.
    def read_model(self):
        suffix = ""
        if self.fw:
            suffix += "FW"
        if self.bw:
            suffix += "BW"
        fname = "{}/{}.hires.{}.txt".format(self.folder, self.dust_type, suffix)
        fname_S1 = "{}/{}.hires.{}.S1.txt".format(self.folder, self.dust_type, suffix)

        wave_grid  = np.loadtxt(fname, max_rows=1)*u.AA
        theta_grid = np.loadtxt(fname, max_rows=1, skiprows=1)*u.deg
        psi_aux    = np.loadtxt(fname, max_rows=1, skiprows=2)*u.deg
        data = np.loadtxt(fname, skiprows=3)
        p_aux = data.reshape([len(wave_grid), len(theta_grid), len(psi_aux)])
        s1_data = np.loadtxt(fname_S1, skiprows=3)
        s1_aux = s1_data.reshape([len(wave_grid), len(theta_grid), len(psi_aux)])

        #Now, we need to add for each wavelength an extra value for psi=0.
        psi_grid = np.zeros(len(psi_aux)+1)*u.deg
        psi_grid[1:] = psi_aux
        p_grid = np.zeros((p_aux.shape[0], p_aux.shape[1], p_aux.shape[2]+1))
        s1_grid = np.zeros((s1_aux.shape[0], s1_aux.shape[1], s1_aux.shape[2]+1))
        for k, wave in enumerate(wave_grid):
            dust = draine_dust_1D(wave, self.dust_type)
            for j, theta in enumerate(theta_grid):
                tmod = torus_model(theta, dust)
                p_grid[k,j,0] = tmod.p_psi0(np.cos(theta), forward_scattering=self.fw, backward_scattering=self.bw)
                p_grid[k,j,1:] = p_aux[k,j]
                s1_grid[k,j,0] = 0.
                s1_grid[k,j,1:] = s1_aux[k,j]

        return wave_grid, theta_grid, psi_grid, p_grid, s1_grid
    
    def p_bb(self, band, ths, psis, spec_lam_obs, spec_flam, z):

        p_bb_output = np.ma.zeros((len(ths), len(psis)))
        lam_grid = spec_lam_obs.to(u.AA).value/(1.+z)

        #Make the binset for the observations. 
        binset_cond = spec_lam_obs.to(u.AA).value<band._model.points[0].max()
        binset_cond = binset_cond & (spec_lam_obs.to(u.AA).value>band._model.points[0].min())
        binset = spec_lam_obs[binset_cond]

        full_spec = SourceSpectrum(Empirical1D, points=spec_lam_obs, lookup_table=spec_flam, keep_neg=True)
        obs_I = Observation(full_spec, band, binset=binset)#force='extrap')
        Ibb = obs_I.effstim(flux_unit='flam').value
        
        for i, psi in enumerate(psis):
            for j, th in enumerate(ths):
                p_aux = self.p((lam_grid, th*np.ones(lam_grid.shape), psi*np.ones(lam_grid.shape)))
                Q_spec = SourceSpectrum(Empirical1D, points=spec_lam_obs, lookup_table=spec_flam * p_aux, keep_neg=True)
                obs_Q = Observation(Q_spec, band, binset=binset)#force='extrap')
                Qbb = obs_Q.effstim(flux_unit='flam').value

                p_bb_output[j,i]= Qbb/Ibb
        
        return p_bb_output

