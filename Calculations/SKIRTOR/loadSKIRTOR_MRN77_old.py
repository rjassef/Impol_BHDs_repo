import numpy as np
import subprocess
import re
import os
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import astropy.units as u
from synphot import SpectralElement, Observation, SourceSpectrum
from synphot.models import Empirical1D
from synphot.exceptions import SynphotError

class LoadSKIRTOR_MRN77(object):

    def __init__(self, folder=None, interp_method='linear'):

        #Set the folder to use. 
        if folder is None:
            self.folder = os.path.dirname(os.path.realpath(__file__))+"/models_old/"
        else:
            self.folder = folder

        #Make a list of all the MRN77 models. 
        ls_output = subprocess.run("ls {}/bHDPol_mrn77_tor_oa*".format(self.folder), shell=True, capture_output=True)
        fnames = ls_output.stdout.decode('utf8').split()

        #Make a list of the torus opening, cone opening and inclination angles.
        tangle = list()
        cangle = list()
        iangle = list()
        for fname in fnames:
            m = re.search("bHDPol_mrn77_tor_oa(.*)_con_oa(.*)-tauV1_i(.*)_sed.dat", fname)
            if m.group(3)[1]!="1":
                tangle.append(float(m.group(1)))
                cangle.append(float(m.group(2)))
                iangle.append(float(m.group(3)))
        self.tang_grid = np.unique(tangle) * u.deg
        self.cang_grid = np.unique(cangle) * u.deg
        self.iang_grid = np.unique(iangle) * u.deg

        #Make a list of the wavelengths. 
        self.lam_grid = np.loadtxt(fnames[0], usecols=[0])*u.um

        #Transform wavelengths from microns to angstroms. 
        self.lam_grid = self.lam_grid.to(u.AA)
        #self.lam_grid *= 10

        #Now make the array holding the polarization grid. 
        self.p_grid = np.ma.zeros((len(self.tang_grid), len(self.cang_grid), len(self.iang_grid), len(self.lam_grid)))
        self.p_grid.mask = np.zeros(self.p_grid.shape, dtype=bool)
        for i, tang in enumerate(self.tang_grid):
            for j, cang in enumerate(self.cang_grid):
                for k, iang in enumerate(self.iang_grid):
                    fname = "{}/bHDPol_mrn77_tor_oa{:.0f}_con_oa{:.0f}-tauV1_i{:.0f}_sed.dat".format(self.folder, tang.value, cang.value, iang.value)
                    if fname in fnames:
                        data = np.loadtxt(fname)
                        self.p_grid[i,j,k] = -data[:,2]/data[:,1]
                    # else:
                    #     #self.p_grid[i,j,k] = np.nan * np.ones(len(self.lam_grid))
                    #     if iang>tang and cang<tang:
                    #        self.p_grid.mask[i,j,k] = np.ones(len(self.lam_grid), dtype=bool)

        #Now, make the interpolator.
        self.p = RegularGridInterpolator((self.tang_grid, self.cang_grid, self.iang_grid, self.lam_grid), self.p_grid, bounds_error=False, fill_value=0.0, method=interp_method)

        return

    def p_bb(self, band, tangs, cangs, iangs, spec_lam_obs, spec_flam, z):

        p_bb_output = np.ma.zeros((len(tangs), len(cangs), len(iangs)))
        lam_grid = spec_lam_obs.to(u.AA).value/(1.+z)

        #Make the binset for the observations. 
        binset_cond = spec_lam_obs.to(u.AA).value<band._model.points[0].max()
        binset_cond = binset_cond & (spec_lam_obs.to(u.AA).value>band._model.points[0].min())
        binset = spec_lam_obs[binset_cond]

        full_spec = SourceSpectrum(Empirical1D, points=spec_lam_obs, lookup_table=spec_flam, keep_neg=True)
        obs_I = Observation(full_spec, band, binset=binset)#force='extrap')
        Ibb = obs_I.effstim(flux_unit='flam').value
        
        for i, tang in enumerate(tangs):
            for j, cang in enumerate(cangs):
                for k, iang in enumerate(iangs):
                    # if cang>=tang or iang<=tang:
                    #     continue
                    p_aux = self.p((tang*np.ones(lam_grid.shape), cang*np.ones(lam_grid.shape), iang*np.ones(lam_grid.shape), lam_grid))
                    Q_spec = SourceSpectrum(Empirical1D, points=spec_lam_obs, lookup_table=spec_flam * p_aux, keep_neg=True)
                    obs_Q = Observation(Q_spec, band, binset=binset)#force='extrap')
                    try:
                        Qbb = obs_Q.effstim(flux_unit='flam').value
                        p_bb_output[i,j,k]= Qbb/Ibb
                    except SynphotError:
                        # print(tang, cang, iang)
                        # print(p_aux)
                        # input()
                        pass
        
        return p_bb_output

