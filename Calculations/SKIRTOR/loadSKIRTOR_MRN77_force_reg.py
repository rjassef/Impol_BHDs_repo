import numpy as np
import subprocess
import re
import os
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d
import astropy.units as u
from synphot import SpectralElement, Observation, SourceSpectrum
from synphot.models import Empirical1D
from synphot.exceptions import SynphotError

class LoadSKIRTOR_MRN77(object):

    def __init__(self, folder=None, interp_method='linear', cone_type="Full"):

        #Set the folder to use. 
        if folder is None:
            self.folder = os.path.dirname(os.path.realpath(__file__))+"/models/"
        else:
            self.folder = folder

        #Make a list of all the MRN77 models. 
        self.cone_type = cone_type
        self.base_fname = "bHDPol_mrn77_"
        if self.cone_type=="Full":
            pass
        elif self.cone_type=="Top":
            self.base_fname += "TopConeOnly_"
        elif self.cone_type=="Bottom":
            self.base_fname += "BottomConeOnly_"
        else:
            print("Unrecognized type of cone ", self.cone_type)
            return
        ls_output = subprocess.run("ls {}/{}tor_oa*".format(self.folder, self.base_fname), shell=True, capture_output=True)
        fnames = ls_output.stdout.decode('utf8').split()

        #Make a list of the wavelengths. 
        self.lam_grid = np.loadtxt(fnames[0], usecols=[0])*u.um

        #Transform wavelengths from microns to angstroms. 
        self.lam_grid = self.lam_grid.to(u.AA)
        #self.lam_grid *= 10

        #Make a list of the torus opening, cone opening and inclination angles.
        tangle = list()
        cangle = list()
        iangle = list()
        for fname in fnames:
            m = re.search("{}tor_oa(.*)_con_oa(.*)-tauV0.1_i(.*)_sed.dat".format(self.base_fname), fname)
            if m.group(3)[1]!="1":
                tangle.append(float(m.group(1)))
                cangle.append(float(m.group(2)))
                iangle.append(float(m.group(3)))
        self.tang_grid = np.unique(tangle) * u.deg
        self.cang_grid = np.unique(cangle) * u.deg
        self.iang_grid = np.unique(iangle) * u.deg

        #Now, we are going to have to combine interpolation methods. Basically, we will use a 1D to interpolate between inclinations (which are mapped not-regularly) and fill in our regular grid.
        self.p_grid = np.zeros((len(self.tang_grid), len(self.cang_grid), len(self.iang_grid), len(self.lam_grid))) 
        for i,tang in enumerate(self.tang_grid):
            for j,cang in enumerate(self.cang_grid):

                #Read in all the files available for these Tang and Cang.
                ls_output =  subprocess.run("ls {}/{}tor_oa{:.1f}_con_oa{:.1f}-tauV0.1_i*_sed.dat".format(self.folder, self.base_fname, tang.value, cang.value), shell=True, capture_output=True)
                fnames_aux = ls_output.stdout.decode('utf8').split()
                if len(fnames_aux)==0:
                    continue

                iang_aux = np.zeros(len(fnames_aux))*u.deg
                p_aux = np.zeros((len(fnames_aux), len(self.lam_grid)))
                for k, fname_aux in enumerate(fnames_aux):
                    m = re.search("{}tor_oa(.*)_con_oa(.*)-tauV0.1_i(.*)_sed.dat".format(self.base_fname), fname_aux)
                    iang_aux[k] = float(m.group(3)) * u.deg
                    data = np.loadtxt(fname_aux)
                    #lam_aux = (data[:,0]*u.micron).to(u.AA)
                    p_aux[k] = (data[:,8]**2+data[:,9]**2)**0.5/data[:,1]

                # #For each wavelength, we will make an interp1d object. 
                # p_interp1d = list()
                # for l in range(len(self.lam_grid)):
                #     p_interp1d.append(interp1d(iang_aux.value, p_aux[:,l]))

                #Finally, fill in p_grid. 
                for l, lam in enumerate(self.lam_grid):
                    p_interp1d = interp1d(iang_aux.value, p_aux[:,l], bounds_error=False, fill_value=0.)
                    self.p_grid[i,j,:,l] = p_interp1d(self.iang_grid.value)
                    
        #Now, make the interpolator.
        self.p = RegularGridInterpolator((self.tang_grid.value, self.cang_grid.value, self.iang_grid.value, self.lam_grid.value), self.p_grid, bounds_error=False, fill_value=0.0, method=interp_method)

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
        
        vt, vc, vi, vl = np.meshgrid(tangs.value, cangs.value, iangs.value, self.lam_grid.value, indexing='ij')
        p_interp = self.p((vt, vc, vi, vl))
        for i, tang in enumerate(tangs):
            for j, cang in enumerate(cangs):
                for k, iang in enumerate(iangs):
                    if cang+5.*u.deg>=tang or iang<=tang:
                    #if cang+10.*u.deg>=tang or iang<=tang:
                        continue
                    p_aux = np.interp(lam_grid, self.lam_grid.value, p_interp[i,j,k])
                    #p_aux = self.p((tang*np.ones(lam_grid.shape), cang*np.ones(lam_grid.shape), iang*np.ones(lam_grid.shape), lam_grid))
                    #p_aux = self.p(tang.value*np.ones(lam_grid.shape), cang.value*np.ones(lam_grid.shape), iang.value*np.ones(lam_grid.shape), lam_grid)
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

