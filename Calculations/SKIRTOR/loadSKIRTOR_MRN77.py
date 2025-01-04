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

        #We need to make lists and do a non regular grid interpolation. 
        self.grid_points = list()
        self.grid_ps = list()
        for fname in fnames:
            m = re.search("{}tor_oa(.*)_con_oa(.*)-tauV0.1_i(.*)_sed.dat".format(self.base_fname), fname)
            tang = float(m.group(1))*u.deg
            cang = float(m.group(2))*u.deg
            iang = float(m.group(3))*u.deg
            data = np.loadtxt(fname)
            lam  = data[:,0]*u.micron
            lam = lam.to(u.AA)
            #puse = -data[:,8]/data[:,1]
            puse = (data[:,8]**2+data[:,9]**2)**0.5/data[:,1]
            #puse = (data[:,2]**2+data[:,3]**2)**0.5/data[:,1]
            for k in range(len(lam)):
                self.grid_points.append((tang.value, cang.value, iang.value, lam[k].value))
                self.grid_ps.append(puse[k])
        self.p = LinearNDInterpolator(self.grid_points, self.grid_ps, fill_value=0.)

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
                    #if cang+5.*u.deg>=tang or iang<=tang:
                    if cang+10.*u.deg>=tang or iang<=tang:
                        continue
                    #p_aux = self.p((tang*np.ones(lam_grid.shape), cang*np.ones(lam_grid.shape), iang*np.ones(lam_grid.shape), lam_grid))
                    p_aux = self.p(tang.value*np.ones(lam_grid.shape), cang.value*np.ones(lam_grid.shape), iang.value*np.ones(lam_grid.shape), lam_grid)
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

