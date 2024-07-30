import numpy as np 
import os
import subprocess
import re
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u

from synphot import SpectralElement, Observation, SourceSpectrum
from synphot.models import Empirical1D
from synphot.exceptions import SynphotError

class LoadSKIRTOR_General(object):

    def __init__(self, gs_min=None, gs_max=None, a=None, d03_mod=None, sil=False, folder=None, interp_method='linear'):

        #Save the input parameters. 
        self.gs_max = gs_max
        self.gs_min = gs_min
        self.a = a
        self.d03_mod = d03_mod

        #Set the folder to use. 
        if folder is None:
            self.folder = os.path.dirname(os.path.realpath(__file__))+"/models/"
        else:
            self.folder = folder

        #Either a Draine (2003) model must be selected, or grain size and power-law index parameters must be selected. 
        if d03_mod is not None:
            self.root_fname = "blueHotDogsPol_{}_i".format(d03_mod)
            self.model_name = d03_mod
        elif gs_min is not None and gs_max is not None and a is not None:
            self.root_fname = "blueHotDogsPol_gs{}-{}_a{}_".format(gs_min, gs_max, a)
            if sil==True:
                self.root_fname += "sil_"
            self.root_fname += "i"
            self.model_name = self.root_fname[15:-2]
            
        else:
            print("Cannot load model without any parameters being specified.")
            print("Models available are: ")
            ls_output = subprocess.run("ls {}/blueHotDogsPol_*".format(self.folder), shell=True, capture_output=True)
            fnames = ls_output.stdout.decode('utf8').split()
            mod_fams = list()
            for fname in fnames:
                m = re.search("blueHotDogsPol_(.*)_i.*?_sed.dat", fname)
                if m.group(1) not in mod_fams:
                    mod_fams.append(m.group(1))
            for mod_fam in mod_fams:
                print(mod_fam)
            return
        
        #List all the models that match the parameters.
        ls_output = subprocess.run("ls {}/{}*".format(self.folder, self.root_fname), shell=True, capture_output=True)
        fnames = ls_output.stdout.decode('utf8').split()
        
        if len(fnames)==0:
            print("No models found with the input parameters.")
            return
        
        #Get the angles we have.
        self.iang_grid = np.zeros(len(fnames))
        for i, fname in enumerate(fnames):
            m = re.search("_i(.*?)_sed.dat", fname)
            self.iang_grid[i] = float(m.group(1))
        self.iang_grid = self.iang_grid * u.deg
        #print(self.iang_grid)

        #Make a list of the wavelengths. 
        self.lam_grid = np.loadtxt(fnames[0], usecols=[0])*u.um

        #Transform wavelengths from microns to angstroms. 
        self.lam_grid = self.lam_grid.to(u.AA)

        #Make the polarization grid. 
        self.p_grid = np.ma.zeros((len(self.iang_grid), len(self.lam_grid)))
        self.p_grid.mask = np.zeros(self.p_grid.shape, dtype=bool)
        for k, iang in enumerate(self.iang_grid):
            fname = "{}/{}{:.0f}_sed.dat".format(self.folder, self.root_fname, iang.value)
            if fname in fnames:
                data = np.loadtxt(fname)
                self.p_grid[k] = -data[:,2]/data[:,1]

        #Finally, make the interpolator.
        self.p = RegularGridInterpolator((self.iang_grid, self.lam_grid), self.p_grid, bounds_error=False, fill_value=0.0, method=interp_method)

        return 
    
    def p_bb(self, band, iangs, spec_lam_obs, spec_flam, z):

        p_bb_output = np.ma.zeros(len(iangs))
        lam_grid = spec_lam_obs.to(u.AA).value/(1.+z)

        #Make the binset for the observations. 
        binset_cond = spec_lam_obs.to(u.AA).value<band._model.points[0].max()
        binset_cond = binset_cond & (spec_lam_obs.to(u.AA).value>band._model.points[0].min())
        binset = spec_lam_obs[binset_cond]

        full_spec = SourceSpectrum(Empirical1D, points=spec_lam_obs, lookup_table=spec_flam, keep_neg=True)
        obs_I = Observation(full_spec, band, binset=binset)#force='extrap')
        Ibb = obs_I.effstim(flux_unit='flam').value
        
        for k, iang in enumerate(iangs):
            # if cang>=tang or iang<=tang:
            #     continue
            p_aux = self.p((iang*np.ones(lam_grid.shape), lam_grid))
            Q_spec = SourceSpectrum(Empirical1D, points=spec_lam_obs, lookup_table=spec_flam * p_aux, keep_neg=True)
            obs_Q = Observation(Q_spec, band, binset=binset)#force='extrap')
            try:
                Qbb = obs_Q.effstim(flux_unit='flam').value
                p_bb_output[k]= Qbb/Ibb
            except SynphotError:
                # print(tang, cang, iang)
                # print(p_aux)
                # input()
                pass
        
        return p_bb_output

