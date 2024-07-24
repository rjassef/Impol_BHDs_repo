import numpy as np
import os
import astropy.units as u
from scipy.interpolate import RectBivariateSpline
import os

class draine_dust(object):

    def __init__(self, type='SMC'):
        #self.dsigma_norm = 1e-24*u.cm**2
        self.dsigma_norm = 1*u.cm**2
        self.type = type

        root_folder = os.path.dirname(os.path.realpath(__file__))+"/draine_models/"

        if type=='SMC':
            p_fname = root_folder+"callscat_init_p.out_SMC_bar"
            s_fname = root_folder+"callscat_init_i.out_SMC_bar"
        elif type=='LMC':
            p_fname = root_folder+"callscat_init_p.out_LMC_avg"
            s_fname = root_folder+"callscat_init_i.out_LMC_avg"
        elif type=='MW':
            p_fname = root_folder+"callscat_init_p.out_MW_3.1"
            s_fname = root_folder+"callscat_init_i.out_MW_3.1"
        else:
            print("Unknown dust type")
            return

        wave_pfrac, theta_S_pfrac, pfrac = self.read_file(p_fname)
        cos_th_S_pfrac = np.cos(theta_S_pfrac*u.deg)
        self.pfrac = RectBivariateSpline(wave_pfrac, cos_th_S_pfrac, pfrac.T)

        wave_dsig, theta_S_dsig, dsig = self.read_file(s_fname)
        cos_th_S_dsig = np.cos(theta_S_dsig*u.deg)
        dsig_use = dsig/self.dsigma_norm.to(u.cm**2).value
        self.diff_cross_section = RectBivariateSpline(wave_dsig, cos_th_S_dsig, dsig_use.T)

        return 

    def read_file(self, fname):
        wave = list()
        regs = ["_uv","","_ir"]
        for i, suffix in enumerate(regs):
            cat = open(fname+suffix)
            #Read the headers.
            k = 0
            for line in cat.readlines():
                k += 1
                if line[:7]=='wav(um)':
                    wave_aux = [float(ix) for ix in line.split()[1:]]
                    if i==0:
                        wave = np.zeros(len(wave_aux)*len(regs))
                    if suffix=="_ir":
                        wave[i*len(wave_aux):(i+1)*len(wave_aux)] = wave_aux
                    else:
                        wave[i*len(wave_aux):(i+1)*len(wave_aux)] = wave_aux[::-1]
                if line[0]=='-':
                    break
            cat.close()
            data = np.loadtxt(fname+suffix, skiprows=k)
            if i==0:
                p = np.zeros((data.shape[0], (data.shape[1]-1)*len(regs)))    
                theta = data[:,0]
            if suffix=="_ir":
                p[:,i*(data.shape[1]-1):(i+1)*(data.shape[1]-1)] = data[:,1:]
            else:
                p[:,i*(data.shape[1]-1):(i+1)*(data.shape[1]-1)] = data[:,-1:0:-1] 
        return wave, theta, p



    
