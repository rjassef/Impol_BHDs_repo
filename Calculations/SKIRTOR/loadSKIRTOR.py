import numpy as np
import subprocess
import re
import os
from scipy.interpolate import RegularGridInterpolator

class LoadSKIRTOR(object):

    def __init__(self, folder=None, interp_method='linear'):

        if folder is None:
            self.folder = os.path.dirname(os.path.realpath(__file__))+"/models/"
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
        self.tang_grid = np.unique(tangle)
        self.cang_grid = np.unique(cangle)
        self.iang_grid = np.unique(iangle)

        #Make a list of the wavelengths. 
        self.lam_grid = np.loadtxt(fnames[0], usecols=[0])

        #Now make the array holding the polarization grid. 
        self.p_grid = np.ma.zeros((len(self.tang_grid), len(self.cang_grid), len(self.iang_grid), len(self.lam_grid)))
        self.p_grid.mask = np.zeros(self.p_grid.shape, dtype=bool)
        for i, tang in enumerate(self.tang_grid):
            for j, cang in enumerate(self.cang_grid):
                for k, iang in enumerate(self.iang_grid):
                    fname = "{}/bHDPol_mrn77_tor_oa{:.0f}_con_oa{:.0f}-tauV1_i{:.0f}_sed.dat".format(self.folder, tang, cang, iang)
                    if fname in fnames:
                        data = np.loadtxt(fname)
                        self.p_grid[i,j,k] = -data[:,2]/data[:,1]
                    else:
                        #self.p_grid[i,j,k] = np.nan * np.ones(len(self.lam_grid))
                        if iang>tang and cang<tang:
                           self.p_grid.mask[i,j,k] = np.ones(len(self.lam_grid), dtype=bool)

        #Now, make the interpolator.
        self.p = RegularGridInterpolator((self.tang_grid, self.cang_grid, self.iang_grid, self.lam_grid), self.p_grid, bounds_error=False, fill_value=0.0, method=interp_method)

        return


