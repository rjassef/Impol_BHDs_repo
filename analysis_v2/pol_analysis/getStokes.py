import numpy as np 
import re
import os

from astropy.io import fits
from astropy.table import Table

class GetStokes(object):

    def __init__(self, pdata):

        #Save the input.
        self.pdata = pdata

        #Load the zero angle corrections. 
        self.zero_angle_corrections = dict()
        cat = open("{}/tables/zero_angle_corrections.dat".format(os.path.dirname(os.path.realpath(__file__))))
        for line in cat:
            if line[0]=="#":
                continue
            x = line.split()
            self.zero_angle_corrections[x[0]] = float(x[1])
        cat.close()

        return
    
    def load_fluxes(self, ob_ids, mjds, chips):

        #Load the fluxes and uncertainties.
        self.esum = dict()
        self.osum = dict()
        self.eerr = dict()
        self.oerr = dict()
        for fname in self.pdata.list_of_filenames(ob_ids=ob_ids, mjds=mjds, chips=chips):
            pname = re.sub(".fits",".phot",fname)
            data = np.loadtxt("{}/{}".format(self.pdata.phot_folder, pname))
            self.esum[fname] = data[:,0]
            self.eerr[fname] = data[:,1]
            self.osum[fname] = data[:,2]
            self.oerr[fname] = data[:,3]
        return


    def get_QU_with_errors(self, e_pos_ref=None):

        #First, add all the observed e-beam and o-beam fluxes. This is what was recommended by A. Cikotta.
        f_e = dict()
        f_o = dict()
        ef_e2 = dict()
        ef_o2 = dict()
        for fname in self.esum.keys():
            h = fits.open("{0:s}/{1:s}".format(self.pdata.rim_folder, fname))
            th = "{0:.1f}".format(h[0].header['HIERARCH ESO INS RETA2 ROT'])
            h.close()
            if th in f_e.keys():
                f_e[th] += self.esum[fname]
                f_o[th] += self.osum[fname]
                ef_e2[th] += self.eerr[fname]**2
                ef_o2[th] += self.oerr[fname]**2
            else:
                f_e[th] = self.esum[fname]
                f_o[th] = self.osum[fname]
                ef_e2[th] = self.eerr[fname]**2
                ef_o2[th] = self.oerr[fname]**2

        #Now, calculate for every angle the normalized flux difference and its error through an MC.
        F = dict()
        F_resamp = dict()
        n_resamp = 1000
        for th in f_e.keys():
            F[th] = (f_o[th]-f_e[th])/(f_o[th]+f_e[th])
            # f_e_resamp = np.random.normal(f_e[th], ef_e2[th]**0.5, (n_resamp, len(f_e[th])))
            # f_o_resamp = np.random.normal(f_o[th], ef_o2[th]**0.5, (n_resamp, len(f_o[th])))
            resamp_shape = np.concatenate([[n_resamp],f_e[th].shape])
            f_e_resamp = np.random.normal(f_e[th], ef_e2[th]**0.5, resamp_shape)
            f_o_resamp = np.random.normal(f_o[th], ef_o2[th]**0.5, resamp_shape)
            F_resamp[th] = (f_o_resamp-f_e_resamp)/(f_o_resamp+f_e_resamp)

        #Finally, estimate the Stokes Q and U parameters.
        Q = 0.
        U = 0.
        Q_resamp = 0.
        U_resamp = 0.
        N = len(F)
        for th in F.keys():
            th_rad = float(th)*np.pi/180.
            Q += (2./N) * F[th] * np.cos(4.*th_rad)
            U += (2./N) * F[th] * np.sin(4.*th_rad)
            Q_resamp += (2./N) * F_resamp[th] * np.cos(4.*th_rad)
            U_resamp += (2./N) * F_resamp[th] * np.sin(4.*th_rad)        

        #Subtract the background. We approximate the position to that of the reference image.
        if e_pos_ref is None:
            e_pos_ref = np.loadtxt("{}/{}".format(self.pdata.phot_folder, re.sub(".fits",".epos",fname)))
            ex = e_pos_ref[:,0]
            ey = e_pos_ref[:,1]
        else:
            ex = np.array([e_pos_ref[0]])
            ey = np.array([e_pos_ref[1]])
        m = re.search("chip(.)", list(self.esum.keys())[0])
        ichip = m.group(1)
        Q_back = self.QU_background(ex, ey, "Q", ichip)
        U_back = self.QU_background(ex, ey, "U", ichip)

        Q -= Q_back
        U -= U_back
        #Q -= U_back
        #U -= Q_back
        self.Q = Q
        self.U = U
        self.pol_frac = (Q**2+U**2)**0.5
        self.pol_angle = 0.5*np.arctan2(U,Q)*180./np.pi

        #Now, we need to correct for the zero polarization angle, using Table 4.7 from the FORS2 manual. For R-band, the correction is -1.19 degrees.
        #self.pol_angle += 1.19
        self.pol_angle -= self.zero_angle_corrections[self.pdata.bband]

        self.pol_angle = np.where(self.pol_angle<0, 180.+self.pol_angle, self.pol_angle)

        Q_resamp -= Q_back
        U_resamp -= U_back
        self.epol_frac = np.std((Q_resamp**2+U_resamp**2)**0.5, axis=0)
        self.epol_angle = np.std(0.5*np.arctan2(U_resamp,Q_resamp)*180./np.pi, axis=0)
        self.dQ = np.std(Q_resamp, axis=0)
        self.dU = np.std(U_resamp, axis=0)

        return 
    
    def QU_background(self, x, y, STK_Par, chip):#, filter="R-band"):

        #Assign the correct filter in the table to the FORS2 name of the filter. 
        if self.pdata.bband=="R_SPECIAL":
            filter="R-band"
        elif self.pdata.bband=="I_BESS":
            filter="I-band"
        elif self.pdata.bband=="v_HIGH":
            filter="V-band"
        else:
            print("Unrecognized filter {} for QU corrections.".format(self.pdata.band))
            return np.zeros(x.shape)

        #Read the corefficients table.
        coeffs = Table.read("{}/tables/{}_Correction_Coefficients.dat".format(os.path.dirname(os.path.realpath(__file__)), STK_Par), format='ascii')
        a  = coeffs["a_{0:s}".format(STK_Par) ][coeffs['Filter']==filter] * 1e3
        b  = coeffs["b_{0:s}".format(STK_Par) ][coeffs['Filter']==filter] * 1e3
        th = coeffs["th_{0:s}".format(STK_Par)][coeffs['Filter']==filter] * np.pi/180.
        x0 = coeffs["x0_{0:s}".format(STK_Par)][coeffs['Filter']==filter] + 1226
        y0 = coeffs["y0_{0:s}".format(STK_Par)][coeffs['Filter']==filter] + 1025

        #Since we are only using the top image, we need to add 1024 to all the y values used in order to use this correction. No need to alter x.
        x_use = x
        if chip=="1":
            y_use = y + 1024
        elif chip=="2":
            y_use = y
        else:
            print("Chip ",chip,"is not valid. Asuming chip 1.")
            y_use = y + 1024

        #Estimate the correction.
        yp = (x_use-x0)*np.sin(th) + (y_use-y0)*np.cos(th)
        xp = (x_use-x0)*np.cos(th) - (y_use-y0)*np.sin(th)
        return (yp/b)**2 - (xp/a)**2
    
    def get_pol(self, ob_ids=None, mjds=None, chips=None):

        self.load_fluxes(ob_ids, mjds, chips)
        self.get_QU_with_errors()

        return 