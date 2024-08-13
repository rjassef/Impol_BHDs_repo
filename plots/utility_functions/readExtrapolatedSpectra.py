import numpy as np 
import os
from scipy.interpolate import CubicSpline
import astropy.units as u
from astropy.constants import c

import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../spec_data/")
from loadSpecs import LoadSpecs

class ReadExtrapolatedSpectra(object):

    def __init__(self, wids=None):

        #Start by reading the spectra.
        self.specs = LoadSpecs()

        #Now, if no wids have been declared, assume all will be used. 
        if wids is None:
            self.wids = list(self.specs.sp.keys())
        else:
            self.wids = wids

        #Save folder where the script lives. 
        self.script_folder = os.path.dirname(os.path.realpath(__file__))

        #Read the best-fit SED models.
        self._read_seds()

        #For the requested sources, get the extrapolated spectra. 
        self._extrapolate_specs()
        self.lam_rest = dict()
        for wid in self.wids:
            self.lam_rest[wid] = self.lam_obs[wid]/(1+self.specs.sp[wid].zspec)

        return

    def _extrapolate_specs(self):

        self.lam_obs = dict()
        self.flam = dict()
        for wid in self.wids:

            #For simplicity, assign the spectrum to be used to spec. 
            spec = self.specs.sp[wid]

            #Create an interpolated version of the sed model. 
            sed_flam_interp = CubicSpline(self.sed_lam_obs[wid].to(spec.lam_obs.unit).value, self.sed_flam[wid].to(spec.flam.unit).value)

            #Output wavelength grid (extend to observed-frame 10000 A) with the same median wavelength resolution of the spectrum.
            lam_max =  1e4*u.AA
            if np.max(spec.lam_obs)>lam_max:
                lam_obs_ex = spec.lam_obs
                self.lam_obs[wid] = lam_obs_ex
                self.flam[wid] = flam_ex 
            median_dlam = np.median(spec.lam_obs[1:]-spec.lam_obs[:-1]).value
            lam_1 = np.max(spec.lam_obs).value
            lam_2 = lam_max.to(spec.lam_obs.unit).value
            lam_obs_ex = np.arange(lam_1+median_dlam, lam_2, median_dlam) * spec.lam_obs.unit
            lam_obs_ex = np.concatenate([spec.lam_obs, lam_obs_ex])

            #Calculate the normalization to join them.
            iw = 30
            norm1 = spec.flam[-iw:].value * sed_flam_interp(spec.lam_obs[-30:].value)
            norm2 = sed_flam_interp(spec.lam_obs[-iw:].value)**2
            norm = np.sum(norm1)/np.sum(norm2)

            #Join the spectra and SED smoothly. 
            c1 = 1. - np.arange(iw)/iw
            flam_ex = np.zeros(len(lam_obs_ex))*spec.flam.unit
            flam_ex[:len(spec.lam_obs)-iw] = spec.flam[:-iw]
            flam_ex[len(spec.lam_obs)-iw:len(spec.lam_obs)] = spec.flam[-iw:]*c1 + (1-c1)*norm*sed_flam_interp(spec.lam_obs[-iw:])*spec.flam.unit
            flam_ex[len(spec.lam_obs):] = norm*sed_flam_interp(lam_obs_ex[len(spec.lam_obs):])*spec.flam.unit

            self.lam_obs[wid] = lam_obs_ex
            self.flam[wid] = flam_ex
        return


    def _read_seds(self):

        #Read the best-fit SED model. 
        self.sed_lam_obs = dict()
        self.sed_flam = dict()
        for wid in self.wids:
            sed = np.loadtxt("{}/../../SED_Modeling/{}.SED.txt".format(self.script_folder, wid))
            sed_lam_obs = sed[:,0] * u.micron * (1+self.specs.sp[wid].zspec)
            sed_nu_obs = c/sed_lam_obs
            sed_fnu = np.sum(sed[:,1:], axis=1) * 1e-13*u.erg/u.s/u.cm**2 
            sed_fnu = sed_fnu/sed_nu_obs
            sed_flam = sed_fnu * c/sed_lam_obs**2
            self.sed_lam_obs[wid] = sed_lam_obs
            self.sed_flam[wid] = sed_flam
        return


