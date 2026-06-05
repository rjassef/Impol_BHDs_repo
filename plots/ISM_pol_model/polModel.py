import numpy as np
import astropy.units as u
import sys
from synphot import Empirical1D, SourceSpectrum, Observation
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, LinearConstraint

sys.path.append("Gas_and_dust/")
from polWaveDust import PolWaveDust
from polWaveGas import PolWaveGas

def get_p_bb(dust_type, specs, bands, wid, fw=True, bw=True, theta_angles=None, psi_angles=None, return_p_bb_interp=True):

    if theta_angles is None:
        theta_angles = np.arange(0., 90.1, 5.)
    if psi_angles is None:
        psi_angles = np.arange(0., 90.1, 5.)

    p_bb = dict()
    p_bb_interp = dict()

    if dust_type=="Gas":
        #model = PolWave_gas(fw=fw,bw=bw)
        model = PolWaveGas(folder="Gas_and_dust/gas_models/",fw=fw,bw=bw)
        for bname in bands.bp.keys():
            p_bb_interp[bname] = model.p
            p_bb[bname] = np.zeros((len(theta_angles), len(psi_angles)))
            p_bb[bname][:,:] = model.p

    else:
        model = PolWaveDust(dust_type, folder="Gas_and_dust/fullspec_dust_models/", fw=fw, bw=bw)

        full_spec = SourceSpectrum(Empirical1D, points=specs.lam_obs[wid], lookup_table=specs.flam[wid], keep_neg=True)

        for bname in bands.bp.keys():

            p_bb[bname] = np.zeros((len(theta_angles), len(psi_angles)))

            obs_I = Observation(full_spec, bands.bp[bname])
            Ibb = obs_I.effstim(flux_unit='flam').value

            for jtheta, theta in enumerate(theta_angles):
                for kpsi, psi in enumerate(psi_angles):
                    th_aux = theta*np.ones(len(specs.lam_obs[wid]))
                    psi_aux = psi*np.ones(len(specs.lam_obs[wid]))
                    p_lam = model.p((specs.lam_rest[wid].to(u.AA).value, th_aux, psi_aux))
            
                    Q_spec = SourceSpectrum(Empirical1D, points=specs.lam_obs[wid], lookup_table=specs.flam[wid] * p_lam, keep_neg=True)

                    obs_Q = Observation(Q_spec, bands.bp[bname])
                    Qbb = obs_Q.effstim(flux_unit='flam').value

                    p_bb[bname][jtheta, kpsi] = Qbb/Ibb
            p_bb_interp[bname] = RegularGridInterpolator((theta_angles, psi_angles), p_bb[bname])
    if return_p_bb_interp:
        return p_bb_interp
    else:
        return p_bb
    

def get_s1_bb(dust_type, specs, bands, wid, fw=True, bw=True, theta_angles=None, psi_angles=None):

    if theta_angles is None:
        theta_angles = np.arange(0., 90.1, 5.)
    if psi_angles is None:
        psi_angles = np.arange(0., 90.1, 5.)

    #s1_bb = np.zeros((len(bands), len(theta_angles), len(psi_angles)))
    s1_bb = dict()

    if dust_type=="gas":
        model = PolWaveGas(folder="Gas_and_dust/gas_models/",fw=fw,bw=bw)
        for bname in bands.bp.keys():
            s1_bb[bname] = np.zeros((len(theta_angles), len(psi_angles)))
            for jtheta, theta in enumerate(theta_angles):
                th_aux = theta*np.ones(len(psi_angles))
                s1_bb[bname][jtheta] = model.s1((th_aux, psi_angles))
                

    else:
        model = PolWaveDust(dust_type, folder="Gas_and_dust/fullspec_dust_models/", fw=fw, bw=bw)

        full_spec = SourceSpectrum(Empirical1D, points=specs.lam_obs[wid], lookup_table=specs.flam[wid], keep_neg=True)

        for bname in bands.bp.keys():

            s1_bb[bname] = np.zeros((len(theta_angles), len(psi_angles)))

            obs_I = Observation(full_spec, bands.bp[bname], force='extrap')
            Ibb = obs_I.effstim(flux_unit='flam').value
            for jtheta, theta in enumerate(theta_angles):
                for kpsi, psi in enumerate(psi_angles):

                    th_aux = theta*np.ones(len(specs.lam_obs[wid]))
                    psi_aux = psi*np.ones(len(specs.lam_obs[wid]))
                    s1_lam = model.s1((specs.lam_rest[wid].to(u.AA).value, th_aux, psi_aux))
           
                    S1_spec = SourceSpectrum(Empirical1D, points=specs.lam_obs[wid], lookup_table=specs.flam[wid] * s1_lam, keep_neg=True)

                    obs_S1 = Observation(S1_spec, bands.bp[bname], force='extrap')
                    S1bb = obs_S1.effstim(flux_unit='flam').value
                    
                    s1_bb[bname][jtheta, kpsi] = S1bb/Ibb
    return s1_bb
    
    