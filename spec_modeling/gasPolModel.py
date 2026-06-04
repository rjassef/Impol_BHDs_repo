import numpy as np
import astropy.units as u
from synphot import SpectralElement, SourceSpectrum, Observation
from synphot.models import Empirical1D
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee
import matplotlib.transforms
from scipy.signal import savgol_filter

import os
gd_folder = os.getcwd()+"/../Calculations/Gas_and_dust/"

import sys 
sys.path.append(gd_folder)
from polWaveGas import PolWaveGas

#########
# Module to only carry out gas polarization modeling of W0116-0505. By focusing on gas only, we can enable a fast MCMC approach. 
#########

# def ln_prob(x, pol_model, scat_obj, min_vals, max_vals, ifix):
#     #x_use = np.where(min_vals==max_vals, min_vals, x)
#     #if np.any(x_use < min_vals) or np.any(x_use > max_vals):
#     if np.any(x < min_vals[~ifix]) or np.any(x>max_vals[~ifix]):
#         return -np.inf
#     x_use = np.copy(min_vals)
#     x_use[~ifix] = np.copy(x)
#     return -0.5 * pol_model.chi2(x_use, scat_obj)

class PolModel(object):

    def __init__(self, spec, spec_model, bands, op):

        #Order of the bands must be v, R and then I. 
        self.bands = list()
        self.p_measured = np.zeros(3)
        self.p_unc = np.zeros(3)
        for j,bname in enumerate(["v_HIGH","R_SPECIAL","I_BESS"]):
            self.bands.append(bands.bp[bname])
            self.p_measured[j] = op.pfrac["W0116-0505"][bname]/100.
            self.p_unc[j] = op.epfrac["W0116-0505"][bname]/100.

        #Save the spec and the spec model loaded for the pol model. 
        self.spec = spec
        self.spec_model = spec_model

        #All the synphot objects need to be precomputed. 
        self.I_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec_model.flam_model(self.spec.lam_rest), keep_neg=True)
        self.A_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec_model.flam_cont_model(self.spec.lam_rest), keep_neg=True)

        flam_pol_lines_model = np.zeros(len(self.spec.lam_rest)) * self.spec_model.flam_model(self.spec.lam_rest).unit
        for i in range(len(self.spec_model.multi_line)):
            if self.spec_model.multi_line[i].pol:
                flam_pol_lines_model += self.spec_model.multi_line[i].flam_line_model(self.spec.lam_rest)
        self.B_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=flam_pol_lines_model, keep_neg=True)

        self.I_BB = np.zeros(len(self.bands))
        self.A_BB = np.zeros(self.I_BB.shape)
        self.B_BB = np.zeros(self.I_BB.shape)
        for j, band in enumerate(self.bands):
            binset_cond = self.spec.lam_obs.to(u.AA).value<band._model.points[0].max()
            binset_cond = binset_cond & (self.spec.lam_obs.to(u.AA).value>band._model.points[0].min())
            binset=self.spec.lam_obs[binset_cond]
            obs_I = Observation(self.I_spec, band, binset=binset)
            obs_A = Observation(self.A_spec, band, binset=binset)
            obs_B = Observation(self.B_spec, band, binset=binset)

            self.I_BB[j] = obs_I.effstim(flux_unit='flam').value
            self.A_BB[j] = obs_A.effstim(flux_unit='flam').value/self.I_BB[j]
            self.B_BB[j] = obs_B.effstim(flux_unit='flam').value/self.I_BB[j]

        return

    def spec_plot(self, smooth=True):

        fig, ax = plt.subplots(1)

        ax.set_ylim([-0.2e-16, 1.1e-16])
        ax.set_xlim([4000., 10000.])

        if smooth:
            flam_aux = savgol_filter(self.spec.flam, 7, 3)
            ax.plot(self.spec.lam_obs, flam_aux, label="SDSS Spec")
        else:
            ax.plot(self.spec.lam_obs, self.spec.flam, label="SDSS Spec")

        ax.plot(self.spec.lam_obs, self.spec_model.flam_model(self.spec.lam_rest))
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for j, band in enumerate(self.bands):
            if j==0:
                norm = 1./100.
            else: 
                norm = 1.
            plt.plot(band.model.points[0], band.model.lookup_table*norm, color='xkcd:grey', transform=trans, alpha=0.5)

        ax.set_ylabel(r'$f_{{\lambda}}$ ({})'.format(self.spec.flam.unit))
        ax.set_xlabel(r'$\lambda$ ({})'.format(self.spec.lam_obs.unit))

        plt.show()

        return 

    #Function to return the broad-band p calculated from the model.
    def model_p(self, x, scat_obj):
        pfrac_A = scat_obj.pfrac_A(x)
        pfrac_B = scat_obj.pfrac_B(x)
        a = self.A_BB * pfrac_A
        b = self.B_BB * pfrac_B
        return (a**2 + b**2 + 2*a*b*np.cos(2*x[0]*u.deg))**0.5

    #Function to return the broad-band polarization angle under the assumption that the continuum polarization angle is 0. 
    def model_chiBB(self, x, scat_obj):
        pfrac_A = scat_obj.pfrac_A(x)
        pfrac_B = scat_obj.pfrac_B(x)
        a = self.A_BB * pfrac_A
        b = self.B_BB * pfrac_B
        return 0.5*np.arctan2(b*np.sin(2*x[0]*u.deg),(a+b*np.cos(2*x[0]*u.deg)))     

    def chi2(self, x, scat_obj):
        p_mod = self.model_p(x, scat_obj)
        return np.sum(((self.p_measured-p_mod)/self.p_unc)**2)
    
    def ln_prob(self, x, scat_obj, min_vals, max_vals, ifix):
        if np.any(x < min_vals[~ifix]) or np.any(x>max_vals[~ifix]):
            return -np.inf
        x_use = np.copy(min_vals)
        x_use[~ifix] = np.copy(x)
        return -0.5 * self.chi2(x_use, scat_obj)

    #Function to find the best-fit to the polarization.
    def fit_pol(self, scat_obj, x0, min_vals, max_vals, method=None):

        #Set the linear constraints.
        G = np.identity(x0.shape[0])
        lincon = LinearConstraint(G, min_vals, max_vals)

        #Run the fit
        self.xopt = minimize(self.chi2, x0=x0, constraints=lincon, args=(scat_obj), method=method)        

        #Save the best-fit model broad-band polarizations.
        self.mod_p   = self.model_p(self.xopt.x, scat_obj)
        self.mod_chi = self.model_chiBB(self.xopt.x, scat_obj)

        #Print the results.
        print(self.xopt.message)
        print(self.xopt.x)
        print(self.xopt.fun)
        if hasattr(self.xopt, "hess_inv"):
            print(np.diagonal((self.xopt.hess_inv))**0.5)
    
        return


    #Function to find the best-fit to the polarization.
    def fit_pol_MCMC(self, scat_obj, x0, min_vals, max_vals, nwalkers=250, nrep=5000, nburn=500, nthread=None):

        ifix = np.zeros(len(x0), dtype=bool)
        ifix[min_vals==max_vals] = True

        #Set the starting point. 
        ndim = len(x0[~ifix])
        p0 = x0[~ifix] + 1e-4 * np.random.randn(nwalkers, ndim)

        #Force fork mode for multiprocessing
        mp.set_start_method('fork', force=True)

        if nthread is None:
            nthread = mp.cpu_count()-1 

        with mp.Pool(processes=nthread) as pool:

            #Set the sampler
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, args=(scat_obj, min_vals, max_vals, ifix), pool=pool)

            #Run the burn-in steps
            state = self.sampler.run_mcmc(p0, nburn, progress=True)
            self.sampler.reset()

            #Now, run the production step
            self.sampler.run_mcmc(state, nrep, progress=True)

        #Once done, get the flat chain. Here we need to reinsert the fixed values. 
        self.flat_chain = self.sampler.get_chain(flat=True)
        self.flat_samples = np.zeros((self.flat_chain.shape[0], len(x0)))
        self.flat_samples[:,~ifix] = self.flat_chain
        self.flat_samples[:,ifix] = min_vals[ifix] 

        return
