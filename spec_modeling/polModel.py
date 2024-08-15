import numpy as np
import astropy.units as u
from synphot import SpectralElement, SourceSpectrum, Observation
from synphot.models import Empirical1D
#from draine_dust_2D import draine_dust
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import os
import matplotlib.transforms
from scipy.signal import savgol_filter

class PolModel(object):

    def __init__(self, spec, spec_model, bands, op):

        # #Load the measured values of p.
        # data = np.loadtxt("pol_measurements.dat",usecols=[1,2])
        # self.p_measured = data[:,0]
        # self.p_unc = data[:,1]

        #Load the filters. 
        # self.load_bands()
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


    def get_a_b(self, pfrac_A, pfrac_B):

        #Load the spectrum into a synphot model.
        full_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec.flam, keep_neg=True)

        #Use the best-fit model to load the spectrum for the I stoke parameters. 
        I_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec_model.flam_model(self.spec.lam_rest), keep_neg=True)

        #The A spec corresponds to the continuum contribution to the stokes parameters. 
        A_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec_model.flam_cont_model(self.spec.lam_rest) * pfrac_A, keep_neg=True)

        #The B spec corresponds to the contribution of the lines with polarization to the Stokes parameters.
        flam_pol_lines_model = np.zeros(len(self.spec.lam_rest)) * self.spec_model.flam_model(self.spec.lam_rest).unit
        for i in range(len(self.spec_model.multi_line)):
            if self.spec_model.multi_line[i].pol:
                flam_pol_lines_model += self.spec_model.multi_line[i].flam_line_model(self.spec.lam_rest)
        B_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=flam_pol_lines_model * pfrac_B, keep_neg=True)

        #Now, precompute the BB parameters. 
        a = np.zeros(len(self.bands))
        b = np.zeros(len(self.bands))
        for j, band in enumerate(self.bands):
            binset_cond = self.spec.lam_obs.to(u.AA).value<band._model.points[0].max()
            binset_cond = binset_cond & (self.spec.lam_obs.to(u.AA).value>band._model.points[0].min())
            binset=self.spec.lam_obs[binset_cond]
            #obs_I = Observation(full_spec, band, binset=binset)
            obs_I = Observation(I_spec, band, binset=binset)
            obs_A = Observation(A_spec, band, binset=binset)
            obs_B = Observation(B_spec, band, binset=binset)

            I_BB = obs_I.effstim(flux_unit='flam').value
            a[j] = obs_A.effstim(flux_unit='flam').value/I_BB
            b[j] = obs_B.effstim(flux_unit='flam').value/I_BB

        return a, b

    #Function to return the broad-band p calculated from the model.
    def model_p(self, x, scat_obj):
        pfrac_A = scat_obj.pfrac_A(x, self.spec.lam_rest)
        pfrac_B = scat_obj.pfrac_B(x, self.spec.lam_rest)
        a, b = self.get_a_b(pfrac_A, pfrac_B)
        return (a**2 + b**2 + 2*a*b*np.cos(2*x[0]*u.deg))**0.5

    #Function to return the broad-band polarization angle under the assumption that the continuum polarization angle is 0. 
    def model_chiBB(self, x, scat_obj):
        pfrac_A = scat_obj.pfrac_A(x, self.spec.lam_rest)
        pfrac_B = scat_obj.pfrac_B(x, self.spec.lam_rest)
        a, b = self.get_a_b(pfrac_A, pfrac_B)
        return 0.5*np.arctan2(b*np.sin(2*x[0]*u.deg),(a+b*np.cos(2*x[0]*u.deg)))     

    def chi2(self, x, scat_obj):
        p_mod = self.model_p(x, scat_obj)
        return np.sum(((self.p_measured-p_mod)/self.p_unc)**2)

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

    # def fit_pol(self, get_pmod, dust_types, bands, spec, x0 = np.array([0.5, 0.5, 70., 0.]), force_forward=False, force_backward = False):
        
    #     self.xopt_all = dict()
    #     self.mod_p = np.zeros(len(self.bands))
    #     for k, dust_type in enumerate(dust_types):

    #         dust = draine_dust(type=dust_type)

    #         #Set the limits for each of the fitted values.
    #         G = np.identity(x0.shape[0])
    #         min_vals = [0., 0., 0., 0.]
    #         max_vals = [1., 1., 90., 180.]

    #         #The code has difficulties converging above and below 90 deg. So let's try both and save the best.
    #         for i in range(2):
    #             if i==0:
    #                 if force_backward:
    #                     continue
    #                 if force_forward:
    #                     min_vals[-1] = 0.
    #                     max_vals[-1] = 90.
    #                 x0[-1] = 60.
    #             if i==1:
    #                 if force_forward:
    #                     continue
    #                 if force_backward:
    #                     min_vals[-1] = 90.
    #                     max_vals[-1] = 180.
    #                 x0[-1] = 170.
    #             lincon = LinearConstraint(G, min_vals, max_vals)
    #             xopt_aux = minimize(self.chi2, x0=x0, constraints=lincon, args=(get_pmod, dust))
    #             if "xopt" not in locals() or xopt_aux.fun < xopt.fun:
    #                 xopt = xopt_aux

    #         print(xopt)
    #         self.mod_p[k] = get_pmod(xopt.x, dust, spec)


    #         self.xopt_all[dust_type] = xopt
    #         del xopt

    #     return

    # def pol_plot(self, mod_p, dust_types):

    #     wave = np.array([5500., 6500., 8000.]) / self.spec.zspec

    #     fig, ax = plt.subplots(1)

    #     ax.errorbar(wave, self.p_measured, yerr=self.p_unc, fmt='ko', label='Measurements')
    #     for k, dust_type in enumerate(dust_types):
    #         ax.plot(wave, mod_p[k], 's', label=dust_type)
    #     ax.legend()
    #     ax.set_xlabel('Wavelength (Angstroms)')
    #     ax.set_ylabel('Polarization fraction')
    #     plt.show()

