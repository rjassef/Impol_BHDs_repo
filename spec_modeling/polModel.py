import numpy as np
import astropy.units as u
from synphot import SpectralElement, SourceSpectrum, Observation
from synphot.models import Empirical1D
#from draine_dust_2D import draine_dust
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt


class PolModel(object):

    def __init__(self, spec, spec_model):

        #Load the measured values of p.
        data = np.loadtxt("pol_measurements.dat",usecols=[1,2])
        self.p_measured = data[:,0]
        self.p_unc = data[:,1]

        #Load the filters. 
        self.load_bands()

        #Save the spec and the spec model loaded for the pol model. 
        self.spec = spec
        self.spec_model = spec_model

        return

    def load_bands(self):
        #Load the filters.
        R_spec = np.loadtxt("M_SPECIAL_R.txt")
        I_bess = np.loadtxt("M_BESS_I.txt")
        v_high = np.loadtxt("v_HIGH.txt", skiprows=2)

        R_spec = R_spec[R_spec[:,1]>0.01]
        I_bess = I_bess[I_bess[:,1]>0.01]
        v_high = v_high[v_high[:,1]>0.01]

        #Transform the wavelengths to angstroms.
        R_spec[:,0] *= 10
        I_bess[:,0] *= 10
        v_high[:,0] *= 10

        v_high = v_high[(v_high[:,0]>4000.) & (v_high[:,0]<8000.)]
        I_bess = I_bess[(I_bess[:,0]>4000.) & (I_bess[:,0]<8000.)]
        R_spec = R_spec[(R_spec[:,0]>4000.) & (R_spec[:,0]<8000.)]

        Rbp= SpectralElement(Empirical1D, points=R_spec[:,0], lookup_table=R_spec[:,1]/100., keep_neg=False)
        Ibp= SpectralElement(Empirical1D, points=I_bess[:,0], lookup_table=I_bess[:,1]/100., keep_neg=False)
        vbp= SpectralElement(Empirical1D, points=v_high[:,0], lookup_table=v_high[:,1]/100., keep_neg=False)

        self.bands = [vbp, Rbp, Ibp]
        return


    def get_a_b(self, theta_A, theta_B):

        #Load the spectrum into a synphot model.
        full_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec.flam, keep_neg=True)

        #Use the best-fit model to load the spectrum for the I stoke parameters. 
        I_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec_model.flam_model(self.spec.lam_rest), keep_neg=True)

        #The A spec corresponds to the continuum contribution to the stokes parameters. 
        A_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=self.spec_model.flam_cont_model(self.spec.lam_rest) * theta_A, keep_neg=True)

        #The B spec corresponds to the contribution of the lines with polarization to the Stokes parameters.
        flam_pol_lines_model = np.zeros(len(self.spec.lam_rest)) * self.spec_model.flam_model(self.spec.lam_rest).unit
        for i in range(len(self.spec_model.multi_line)):
            if self.spec_model.multi_line[i].pol:
                flam_pol_lines_model += self.spec_model.multi_line[i].flam_line_model(self.spec.lam_rest)
        B_spec = SourceSpectrum(Empirical1D, points=self.spec.lam_obs, lookup_table=flam_pol_lines_model * theta_B, keep_neg=True)

        #Now, precompute the BB parameters. 
        a = np.zeros(len(self.bands))
        b = np.zeros(len(self.bands))
        for j, band in enumerate(self.bands):
            obs_I = Observation(full_spec, band)
            obs_A = Observation(A_spec, band)
            obs_B = Observation(B_spec, band)

            I_BB = obs_I.effstim(flux_unit='flam').value
            a[j] = obs_A.effstim(flux_unit='flam').value/I_BB
            b[j] = obs_B.effstim(flux_unit='flam').value/I_BB

        return a, b

    #Function to return the broad-band p calculated from the model.
    def model_p(self, x, scat_obj):
        theta_A = scat_obj.get_theta_A(x, self.spec.lam_rest)
        theta_B = scat_obj.get_theta_B(x, self.spec.lam_rest)
        a, b = self.get_a_b(theta_A, theta_B)
        return ((x[0]*a)**2 + (x[1]*b)**2 + 2*x[0]*x[1]*a*b*np.cos(2*x[2]*u.deg))**0.5

    def chi2(self, x, scat_obj):
        #p_mod = get_pmod(x, dust, self.spec, self.spec_model)
        p_mod = self.model_p(x, scat_obj)
        return np.sum(((self.p_measured-p_mod)/self.p_unc)**2)

    #Function to find the best-fit to the polarization.
    def fit_pol(self, scat_obj, x0, min_vals, max_vals):

        #Set the linear constraints.
        G = np.identity(x0.shape[0])
        lincon = LinearConstraint(G, min_vals, max_vals)

        #Run the fit
        self.xopt = minimize(self.chi2, x0=x0, constraints=lincon, args=(scat_obj))        

        #Save the best-fit model broad-band polarizations.
        self.mod_p = self.model_p(self.xopt.x, scat_obj)
    
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
