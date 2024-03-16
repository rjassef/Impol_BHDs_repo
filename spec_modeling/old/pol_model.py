import numpy as np
import astropy.units as u
from synphot import SpectralElement, SourceSpectrum, Observation
from synphot.models import Empirical1D
from draine_dust_2D import draine_dust
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

def load_bands():
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

    bands = [vbp, Rbp, Ibp]
    return bands


def get_a_b(theta_A, theta_B, bands, spec, model):

    #Load the spectrum into a synphot model.
    full_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=spec.flam, keep_neg=True)

    #Use the best-fit model to load the spectrum for the I stoke parameters. 
    I_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=model.flam_model(spec.lam_rest), keep_neg=True)

    #The A spec corresponds to the continuum contribution to the stokes parameters. 
    A_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=model.flam_cont_model(spec.lam_rest) * theta_A, keep_neg=True)

    #The B spec corresponds to the contribution of the lines with polarization to the Stokes parameters.
    flam_pol_lines_model = np.zeros(len(spec.lam_rest)) * model.flam_model(spec.lam_rest).unit
    for i in range(len(model.multi_line)):
        if model.multi_line[i].pol:
            flam_pol_lines_model += model.multi_line[i].flam_line_model(spec.lam_rest)
    B_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=flam_pol_lines_model * theta_B, keep_neg=True)

    #Now, precompute the BB parameters. 
    a = np.zeros(len(bands))
    b = np.zeros(len(bands))
    for j, band in enumerate(bands):
        obs_I = Observation(full_spec, band)
        obs_A = Observation(A_spec, band)
        obs_B = Observation(B_spec, band)

        I_BB = obs_I.effstim(flux_unit='flam').value
        a[j] = obs_A.effstim(flux_unit='flam').value/I_BB
        b[j] = obs_B.effstim(flux_unit='flam').value/I_BB

    return a, b

#Function to return the broad-band p calculated from the model.
def model_p(x, a, b):
    return ((x[0]*a)**2 + (x[1]*b)**2 + 2*x[0]*x[1]*a*b*np.cos(2*x[2]*u.deg))**0.5


def chi2(x, get_pmod, p_measured, p_unc, dust, spec, model):
    p_mod = get_pmod(x, dust, spec, model)
    return np.sum(((p_measured-p_mod)/p_unc)**2)

#Function to find the best-fit to the polarization.
def fit_pol(get_pmod, p_measured, p_unc, dust_types, bands, spec, model, x0 = np.array([0.5, 0.5, 70., 0.]), force_forward=False, force_backward = False):
    
    xopt_all = dict()
    mod_p = np.zeros((len(dust_types),len(bands)))
    for k, dust_type in enumerate(dust_types):

        dust = draine_dust(type=dust_type)

        G = np.identity(x0.shape[0])
        min_vals = [0., 0., 0., 0.]
        max_vals = [1., 1., 90., 180.]

        #The code has difficulties converging above and below 90 deg. So let's try both and save the best.
        for i in range(2):
            if i==0:
                if force_backward:
                    continue
                if force_forward:
                    min_vals[-1] = 0.
                    max_vals[-1] = 90.
                x0[-1] = 60.
            if i==1:
                if force_forward:
                    continue
                if force_backward:
                    min_vals[-1] = 90.
                    max_vals[-1] = 180.
                x0[-1] = 170.
            lincon = LinearConstraint(G, min_vals, max_vals)
            xopt_aux = minimize(chi2, x0=x0, constraints=lincon, args=(get_pmod, p_measured, p_unc, dust, spec, model))
            if "xopt" not in locals() or xopt_aux.fun < xopt.fun:
                xopt = xopt_aux

        print(xopt)

        # phi = xopt.x[3]
        # theta = dust.pfrac(spec.lam_rest.to(u.um).value, phi).flatten()
        # a, b = get_a_b(theta, theta, bands, spec, model)

        # for j in range(len(bands)):
        #     mod_p[k,j] = model_p(xopt.x[:3], a[j], b[j])
            #print(mod_p[k,j])
        mod_p[k] = get_pmod(xopt.x, dust, spec)


        xopt_all[dust_type] = xopt
        del xopt

    return xopt_all, mod_p

def pol_plot(mod_p, p_measured, p_unc, spec, dust_types):

    wave = np.array([5500., 6500., 8000.]) / spec.zspec

    fig, ax = plt.subplots(1)

    ax.errorbar(wave, p_measured, yerr=p_unc, fmt='ko', label='Measurements')
    for k, dust_type in enumerate(dust_types):
        ax.plot(wave, mod_p[k], 's', label=dust_type)
    ax.legend()
    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('Polarization fraction')
    plt.show()

