import numpy as np
import astropy.units as u
from synphot import SpectralElement, SourceSpectrum, Observation
from synphot.models import Empirical1D

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


def get_a_b(theta, bands, spec, model):

    #Load the spectrum into a synphot model.
    full_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=spec.flam, keep_neg=True)

    #Use the best-fit model to load the spectrum for the I stoke parameters. 
    I_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=model.flam_model(spec.lam_rest), keep_neg=True)

    #The A spec corresponds to the continuum contribution to the stokes parameters. 
    A_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=model.flam_cont_model(spec.lam_rest) * theta(spec.lam_rest), keep_neg=True)

    #The B spec corresponds to the contribution of the lines with polarization to the Stokes parameters.
    flam_pol_lines_model = np.zeros(len(spec.lam_rest)) * model.flam_model(spec.lam_rest).unit
    for i in range(len(model.multi_line)):
        if model.multi_line[i].pol:
            flam_pol_lines_model += model.multi_line[i].flam_line_model(spec.lam_rest)
    B_spec = SourceSpectrum(Empirical1D, points=spec.lam_obs, lookup_table=flam_pol_lines_model * theta(spec.lam_rest), keep_neg=True)

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

