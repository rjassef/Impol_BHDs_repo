import sys
import os
SPEC_PIPE_LOC = "{}/JPL/W12_Drops/spec_paper/Spec_pipeline".format(os.path.expanduser("~"))
sys.path.append(SPEC_PIPE_LOC)
os.environ['SPEC_PIPE_LOC'] = SPEC_PIPE_LOC

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from Spec_pipeline import SDSS_Spec, Powc_Line_fit

def model_W0116_spec(specs):

    #Read the spectrum.
    spec = specs.specs.sp['W0116-0505']

    #Create the model.
    model = Powc_Line_fit(spec=spec)

    #Add the non-polarized emission lines.
    model.add_line('Lyalpha')
    model.multi_line[-1].line_center[0] = 1220.*u.AA
    model.multi_line[-1].pol = False
    model.add_line('Lyalpha')
    model.multi_line[-1].pol = False

    #Add the polarized emission lines.
    model.add_line('CIV')
    model.multi_line[-1].pol = True
    model.add_line('NV')
    model.multi_line[-1].pol = True
    model.add_line('SiIV-OIV]')
    model.multi_line[-1].pol = True
    #model.multi_line[-1].line_center[0] = 1398.*u.AA

    #Replace the continuum regions so that the full spectrum is used. 
    full_range = np.zeros(2)*u.AA
    full_range[0] = 1260.*u.AA
    full_range[1] = np.max(spec.lam_rest)
    model.multi_line[-1].continuum_regions[1] = full_range

    #Run the fitting
    model.run_fit()

    #This line is necessary to not have issues with plotting.
    model.line_name = "" 

    return spec, model