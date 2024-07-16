#!/usr/bin/env python 

import numpy as np
from astropy.io import fits
from astroscrappy import detect_cosmics

#Read in the f555w image. 
h = fits.open("hst_14358_03_wfc3_uvis_f555w_drz.fits")

# for i in range(1,len(h)):
#     print(h[i].header['EXTNAME'])

crmask, clean_im = detect_cosmics(h['SCI'].data * h['EXP'].data, gain=1.0, readnoise=3.25, objlim=5.0, sigclip=4.5, cleantype='median')

h['SCI'].data = clean_im / h['EXP'].data
h.writeto("test.fits", overwrite=True)
