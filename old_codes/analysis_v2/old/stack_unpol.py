#!/usr/bin/env python 

import numpy as np
import re
from astropy.io import fits

from crz import crz_clean
from util import join_chips

#Setup important parameters.
mask = None
root = "/Volumes/Maxell_SSD/FORS2/"
rim_folder = root+"proc/unpol_reduced_images_with_skyflat/"
crz_folder = root+"analysis/unpol_crz_with_skyflat/"

#Set up the files to process.
mjds = [59135, 59136, 59137]
fnames = []
for mjd in mjds:
    fnames.append([])
    for ichip in (1,2):
        fnames[-1].append("science_reduced_img.{0:d}.chip{1:d}.1.fits".format(mjd,ichip))

#First, clean the cosmic rays. 
for fnames_mjd in fnames:
    for fname in fnames_mjd: 
        crz_clean(fname, mask, rim_folder, crz_folder)

#Now, join the chips for each MJD.
for k, mjd in enumerate(mjds):    
    im1 = fits.open(crz_folder+re.sub(".fits",".crz.fits",fnames[k][0]))
    im2 = fits.open(crz_folder+re.sub(".fits",".crz.fits",fnames[k][1]))
    im_combined = join_chips(im1, im2)
    fname_out = "stacked/unpol_{0:d}.fits".format(mjd)
    fits.writeto(fname_out, im_combined)

