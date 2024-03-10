#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import subprocess
import os
from astropy.table import Table

#Make a list of all the fits files.
ls_output = subprocess.run("ls rawdata/*.fits", shell=True, capture_output=True)
fnames = ls_output.stdout.decode('utf8').split()

#Load the blacklist of files. 
black_list = Table.read("blacklist.txt", format='ascii.no_header')

#Now, go by file and put them in the directory they belong.
for fname in fnames:

    #Check if the file is blacklisted. If so, skip. 
    if fname[8:] in black_list['col1']:
        continue

    #Open the fits file.
    h = fits.open(fname)

    #Get the object name.
    obj = h[0].header['OBJECT']

    #All observations where done in 2x2 binning, but some biases where obtained in 1x1. Do not distribute those. 
    if obj=="BIAS":
        if h[0].header['HIERARCH ESO DET WIN1 BINX']!=2 or h[0].header['HIERARCH ESO DET WIN1 BINY']!=2:
            continue

    #Chip
    chip = h[0].header['EXTNAME']

    #Type of observation.
    obs_type = h[0].header['HIERARCH ESO DPR TECH']

    #MJD of the observations.
    mjd = "{0:.0f}".format(np.round(h[0].header['MJD-OBS'],0))

    #Filter
    try:
        filt = h[0].header['HIERARCH ESO INS FILT1 NAME']
    except KeyError:
        filt = None

    out_path_elements = ["proc",obj,mjd,obs_type,chip,filt]
    out_path = ""
    for element in out_path_elements:
        if element is None:
            continue
        out_path += element+"/"
        if not os.path.exists(out_path):
            subprocess.call(["mkdir",out_path])

    #Finally, copy the files.
    subprocess.run(["cp",fname,out_path])
    h.close()
