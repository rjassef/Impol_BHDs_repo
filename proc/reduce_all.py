#!/usr/bin/env python

import numpy as np
import subprocess
import os
import re

from proc_routines import *

#Define important folders
root = "/home/rjassef/Impol_Blue_HotDOGs/reduction/proc/"

conf_folder = root+"conf_files"
bias_folder = root+"BIAS"
skyflat_folder = root+"FLAT,SKY"
std_folder  = root+"STD"
#scipol_folder = root+"W0116-0505"
scipol_folders = [
    root+"W0116-0505",
    root+"W0220+0137",
    root+"W0019-1046",
    root+"W0204-0506",
    root+"W0831+0140",
]

cal_folder  = root+"reduced_calibrations"
rim_folder  = root+"reduced_images"
phot_folder = root+"phot_cats"

#Create the output folders if needed.
subprocess.call(["mkdir",cal_folder])
subprocess.call(["mkdir",rim_folder])
subprocess.call(["mkdir",phot_folder])

#Start by reducing the biases.
mb_dates = reduce_bias(bias_folder, conf_folder, cal_folder)

#Next, reduce the sky flats.
sf_dates = None
#sf_dates = reduce_sf(skyflat_folder, mb_dates, conf_folder, cal_folder)

#Now, reduce the standards.
std_dates = reduce_scipol(std_folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder)

#Finally, reduce the polarimetry science images, both from the standards and the main target.
for scipol_folder in scipol_folders:
    sci_pol_dates = reduce_scipol(scipol_folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder)
