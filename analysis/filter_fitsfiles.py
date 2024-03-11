import numpy as np
import re
import subprocess
from astropy.io import fits
import os

def filter_fitsfiles(obj_id, rim_folder, filter, mjd_all=None, ichip="1", ob_all=None):

    ls_output = subprocess.run("ls {0:s}/*.chip{1:s}.*.fits".format(rim_folder, ichip), shell=True, capture_output=True)
    fnames = ls_output.stdout.decode('utf8').split()

    if not isinstance(mjd_all,list):
        mjd_all = [mjd_all]
    if not isinstance(ob_all, list):
        ob_all = [ob_all]

    output_fnames = []
    for i, mjd in enumerate(mjd_all):
        ob = ob_all[i]

        for fname in fnames:
            h = fits.open(fname)
            if h[0].header['HIERARCH ESO OBS TARG NAME']==obj_id:
                if ob is not None and h[0].header['HIERARCH ESO OBS ID']!=ob:
                    continue
                if mjd is not None and not re.search(mjd,fname):
                    continue
                if h[0].header['HIERARCH ESO INS FILT1 NAME']!=filter:
                    continue
                output_fnames.append(re.sub(r'^.*/(.*?)$', r'\1', fname))

    return(output_fnames)


def default_folders(use_skyflats=False):

    root = "{}/Impol_Blue_HotDOGs/Impol_BHDs_repo/".format(os.path.expanduser("~"))

    mask_folder = root+"analysis/masks"
    crz_folder = root+"analysis/crz"
    phot_folder = root+"analysis/phot"
    rim_folder = root+"proc/reduced_images"

    if use_skyflats:
        mask_folder += "_with_skyflat"
        crz_folder  += "_with_skyflat"
        phot_folder += "_with_skyflat"
        rim_folder  += "_with_skyflat"

    subprocess.call(["mkdir",mask_folder], stderr=subprocess.DEVNULL)
    subprocess.call(["mkdir",crz_folder], stderr=subprocess.DEVNULL)
    subprocess.call(["mkdir",phot_folder], stderr=subprocess.DEVNULL)

    return mask_folder, crz_folder, phot_folder, rim_folder
