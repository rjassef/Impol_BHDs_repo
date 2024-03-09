import numpy as np
import re
import subprocess
from astropy.io import fits

def filter_fitsfiles(obj_id, rim_folder, filter, mjd=None, ichip="1", ob=None):

    cat = open("aux.dat","w")
    subprocess.call("ls {0:s}/*.chip{1:s}.*.fits".format(rim_folder, ichip), shell=True, stdout=cat)
    cat.close()

    fnames = np.genfromtxt("aux.dat", dtype="U")
    subprocess.call(["rm", "aux.dat"])

    output_fnames = []
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

    root = "/home/rjassef/Impol_Blue_HotDOGs/reduction/"

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
