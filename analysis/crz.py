import numpy as np
import re
from astropy.io import fits
from astroscrappy import detect_cosmics

def crz_clean(fname, mask, rim_folder, crz_folder, force=False):

    #Compute the CR corrected image if needed.
    crzname = re.sub(".fits",".crz.fits",fname)
    try:
        if force:
            raise FileNotFoundError
        cr_hdu = fits.open("{0:s}/{1:s}".format(crz_folder, crzname))
        cr_hdu.close()
    except FileNotFoundError:
        print("Cleaning cosmic rays in ",fname)
        h = fits.open("{0:s}/{1:s}".format(rim_folder, fname))
        crmask, clean_im = detect_cosmics(h[0].data, inmask=mask, objlim=10)
        h[0].data = clean_im
        h.writeto("{0:s}/{1:s}".format(crz_folder, crzname), overwrite=True)
        h.close()
    return crzname
