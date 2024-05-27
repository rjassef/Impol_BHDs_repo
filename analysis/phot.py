import numpy as np
import re

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip

from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_sources, centroid_com
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils import Background2D, SExtractorBackground

####

pix_scale = 0.25 #arcsec/pixel
fwhm_pix  = 1./pix_scale #Seeing was about 1".


####

def filter_by_shift(pos, pos_ref):
    dpos = pos-pos_ref
    mean, median, std = sigma_clipped_stats(dpos, axis=0)
    pos_filtered_2d = (np.abs(dpos-mean)/std<3.)
    pos_filtered = (pos_filtered_2d[:,0]) & (pos_filtered_2d[:,1])
    print("filtered",len(pos[~pos_filtered,:]),"by excessive position shift.")
    return pos_filtered


def ebeam_dao_find(fname, emask, data_folder, phot_folder, force=False):
    pos_name = re.sub(".fits",".epos",fname)
    try:
        if force:
            raise OSError
        pos = np.loadtxt("{0:s}/{1:s}".format(phot_folder, pos_name))
        ex = pos[:,0]
        ey = pos[:,1]
    except OSError:
        print("Finding sources for image",fname)
        h = fits.open("{0:s}/{1:s}".format(data_folder, fname))
        mean, median, std = sigma_clipped_stats(h[0].data, mask=emask, sigma=3.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        sources = daofind(h[0].data - median, mask=emask)
        ex = sources['xcentroid']
        ey = sources['ycentroid']
        h.close()
        np.savetxt("{0:s}/{1:s}".format(phot_folder, pos_name), np.array([ex, ey]).T)
    e_positions = np.vstack((ex,ey)).T
    return e_positions

def ebeam_dao_recenter(fname, e_pos_ref, emask, data_folder, phot_folder, force=False, box_size=21):
    pos_name = re.sub(".fits",".epos",fname)
    try:
        if force:
            raise OSError
        pos = np.loadtxt("{0:s}/{1:s}".format(phot_folder, pos_name))
        ex = pos[:,0]
        ey = pos[:,1]
        e_positions = np.vstack((ex,ey)).T
    except OSError:
        print("Recentering e-beam sources for image",fname)
        e_positions = dao_recenter(fname, e_pos_ref, emask, "e", data_folder, box_size=box_size)
        np.savetxt("{0:s}/{1:s}".format(phot_folder, pos_name), np.array([e_positions[:,0], e_positions[:,1]]).T)
    return e_positions

def obeam_dao_recenter(fname, e_pos_ref, omask, data_folder, phot_folder, force=False, box_size=21):
    pos_name = re.sub(".fits",".opos",fname)
    try:
        if force:
            raise OSError
        pos = np.loadtxt("{0:s}/{1:s}".format(phot_folder, pos_name))
        ox = pos[:,0]
        oy = pos[:,1]
        o_positions = np.vstack((ox,oy)).T
    except OSError:
        print("Recentering o-beam sources for image",fname)
        o_positions = dao_recenter(fname, e_pos_ref, omask, "o", data_folder, box_size=box_size)
        np.savetxt("{0:s}/{1:s}".format(phot_folder, pos_name), np.array([o_positions[:,0], o_positions[:,1]]).T)
    return o_positions

def dao_recenter(fname, e_pos_ref, mask, beam, data_folder, box_size=11, back_sub=True): #box_size=21):
    h = fits.open("{0:s}/{1:s}".format(data_folder, fname))
    if back_sub:
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = SExtractorBackground()
        bkg = Background2D(h[0].data, (50,50), filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, coverage_mask=mask.astype(bool))
        h[0].data -= bkg.background
    x_ref = np.copy(e_pos_ref[:,0])
    y_ref = np.copy(e_pos_ref[:,1])
    if beam=="o":
        y_ref += 90
    x, y = centroid_sources(h[0].data, x_ref, y_ref, mask=mask,     box_size=box_size, centroid_func=centroid_com)
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            print("Could not recenter {}-beam position for source {} in file {}. Reverting to reference position.".format(beam, i, fname))
            x[i] = x_ref
            y[i] = y_ref
    #x, y = centroid_sources(h[0].data, x_ref, y_ref, mask=mask, box_size=10, centroid_func=centroid_com)
    h.close()

    pos = np.vstack((x,y)).T

    return pos

#####

def run_phot_all(fname, beam, mask, pos, r_ap, r_an_in, r_an_out, data_folder, phot_folder, force=False):

    pname = re.sub(".fits", ".{0:.1f}.{1:.1f}.{2:.1f}.{3:s}beam.phot".format(r_ap, r_an_in, r_an_out, beam), fname)

    try:
        if force:
            raise OSError
        phot = np.loadtxt("{0:s}/{1:s}".format(phot_folder, pname))
        final_sum = phot[:,0]
        final_error = phot[:,1]

    except OSError:

        print("Getting photometry for",fname)

        final_sum, final_error = run_phot(fname, mask, pos, r_ap, r_an_in, r_an_out, data_folder)

        np.savetxt("{0:s}/{1:s}".format(phot_folder, pname), np.array([final_sum, final_error]).T)

    return final_sum, final_error

def run_phot(fname, mask, pos, r_ap, r_an_in, r_an_out, data_folder):

    #Set the apertures and annuli to be used.
    r_ap_use     = r_ap/pix_scale
    r_an_in_use  = r_an_in/pix_scale
    r_an_out_use = r_an_out/pix_scale
    aps   = CircularAperture(pos, r=r_ap_use)
    anns  = CircularAnnulus(pos, r_in=r_an_in_use, r_out=r_an_out_use)
    #apers = [aps, anns]

    #Open the image.
    h = fits.open("{0:s}/{1:s}".format(data_folder, fname))
    h[1].data[mask]=0

    #Estimate the background and its error using a 3 sigma clipping.
    annulus_masks = anns.to_mask(method='center')
    bkg_mean = np.zeros(len(pos))
    bkg_sig  = np.zeros(len(pos))
    bkg_mean_err2 = np.zeros(len(pos))
    for k, ann_mask in enumerate(annulus_masks):
        ann_data = ann_mask.multiply(h[0].data*np.where(mask,0,1))
        ann_data_1d = ann_data[ann_data>0]
        bkg_mean[k], bkg_median, bkg_sig[k] = sigma_clipped_stats(ann_data_1d)

    #Calculate the photometry.
    phot_table = aperture_photometry(h[0].data, [aps], mask=mask)
    err_table  = aperture_photometry((h[1].data)**2, [aps], mask=mask)

    #Close the image.
    h.close()

    #Subtract the background.
    bkg_sum = bkg_mean * aps.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum

    #Get the uncertainty.
    bkg_sum_err2 = bkg_sig**2 * (aps.area)**2 / (anns.area)**2
    final_error = (err_table['aperture_sum_0'] + bkg_sum_err2)**0.5
    #final_error = (err_table['aperture_sum_0'])**0.5

    return final_sum, final_error
