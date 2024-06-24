import numpy as np 
import re
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
from astropy.utils.exceptions import AstropyWarning

from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_sources, centroid_com
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils import Background2D, SExtractorBackground
from photutils.segmentation import SourceFinder, SourceCatalog, make_2dgaussian_kernel

class RunPhot(object):

    def __init__(self, pdata, force_new=False):

        #Save the input. 
        self.pdata = pdata
        self.force_new = force_new

        return
    
    def subtract_background(self, box_size = (50,50)):

        #Get all the filenames. 
        fnames = self.pdata.list_of_filenames()

        #Subtract the background for each of the images using the SExtractorBackground method.
        for fname in fnames:

            #Output filename. If it exists and we are not forcing new calculations, skip. 
            bkg_fname = re.sub(".fits",".bkg.fits",fname)
            if not self.force_new and Path("{}/{}".format(self.pdata.bkg_folder, bkg_fname)).exists():
                continue

            print("Subtracting the background for image ",fname)

            #Load the masks. 
            mask, omask, emask = self.pdata.mask_obj.read_masks(fname)

            #Open the CRZ image.
            h = fits.open("{}/{}".format(self.pdata.crz_folder, re.sub(".fits",".crz.fits",fname)))

            #Create the background and sigmaclip objects.
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = SExtractorBackground(sigma_clip)

            #First subtract the background for the e-beam, and then repeat for the o-beam. 
            ebkg = Background2D(h[0].data, box_size , filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, coverage_mask=emask.astype(bool))
            obkg = Background2D(h[0].data, box_size, filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, coverage_mask=omask.astype(bool))
            h[0].data -= ebkg.background*(1-emask) + obkg.background*(1-omask)
            h[0].data[mask] = np.nan

            bkg_mod = ebkg.background*(1-emask) + obkg.background*(1-omask)
            bkg_mod[mask] = np.nan

            #Save the background subtracted image.
            h.writeto("{}/{}".format(self.pdata.bkg_folder, bkg_fname), overwrite=True)
            bkg_model_fname = re.sub(".fits",".bkg_mod.fits",fname)
            fits.writeto("{}/{}".format(self.pdata.bkg_folder, bkg_model_fname), bkg_mod, overwrite=True)

        return

    def get_source_positions(self, box_size=11):

        #Get all the filenames. 
        fnames = self.pdata.list_of_filenames()

        for i, fname in enumerate(fnames):

            #Position file names.
            epos_fname = re.sub(".fits",".epos",fname)
            opos_fname = re.sub(".fits",".opos",fname)

            #First do the positions in the ebeam. If source positions are not known for this file, then either find them if it is the first file, or recenter them if we already have the reference positions.
            if self.force_new or not Path("{}/{}".format(self.pdata.phot_folder, epos_fname)).exists():

                if i==0:
                    epos = self.ebeam_dao_find(fname)
                    if self.pdata.bband == "v_HIGH":
                        epos = epos[epos[:,1]<500.]
                else: 
                    epos = self.dao_recenter(fname, self.e_pos_ref, "e", box_size)           
                np.savetxt("{0:s}/{1:s}".format(self.pdata.phot_folder, epos_fname), epos)

                opos = self.dao_recenter(fname, epos, "o", box_size)
                np.savetxt("{0:s}/{1:s}".format(self.pdata.phot_folder, opos_fname), opos)

            #If we are not finding/centering sources for the first file, then read it, as we will need it for the reference positions.
            elif i==0:
                epos = np.loadtxt("{}/{}".format(self.pdata.phot_folder, epos_fname))

            if i==0:
                self.e_pos_ref = np.copy(epos)


        return

            
    def ebeam_dao_find(self, fname):

        print("Finding sources for image",fname)

        #Read the masks. 
        _, _, emask = self.pdata.mask_obj.read_masks(fname)

        #Open the bakground subtracted file. 
        bkg_fname = re.sub(".fits",".bkg.fits",fname)
        h = fits.open("{0:s}/{1:s}".format(self.pdata.bkg_folder, bkg_fname))

        #Run the DAOStarFinder.
        # _, _, std = sigma_clipped_stats(h[0].data, mask=emask, sigma=3.0)
        # daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        #sources = daofind(h[0].data, mask=emask)
        #ex = sources['xcentroid']
        
        #Run the segmentation map detection,following https://photutils.readthedocs.io/en/stable/segmentation.html#image-segmentation

        #Start by assuming a 0.7" PSF to create the convolution. 
        pixscale = h[0].header["HIERARCH ESO INS PIXSCALE"]*h[0].header["HIERARCH ESO DET WIN1 BINX"]
        kernel = make_2dgaussian_kernel(0.7/pixscale, size=2*int(0.7/pixscale)-1)#5)
        convolved_data = convolve(h[0].data, kernel, mask=emask)

        #Now, create the segmentation map.
        _, _, std = sigma_clipped_stats(h[0].data, mask=emask, sigma=3.0)
        finder = SourceFinder(npixels=20)
        segment_map = finder(convolved_data, 3.0*std, mask=emask.astype(bool))

        #Run the source selection. 
        cat = SourceCatalog(h[0].data, segment_map, convolved_data=convolved_data, mask=emask.astype(bool))
        tbl = cat.to_table()

        #Get the source positions.
        ex = tbl['xcentroid'].data
        ey = tbl['ycentroid'].data

        h.close()
        e_positions = np.vstack((ex,ey)).T 
        
        return e_positions


    def dao_recenter(self, fname, e_pos_ref, beam, box_size):

        #Load the masks
        mask, omask, emask = self.pdata.mask_obj.read_masks(fname)
        if beam=="o":
            mask_use = omask
        else:
            mask_use = emask

        #Open the bakground subtracted file. 
        bkg_fname = re.sub(".fits",".bkg.fits",fname)
        h = fits.open("{0:s}/{1:s}".format(self.pdata.bkg_folder, bkg_fname))

        x_ref = np.copy(e_pos_ref[:,0])
        y_ref = np.copy(e_pos_ref[:,1])
        if beam=="o":
            y_ref += 90
        x, y = centroid_sources(h[0].data, x_ref, y_ref, mask=mask_use,     box_size=box_size, centroid_func=centroid_com)

        for i in range(len(x)):
            if (x[i]-x_ref[i])**2 + (y[i]-y_ref[i])**2 > box_size**2:
                print("Could not recenter {}-beam position for source {} in file {}. Reverting to reference position.".format(beam, i, fname))
                x[i] = x_ref[i]
                y[i] = y_ref[i]
                      
        h.close()

        pos = np.vstack((x,y)).T

        return pos
    
    def get_phot(self, r_ap=1.0, resubtract_background=False, force_new=True, apply_convolution=False, ob_ids=None, mjds=None, chips=None):

        for ifname, fname in enumerate(self.pdata.list_of_filenames(ob_ids=ob_ids, mjds=mjds, chips=chips)):

            #Check if photometry has already been calculated.
            pname = re.sub(".fits",".phot",fname)
            if not force_new and not self.force_new and Path("{}/{}".format(self.pdata.phot_folder, pname)).exists():
                continue

            #Open the crz and bakground subtracted images. 
            bkg_fname = re.sub(".fits",".bkg.fits",fname)
            h = fits.open("{0:s}/{1:s}".format(self.pdata.bkg_folder, bkg_fname))

            #Set the pixel scale. 
            pixscale = h[0].header["HIERARCH ESO INS PIXSCALE"]*h[0].header["HIERARCH ESO DET WIN1 BINX"]

            #Load the masks. 
            mask, omask, emask = self.pdata.mask_obj.read_masks(fname)

            if apply_convolution:
                target_seeing = np.ceil(np.max(self.seeing)*10)/10.
                kernel_fwhm = (target_seeing**2-self.seeing[ifname]**2)**0.5/pixscale
                kernel_size = 2*int(target_seeing/pixscale)-1
                kernel = make_2dgaussian_kernel(kernel_fwhm, size=kernel_size)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyWarning)
                    convolved_data = convolve(h[0].data, kernel, mask=mask)
                    convolved_err  = convolve(h[1].data**2, kernel, mask=mask)**0.5
                h[0].data = convolved_data
                h[1].data = convolved_err

                h.writeto("{}/{}".format(self.pdata.bkg_folder, re.sub(".fits",".bkg_conv.fits",fname)), overwrite=True)

            #Load the e and o beam source positions. 
            epos = np.loadtxt("{}/{}".format(self.pdata.phot_folder, re.sub(".fits",".epos",fname)))
            opos = np.loadtxt("{}/{}".format(self.pdata.phot_folder, re.sub(".fits",".opos",fname)))
            e_aps = CircularAperture(epos, r=r_ap/pixscale)
            o_aps = CircularAperture(opos, r=r_ap/pixscale)

            #Calculate the photometry.
            e_phot_table = aperture_photometry(h[0].data, [e_aps], mask=emask, error=h[1].data)
            o_phot_table = aperture_photometry(h[0].data, [o_aps], mask=omask, error=h[1].data)

            if resubtract_background:

                #Define the anuli.  
                r_an_in_use = 4./pixscale
                r_an_out_use = 7./pixscale
                e_anns  = CircularAnnulus(epos, r_in=r_an_in_use, r_out=r_an_out_use)
                o_anns  = CircularAnnulus(opos, r_in=r_an_in_use, r_out=r_an_out_use)

                #Estimate the background and its error using a 3 sigma clipping.
                e_annulus_masks = e_anns.to_mask(method='center')
                e_bkg_mean = np.zeros(len(epos))
                e_bkg_sig  = np.zeros(len(epos))
                for k, ann_mask in enumerate(e_annulus_masks):
                    ann_data = ann_mask.multiply(h[0].data*np.where(emask,0,1))
                    ann_data_1d = ann_data[ann_data>0]
                    e_bkg_mean[k], _, e_bkg_sig[k] = sigma_clipped_stats(ann_data_1d)
                o_annulus_masks = o_anns.to_mask(method='center')
                o_bkg_mean = np.zeros(len(opos))
                o_bkg_sig  = np.zeros(len(opos))
                for k, ann_mask in enumerate(o_annulus_masks):
                    ann_data = ann_mask.multiply(h[0].data*np.where(omask,0,1))
                    ann_data_1d = ann_data[ann_data>0]
                    o_bkg_mean[k], _, o_bkg_sig[k] = sigma_clipped_stats(ann_data_1d)

                #Get the background level.
                e_bkg_sum = e_bkg_mean * e_aps.area
                o_bkg_sum = o_bkg_mean * o_aps.area

                #Get the background uncertainty
                e_bkg_sum_err2 = e_bkg_sig**2 * (e_aps.area)**2 / (e_anns.area)**2
                o_bkg_sum_err2 = o_bkg_sig**2 * (o_aps.area)**2 / (o_anns.area)**2

            else:
                e_bkg_sum = 0.
                o_bkg_sum = 0.
                e_bkg_sum_err2 = 0.
                o_bkg_sum_err2 = 0.

            esum = e_phot_table['aperture_sum_0'] - e_bkg_sum
            osum = o_phot_table['aperture_sum_0'] - o_bkg_sum
            esum_err = (e_phot_table['aperture_sum_err_0']**2 + e_bkg_sum_err2)**0.5
            osum_err = (o_phot_table['aperture_sum_err_0']**2 + o_bkg_sum_err2)**0.5

            #Close the image. 
            h.close()

            #Save the photometry
            np.savetxt("{}/{}".format(self.pdata.phot_folder, pname), np.array([esum.data, esum_err.data, osum.data, osum_err.data]).T)

        return


    def find_seeing(self, ex_ref0, ey_ref0, x_size=24, y_size=24, stddev_0 = 1.1, x_mean_0=None, y_mean_0=None, show_plots=False, ob_ids=None, mjds=None, chips=None):

        im_fnames = self.pdata.list_of_filenames(ob_ids=ob_ids, mjds=mjds, chips=chips)
        nf = len(im_fnames)

        self.seeing = np.zeros(nf)

        for i in range(nf):
            bkg_name = re.sub(".fits",".bkg.fits",im_fnames[i])
            im = fits.open("{}/{}".format(self.pdata.bkg_folder, bkg_name))

            #Find the position of the closest object to the star choosen. 
            epos = np.loadtxt("{}/{}".format(self.pdata.phot_folder, re.sub(".fits",".epos", im_fnames[i])))
            dist2 = (epos[:,0]-ex_ref0)**2 + (epos[:,1]-ey_ref0)**2
            k = np.argmin(dist2)
            ex_ref, ey_ref = epos[k]

            ix1 = int(ex_ref-x_size/2)
            ix2 = int(ex_ref+x_size/2)
            iy1 = int(ey_ref-y_size/2)
            iy2 = int(ey_ref+y_size/2)
            y, x = np.mgrid[:x_size, :y_size]
            z = im[0].data[iy1:iy2, ix1:ix2]
            z[np.isnan(z)] = 0.
            #z_err = im[1].data[iy1:iy2, ix1:ix2]
            #z -= np.median(z)

            if x_mean_0 is None:
                x_mean_0 = x_size/2
            if y_mean_0 is None:
                y_mean_0 = y_size/2
            p_init = models.Gaussian2D(x_mean=x_mean_0, y_mean=y_mean_0, x_stddev=stddev_0[i], y_stddev=stddev_0[i], amplitude=np.max(z))#x_stddev=0.95, y_stddev=0.95)
            stddev_tied = lambda model: model.x_stddev

            p_init.y_stddev.tied = stddev_tied
            p_init.x_mean.min = 0
            p_init.x_mean.max = x_size
            p_init.y_mean.min = 0
            p_init.y_mean.max = y_size
            p_init.x_stddev.min = 0.
            p_init.x_stddev.max = 5.
            p_init.y_stddev.min = 0.
            p_init.y_stddev.max = 5.

            fit_p = fitting.LevMarLSQFitter()
            #fit_p = fitting.TRFLSQFitter()
            p = fit_p(p_init, x, y, z)#, weights=1./z_err)
            self.seeing[i] = np.mean([p.x_fwhm,p.y_fwhm])*im[0].header["HIERARCH ESO INS PIXSCALE"]*im[0].header["HIERARCH ESO DET WIN1 BINX"]
            #print(p.x_fwhm, p.y_fwhm)

            if show_plots:
                norm = ImageNormalize(z, interval=ZScaleInterval(), stretch=LinearStretch())
                fig, axs = plt.subplots(1,2)
                axs[0].imshow(z, norm=norm, cmap='gray')
                axs[1].imshow(p(x,y), norm=norm, cmap='gray')
                plt.show()

        return



