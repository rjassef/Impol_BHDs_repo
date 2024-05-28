import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from astropy.wcs import WCS
from photutils.background import Background2D, SExtractorBackground
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve
import astropy.units as u
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
import matplotlib.pyplot as plt
import subprocess
import re

import sys
sys.path.append("../analysis")
import phot

class ResolvedPol(object):

    def __init__(self, object, band, mjd, ichip=1, data_folder = "../analysis/crz", mask_folder = "../analysis/masks", stamps_folder="stamps"):

        #Save input parameters. 
        self.object = object
        self.band = band
        self.mjd = mjd
        self.ichip = ichip
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.stamps_folder = stamps_folder

        subprocess.call(["mkdir", self.stamps_folder])
        
        #Common part of output file names. 
        self.ref_name = "{}.{}.{}".format(object, mjd, band)

        #Search for all the files that match the input parameter criteria.
        if mjd=="All": 
            ls_output = subprocess.run("ls {}/science_reduced_img.{}.*.{}.chip{}.*.fits".format(data_folder,object,band,ichip), shell=True, capture_output=True)
        else:
            ls_output = subprocess.run("ls {}/science_reduced_img.{}.{}.{}.chip{}.*.fits".format(data_folder,object,mjd,band,ichip), shell=True, capture_output=True)
        self.im_fnames = ls_output.stdout.decode('utf8').split()
        self.mask_fnames = list()
        for im_fname in self.im_fnames:
            im_fname = re.sub(data_folder, mask_folder, im_fname)
            im_fname = re.sub("crz","{}",im_fname)
            self.mask_fnames.append(im_fname)

        #Set a variabile with the number of files. 
        self.nf = len(self.im_fnames)

        #Array to hold the seeing values.
        self.seeing = None

        #Zoom in region around the target. 
        self.ix1_z =   30# 25
        self.ix2_z =  110
        self.iy1_z =  960
        self.iy2_z = 1080

        #Default values of the offsets. Measured on the images of W0204-0506 which has a nearby star to the source. 
        self.dx_use = -0.1 
        self.dy_use = -90.5
        self.e_pos = None
        self.o_pos = None

        return
    
    def find_shift(self, ex_ref, ey_ref):

        e_pos_ref = np.vstack((ex_ref,ey_ref)).T

        dx = np.zeros(self.nf)
        dy = np.zeros(self.nf)
        self.e_pos = list()
        self.o_pos = list()

        for i in range(self.nf):

            omask = fits.open(self.mask_fnames[i].format("omask"))
            emask = fits.open(self.mask_fnames[i].format("emask"))
            fname = self.im_fnames[i]
            self.e_pos.append(phot.dao_recenter(fname, e_pos_ref, emask[0].data, "e", ".", box_size=7))
            self.o_pos.append(phot.dao_recenter(fname, e_pos_ref, omask[0].data, "o", ".", box_size=7))

            dx[i], dy[i] = (self.e_pos[-1]-self.o_pos[-1])[0]

        #Save the median offsets.
        self.dx_use = np.round(np.median(dx),1)
        self.dy_use = np.round(np.median(dy),1)

        return     

    def find_seeing(self, ex_ref, ey_ref, x_size=24, y_size=24, stddev_0 = 1.1, show_plots=False):

        self.seeing = np.zeros(self.nf)

        for i in range(self.nf):
            im = fits.open(self.im_fnames[i])

            ix1 = int(ex_ref-x_size/2)
            ix2 = int(ex_ref+x_size/2)
            iy1 = int(ey_ref-y_size/2)
            iy2 = int(ey_ref+y_size/2)
            y, x = np.mgrid[:x_size, :y_size]
            z = im[0].data[iy1:iy2, ix1:ix2]
            #z_err = im[1].data[iy1:iy2, ix1:ix2]
            z -= np.median(z)

            p_init = models.Gaussian2D(x_mean=x_size/2, y_mean=y_size/2, x_stddev=stddev_0, y_stddev=stddev_0)#x_stddev=0.95, y_stddev=0.95)
            #stddev_tied = lambda model: model.x_stddev

            #p_init.y_stddev.tied = stddev_tied
            p_init.x_mean.min = 0
            p_init.x_mean.max = x_size
            p_init.y_mean.min = 0
            p_init.y_mean.max = y_size
            p_init.x_stddev.min = 0.
            p_init.x_stddev.max = 5.
            p_init.y_stddev.min = 0.
            p_init.y_stddev.max = 5.

            fit_p = fitting.LevMarLSQFitter()
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
    
    def get_pol(self, regularize_psf=False, target_fwhm=None):

        self.th = np.zeros(self.nf)

        for i in range(self.nf):
            im = fits.open(self.im_fnames[i])
            mname = self.mask_fnames[i]
            mask  = fits.open(mname.format("mask"))
            omask = fits.open(mname.format("omask"))
            emask = fits.open(mname.format("emask"))

            self.th[i] = im[0].header["HIERARCH ESO INS RETA2 ROT"]

            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = SExtractorBackground()
            bkg = Background2D(im[0].data, (50,50), filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, coverage_mask=mask[0].data.astype(bool))
            im[0].data -= bkg.background

            if i==0:
                oim = np.zeros((self.nf, im[0].data.shape[0], im[0].data.shape[1]))
                eim = np.zeros(oim.shape)
            oim[i] = im[0].data*(1-omask[0].data)
            eim[i] = im[0].data*(1-emask[0].data)

            input_wcs = WCS(naxis=2)
            if self.e_pos is None:
                input_wcs.wcs.crpix = 1021.5, 70.0 #o_pos[0] #128.5, 128.5
            else:
                input_wcs.wcs.crpix = self.o_pos[i][0]
            input_wcs.wcs.cdelt = 1., 1.#-0.01, 0.01

            output_wcs_o = WCS(naxis=2)
            if self.o_pos is None:
                output_wcs_o.wcs.crpix = input_wcs.wcs.crpix + (self.dx_use, self.dy_use)
            else:
                output_wcs_o.wcs.crpix = self.o_pos[0][0]+ (self.dx_use, self.dy_use)
            output_wcs_o.wcs.cdelt = input_wcs.wcs.cdelt

            output_wcs_e = WCS(naxis=2)
            if self.e_pos is None:
                output_wcs_e.wcs.crpix = input_wcs.wcs.crpix
            else:
                output_wcs_e.wcs.crpix = self.o_pos[0][0]
            output_wcs_o.wcs.cdelt = input_wcs.wcs.cdelt

            oim[i], _ = reproject_interp((oim[i], input_wcs), output_wcs_o, shape_out=eim[i].shape, order='bicubic') #order='nearest-neighbor')
            if self.e_pos is not None:
                eim[i], _ = reproject_interp((eim[i], input_wcs), output_wcs_e, shape_out=eim[i].shape, order='bicubic') #order='nearest-neighbor')

            #If requested, convolve images to a common PSF.
            if regularize_psf:

                if i==0:
                    if target_fwhm is None:
                        target_fwhm = np.ceil(np.max(self.seeing)*10)/10.
                    self.target_fwhm = target_fwhm
                    self.target_fwhm_pix = target_fwhm/(im[0].header["HIERARCH ESO INS PIXSCALE"]*im[0].header["HIERARCH ESO DET WIN1 BINX"])
                    print("Target FWHM: ",target_fwhm)

                if self.seeing[i] < target_fwhm:
                    stddev_image = gaussian_fwhm_to_sigma*self.seeing[i]/(im[0].header["HIERARCH ESO INS PIXSCALE"]*im[0].header["HIERARCH ESO DET WIN1 BINX"])
                    stddev_targ = gaussian_fwhm_to_sigma*target_fwhm/(im[0].header["HIERARCH ESO INS PIXSCALE"]*im[0].header["HIERARCH ESO DET WIN1 BINX"])
                    stddev = (stddev_targ**2-stddev_image**2)**0.5
                    print(stddev, stddev * im[0].header["HIERARCH ESO INS PIXSCALE"]*im[0].header["HIERARCH ESO DET WIN1 BINX"])
                    gauss = Gaussian2DKernel(x_stddev=stddev)
                    eim[i] = convolve(eim[i], gauss, mask=emask[0].data)
                    oim[i] = convolve(oim[i], gauss, mask=emask[0].data)

            fits.writeto("{}/oim.{}.{}.fits".format(self.stamps_folder, self.ref_name, i), oim[i, self.ix1_z:self.ix2_z,self.iy1_z:self.iy2_z], overwrite=True)
            fits.writeto("{}/eim.{}.{}.fits".format(self.stamps_folder, self.ref_name, i), eim[i, self.ix1_z:self.ix2_z,self.iy1_z:self.iy2_z], overwrite=True)

        #Get the Stoke parameter images. 
        all_th_S = np.unique(self.th)
        self.Q = np.zeros(oim.shape[1:])
        self.U = np.zeros(self.Q.shape)
        for j, th_S in enumerate(all_th_S):
            k = np.where(self.th==th_S)
            fo = np.sum(oim[k], axis=0)
            fe = np.sum(eim[k], axis=0)
            F = (fo-fe)/(fo+fe)
            self.Q += (2/len(all_th_S)) * F * np.cos(4*th_S*u.deg)
            self.U += (2/len(all_th_S)) * F * np.sin(4*th_S*u.deg)

        #Calculate the polarization image. 
        pol_frac_unmasked = (self.Q**2+self.U**2)**0.5
        pol_angle_unmasked = 0.5*np.arctan2(self.U,self.Q)*180./np.pi

        #Make a stack of everything. 
        self.stack = (oim.sum(axis=0) + eim.sum(axis=0))[self.ix1_z:self.ix2_z,self.iy1_z:self.iy2_z]
        fits.writeto("{}/stack.{}.fits".format(self.stamps_folder, self.ref_name), self.stack, overwrite=True)
        self.diff = (oim.sum(axis=0) - eim.sum(axis=0))[self.ix1_z:self.ix2_z,self.iy1_z:self.iy2_z]
        fits.writeto("{}/diff.{}.fits".format(self.stamps_folder, self.ref_name), self.diff, overwrite=True)

        #Make source mask from the stack and save the masked polarization and polarization angle images.
        rms = np.std(self.stack)
        #print(rms)
        source_mask = np.ones(self.stack.shape)
        source_mask[self.stack>2*rms] = 0
        self.pol_frac = pol_frac_unmasked[self.ix1_z:self.ix2_z,self.iy1_z:self.iy2_z] * (1-source_mask)
        fits.writeto("{}/masked_pol_frac.{}.fits".format(self.stamps_folder, self.ref_name), self.pol_frac, overwrite=True)

        self.pol_angle = pol_angle_unmasked[self.ix1_z:self.ix2_z,self.iy1_z:self.iy2_z] * (1-source_mask)
        self.pol_angle = np.where(source_mask, np.nan, self.pol_angle)
        fits.writeto("{}/masked_pol_angle.{}.fits".format(self.stamps_folder, self.ref_name), self.pol_angle, overwrite=True)

        return
    
