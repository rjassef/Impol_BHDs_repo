#!/usr/bin/env python

import numpy as np
from scipy import ndimage
from astropy.io import fits

import subprocess

from masks import bright_star_mask
from util import join_chips

####

#This is basically just a hack to remove the masking from the bright stars and ghosts for stacking.

def remove_bright_star_masking(mask, chip):

    mask_stars = np.zeros(mask[0].data.shape, dtype=np.bool)

    if chip==1:
        #Mask the three saturated stars.
        #Star on the center.
        mask_stars = bright_star_mask(1232, 408, 50, 40, mask_stars)
        #Star on the NE
        mask_stars = bright_star_mask( 418, 600, 50, 40, mask_stars)
        #Star on the center top.
        mask_stars = bright_star_mask( 890, 640, 30, 30, mask_stars)

    elif chip==2:
        mask_stars = bright_star_mask( 260, 739,  40,  40, mask_stars)
        mask_stars = bright_star_mask( 410, 403,  40,  40, mask_stars)
        mask_stars = bright_star_mask( 573, 344,  20,  20, mask_stars)
        mask_stars = bright_star_mask(1051, 344,  40,  40, mask_stars)
        mask_stars = bright_star_mask(1823, 350,  60,  60, mask_stars)

    return mask_stars

####

#Function to read the images with the masks.
def load_images(mjd, j):
    im1 = fits.open("crz/science_reduced_img.{}.chip1.{}.crz.fits".format(mjd, j))
    im1_emask = fits.open("masks/science_reduced_img.{}.chip1.{}.emask.fits".format(mjd, j))
    im1_omask = fits.open("masks/science_reduced_img.{}.chip1.{}.omask.fits".format(mjd, j))

    im2 = fits.open("crz/science_reduced_img.{}.chip2.{}.crz.fits".format(mjd, j))
    im2_emask = fits.open("masks/science_reduced_img.{}.chip2.{}.emask.fits".format(mjd, j))
    im2_omask = fits.open("masks/science_reduced_img.{}.chip2.{}.omask.fits".format(mjd, j))

    im1_stars_mask = remove_bright_star_masking(im1_emask, 1)
    im1_emask[0].data[im1_stars_mask] = 0
    im1_omask[0].data[im1_stars_mask] = 0

    im2_stars_mask = remove_bright_star_masking(im2_emask, 2)
    im2_emask[0].data[im2_stars_mask] = 0
    im2_omask[0].data[im2_stars_mask] = 0

    #Now, we need to make sure that we are not unmasking the stripes. x=215 does not intersect with any masks, so we can use that column to repopulate the stripe masking.
    im1_emask_stripes = np.tile(im1_emask[0].data[:,215], (im1_emask[0].data.shape[1], 1)).T
    im1_omask_stripes = np.tile(im1_omask[0].data[:,215], (im1_omask[0].data.shape[1], 1)).T
    im1_emask[0].data[im1_emask_stripes==1] = 1
    im1_omask[0].data[im1_omask_stripes==1] = 1

    im2_emask_stripes = np.tile(im2_emask[0].data[:,215], (im2_emask[0].data.shape[1], 1)).T
    im2_omask_stripes = np.tile(im2_omask[0].data[:,215], (im2_omask[0].data.shape[1], 1)).T
    im2_emask[0].data[im2_emask_stripes==1] = 1
    im2_omask[0].data[im2_omask_stripes==1] = 1


    image = join_chips(im1, im2)
    omask = join_chips(im1_omask, im2_omask, is_mask=True)
    emask = join_chips(im1_emask, im2_emask, is_mask=True)

    o_image = image * (1-omask)
    e_image = image * (1-emask)

    im1.close()
    im2.close()

    return o_image, e_image

#####

mjds  = [59135, 59136, 59137]

for mjd in mjds:

    #Read the first image to figure out the normalization.
    o_im_norm, e_im_norm = load_images(mjd, 1)

    #We'll normalize by the median of the image within the mask.
    norm_e = np.median(e_im_norm[e_im_norm>0])
    norm_o = np.median(o_im_norm[o_im_norm>0])

    #Now, let's go through all the images.
    e_images = list()
    o_images = list()
    for i in range(1,9):
        o_im, e_im = load_images(mjd, i)

        fact_e = norm_e / np.median(e_im[e_im>0])
        fact_o = norm_o / np.median(o_im[o_im>0])
        e_im *= fact_e
        o_im *= fact_o

        e_images.append(e_im)
        o_images.append(o_im)


    #Get the median and save the stacked images.
    e_image = np.median(e_images, axis=0)
    o_image = np.median(o_images, axis=0)

    subprocess.call(["mkdir","stacked"], stderr=subprocess.DEVNULL)
    fits.writeto("stacked/stacked_{}.e.fits".format(mjd), e_image, overwrite=True)
    fits.writeto("stacked/stacked_{}.o.fits".format(mjd), o_image, overwrite=True)
