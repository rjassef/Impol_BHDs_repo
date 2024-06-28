import os
import numpy as np
import subprocess
import re
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.visualization import LinearStretch, ZScaleInterval, ImageNormalize

from reproject import reproject_interp

from photutils.segmentation import make_2dgaussian_kernel

import sys
sys.path.append("../analysis_v2")
from pol_analysis.polData import PolData
from pol_analysis.runPhot import RunPhot
from pol_analysis.getStokes import GetStokes


class ResolvedPol(object):

    def __init__(self, object, band, star_coords, bhd_coords=(1022.5, 70.), root_folder=None, stamps_folder="stamps", pdata_force_new=False, phot_force_new=False, centroid_box_size=5):

        #Save input parameters. 
        self.object = object
        self.band = band
        self.stamps_folder = stamps_folder
        self.root_folder = root_folder
        self.centroid_box_size = centroid_box_size
        self.star_coords = star_coords
        self.bhd_coords = bhd_coords

        subprocess.call(["mkdir", self.stamps_folder])
        
        if band=="R_SPECIAL":   
            self.latex_band = r"${\boldsymbol{R_{\rm Special}}}$"
        elif band=="I_BESS":
            self.latex_band = r"${\boldsymbol{I_{\rm Bessel}}}$"
        elif band=="v_HIGH":
            self.latex_band = r"${\boldsymbol{v_{\rm High}}}$"
        else:
            print("Unrecognized band ",band)
            return

        #Initialize the data object. 
        if self.root_folder is None:
            self.root_folder = os.getcwd()+"/../analysis_v2"
        self.pdata = PolData(object, band, root=self.root_folder, force_new=pdata_force_new)

        #Load the pixscale. 
        h = fits.open("{}/{}".format(self.pdata.rim_folder, self.pdata.list_of_filenames()[0]))
        #Set the pixel scale. 
        self.pixscale = h[0].header["HIERARCH ESO INS PIXSCALE"]*h[0].header["HIERARCH ESO DET WIN1 BINX"]
        h.close()

        #Initialize the photometry object. 
        self.pphot = RunPhot(self.pdata, force_new=phot_force_new)
        self.pphot.subtract_background(box_size=(25,25))
        self.pphot.get_source_positions(box_size=centroid_box_size)

        #Get the seeing values for each pair of OB/MJD
        self.seeing = dict()
        for ob_pair in self.pdata.ob_pairs:
            self.pphot.find_seeing(self.star_coords[0], self.star_coords[1], ob_ids=[ob_pair[0]], mjds=[ob_pair[1]])
            fnames = self.pdata.list_of_filenames(ob_ids=[ob_pair[0]], mjds=[ob_pair[1]])
            for i, fname in enumerate(fnames):
                self.seeing[fname] = self.pphot.seeing[i]

        #Zoom in region around the target. 
        self.iy1_z =   30# 25
        self.iy2_z =  110
        self.ix1_z =  960
        self.ix2_z = 1080

        #Get the offsets of the positions. 
        self.find_shift()

        #Make the Stokes object.
        self.stk = GetStokes(self.pdata)

        #Create a work folder for imagematch.
        subprocess.call("mkdir work", shell=True)

        return
    
    def find_shift(self):

        #Find the centroid of the source in each image, in each beam. 
        self.e_pos = dict()
        self.o_pos = dict()

        #Axiliary lists to keep the position shifts.
        dx = list()
        dy = list()

        for ob_pair in self.pdata.ob_pairs:

            #Get the file names.
            fnames = self.pdata.list_of_filenames(ob_ids=[ob_pair[0]], mjds=[ob_pair[1]])

            for i, fname in enumerate(fnames):
                #Find the centroid of the source. 
                epos = np.loadtxt("{}/{}".format(self.pdata.phot_folder, re.sub(".fits",".epos",fname)))
                opos = np.loadtxt("{}/{}".format(self.pdata.phot_folder, re.sub(".fits",".opos",fname)))

                dist_e2 = (epos[:,0]-self.bhd_coords[0])**2 + (epos[:,1]-self.bhd_coords[1])**2
                k = np.argmin(dist_e2)
                self.e_pos[fname] = epos[k]
                self.o_pos[fname] = opos[k]
                dx.append(epos[k,0]-opos[k,0])
                dy.append(epos[k,1]-opos[k,1])

        self.dx_use = np.median(dx)
        self.dy_use = np.median(dy)

        return     

    def get_pol(self, regularize_PSF=True, special_ob_pairs=None, align_images=True, no_processing=False, common_seeing_all_obs=True):

        #Do this for each OB/MJD pair, and then do it for all combined and for all special ob_pairs requested. 
        ob_combs = list()
        ob_comb_names = list()
        for ob_pair in self.pdata.ob_pairs:
            ob_combs.append([[ob_pair[0]],[ob_pair[1]]])
            ob_comb_names.append("{}.{}".format(ob_pair[0], ob_pair[1]))
        if len(self.pdata.ob_pairs)>1:
            ob_combs.append([None, None])
            ob_comb_names.append("All")

        #Add special combinations if provided. 
        if special_ob_pairs is not None:
            ob_combs.extend(special_ob_pairs)
            for iob in range(len(special_ob_pairs)):
                ob_comb_names.append("sp{}".format(iob+1))

        self.target_seeing = dict()
        if common_seeing_all_obs:
            fnames = self.pdata.list_of_filenames()
            seeing_use = np.zeros(len(fnames))
            for i, fname in enumerate(fnames):
                seeing_use[i] = self.seeing[fname]

        #Run through all the combinations. 
        for iob, ob_comb in enumerate(ob_combs):

            print()
            print("#######")
            print(ob_comb_names[iob])
            print("#######")
            print()

            #Get the file names. 
            fnames = self.pdata.list_of_filenames(ob_ids=ob_comb[0], mjds=ob_comb[1])

            seeing_use = np.zeros(len(fnames))
            for i, fname in enumerate(fnames):
                seeing_use[i] = self.seeing[fname]
            worse_psf_fname = fnames[np.argmax(seeing_use)]
            self.target_seeing[ob_comb_names[iob]] = self.seeing[worse_psf_fname]
            if no_processing:
                continue
            print("Matching to image {}.".format(worse_psf_fname))

            self.stk.esum = dict()
            self.stk.osum = dict()
            self.stk.eerr = dict()
            self.stk.oerr = dict()

            #For aligning, set the output WCS axes for the o and e beam images.
            output_wcs_o = WCS(naxis=2)
            output_wcs_o.wcs.crpix = self.o_pos[worse_psf_fname]+ (self.dx_use, self.dy_use)
            output_wcs_o.wcs.cdelt = 1., 1.

            output_wcs_e = WCS(naxis=2)
            output_wcs_e.wcs.crpix = self.o_pos[worse_psf_fname]
            output_wcs_o.wcs.cdelt = 1., 1.

            #Write the image with the worse PSF in the work folder in the right format.
            h = fits.open("{}/{}".format(self.pdata.bkg_folder, re.sub(".fits",".bkg.fits",worse_psf_fname)))
            fits.writeto("work/im.fits", h[0].data[40:900,300:1700], overwrite=True)
            h.close()
            
            #Iterate through them. 
            for fname in fnames:

                h = fits.open("{}/{}".format(self.pdata.bkg_folder, re.sub(".fits",".bkg.fits",fname)))
                
                #Load the masks. 
                mask, omask, emask = self.pdata.mask_obj.read_masks(fname)

                if fname!=worse_psf_fname:
                    fits.writeto("work/t.fits", h[0].data[40:900,300:1700], overwrite=True)
    
                    subprocess.call("ImageMatch -v -preserve -cfg sample.cfg -m work/t.fits work/im.fits", shell=True, executable='/bin/zsh', stdout=subprocess.DEVNULL)

                    #Plot the original, convolved and difference images.
                    self.plot_imagematch(fname=fname)

                    hh = fits.open("work/imtemp.fits")
                    h[0].data[40:900,300:1700] = hh[0].data

                #Separate the e and o beam images.
                oim     = h[0].data*(1-omask.astype(int))
                oim_err = h[1].data*(1-omask.astype(int))
                eim     = h[0].data*(1-emask.astype(int))
                eim_err = h[1].data*(1-emask.astype(int))

                #Align them. 
                if align_images:

                    oim[np.isnan(oim)] = 0.
                    eim[np.isnan(eim)] = 0.
                    oim_err[np.isnan(oim_err)] = 0.
                    eim_err[np.isnan(eim_err)] = 0.

                    input_wcs = WCS(naxis=2)
                    input_wcs.wcs.crpix = self.o_pos[worse_psf_fname]
                    input_wcs.wcs.cdelt = 1., 1.

                    oim, _     = reproject_interp((oim    , input_wcs), output_wcs_o, shape_out=oim.shape, order='bicubic') 
                    oim_err, _ = reproject_interp((oim_err**2, input_wcs), output_wcs_o, shape_out=oim.shape, order='bicubic')
                    oim_err[oim_err<0.] = 0.
                    oim_err = oim_err**0.5 
                    eim, _     = reproject_interp((eim    , input_wcs), output_wcs_e, shape_out=oim.shape, order='bicubic')
                    eim_err, _ = reproject_interp((eim_err**2, input_wcs), output_wcs_e, shape_out=oim.shape, order='bicubic')
                    eim_err[eim_err<0.] = 0.
                    eim_err = eim_err**0.5 

                #Finally, load the cutouts. 
                self.stk.esum[fname] = eim[self.iy1_z:self.iy2_z, self.ix1_z:self.ix2_z]
                self.stk.osum[fname] = oim[self.iy1_z:self.iy2_z, self.ix1_z:self.ix2_z]
                self.stk.eerr[fname] = eim_err[self.iy1_z:self.iy2_z, self.ix1_z:self.ix2_z]
                self.stk.oerr[fname] = oim_err[self.iy1_z:self.iy2_z, self.ix1_z:self.ix2_z]

                #Save the stamps of the cutouts.
                fits.writeto("{}/{}".format(self.stamps_folder, re.sub(".fits",".eim.cutout.fits", fname)), self.stk.esum[fname], overwrite=True)
                fits.writeto("{}/{}".format(self.stamps_folder, re.sub(".fits",".oim.cutout.fits", fname)), self.stk.osum[fname], overwrite=True)
                fits.writeto("{}/{}".format(self.stamps_folder, re.sub(".fits",".eim.err.cutout.fits", fname)), self.stk.eerr[fname], overwrite=True)
                fits.writeto("{}/{}".format(self.stamps_folder, re.sub(".fits",".oim.err.cutout.fits", fname)), self.stk.oerr[fname], overwrite=True)
                
            #Finally, calculate the polarization parameters. 
            self.stk.get_QU_with_errors(e_pos_ref=self.bhd_coords)

            #The maps are quite dirty, so we need to apply a source mask based on the stack of all the cutouts.
            for i, fname in enumerate(fnames):
                if i==0:
                    stack = np.zeros(self.stk.esum[fname].shape)
                stack += self.stk.esum[fname]
            _, _, rms = sigma_clipped_stats(stack, sigma=3.0)
            source_mask = np.ones(stack.shape)
            #source_mask[stack<3.0*rms] = np.nan          

            po_fname = "{}/{}.{}.{}".format(self.stamps_folder, self.object, self.band, ob_comb_names[iob])
            fits.writeto(po_fname+".stack.fits"  , stack, overwrite=True)
            fits.writeto(po_fname+".pfrac.fits"  , self.stk.pol_frac  * source_mask, overwrite=True)
            fits.writeto(po_fname+".epfrac.fits" , self.stk.epol_frac * source_mask, overwrite=True)
            fits.writeto(po_fname+".pangle.fits" , self.stk.pol_angle * source_mask, overwrite=True)
            fits.writeto(po_fname+".epangle.fits", self.stk.epol_angle *source_mask, overwrite=True)
            fits.writeto(po_fname+".Q.fits", self.stk.Q, overwrite=True)
            fits.writeto(po_fname+".U.fits", self.stk.U, overwrite=True)
        return

    def plot_imagematch(self, fname):

        original = fits.getdata("work/t.fits")
        blurred  = fits.getdata("work/imtemp.fits")
        target   = fits.getdata("work/im.fits")
        diff     = target-blurred

        #BHD
        #e image
        iy1_be = 0  #self.ix1_z-40
        iy2_be = 70 #self.ix2_z-40
        ix1_be = 685 #self.iy1_z-300
        ix2_be = 755 #self.iy2_z-300

        #Star used for seeing. 
        iy1_se = int(self.star_coords[1]) - 35 - 40
        iy2_se = int(self.star_coords[1]) + 35 - 40
        ix1_se = int(self.star_coords[0]) - 35 - 300
        ix2_se = int(self.star_coords[0]) + 35 - 300


        fig, axs = plt.subplots(4, 3, figsize=(24,35))
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(axs.shape[0]):

            if i<2:
                iy1, iy2, ix1, ix2 = (iy1_be, iy2_be, ix1_be, ix2_be)
                obj = "BHD"
            else:
                iy1, iy2, ix1, ix2 = (iy1_se, iy2_se, ix1_se, ix2_se)
                obj = "Star"
            beam = "e-beam"
            if i==1 or i==3:
                beam = "o-beam"
                iy1 -= int(self.dy_use)
                iy2 -= int(self.dy_use)

            norm_ims = ImageNormalize(original[iy1:iy2, ix1:ix2], stretch=LinearStretch(), interval=ZScaleInterval())
            norm_diff = ImageNormalize(diff[iy1:iy2, ix1:ix2], stretch=LinearStretch(), interval=ZScaleInterval())
            axs[i,0].imshow(original[iy1:iy2, ix1:ix2], norm=norm_ims, cmap='gray')
            axs[i,1].imshow(blurred[iy1:iy2, ix1:ix2], norm=norm_ims, cmap='gray')
            axs[i,2].imshow(diff[iy1:iy2, ix1:ix2], norm=norm_ims, cmap='gray')

            axs[i,0].text(0.1, 0.9, obj , fontsize=36, weight='bold', color='white', transform=axs[i,0].transAxes)
            axs[i,0].text(0.1, 0.8, beam, fontsize=32, weight='bold', color='white', transform=axs[i,0].transAxes)
                

        fig.suptitle(fname, fontsize=36, weight='bold')

        plt.tight_layout()
        plt.show()

        return 

    def plot_pol(self, pmin=0., pmax=50., chimin=-90., chimax=90., ob_names=None, size=20, cmap_pfrac='plasma_r', cmap_pangle='hsv', z=None, figsize=(17,16), save_fig=False, snr_stack_lim=6.0):

        if ob_names is None:
            ob_names = list()
            for ob_pair in sorted(self.pdata.ob_pairs, key=lambda l:l[1]):
                ob_names.append("{}.{}".format(ob_pair[0], ob_pair[1]))
            if len(ob_names)>1:
                ob_names.append("All")

        fig, axs = plt.subplots(3, len(ob_names), figsize=figsize, sharex=True, sharey=True, squeeze=False)
        plt.subplots_adjust(wspace=0.)
        for ax in axs.flat:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])

        #Load the images
        po_fname = "{}/{}.{}.".format(self.stamps_folder, self.object, self.band)
        stacks = list()
        pfracs = list()
        pangles = list()
        epfracs = list()
        epangles = list()
        for ob_name in ob_names:
            stacks.append(fits.getdata(po_fname+ob_name+".stack.fits"))
            pfracs.append(fits.getdata(po_fname+ob_name+".pfrac.fits")*100)
            pangles.append(fits.getdata(po_fname+ob_name+".pangle.fits"))
            epfracs.append(fits.getdata(po_fname+ob_name+".epfrac.fits")*100)
            epangles.append(fits.getdata(po_fname+ob_name+".epangle.fits"))
        
        #Apply masking based on the error images. Note that showning only pixels detected at SNR 4 in the full stack basically means that in the stack of a given angle, the pixel would be detected with SNR 2 in the stack of a given retarder plate angle. If we want to be SNR 2 in each of the o and e beam stacks, we would need to require SNR 6 in the full stack.  
        for i in range(len(ob_names)):
            # pmask = np.where(pfracs[i]>epfracs[i], False, True)
            # _, _, rms = sigma_clipped_stats(stacks[i], sigma=3.0)
            # print(rms)
            # pmask[stacks[i]<3.0*rms] = True
            _, _, rms = sigma_clipped_stats(stacks[i], sigma=3.0)
            pmask = np.where(stacks[i]<snr_stack_lim*rms, True, False)
            #stacks[i][pmask]   = np.nan
            #pmask[pfracs[i]<epfracs[i]] = True
            pfracs[i][pmask]   = np.nan
            epfracs[i][pmask]  = np.nan
            pangles[i][pmask]  = np.nan
            epangles[i][pmask] = np.nan

        ix1 = int(stacks[0].shape[1]/2 - size/2)
        ix2 = int(stacks[0].shape[1]/2 + size/2)
        iy1 = int(stacks[0].shape[0]/2 + size/2)
        iy2 = int(stacks[0].shape[0]/2 - size/2)
        #print(ix1, ix2, iy1, iy2)

        for i, stack in enumerate(stacks):
            norm = ImageNormalize(stack[iy1:iy2:-1,ix1:ix2], stretch=LinearStretch(), interval=ZScaleInterval())
            axs[0,i].imshow(stack[iy1:iy2:-1,ix1:ix2], norm=norm, cmap='gray_r')

        for i, pfrac in enumerate(pfracs):
            cm_pf = axs[1,i].imshow(pfrac[iy1:iy2:-1,ix1:ix2], cmap=cmap_pfrac, vmin=pmin, vmax=pmax)
        divider_pf = make_axes_locatable(axs[1,-1])
        #cax_pf = divider_pf.append_axes("right", size="5%", pad=0.05)
        cax_pf = inset_axes(axs[1,2], width="5%", height="97%", loc=4) 
        fig.colorbar(cm_pf, cax=cax_pf)

        for i, pangle in enumerate(pangles):
            cm_pa = axs[2,i].imshow(pangle[iy1:iy2:-1,ix1:ix2], cmap=cmap_pangle, vmin=chimin, vmax=chimax)
        divider_pa = make_axes_locatable(axs[2,-1])
        #cax_pa = divider_pa.append_axes("right", size="5%", pad=0.05)
        cax_pa = inset_axes(axs[2,2], width="5%", height="97%", loc=4) 
        fig.colorbar(cm_pa, cax=cax_pa)

        for j in range(3):
            for i in range(len(ob_names)):
                beam  = plt.Circle((size*0.15,size*0.8),self.target_seeing[ob_names[i]]/2/self.pixscale, color='k', fill=False)
                axs[j,i].add_patch(beam)

        for i, ob_name in enumerate(ob_names):
            x = ob_name.split(".")
            if len(x)==2:
                txt = axs[0,i].text(size*0.025, size*0.05, "OB : "+x[0], fontsize=28, weight='bold', color='white')
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                txt = axs[0,i].text(size*0.025, size*0.15, "MJD: "+x[1], fontsize=28, weight='bold', color='white')
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                if i==0:
                    txt = axs[0,i].text(size*0.025, size*0.25, self.latex_band, fontsize=36, color='white')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            else:
                ob_name_use = ob_name
                if ob_name=="All":
                    ob_name_use = "Combined"
                txt = axs[0,i].text(size*0.05, size*0.05, ob_name_use, fontsize=28, weight='bold', color='white')
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                
        if z is not None:
            from astropy.cosmology import FlatLambdaCDM
            import astropy.units as u
            cosmo = FlatLambdaCDM(H0=70., Om0 = 0.3)
            for j in range(1,3):
                for i in range(len(ob_names)):
                    if i==2:
                        continue
                    if j>0:
                        bar_size = (10*u.kpc/cosmo.angular_diameter_distance(z))*u.rad
                        bar_size_pix = bar_size.to(u.arcsec).value / self.pixscale
                        axs[j,i].errorbar([size*0.8],[size*0.9],xerr=[bar_size_pix/2.], fmt='none', capsize=10.0)
                        axs[j,i].text(size*0.8,size*0.9,"10 kpc",ha='center',va='bottom', fontsize=14)

        fig.tight_layout()
        plt.show()

        if save_fig:
            fig.savefig("{}.{}.2Dpol.png".format(self.object, self.band), dpi=200)
