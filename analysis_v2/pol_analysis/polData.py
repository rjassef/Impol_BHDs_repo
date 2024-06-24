import numpy as np
import os
import re
import subprocess
from pathlib import Path

from astropy.table import Table
from astropy.io import fits
from astroscrappy import detect_cosmics

from .polMasks import PolMasks


class PolData(object):

    def __init__(self, obj_id, bband, show_plots=False, use_skyflats=False, force_new=False, rim_folder=None, root=None, use_masks=True, crz_objlim=10, filenames_for_mask_creation=None):

        #Save the input parameters.
        for k,v in locals().items():
            if k!='self':
                setattr(self, k, v)

        #Set the default folders to use.
        suffix = ""
        if self.use_skyflats:
            suffix = "_with_skyflat"
        if self.root is None:
            self.root = os.getcwd()
        self.mask_folder = self.root+"/masks"+suffix
        self.crz_folder = self.root+"/crz"+suffix
        self.phot_folder = self.root+"/phot"+suffix
        self.bkg_folder = self.root+"/bkg"+suffix
        self.bstar_folder = self.root+"/bright_stars_lists"

        #Create folders that do not exist. 
        for fold in [self.mask_folder, self.crz_folder, self.phot_folder, self.bkg_folder]:
            if not Path(fold).exists():
                subprocess.run("mkdir {}".format(fold), shell=True)

        #This folder needs to point to the reduced images. The default is in Impol_BHDs_repo/proc/reduced_images
        if self.rim_folder is None:
            m = re.search("^(.*/Impol_BHDs_repo).*$",self.root)
            self.rim_folder = m.group(1)+"/proc/reduced_images"+suffix

        #Each set of observations is defined for a pair of OB_ID and MJD. Let's find from the log all the relevant pairs for the given objects and broad-band. 
        log = Table.read(self.rim_folder+"/../log.fits")
        self.ob_pairs = np.unique(log['OB_ID','MJD_folder'][(log['Target']==self.obj_id) & (log['Filter']==self.bband)])

        #Find the reduced file names associated to each set of observations. 
        self.filter_fitsfiles()

        #If masks are requested, create them if they do not already exist. 
        self.mask_obj = PolMasks(self)

        #Finally, clean the cosmic rays. 
        self.crz_clean()

        return 
    
    def filter_fitsfiles(self):

        #Create the structure to hold all the filenames. 
        self.file_names = dict()

        #Get the filenames corresponding to each pair. 
        for ob_pair in self.ob_pairs:
            ob_id, mjd = ob_pair
            if ob_id not in self.file_names:
                self.file_names[ob_id] = dict()
            self.file_names[ob_id][mjd] = dict()

            #Get separately the file names for each chip. 
            for ichip in ("1","2"):
                
                #We want to save a list. 
                self.file_names[ob_id][mjd][ichip] = list()

                #List all the files matching the criteria. 
                ls_output = subprocess.run("ls {}/*.{}.{}.{}.chip{}.*.fits".format(self.rim_folder, self.obj_id, mjd, self.bband, ichip), shell=True, capture_output=True)
                fnames = ls_output.stdout.decode('utf8').split()

                #Save only the files that have the correct OB ID. 
                for fname in fnames:
                    h = fits.open(fname)
                    if h[0].header['HIERARCH ESO OBS ID']!=ob_id:
                        continue
                    self.file_names[ob_id][mjd][ichip].append(re.sub(r'^.*/(.*?)$', r'\1', fname))

        return
    
    #List filenames for a set of OB IDs and MJDs. If none specified, all files will be listed.
    def list_of_filenames(self, ob_ids=None, mjds=None, chips=None):
        fname_list = list()
        if ob_ids is None:
            ob_ids = list(self.file_names.keys())
        for ob_id in ob_ids:
            if mjds is None:
                mjds_use = list(self.file_names[ob_id].keys())
            else:
                mjds_use = mjds
            for mjd in mjds_use:
                if mjd not in self.file_names[ob_id]:
                    continue
                if chips is None:
                    chips_use = list(self.file_names[ob_id][mjd].keys())
                else:
                    chips_use = chips
                for chip in chips_use:
                    fname_list.extend(self.file_names[ob_id][mjd][chip])

        return fname_list

    def crz_clean(self):

        #Get all the filenames.
        for fname in self.list_of_filenames():

            #Compute the CR corrected image if needed.
            crzname = re.sub(".fits",".crz.fits",fname)
            if self.force_new or not Path("{}/{}".format(self.crz_folder, crzname)).exists():
                print("Cleaning cosmic rays in ",fname)
                h = fits.open("{0:s}/{1:s}".format(self.rim_folder, fname))
                mask, omask, emask = self.mask_obj.read_masks(fname)
                crmask, clean_im = detect_cosmics(h[0].data, inmask=mask, objlim=self.crz_objlim)
                h[0].data = clean_im
                h.writeto("{0:s}/{1:s}".format(self.crz_folder, crzname), overwrite=True)
                h.close()

                mname, omname, emname = self.mask_obj.get_mask_names(fname)
                mask = mask | crmask
                emask = emask | crmask
                omask = omask | crmask
                fits.writeto("{0:s}/{1:s}".format(self.mask_folder,mname),  mask.astype(int),overwrite=True)
                fits.writeto("{0:s}/{1:s}".format(self.mask_folder,omname),  omask.astype(int),overwrite=True)
                fits.writeto("{0:s}/{1:s}".format(self.mask_folder,emname),  emask.astype(int),overwrite=True)                

        return #crzname