#!/usr/bin/env python

import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

import masks
import crz
import phot
import filter_fitsfiles
import stokes
from myparse import myparse

###

args = myparse()

#Folders
mask_folder, crz_folder, phot_folder, rim_folder = filter_fitsfiles.default_folders(args.use_skyflats)

###

#DAO recenter box size.
box_size = 21

#Photometry parameters
r_ap = 1.0
r_an_in = 4.
r_an_out = 7.

###

chips = ["1", "2"]
for kchip, chip in enumerate(chips):
    #Filter the fits_files provided so that we only use those in which the object is the input object.
    fnames = filter_fitsfiles.filter_fitsfiles("W0116-0505", rim_folder, ichip=chip)
    osum = dict()
    oerr = dict()
    esum = dict()
    eerr = dict()
    e_pos = dict()
    o_pos = dict()

    #First, we'll need to get the reference positions. We will start with the positions in the first image, and then eliminate any object whose position cannot be found in subsequent images.
    for i, fname in enumerate(fnames):
        #Get the masks if needed.
        mask, omask, emask = masks.read_masks(fname, mask_folder, rim_folder, force=args.force_new, chip=chip)

        #Clean the cosmic rays if needed.
        crzname = crz.crz_clean(fname, mask, rim_folder, crz_folder, force=args.force_new)

        #Find the sources. Only do this if it is the first file being processed.
        if i==0:
            e_pos = phot.ebeam_dao_find(crzname, emask, crz_folder, phot_folder, force=True)

            #Important for chip2, since the top e-beam image does not have a corresponding o-beam image.
            e_pos_ref = e_pos[e_pos[:,1]<1024-91,:]

        e_pos = phot.ebeam_dao_recenter(crzname, e_pos_ref, emask, crz_folder, phot_folder, force=True, box_size=box_size)
        e_pos_ref = e_pos_ref[(~np.isnan(e_pos[:,0])) | (~np.isinf(e_pos[:,0])),:]
        shift_filter = phot.filter_by_shift(e_pos, e_pos_ref)
        e_pos_ref = e_pos_ref[shift_filter,:]
        e_pos     = e_pos[shift_filter,:]

        o_pos = phot.obeam_dao_recenter(crzname, e_pos, omask, crz_folder, phot_folder, force=True, box_size=box_size)
        e_pos_ref = e_pos_ref[(~np.isnan(o_pos[:,0])) | (~np.isinf(o_pos[:,0])),:]
        shift_filter = phot.filter_by_shift(o_pos, e_pos)
        e_pos_ref = e_pos_ref[shift_filter]

    neg_flux = np.zeros(len(e_pos_ref[:,0]), dtype=np.int32)
    #Now, get the photometry
    for i, fname in enumerate(fnames):
        #Get the masks if needed.
        mask, omask, emask = masks.read_masks(fname, mask_folder, rim_folder)

        #Clean the cosmic rays if needed.
        crzname = crz.crz_clean(fname, mask, rim_folder, crz_folder)

        #Recenter the sources.
        e_pos = phot.ebeam_dao_recenter(crzname, e_pos_ref, emask, crz_folder, phot_folder, force=True, box_size=box_size)

        o_pos = phot.obeam_dao_recenter(crzname, e_pos, omask, crz_folder, phot_folder, force=True, box_size=box_size)

        #Get the e_beam_photometry
        esum[crzname], eerr[crzname] = phot.run_phot_all(crzname, "e", emask, e_pos, r_ap, r_an_in, r_an_out, crz_folder, phot_folder, force=True)

        #Get the o_beam_photometry
        osum[crzname], oerr[crzname] = phot.run_phot_all(crzname, "o", omask, o_pos, r_ap, r_an_in, r_an_out, crz_folder, phot_folder, force=True)

        neg_flux = np.where((osum[crzname]<0.) | (esum[crzname]<0.), 1, 0)

    #Now, with all the photometry done, we just need to calculate the Stokes parameters.
    pol_frac, pol_angle, epol_frac, epol_angle = stokes.get_QU_with_errors(esum, eerr, osum, oerr, e_pos_ref, crz_folder)
    if kchip==0:
        pol_frac_all   = np.copy(pol_frac)
        pol_angle_all  = np.copy(pol_angle)
        epol_frac_all  = np.copy(epol_frac)
        epol_angle_all = np.copy(epol_angle)
        e_pos_ref_all  = np.copy(e_pos_ref)
        neg_flux_all   = np.copy(neg_flux)
    else:
        pol_frac_all   = np.concatenate([pol_frac_all  , pol_frac  ])
        pol_angle_all  = np.concatenate([pol_angle_all , pol_angle ])
        epol_frac_all  = np.concatenate([epol_frac_all , epol_frac ])
        epol_angle_all = np.concatenate([epol_angle_all, epol_angle])
        e_pos_ref_all  = np.concatenate([e_pos_ref_all , e_pos_ref ])
        neg_flux_all   = np.concatenate([neg_flux_all  , neg_flux  ])

    # np.savetxt("Chip{0:s}_field_W0116_pol.dat".format(chip), np.array([pol_frac[neg_flux==0], pol_angle[neg_flux==0], epol_frac[neg_flux==0], epol_angle[neg_flux==0], e_pos_ref[neg_flux==0,0], e_pos_ref[neg_flux==0,1]]).T)

pol_frac_all   = pol_frac_all[neg_flux_all==0]
epol_frac_all  = epol_frac_all[neg_flux_all==0]
pol_angle_all  = pol_angle_all[neg_flux_all==0]
epol_angle_all = epol_angle_all[neg_flux_all==0]
e_pos_ref_all  = e_pos_ref_all[neg_flux_all==0]
print("Highest polarization object: pol_frac={0:.2f}, angle={1:.1f}".format(np.max(pol_frac_all), pol_angle_all[np.argmax(pol_frac_all)]))

if args.show_plots:

    plt.hist(pol_frac_all, bins=20)
    plt.ylabel("N")
    plt.xlabel("Polarization Fraction")
    plt.show()

    plt.plot(epol_frac_all, pol_frac_all, 'o')
    plt.ylabel("Polarization Fraction")
    plt.xlabel("Polarization Fraction Error")
    plt.show()

out_file = "All_field_W0116_pol"
if args.use_skyflats:
    out_file += "_with_skyflat"
out_file += ".dat"
np.savetxt(out_file, np.array([pol_frac_all, pol_angle_all, epol_frac_all, epol_angle_all, e_pos_ref_all[:,0], e_pos_ref_all[:,1]]).T)
