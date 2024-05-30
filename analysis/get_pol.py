#!/usr/bin/env python

import numpy as np

import masks
import crz
import phot
import filter_fitsfiles
import stokes
import im_display

def get_pol(obj_id, ex_ref, ey_ref, rim_folder, filter, mask_folder, crz_folder, ob_ids=[None], mjds=None, chip="1", force=False, r_ap=2., r_an_in=4., r_an_out=7., use_masks=True, back_sub_dao=True, show_recenter_figs=False, objlim=10):

    if mjds is None:
        mjds = [None]*len(ob_ids)

    e_pos_ref = np.vstack((ex_ref,ey_ref)).T

    pol_frac   = np.zeros((e_pos_ref.shape[0], len(ob_ids)))
    pol_angle  = np.zeros((e_pos_ref.shape[0], len(ob_ids)))
    epol_frac  = np.zeros((e_pos_ref.shape[0], len(ob_ids)))
    epol_angle = np.zeros((e_pos_ref.shape[0], len(ob_ids)))

    for kid, ob_id in enumerate(ob_ids):

        esum = dict()
        eerr = dict()
        osum = dict()
        oerr = dict()

        #Select the appropriate files.
        fnames = filter_fitsfiles.filter_fitsfiles(obj_id, rim_folder, filter, ob_all=ob_id, ichip=chip, mjd_all=mjds[kid])

        for i, fname in enumerate(fnames):
            #Get the masks if needed.
            mask, omask, emask = masks.read_masks(fname, mask_folder, rim_folder, force=force, chip=chip)
            if not use_masks:
                # mask = np.zeros(mask.shape, dtype=np.int32)
                # omask = np.zeros(mask.shape, dtype=np.int32)
                # emask = np.zeros(mask.shape, dtype=np.int32)
                mask = np.zeros(mask.shape, dtype=bool)
                omask = np.zeros(mask.shape, dtype=bool)
                emask = np.zeros(mask.shape, dtype=bool)
            
            #Clean the cosmic rays if needed.
            crzname = crz.crz_clean(fname, mask, rim_folder, crz_folder, force=force, objlim=objlim)

            #Recenter the source.
            e_pos = phot.dao_recenter(crzname, e_pos_ref, emask, "e", crz_folder, back_sub=back_sub_dao)

            #Get the e_beam_photometry
            esum[crzname], eerr[crzname] = phot.run_phot(crzname, emask, e_pos, r_ap, r_an_in, r_an_out, crz_folder)

            #Now find the o-beam sources.
            o_pos = phot.dao_recenter(crzname, e_pos_ref, omask, "o", crz_folder, back_sub=back_sub_dao)

            if show_recenter_figs:
                im_display.im_display(crzname, crz_folder, e_pos, o_pos)

            #Get the o_beam_photometry
            osum[crzname], oerr[crzname] = phot.run_phot(crzname, omask, o_pos, r_ap, r_an_in, r_an_out, crz_folder)

            #print(e_pos, o_pos)

        #Now, with all the photometry done, we just need to calculate the Stokes parameters.
        pol_frac[:,kid], pol_angle[:,kid], epol_frac[:,kid], epol_angle[:,kid] = stokes.get_QU_with_errors(esum, eerr, osum, oerr, e_pos_ref, crz_folder)
        print("{0:.2f}% +/- {1:.2f}% {2:.1f} +/- {3:.1f} degrees".format(pol_frac[0,kid]*100, epol_frac[0,kid]*100, pol_angle[0,kid], epol_angle[0,kid]))

    return pol_frac, pol_angle, epol_frac, epol_angle
