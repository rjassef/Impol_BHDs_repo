#!/usr/bin/env python

from get_pol import get_pol
import filter_fitsfiles
from myparse import myparse

###

args = myparse()

#Folders
mask_folder, crz_folder, phot_folder, rim_folder = filter_fitsfiles.default_folders(args.use_skyflats)

###

#Expected position for the source.
ex_ref = 1021.5
ey_ref = 70.0

filters = ["R_SPECIAL"]
for filter in filters:

    if filter=="R_SPECIAL":
        ob_ids = [3565057, 3565577, None]
        mjds = ["60201", "60202", None]


    pol_frac, pol_angle, epol_frac, epol_angle = get_pol("W0019-1046", ex_ref, ey_ref, rim_folder, filter, mask_folder, crz_folder, ob_ids=ob_ids, mjds=mjds, r_ap=1.0, force=args.force_new)

    fname = "W0019_pol_{}".format(filter)
    if args.use_skyflats:
        fname += "_with_skyflat"
    fname += ".dat"
    cato = open(fname,"w")
    for k, ob_id in enumerate(ob_ids):
        if ob_id is None:
            cato.write("{0:15s} ".format("All"))
        else:
            cato.write("{0:15d} ".format(ob_id))
        cato.write("{0:10.4f} {1:10.4f} {2:10.2f} {3:10.2f}\n".format(pol_frac[0][k], epol_frac[0][k], pol_angle[0][k], epol_angle[0][k]))
    cato.close()
