#!/usr/bin/env python

from get_pol import get_pol
import filter_fitsfiles
from myparse import myparse

###

args = myparse()

#Folders
mask_folder, crz_folder, phot_folder, rim_folder = filter_fitsfiles.default_folders(args.use_skyflats)

###

#The observing blocks are:
#W0116-0505  59135 2886768
#W0116-0505  59136 2886765
#W0116-0505  59136 2886772
#W0116-0505  59137 2886622
#ob_ids = [3564862, 3565005, None]
#ob_ids = [3564847, None]

#Expected position for the source.
ex_ref = 1021.5
ey_ref = 70.0

filters = ["v_HIGH", "R_SPECIAL", "I_BESS"]
for filter in filters:

    if filter=="R_SPECIAL":
        ob_ids = [2886768, 2886765, 2886772, 2886622, None]
        mjds = ["59135", "59136", "59136", "59137", None]

    if filter=="I_BESS":
        ob_ids = [3564862, 3564862, 3565005, None, None]
        mjds = ["60146", "60148", "60148", "60148", None]

    if filter=="v_HIGH":
        ob_ids = [3564847, None]
        mjds = ["60143", None]


    pol_frac, pol_angle, epol_frac, epol_angle = get_pol("W0116-0505", ex_ref, ey_ref, rim_folder, filter, mask_folder, crz_folder, ob_ids=ob_ids, mjds=mjds, r_ap=1.0, force=args.force_new)

    fname = "W0116_pol_{}".format(filter)
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
