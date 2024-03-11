#!/usr/bin/env python

from get_pol import get_pol
import filter_fitsfiles
import re

from myparse import myparse

###

args = myparse()

#Folders
mask_folder, crz_folder, phot_folder, rim_folder = filter_fitsfiles.default_folders(args.use_skyflats)

###

#Expected position for the source.
ex_ref = 1022.5
ey_ref = 70.0
#ex_ref = 1021.0
#ey_ref = 80.0

pol_frac = dict()
pol_angle = dict()
epol_frac = dict()
epol_angle = dict()

obj_ids = list()
filters = list()
mjds = list()

fname = "STDs_pol"
if args.use_skyflats:
    fname += "_with_skyflat"
fname += ".dat"

#Read the objects already processed. 
try:
    cat = open(fname)
    for line in cat:
        obj_id = line[:28].strip()
        x = line[28:].split()
        pol_frac[obj_id] = [[float(x[0])]]
        pol_angle[obj_id] = [[float(x[2])]]
        epol_frac[obj_id] = [[float(x[1])]]
        epol_angle[obj_id] = [[float(x[3])]]
    cat.close()
except FileNotFoundError:
    print("No previous processing found.")

#Read from the log the standards to process. 
cat = open("../proc/small_log.txt")
for line in cat:
    x = line.split()
    if not re.search("IPOL", x[-5]):
        continue

    obj_id1 = x[0]
    if len(x)==9:
        obj_id1 += " "+x[1]
    
    #mjd = "{}".format(int(float(x[-4])))
    mjd = "{:.0f}".format(round(float(x[-4]),0))
    filt = x[-1]

    if obj_id1 not in obj_ids:
        obj_ids.append(obj_id1)
    if mjd not in mjds:
        mjds.append(mjd)
    if filt not in filters:
        filters.append(filt)

    obj_id = "{}.{}.{}".format(obj_id1, filt, mjd)
    if obj_id in pol_frac:
        continue

    pol_frac[obj_id], pol_angle[obj_id], epol_frac[obj_id], epol_angle[obj_id] = get_pol(obj_id1, ex_ref, ey_ref, rim_folder, filt, mask_folder, crz_folder, r_ap=1.0, force=args.force_new, mjds=[mjd], use_masks=False)
cat.close()

# obj_ids = ["BD-12 5133", "WD 1344+106"]
# mjds = ["60147", "60148", "60158"]#, None]
# pol_frac = dict()
# pol_angle = dict()
# epol_frac = dict()
# epol_angle = dict()
# #filter="I_BESS"
# #filter="v_HIGH"
# filters = ["v_HIGH","I_BESS"]
# for filt in filters:
#     for mjd in mjds:
#         for obj_id1 in obj_ids:
#             obj_id = "{}.{}.{}".format(obj_id1, filt, mjd)
#             pol_frac[obj_id], pol_angle[obj_id], epol_frac[obj_id], epol_angle[obj_id] = get_pol(obj_id1, ex_ref, ey_ref, rim_folder, filt, mask_folder, crz_folder, r_ap=1.0, force=args.force_new, mjds=[mjd])


cato = open(fname,"w")
for filt in filters:
    for mjd in mjds:
        for obj_id1 in obj_ids:
            obj_id = "{}.{}.{}".format(obj_id1, filt, mjd)
            if obj_id not in pol_frac:
                continue
            cato.write("{0:28s} {1:10.4f} {2:10.4f} {3:10.2f} {4:10.2f}\n".format(obj_id, pol_frac[obj_id][0][0], epol_frac[obj_id][0][0], pol_angle[obj_id][0][0], epol_angle[obj_id][0][0]))
            #print("{0:28s} {1:10.4f} {2:10.4f} {3:10.2f} {4:10.2f}".format(obj_id, pol_frac[obj_id][0][0], epol_frac[obj_id][0][0], pol_angle[obj_id][0][0], epol_angle[obj_id][0][0]))
cato.close()
