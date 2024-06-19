#!/usr/bin/env python 

import numpy as np
from astropy.table import Table
from astropy.io import fits
import subprocess

#Get the header items we will pull from each fits file.
hkws = {
    "Target"      : "HIERARCH ESO OBS TARG NAME",
    "Program_ID"  : "HIERARCH ESO OBS PROG ID",
    "OB_Name"     : "HIERARCH ESO OBS NAME",
    "OB_ID"       : "HIERARCH ESO OBS ID",
    "Ret Angle"   : "HIERARCH ESO INS RETA2 ROT",
    "Exptime_real": "HIERARCH ESO DET WIN1 DIT1", #"HIERARCH ESO INS SHUT EXPTIME",
    "Exptime_req" : "HIERARCH ESO DET WIN1 UIT1",
    "AIRMASS_1"   : "HIERARCH ESO TEL AIRM START",
    "AIRMASS_2"   : "HIERARCH ESO TEL AIRM END",
    "MJD"         : "MJD-OBS",
    "SEEING_1"    : "HIERARCH ESO TEL AMBI FWHM START",
    "SEEING_2"    : "HIERARCH ESO TEL AMBI FWHM END",
    "Filter"      : "HIERARCH ESO INS FILT1 NAME",
    "MJD_folder"  : None,
}


#These are the folders we will explore for fits files.
# mjd_folders = dict()
# mjd_folders["W0116-0505"] = [
#     59135,
#     59136,
#     59137,
#     60143,
#     60146,
#     60148,
#     60201,
# ]
# mjd_folders["STD"] = [
#     59117,
#     59134,
#     60147,
#     60148,
#     60158,
#     60201,
#     60223,
#     60238,
#     60293,
# ]
# mjd_folders["W0019-1046"] = [
#     60201,
#     60202,
# ]
# mjd_folders["W0220+0137"] = [
#     60201,
#     60207,
# ]
# mjd_folders["W0204-0506"] = [
#     60207,
#     60209,
# ]
# mjd_folders["W0831+0140"] = [
#     60290,
#     60291,
# ]

#These are the folders we will explore for fits files.
mjd_folders = dict()
ls_output = subprocess.run("ls -d */*/POLARIMETRY", shell=True, capture_output=True)
for fold in ls_output.stdout.decode('utf8').split():
    obj, mjd, _ = fold.split("/")
    if obj not in mjd_folders:
        mjd_folders[obj] = list()
    mjd_folders[obj].append(int(float(mjd)))


for fold1 in mjd_folders.keys():
    for fold2 in mjd_folders[fold1]:

        #List all of the fits files in the folder.
        ls_output = subprocess.run("ls {}/{}/POLARIMETRY/CHIP1/*/*.fits".format(fold1, fold2), shell=True, capture_output=True)
        fnames = ls_output.stdout.decode('utf8').split()

        #Go through every file and get the requested keywords from the headers.
        for fname in fnames:

            #Open the fits file and get the keywords from the header. 
            hdu = fits.open(fname)
            new_row = []
            for key in hkws:
                if key == "MJD_folder":
                    new_row.append(fold2)
                else:
                    try:
                        new_row.append(hdu[0].header[hkws[key]])
                    except KeyError:
                        print(fname, key)
            hdu.close()

            #If the table to hold the log does no exist, create it. 
            try:
                log
            except NameError:
                dtypes = []
                for x in new_row:
                    dtypes.append(type(x))
                log = Table(names=tuple(hkws.keys()), dtype=dtypes)

            #Add the file information to the log table.
            log.add_row(new_row)

#Save the log. 
log.write("log.fits", format='fits', overwrite=True)

#Now, go through the different targets, and for each OB get the data of interest.
cato = open("small_log.txt","w")
for target in np.unique(log['Target']):
    target_log = log[log['Target']==target]
    for ob_id in np.unique(target_log['OB_ID']):
        ob_log1 = target_log[target_log['OB_ID']==ob_id]
        for mjd_folder in np.unique(ob_log1['MJD_folder']):
            ob_log2 = ob_log1[ob_log1['MJD_folder']==mjd_folder]
            for filter in np.unique(ob_log2['Filter']):
                ob_log = ob_log2[ob_log2['Filter']==filter]

                #Mean airmass.
                airmass = np.round(np.mean([ob_log['AIRMASS_1'], ob_log['AIRMASS_2']]), 3)

                #Mean seeing.
                seeing = np.round(np.mean([ob_log['SEEING_1'], ob_log['SEEING_2']]), 2)

                #Mean MJD
                mjd = np.round(np.mean(ob_log['MJD']), 2)

                #Program ID.
                prog_ID = ob_log['Program_ID'][0]

                #OB Name
                ob_name = ob_log['OB_Name'][0]

                #Filter
                filter = ob_log['Filter'][0]

                #Write the small log line.
                cato.write("{0:15s} {1:15s} {2:10d} {3:20s} {4:8.2f} {5:8.3f} {6:8.2f} {7:s}\n".format(target, prog_ID, ob_id, ob_name, mjd, airmass, seeing, filter))
cato.close()
