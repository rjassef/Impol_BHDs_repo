import numpy as np
from astropy.table import Table
from run_pol import run_pol
import re

#Load the log. 
log = Table.read("../proc/log.fits")

#Separate the standards from the science targets. The standards all have the same program ID. 
stds_log = log[log['Program_ID']=='60.A-9203(E)']

#Get the target names. 
obj_names = np.unique(stds_log['Target'])

#Set the reference location to find them. They should all be around the sample place. 
ex_ref = 1022.5
ey_ref = 70.

#Go through each of the objects and call run_pol. 
for obj_name in obj_names:
    #if obj_name != "BD-14 4922":
    #    continue
    print(obj_name)
    aux_log = stds_log[stds_log['Target']==obj_name]

    filts = np.unique(aux_log['Filter'])
    obs_params = dict()
    for filt in filts:
        obs_params[filt] = dict()
        aux_aux_log = aux_log[aux_log['Filter']==filt]
        obs_params[filt]["mjds"], ku = np.unique(aux_aux_log['MJD_folder'], return_index=True)
        obs_params[filt]["ob_ids"] = aux_aux_log['OB_ID'][ku].data.tolist()
        obs_params[filt]["mjds"] = obs_params[filt]["mjds"].data.astype(str).tolist()
        obs_params[filt]["name"] = [None]*len(obs_params[filt]["mjds"])
        obs_params[filt]["mean_mjd"] = list()
        for mjd_folder in obs_params[filt]["mjds"]:
            mjd_folder = int(float(mjd_folder))
            obs_params[filt]["mean_mjd"].append(np.mean(aux_aux_log['MJD'][aux_aux_log['MJD_folder']==mjd_folder]))

    objlim = 10
    r_ap = 1.0
    if obj_name == "BD-14 4922":
        objlim = 100000000
        r_ap = 0.7

    run_pol(obj_name, obs_params, ex_ref, ey_ref, use_masks=False, show_recenter_figs=False, objlim=objlim, r_ap=r_ap)
