from get_pol import get_pol
import filter_fitsfiles
from myparse import myparse

def run_pol(obj_name, obs_params, ex_ref, ey_ref):

    args = myparse()

    #Folders
    mask_folder, crz_folder, phot_folder, rim_folder = filter_fitsfiles.default_folders(args.use_skyflats)

    for filter in obs_params.keys():


        ob_ids = obs_params[filter]["ob_ids"]
        mjds = obs_params[filter]["mjds"]

        pol_frac, pol_angle, epol_frac, epol_angle = get_pol(obj_name, ex_ref, ey_ref, rim_folder, filter, mask_folder, crz_folder, ob_ids=ob_ids, mjds=mjds, r_ap=1.0, force=args.force_new)

        fname = "{}_pol_{}".format(obj_name, filter)
        if args.use_skyflats:
            fname += "_with_skyflat"
        fname += ".dat"
        cato = open(fname,"w")
        for k, ob_id in enumerate(ob_ids):
            if obs_params[filter]["name"][k] is not None:
                cato.write("{0:15s} ".format(obs_params[filter]["name"][k]))
            else:
                if ob_id is None:
                    cato.write("{0:15s} ".format("All"))
                else:
                    cato.write("{0:15d} ".format(ob_id))
            cato.write("{0:10.4f} {1:10.4f} {2:10.2f} {3:10.2f}\n".format(pol_frac[0][k], epol_frac[0][k], pol_angle[0][k], epol_angle[0][k]))
        cato.close()

    return
