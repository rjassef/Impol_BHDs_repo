from run_pol import run_pol

obj_name = "W0220+0137"
ex_ref = 1021.5
ey_ref = 70.0

obs_params = dict()

filt = "R_SPECIAL"
obs_params[filt] = dict()
obs_params[filt]["ob_ids"] = [3565583, 3565623, None]
obs_params[filt]["mjds"] = ["60201", "60207", None]
obs_params[filt]["name"] = [None]*len(obs_params[filt]["mjds"])

run_pol(obj_name, obs_params, ex_ref, ey_ref)
