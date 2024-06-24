from run_pol import run_pol

obj_name = "W0204-0506"
ex_ref = 1021.5
ey_ref = 70.5

obs_params = dict()

filt = "R_SPECIAL"
obs_params[filt] = dict()
obs_params[filt]["ob_ids"] = [3565580, 3565504, None]
obs_params[filt]["mjds"] = ["60207", "60209", None]
obs_params[filt]["name"] = [None]*len(obs_params[filt]["mjds"])

run_pol(obj_name, obs_params, ex_ref, ey_ref)
