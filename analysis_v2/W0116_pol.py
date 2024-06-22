from run_pol import run_pol

obj_name = "W0116-0505"
ex_ref = 1021.5
ey_ref = 70.0

obs_params = dict()

filt = "R_SPECIAL"
obs_params[filt] = dict()
obs_params[filt]["ob_ids"] = [2886768, 2886765, 2886772, 2886622, None]
obs_params[filt]["mjds"]   = ["59135", "59136", "59136", "59137", None]
obs_params[filt]["name"]   = [None]*len(obs_params[filt]["mjds"])

filt = "I_BESS"
obs_params[filt] = dict()
obs_params[filt]["ob_ids"] = [3564862, 3564862, 3565005, None, 3564862, None]
obs_params[filt]["mjds"]   = ["60146", "60148", "60148", "60148", "60201",None]
obs_params[filt]["name"]   = [None]*len(obs_params[filt]["mjds"])

#Add some custom combinations for I_BESS.
obs_params[filt]["ob_ids"].extend([
    [3564862, 3565005],
    [3564862, 3565005, 3564862]
])
obs_params[filt]["mjds"].extend([
    ["60201", "60148"],
    ["60201", "60148", "60148"]
])
obs_params[filt]["name"].extend([
    "Best2",
    "Best3"
])


filt="v_HIGH"
obs_params[filt] = dict()
obs_params[filt]["ob_ids"] = [3564847, None]
obs_params[filt]["mjds"]   = ["60143", None]
obs_params[filt]["name"]   = [None]*len(obs_params[filt]["mjds"])

run_pol(obj_name, obs_params, ex_ref, ey_ref)