from run_pol import run_pol

obj_name = "W0019-1046"
ex_ref = 1021.5
ey_ref = 70.0

obs_params = dict()

filt = "R_SPECIAL"
obs_params[filt] = dict()
obs_params[filt]["ob_ids"] = [
    3565057, 
    3565577, 
    [3565057, 3565577],
    None
]
obs_params[filt]["mjds"] = [
    "60201", 
    "60202",
    ["60201", "60202"],
    None  
]
obs_params[filt]["name"] = [
    None,
    None,
    "Combined",
    None
]

run_pol(obj_name, obs_params, ex_ref, ey_ref, r_ap=0.5, show_recenter_figs=True, save_output=False)
