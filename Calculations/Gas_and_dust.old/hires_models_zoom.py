import numpy as np
import astropy.units as u
import multiprocessing as mp
from functools import partial
import subprocess
import os

from torus_model import torus_model
from free_electrons import free_electrons
from draine_dust import draine_dust

###

def get_S(psi_angles, pobj, theta_scattering_angles):
    #This is the array that will hold the S values. 
    S = np.zeros((len(theta_scattering_angles), len(psi_angles), 3))*pobj.dsigma_norm.unit
    for k, theta in enumerate(theta_scattering_angles):
        tmod = torus_model(theta, pobj)
        tmod.get_integrals(psi_angles)
        S[k,:,0] = tmod.S1
        S[k,:,1] = tmod.S2
        S[k,:,2] = tmod.S3
    return S


###

bhds = dict()

wid = 'W0116-0505'
bhds[wid] = dict()
bhds[wid]['z'] = 3.173
bhds[wid]['bands'] = ["v", "R", "I"]
bhds[wid]['pfrac'] = [0.0973, 0.109, 0.147]

wid = 'W0019-1046'
bhds[wid] = dict()
bhds[wid]['z'] = 1.641
bhds[wid]['bands'] = ["R"]
bhds[wid]['pfrac'] = [0.064]

wid = 'W0220+0137'
bhds[wid] = dict()
bhds[wid]['z'] = 3.122
bhds[wid]['bands'] = ["R"]
bhds[wid]['pfrac'] = [0.129] 

wid = 'W0204-0506'
bhds[wid] = dict()
bhds[wid]['z'] = 2.100
bhds[wid]['bands'] = ["R"]
bhds[wid]['pfrac'] = [0.253]

band_centers = dict()
band_centers['v'] = 5500.*u.AA
band_centers['R'] = 6500.*u.AA
band_centers['I'] = 8000.*u.AA

# wids = ['W0116-0505', 'W0019-1046', 'W0220+0137', 'W0204-0506']
# zs = [3.173, 1.641, 3.122, 2.100]
# bands = ["v","R","I"]
# band_centers = np.array([5500., 6500., 8000.])*u.AA

for iwid, wid in enumerate(list(bhds.keys())):
    subprocess.call("mkdir {}".format(wid), shell=True)
    for jband, band in enumerate(bhds[wid]['bands']):
        lam_targ = band_centers[band]/(1+bhds[wid]['z'])

        #Set the particle objects for all models we'll run.
        model_names = ["gas", "SMC_dust", "LMC_dust", "MW_dust"]
        pobjs = list()
        for model_name in model_names:
            if model_name == "gas":
                #For gas we'll assume only free electrons doing the scattering.
                pobjs.append(free_electrons())
            elif model_name == "SMC_dust":
                pobjs.append(draine_dust(lam_targ, "SMC"))
            elif model_name == "LMC_dust":
                pobjs.append(draine_dust(lam_targ, "LMC"))
            elif model_name == "MW_dust":
                pobjs.append(draine_dust(lam_targ, "MW"))
            else:
                continue

        #This is the range of angles for which we'll calculate the integrals. 
        psi_angles = np.arange(0.1,40.01,0.1)*u.deg

        #This is the range of inclination angles we'll consider. 
        #theta_scattering_angles = np.arange(1.0, 90.1, 1.0)*u.deg

        #Run through evey particle object.
        for i, pobj in enumerate(pobjs):

            fname = "{}/{}_S.hires.{}.zoom.txt".format(wid,model_names[i], band)
            if os.path.exists(fname):
                continue

            print(wid, band, model_names[i])

            #This is the range of inclination angles we'll consider.
            tmod_aux = torus_model(None, pobj)
            theta_s_min = tmod_aux.find_th_min(bhds[wid]['pfrac'][jband])[0].to(u.deg).value
            dtheta_s = 0.1
            theta_scattering_angles = np.arange(theta_s_min+dtheta_s, theta_s_min+2.0, dtheta_s)*u.deg

            #Start the multiprocessing.
            Ncpu = mp.cpu_count()
            Pool = mp.Pool(Ncpu)
            func = partial(get_S, psi_angles, pobj)

            #Produce the data chunks.
            th_s_split = np.array_split(theta_scattering_angles, Ncpu)
            S = Pool.map(func, th_s_split)
            S = np.vstack(S)
            Pool.close()

            #Save S.
            S = S.to(u.cm**2).value
            cat = open(fname,"w")
            for l,theta in enumerate(theta_scattering_angles):
                for k,psi in enumerate(psi_angles):
                    cat.write("{0:10.1f} {1:10.2f} {2:15.6e} {3:15.6e} {4:15.6e}\n".format(
                        theta.to(u.deg).value, psi.to(u.deg).value, S[l,k,0], S[l,k,1], S[l,k,2]))
            cat.close()
