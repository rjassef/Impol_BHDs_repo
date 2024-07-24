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

def get_S(psi_angles, pobj, forward, backward, theta_scattering_angles):
    #This is the array that will hold the S values. 
    S = np.zeros((len(theta_scattering_angles), len(psi_angles), 3))*pobj.dsigma_norm.unit
    for k, theta in enumerate(theta_scattering_angles):
        tmod = torus_model(theta, pobj)
        tmod.get_integrals(psi_angles, forward_scattering=forward, backward_scattering=backward)
        S[k,:,0] = tmod.S1
        S[k,:,1] = tmod.S2
        S[k,:,2] = tmod.S3
    return S


###

wids = ['W0116-0505', 'W0019-1046', 'W0220+0137', 'W0204-0506', 'W0831+0140']
zs = [3.173, 1.641, 3.122, 2.100, 3.913]
bands = ["v","R","I"]
band_centers = np.array([5500., 6500., 8000.])*u.AA

for iwid, wid in enumerate(wids):
    subprocess.call("mkdir {}".format(wid), shell=True)
    for iband, band in enumerate(bands):
        lam_targ = band_centers[iband]/(1+zs[iwid])

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
        psi_angles = np.arange(1.0,90.1,1.0)*u.deg

        #This is the range of inclination angles we'll consider. 
        theta_scattering_angles = np.arange(1.0, 90.1, 1.0)*u.deg

        #Run through evey particle object.
        for i, pobj in enumerate(pobjs):

            print(wid, band, model_names[i])

            #Do forwards and backwards scattering separately. 
            for j in range(3):
                if j==0:
                    forward=True
                    backward=False
                    suffix = "FW"
                elif j==1:
                    forward=False
                    backward=True
                    suffix  = "BW"
                elif j==2:
                    forward=True
                    backward=True
                    suffix = "FWBW"

                if model_names[i]=="gas":
                    fname = "{}/{}_S.hires.{}.txt".format(wid,model_names[i],suffix)
                else:
                    fname = "{}/{}_S.hires.{}.{}.txt".format(wid,model_names[i], band, suffix)
                if os.path.exists(fname):
                    continue

                #Start the multiprocessing.
                Ncpu = mp.cpu_count()
                Pool = mp.Pool(Ncpu)
                func = partial(get_S, psi_angles, pobj, forward, backward)

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
