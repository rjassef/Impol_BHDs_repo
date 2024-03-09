import numpy as np
import subprocess
import os
from astropy.io import fits
import re

def get_dates(folder):
    cat = open("aux.dat","w")
    subprocess.call("ls -d {0:s}/60* {0:s}/59*".format(folder), shell=True, stdout=cat)
    cat.close()

    dates = list()
    cat = open("aux.dat")
    for line in cat:
        dates.append(line.rstrip()[-5:])
    cat.close()

    dates = np.array(dates,dtype="U")
    subprocess.call(["rm","aux.dat"])

    return dates

def get_list_of_files(folder):

    if os.path.isdir(folder):
        cat = open("aux.dat","w")
        subprocess.call("ls {0:s}/*".format(folder), shell=True, stdout=cat)
        cat.close()

        fnames = np.genfromtxt("aux.dat",dtype="U")
        subprocess.call(["rm","aux.dat"])
    else:
        fnames = list()

    return fnames

def create_mb_sf_sof(folder, date, ichip, type, filter=None, mb_file=""):

    #Set the appropriate subfolder.
    if type=="bias":
        subfolder = "IMAGE/CHIP{0:d}".format(ichip)
        im_type = "BIAS"
    elif type=="skyflat":
        subfolder = "IMAGE/CHIP{0:d}/{1:s}".format(ichip, filter)
        im_type = "SKY_FLAT_IMG"
        if mb_file=="":
            print("To crate a skyflat SOF, mb_file is needed")
            exit()
    else:
        print("type",type,"not implemented")
        exit()

    #List all the relevant files.
    fnames = get_list_of_files("{0:s}/{1:s}/{2:s}/".format(folder, date, subfolder))

    #Put all the relevant files in the SOF.
    sof_file = "data.{0:s}.chip{1:d}.sof".format(date,ichip)
    cato = open(sof_file,"w")
    for fname in fnames:
        cato.write("{0:s} {1:s}\n".format(fname,im_type))

    #If sky flat, add the bias.
    if type=="skyflat":
        cato.write("{0:s} MASTER_BIAS\n".format(mb_file))
    cato.close()

    return sof_file

def create_scipol_sof(fname, ifile, date, mb_file, sf_file, ichip):

    sof_file = "data.{0:s}.chip{1:d}.{2:d}.sof".format(date,ichip,ifile+1)
    cato = open(sof_file,"w")
    cato.write("{0:s} SCIENCE_IMG\n".format(fname))
    cato.write("{0:s} MASTER_BIAS\n".format(mb_file))
    if sf_file is not None:
        cato.write("{0:s} MASTER_SKY_FLAT_IMG".format(sf_file))
    cato.close()

    return sof_file


######

def reduce_bias(folder, conf_folder, cal_folder):

    #Get the dates.
    dates = get_dates(folder)

    for date in dates:
        for ichip in (1,2):

            #Create the output name file. If it exists, skip creating it.
            output_file = "{0:s}/master_bias.{1:s}.chip{2:d}.fits".format(cal_folder, date, ichip)
            try:
                open(output_file)
                print(output_file.split("/")[-1],"exists. Skipping reduction.")
                continue
            except IOError:
                print("Creating",output_file)

            #First create the data sof file.
            sof_file = create_mb_sf_sof(folder,date,ichip,type="bias")

            #Now, run esorex
            esorex_cmd = "esorex --config={0:s}/myesorex.rc fors_bias {1:s}".format(conf_folder, sof_file)
            log_file = open("log.txt", "a")
            subprocess.call(esorex_cmd, shell=True, stdout=log_file)
            log_file.close()

            #Change the name of the master_bias created.
            subprocess.call(["mv","master_bias.fits",output_file])

            #Remove the sof file.
            subprocess.call(["rm",sof_file])

    return dates

#####

def reduce_sf(folder, mb_dates, conf_folder, cal_folder):

    #Get the dates.
    dates = get_dates(folder)

    #Reduce the sky flat in each date for each chip.
    for date in dates:
        #Figure out the closest master_bias mjd.
        k = np.argmin(np.abs(np.float32(mb_dates)-float(date)))
        for ichip in (1,2):

            for filter in ("I_BESS", "R_SPECIAL", "v_HIGH"):

                #Create the output name file. If it exists, skip creating it.
                output_file = "{0:s}/master_sky_flat_img.{1:s}.chip{2:d}.fits".format(cal_folder, date, ichip)
                try:
                    open(output_file)
                    print(output_file.split("/")[-1],"exists. Skipping reduction.")
                    continue
                except IOError:
                    print("Creating",output_file)

                #First create the data sof file.
                mb_file = "{0:s}/master_bias.{1:s}.chip{2:d}.fits".format(cal_folder, mb_dates[k], ichip)
                sof_file = create_mb_sf_sof(folder, date, ichip, type="skyflat", mb_file=mb_file)

                #Now, run esorex.
                log_file = open("log.txt", "a")
                esorex_cmd = "esorex --config={0:s}/myesorex.rc fors_img_sky_flat {1:s}".format( conf_folder, sof_file)
                subprocess.call(esorex_cmd,shell=True,stdout=log_file)
                log_file.close()
                subprocess.call(["mv", "master_sky_flat_img.fits", output_file])

                #Remove the sof file.
                subprocess.call(["rm",sof_file])

                #Remove the qc file.
                subprocess.call(["rm","qc0000.paf"])

    return dates

####

def reduce_scipol(folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder):
    reduce_general(folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder, pol=True)
    return 

def reduce_unpol(folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder):
    reduce_general(folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder, pol=False)
    return 

def reduce_general(folder, mb_dates, sf_dates, conf_folder, cal_folder, rim_folder, phot_folder, pol=True):

    #Set whether we'll go through the regular or the polarimetry images.
    if pol:
        sci_type = 'POLARIMETRY'
    else:
        sci_type = 'IMAGE'

    #Get the dates.
    dates = get_dates(folder)

    #Process each file separately.
    for date in dates:

        #Figure out the closest master_bias mjd.
        k_mb = np.argmin(np.abs(np.float32(mb_dates)-float(date)))

        #Figure out the closest master_sky_flat mjd is using sky flats.
        if sf_dates is None or len(sf_dates)==0:
            k_sf = None
        else:
            k_sf = np.argmin(np.abs(np.float32(sf_dates)-float(date)))

        for ichip in (1,2):

            for filter in ("I_BESS", "R_SPECIAL", "v_HIGH"):

                #Figure out the master bias and master flats names.
                mb_file = "{0:s}/master_bias.{1:s}.chip{2:d}.fits".format(cal_folder, mb_dates[k_mb], ichip)

                sf_file = None
                if k_sf is not None:
                    sf_file = "{0:s}/master_sky_flat_img.{1:s}.chip{2:d}.fits".format(cal_folder, sf_dates[k_sf], ichip)

                #We have to go file by file here running the reduction.
                fnames = get_list_of_files("{0:s}/{1:s}/{2:s}/CHIP{3:d}/{4:s}".format(folder,date, sci_type, ichip, filter))

                #If there is a single image, the previous function will not return a list. If that is the case, make it a list.
                if not isinstance(fnames, list) and not isinstance(fnames, np.ndarray):
                    fnames = [fnames]


                for ifile, fname in enumerate(fnames):

                    #Save the source name.
                    h = fits.open(fname)
                    source_name = h[0].header['HIERARCH ESO OBS TARG NAME']
                    h.close()
                    source_name = re.sub(" ","_",source_name)

                    #Create the output name file. If it exists, skip creating it.
                    suffix = "{0:s}.{1:s}.{2:s}.chip{3:d}.{4:d}".format(source_name, date, filter,ichip, ifile+1)
                    output_file = "{0:s}/science_reduced_img.{1:s}.fits".format(rim_folder, suffix)
                    try:
                        open(output_file)
                        print(output_file.split("/")[-1],"exists. Skipping reduction.")
                        continue
                    except IOError:
                        print("Creating",output_file)

                    #Create the sof file.
                    sof_file = create_scipol_sof(fname, ifile, date, mb_file, sf_file, ichip)

                    #Now, run esorex.
                    log_file = open("log.txt", "a")
                    if sf_file is None:
                        esorex_cmd = "esorex --config={0:s}/myesorex.rc fors_remove_bias {1:s}".format(conf_folder, sof_file)
                    else:
                        esorex_cmd = "esorex --config={0:s}/myesorex.rc fors_img_science --sex_config={0:s}/my_fors.sex {1:s}".format(conf_folder, sof_file)
                    subprocess.call(esorex_cmd,shell=True, stdout=log_file)
                    log_file.close()

                    #Finally, save the image and the sextractor catalog.
                    if sf_file is None:
                        subprocess.call("mv science_img_unbias.fits {0:s}/science_reduced_img.{1:s}.fits".format(rim_folder, suffix),shell=True)

                    else:

                        subprocess.call("mv science_reduced_img.fits {0:s}/science_reduced_img.{1:s}.fits".format(rim_folder, suffix),shell=True)

                        subprocess.call("mv sources_sci_img.fits {0:s}/sources_sci_img.{1:s}.fits".format(phot_folder, suffix),shell=True)

                        #Remove the auxiliary and uneeded files.
                        subprocess.call(["rm","qc0000.paf","object_table_sci_img.fits", "phot_background_sci_img.fits"])

                    #Remove the sof file.
                    subprocess.call(["rm",sof_file])

    return dates
