import numpy as np
import os
from synphot import SpectralElement, Empirical1D

class ReadBands(object):

    def __init__(self, bandnames=None):

        #If not filters specifically are requested, load them all. 
        if bandnames is None:
            self.bandnames = ["R_SPECIAL", "I_BESS", "v_HIGH"]
        else:
            self.bandnames = bandnames

        #Folder of this script. 
        self.script_folder = os.path.dirname(os.path.realpath(__file__))

        #Load the bands with synphot.
        self.filt_files = dict()
        cat = open(self.script_folder+"/filt_files.txt")
        for line in cat:
            x = line.split()
            if x[0] in self.bandnames:
                self.filt_files[x[0]] = x[1]
        cat.close()

        self.bp = dict()
        for band in self.bandnames:
            skiprows=0
            if band=="v_HIGH":
                skiprows=2
            data = np.loadtxt("{}/{}".format(self.script_folder,self.filt_files[band]),skiprows=skiprows)
            wave = data[:,0]*10
            thru = data[:,1]

            #We remove wavelengths above 1um to not have issues with the spectral overlap, and to not overly rely on the SED model extrapolation. Removing the wavelengths below 4000 and throughput below 0.01 has a measurable effect in some targets. This is how it was being done before, but there is no real reason to. Change the commented line to reproduce those earlier results. 
            #cond = (wave<10000.) & (thru>0.01) & (wave>4000.)
            cond = (wave<10000.)
            self.bp[band] = SpectralElement(Empirical1D, points=wave[cond], lookup_table=thru[cond], keep_neg=True)

        return 