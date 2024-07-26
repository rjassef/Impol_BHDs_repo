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
            self.bp[band] = SpectralElement(Empirical1D, points=wave[wave<10000.], lookup_table=thru[wave<10000.], keep_neg=True)

        return 