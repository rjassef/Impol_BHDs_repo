import numpy as np
import os

class ObjectProperties(object):

    def __init__(self, wids=None):

        #WISE short IDs of the objects we will read. 
        if wids is None:
            self.wids = ["W0019-1046","W0204-0506","W0220+0137","W0831+0140","W0116-0505"]
        else:
            self.wids = wids

        #Set the bands that will be used for each object. All have R_SPECIAL observations, and W0116-0505 also has I and v. 
        self.filters = dict()
        for wid in self.wids:
            self.filters[wid] = ["R_SPECIAL"]
        if "W0116-0505" in self.wids:
            self.filters["W0116-0505"].extend(["I_BESS","v_HIGH"])

        #Current folder where this script resides:
        self.script_folder = os.path.dirname(os.path.realpath(__file__))

        #Read the data files. 
        self._read_pol()
        self._read_redshift()

        return

    def _read_pol(self):

        #Read the polarization fraction values. 
        self.pfrac = dict()
        self.epfrac = dict()
        for wid in self.wids:
            self.pfrac[wid] = dict()
            self.epfrac[wid] = dict()
            for filt in self.filters[wid]:
                data = np.genfromtxt("{}/../../analysis_v3/{}_pol_{}.dat".format(self.script_folder,wid,filt), usecols=[0,1,7], dtype=[('pfrac', '<f8'), ('epfrac','<f8'),('OBID','<U10')])
                if len(data.shape)>0:
                    k = np.argwhere(data['OBID']=="All")[0][0]
                    self.pfrac[wid][filt] = data['pfrac'][k]
                    self.epfrac[wid][filt] = data['epfrac'][k]
                else:
                    self.pfrac[wid][filt] = data['pfrac']
                    self.epfrac[wid][filt] = data['epfrac']
        return
    
    def _read_redshift(self):

        #Find the redshift of each object. 
        self.z = dict()
        d20 = open("{}/../SED_models/double.20".format(self.script_folder))
        d22 = open("{}/../SED_models/double.22".format(self.script_folder))
        for line in d20:
            x = line.split()
            y = d22.readline().split()
            if y[-1] in self.wids:
                self.z[y[-1]] = float(x[1])
        d20.close()
        d22.close()
