import numpy as np

import os
import sys
SPEC_PIPE_LOC = "{}/JPL/W12_Drops/spec_paper/Spec_pipeline".format(os.path.expanduser("~"))
sys.path.append(SPEC_PIPE_LOC)
os.environ['SPEC_PIPE_LOC'] = SPEC_PIPE_LOC
from Spec_pipeline import SDSS_Spec, GMOS_Spec, DBSP_Spec

class LoadSpecs(object):

    def __init__(self):

        #Save the current folder.
        self.folder = os.path.dirname(os.path.realpath(__file__))

        #Read the spec data file. 
        self.spec_data = dict()
        cat = open(self.folder+"/spec_data.txt")
        for line in cat:
            x = line.split()
            self.spec_data[x[0]] = dict()
            self.spec_data[x[0]]["inst"] = x[1]
            self.spec_data[x[0]]["fname"] = x[2]
            if len(x)==4:
                self.spec_data[x[0]]["kws"] = ","+x[3]
            else:
                self.spec_data[x[0]]["kws"] = ""
        cat.close()

        #Read the redshifts of all the objects.
        d20 = open(self.folder+"/../plots/SED_models/double.20")
        d22 = open(self.folder+"/../plots/SED_models/double.22")
        for line in d20:
            x = line.split()
            y = d22.readline().split()
            if y[-1] in self.spec_data:
                self.spec_data[y[-1]]["z"] = float(x[1])
        d20.close()
        d22.close()

        #Now, load all the spectra.
        self.sp = dict()
        for obj in self.spec_data:
            cmd = "self.sp[\"{}\"] = {}_Spec(\"{}\", {}, \"{}\"{})".format(obj, self.spec_data[obj]["inst"], obj, self.spec_data[obj]["z"],  self.spec_data[obj]["fname"], self.spec_data[obj]["kws"])
            exec(cmd)


        return
