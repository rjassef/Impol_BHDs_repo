#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=2:
    print "Correct use: python",sys.argv[0],"object id"
    sys.exit()

i = np.int32(sys.argv[1])
nsamp = 1000
i1 = nsamp*(i-1)
i2 = nsamp*i

data = np.loadtxt("double.22",usecols=(1,2))

ebv1 = data[i1:i2,0] 
ebv2 = data[i1:i2,1] 

ebvh = np.where(ebv1>ebv2,ebv1,ebv2)
ebvl = np.where(ebv1>ebv2,ebv2,ebv1)

plt.hist(ebvh)
#plt.hist(ebvl)
plt.show(block=True)

