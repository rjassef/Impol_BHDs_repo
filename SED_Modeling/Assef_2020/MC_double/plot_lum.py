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

data = np.loadtxt("kc_components.dat")

ll6um1 = np.log10(data[i1:i2,3]) + np.log10(3.839) + 33 + 14
ll6um2 = np.log10(data[i1:i2,4]) + np.log10(3.839) + 33 + 14

ll6umh = np.where(ll6um1>ll6um2,ll6um1,ll6um2)
ll6uml = np.where(ll6um1>ll6um2,ll6um2,ll6um1)

plt.hist(ll6umh)
#plt.hist(ll6uml)
plt.show(block=True)

