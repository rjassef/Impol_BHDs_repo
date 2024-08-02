#!/usr/bin/python

import numpy as np

acat = open("agn_lum_corr.dat")
acorr = float(acat.readline().split()[-1])
acat.close()

#Number of resamples create created for fake cat.
nsamp = 1000

cat  = open("kc_components.dat")
rcat = open("../kc_components.double.dat")
cato = open("errors.dat","w")
f22  = open("../double.22")
for line in rcat.readlines():

    x = [float(ix) for ix in line.split()]
    name   = f22.readline().split()[4]

    ebv_h  = []
    L6um_h = []
    ebv_l  = []
    L6um_l = []
    for i in range(nsamp):
        y = [float(iy) for iy in cat.readline().split()]
        ebv_l.append(y[1])
        ebv_h.append(y[2])
        L6um_l.append(y[3])
        L6um_h.append(y[4])

    oline  = "%10d %15s" % (x[0],name)
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        x[1],np.mean(ebv_l),np.median(ebv_l),np.std(ebv_l))
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        x[2],np.mean(ebv_h),np.median(ebv_h),np.std(ebv_h))
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        x[3],np.mean(L6um_l),np.median(L6um_l),np.std(L6um_l))
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        x[4],np.mean(L6um_h),np.median(L6um_h),np.std(L6um_h))
    oline += "\n"
    cato.write(oline)
