#!/usr/bin/python

import numpy as np

###

def get_90p(x,x_best):

    #Start by separating the list into values above and below the best-fit.
    x_low = x[x<x_best]
    x_hig = x[x>x_best]

    #Now, sort them. 
    if x_best>0 and len(x_low)>0:
        x_low.sort()
        xmin = x_best - x_low[np.int32((len(x_low)-1)*(1.-0.9))]
    else: 
        xmin = 0
    if len(x_hig)>0:
        x_hig.sort()
        xmax = x_hig[np.int32((len(x_hig)-1)*0.9)] - x_best
    else:
        xmax = 0

    #Return the 95.4% ranges. 
    return xmin,xmax

###

acat = open("agn_lum_corr.dat")
acorr = float(acat.readline().split()[-1])
acat.close()

#Number of resamples create created for fake cat.
nsamp = 1000

cat  = open("kc_components.dat")
rcat = open("../kc_components.double.dat")
cato = open("90percent_errors.dat","w")
f22  = open("../double.22")
for line in rcat.readlines():

    x = [float(ix) for ix in line.split()]
    name   = f22.readline().split()[4]

    ebv_h  = np.zeros(nsamp)
    L6um_h = np.zeros(nsamp)
    ebv_l  = np.zeros(nsamp)
    L6um_l = np.zeros(nsamp)
    for i in range(nsamp):
        y = [float(iy) for iy in cat.readline().split()]
        ebv_l[i]  = y[1]
        ebv_h[i]  = y[2]
        L6um_l[i] = np.log10(y[3]) + np.log10(3.839) + 33 + 14
        L6um_h[i] = np.log10(y[4]) + np.log10(3.839) + 33 + 14

    ebv_l_best  = x[1]
    ebv_h_best  = x[2]
    L6um_l_best = np.log10(x[3]) + np.log10(3.839) + 33 + 14
    L6um_h_best = np.log10(x[4]) + np.log10(3.839) + 33 + 14

    ebv_l_err_do , ebv_l_err_up  = get_90p(ebv_l ,ebv_l_best )
    ebv_h_err_do , ebv_h_err_up  = get_90p(ebv_h ,ebv_h_best )
    L6um_l_err_do, L6um_l_err_up = get_90p(L6um_l,L6um_l_best)
    L6um_h_err_do, L6um_h_err_up = get_90p(L6um_h,L6um_h_best)


    oline  = "%10d %15s" % (x[0],name)
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        ebv_l_best,np.median(ebv_l),ebv_l_err_do,ebv_l_err_up)
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        ebv_h_best,np.median(ebv_h),ebv_h_err_do,ebv_h_err_up)
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        L6um_l_best,np.median(L6um_l),L6um_l_err_do,L6um_l_err_up)
    oline += "%20.6e %20.6e %20.6e %20.6e" % (
        L6um_h_best,np.median(L6um_h),L6um_h_err_do,L6um_h_err_up)
    oline += "\n"
    cato.write(oline)
