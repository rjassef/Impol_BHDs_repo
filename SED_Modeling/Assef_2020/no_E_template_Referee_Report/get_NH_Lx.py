#!/usr/bin/env python 

import numpy as np

####

def get_lL2_10(lL6um):
    #log L(2-10keV) = 41.034 + 0.996x - 0.044x2, x=log L6um/1e41 erg/s
    a = -0.044
    b = 0.996
    c = 41.034
    x = lL6um-41.
    lL2_10 = a*x*x + b*x + c
    return lL2_10


def get_NH(ebv):
    return ebv/1.5e-23

####


cat = open("kc_components.double.dat")
cato = open("NH_Lx.dat","w")
for line in cat:

    x = [float(ix) for ix in line.split()]
    x[0] = int(x[0])

    ebv1 = x[1]
    ebv2 = x[2]
    if x[3]>0:
        lL6um_1  = np.log10(x[3]) + np.log10(3.839) + 33 + 14
        lL2_10_1 = get_lL2_10(lL6um_1)
    else:
        lL6um_1  = -1.
        lL2_10_1 = -1.
    lL6um_2 = np.log10(x[4]) + np.log10(3.839) + 33 + 14
    lL2_10_2 = get_lL2_10(lL6um_2)
    cato.write("{0:5d} {1:8.2e} {2:8.2e} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.2f} {8:6.2f}\n".format(
            x[0],get_NH(ebv1),get_NH(ebv2),lL2_10_1,lL2_10_2,
               ebv1,ebv2,lL6um_1,lL6um_2))

cat.close()
cato.close()
