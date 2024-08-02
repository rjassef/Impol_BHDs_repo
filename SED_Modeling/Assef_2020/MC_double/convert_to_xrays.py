#!/usr/bin/env python 

import numpy as np

#####

def get_lL2_10(lL6um,dlL6um):
    #log L(2-10keV) = 41.034 + 0.996x - 0.044x2, x=log L6um/1e41 erg/s
    a = -0.044
    b = 0.996
    c = 41.034
    x = lL6um-41.
    dx= dlL6um
    lL2_10 = a*x**2 + b*x + c
    dlL2_10 = (2.*a*x + b)*dx
    #dlL2_10 = 0.5*((a*(x+dx)**2+b*(x+dx)+c) - (a*(x-dx)**2+b*(x-dx)+c))

    #Add the scatter of the relation.
    dlL2_10 = (dlL2_10**2+0.37**2)**0.5

    return lL2_10, dlL2_10

#####

cat = open("1_sigma_errors.dat")

for line in cat:
    x = line.split()

    x[0] = np.int32(x[0])
    x[2:] = np.float32(x[2:])

    print 
    print x[1]
    print "NH_low"
    print x[2]/1.5e-23, x[4]/1.5e-23, x[5]/1.5e-23
    print "NH_high"
    print x[6]/1.5e-23, x[8]/1.5e-23, x[9]/1.5e-23

    lL6um_l  = x[10]
    dlL6um_l = 0.5*(x[12]+x[13])
    print "L2_10_low"
    print get_lL2_10(lL6um_l,dlL6um_l)

    lL6um_h  = x[14]
    dlL6um_h = 0.5*(x[16]+x[17])
    print "L2_10_high"
    print get_lL2_10(lL6um_h,dlL6um_h)

cat.close()

