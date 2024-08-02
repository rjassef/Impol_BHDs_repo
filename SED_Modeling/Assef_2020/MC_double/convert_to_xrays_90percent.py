#!/usr/bin/env python 

import numpy as np
from scipy.integrate import quad

#####

#Function to get the number of sigmas that encompass 90% of the probability.
def func(x):
    return np.exp(-0.5*x**2)/(2.*np.pi)**0.5

def get_sigfac(p_targ):
    #Bisect to the solution.
    amin = 0.1
    amax = 10.
    for i in range(100):
        a = 0.5*(amin+amax)
        p = quad(func,-a,a)[0]
        if np.abs(p-p_targ)<1e-6:
            break
        if p>p_targ:
            amax = a
        else:
            amin = a
    return a

#####

def get_lL2_10(lL6um,dlL6um,sigfac):
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
    dlL2_10 = (dlL2_10**2+(sigfac*0.37)**2)**0.5

    return lL2_10, dlL2_10

#####

cat = open("90percent_errors.dat")

#We'll assume all behaves 

for line in cat:
    x = line.split()

    x[0] = np.int32(x[0])
    x[2:] = np.float32(x[2:])

    a = get_sigfac(0.9)
    print a

    print 
    print x[1]
    print "NH_low"
    print x[2]/1.5e-23, x[4]/1.5e-23, x[5]/1.5e-23
    print "NH_high"
    print x[6]/1.5e-23, x[8]/1.5e-23, x[9]/1.5e-23

    lL6um_l  = x[10]
    dlL6um_l = 0.5*(x[12]+x[13])
    print "L2_10_low"
    print get_lL2_10(lL6um_l,dlL6um_l,a)

    lL6um_h  = x[14]
    dlL6um_h = 0.5*(x[16]+x[17])
    print "L2_10_high"
    print get_lL2_10(lL6um_h,dlL6um_h,a)

cat.close()

