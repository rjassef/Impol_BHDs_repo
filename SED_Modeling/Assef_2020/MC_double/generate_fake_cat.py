#!/usr/bin/python

import random
import numpy as np

#Set the seed
random.seed(-75)

#Number of photomeric channels.
nchan = 16

#Number of resamples to create.
nsamp = 1000

#Open the relevant files. 
f20 = open("../double.20","r")
f21 = open("../double.21","r")

cato = open("BHDs.ran.phot","w")

for line in f20.readlines():
    
    x = line.split()
    z = float(x[1])
    chi2 = float(x[2])
    id = int(x[0])

    jy    = np.zeros(nchan)
    jymod = np.zeros(nchan)
    ejy   = np.zeros(nchan)
    jyuse = np.zeros(nchan)
    for j in range(nchan):
        y = f21.readline().split()
        jy[j]    = float(y[1])
        jymod[j] = float(y[2])
        ejy[j]   = float(y[3])
        jyuse[j] = int(y[4])

    #An issue is that the chi-squared values are somewhat large. So,
    #we should increase the errorbars to make the chi-squared equal to
    #the number of degrees of freedom.
    ndof = len(jyuse[jyuse>0])-7
    ejy *= (chi2/float(ndof))**0.5

    for i in range(nsamp):
        
        njy = []
        enjy = []
        njyuse = []
        for j in range(len(jy)):
            if int(jyuse[j])==1:
                y = random.gauss(float(jy[j]),float(ejy[j]))
                if y<=0:
                    njyuse.append(2)
                    njy.append(0.)
                    enjy.append(float(ejy[j]))
                else:
                    njyuse.append(1)
                    njy.append(y)
                    enjy.append(float(ejy[j])*np.sqrt(float(jy[j])/njy[j]))
            elif int(jyuse[j])==2:
                enjy.append(float(ejy[j]))
                njy.append(0.)
                njyuse.append(2)
            else:
                njy.append(0.)
                enjy.append(0.)
                njyuse.append(0.)
            
        value = "%10d %20.6e" % (id*1000+i,z) 
        for j in range(len(jy)):
            value += "%20.6e" % (njy[j])
        for j in range(len(jy)):
            value += "%20.6e" % (enjy[j])
        for j in range(len(jy)):
            value += "%10d" % (int(njyuse[j]))
        value += "\n"

        cato.write(str(value))
