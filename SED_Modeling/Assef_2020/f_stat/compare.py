#!/usr/bin/python 

import subprocess
from scipy.special import betainc

cat1 = open("../single.20")
cat2 = open("../double.20")

phot1 = open("../single.21")
phot2 = open("../double.21")

f22   = open("../single.22")

nchan = 16

cato = open("comp.dat","w")
for line1 in cat1.readlines():

    x1 = [float(ix) for ix in line1.split()]
    x2 = [float(ix) for ix in cat2.readline().split()]

    name = f22.readline().split()[4]

    m1 = 0
    m2 = 0
    for j in range(nchan):
        if float(phot1.readline().split()[4])>0:
            m1+=1
        if float(phot2.readline().split()[4])>0:
            m2+=1

    if x1[1]<=1.:
        continue

    if m2<=7:
        continue

    if x2[2]>x1[2]:
        F1 = -1.
        F2 = -1.
        p1 = -1.
        p2 = -1.
    else:
        nu1 = m1 - 5
        nu2 = m2 - 7
        #Maybe we should consider the IGM strength too. It is a
#        parameter, although it is something tha affects, at most, one
#        of the bands. Not sure this is worth taking into account...
#        nu1 = m1 - 6 
#        nu2 = m2 - 8
        if nu1<=0:
            F1 = -1.
            F2 = -1.
            p1 = -1.
            p2 = -1.
        else:
            F1 = ((x1[2]-x2[2])/float(nu1-nu2)) / (x1[2]/(1.*nu1))
            nnu1 = float(nu1-nu2)
            nnu2 = float(nu1)
            w = nnu1*F1/(nnu1*F1+nnu2)
            p1 = 1.-betainc(nnu1/2.,nnu2/2.,w)

            if nu2>0:
                F2 = ((x1[2]-x2[2])/float(nu1-nu2)) / (x2[2]/(1.*nu2))
                nnu1 = float(nu1-nu2)
                nnu2 = float(nu2)
            #w = nnu2/(nnu2+nnu1*F2)
                w = nnu1*F2/(nnu1*F2+nnu2)
                p2 = 1.-betainc(nnu1/2.,nnu2/2.,w)
            else:
                F2 = -1.
                p2 = -1.
    
    oline = "%s %10d %12.3f %20.6e %20.6e %10d %10d %20.6e %20.6e %20.6e %20.6e\n" % \
        (name,x1[0],x1[1],x1[2],x2[2],m1,m2,F1,F2,p1,p2)
    cato.write(oline)

cat1.close()
cat2.close()
cato.close()

subprocess.call("sort comp.dat -o comp.sort -k 11 -g -r",shell=True)

