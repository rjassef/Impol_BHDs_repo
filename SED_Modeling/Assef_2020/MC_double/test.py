#!/usr/bin/env python 

import numpy as np
from scipy.integrate import quad

def func(x):
    return np.exp(-0.5*x**2)/(2.*np.pi)**0.5

#Bisect to the solution.
p_targ = 0.90
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

print a, quad(func,-a,a)[0], p_targ

