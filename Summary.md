Main results
------------

* All new objects show very significant polarization fractions in the $R_{\rm Special}$ band

    * W0019-1046:  6.4 $\pm$ 0.6%
    * W0204-0505: 25.3 $\pm$ 0.6%
    * W0220+0137: 14.0 $\pm$ 0.3%
    * W0831+0140:  6.9 $\pm$ 0.4%

* W0116-0505 shows highly significant polarization in $V$ and $I$ as well. 

    * V:  9.7 $\pm$ 0.3%
    * R: 10.9 $\pm$ 0.2%
    * I: 14.7 $\pm$ 0.4%


* We have extended the models from Assef et al. (2022) such that we can now consider the variations on polarization as a function of wavelength. 

* We find that both gas and dust models can explain the polarization in all three bands of W0116-0505, however with the strict requirement of a significant difference in the polarization angle of the emission lines and continuum, high inclination and small opening angles. Models with forward and backward scattering can explain the data, although the LMC dust mixture does not seem to work well. 

* W0204-0506 shows a higher polarization fraction than possible with most dust mixtures. However gas is unlikely as the UV is very extended in HST. The only dust model that approaches 25% polarization in the R-band at z=2.1 is LMC dust with only backscattering. 

* Use a smaller aperture, except for W0204-0506, for the spatially integrated measurements of the polarization. 


Still To Do
-----------

* Add errors to the SED parameters table.

* The SED fits should include the HST photometry for the three objects that have it (can just copy from Assef et al. 2020 fits). For fraction of light reflected, it does make some difference at least for W0116. 

* Make NH plots. Need to consider bluening + reddening of the input spectrum for I. Could affect the estimates of p as well. 

* Build an outflow reflection model for the scattering through the same formalism of the torus scattering model. 

* Check if some bad fits to W0116 are simply lack of convergence rather than no model possible. Actually, do an MCMC fit to ensure we are properly mapping the parameter space.  

* W0204-0506 spatially resolved polarization map. Is it resolved?

* Understand what causes the backscattering more in detail to understand if small changes in the dust properties could increase the expected polarization fraction. 

* Recentering of sources seem to be somewhat uncertain. Changed the box size from 21 to 11 and enabled the background subtraction, and that seems to potentially help, but need to determine in detail how well the re-centering is working. 

Solved
------

* There seems to be an issue with the first image of the second set of observations of W0220+0137. The source seems to be almost gone, likely a cosmic ray rejection issue. Check all other sources/images.

    * 28/05/2024: One image of W0116 in I was also affected. Changed objlim in astroscrappy.detec_cosmics to 10 and that solved the issues. 
