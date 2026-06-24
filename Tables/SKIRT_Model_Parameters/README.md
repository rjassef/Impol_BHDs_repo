# SKIRT Simulation Parameters

## Common Parameters

Wavelength range: 1200-3000 A in steps of 150 A, with additional points close to the effective wavelengths of UBVRI for comparison with polarization properties of low redsshift AGNs. 

Optical depth: $\tau = 0.1$ at 5500 A (about V-band)

From Assef et al. (2025):

"We assume a geometry where the scattering medium corresponds to a bi-conical polar outflow wind of half-opening angle
$\psi_{\rm Cone}$ where dust is present in the outer 10 deg of the cone, on top of an accretion disk surrounded by an optically thick dust torus of half-opening angle $\psi_{\rm Torus}$. The torus and the polar outflow cones have the same polar axis. The emission of the accretion disk is given by the model of Stalevski et al. (2016), and the outflow cone starts at its sublimation radius."

## Initial Mixtures

All have: 

$\psi_{\rm Cone} = 30~\rm deg$

$\psi_{\rm Torus} = 50~\rm deg$

$\eta = 55-85~\rm deg$ in steps of 10 deg

### 1. $MRN77_{\rm gra+sil}$

* Standard MRN77 power-law size distribution $\propto a^{-3.5}$. 

* Range of $a = 0.005 - 0.25~\mu\rm m$. 

* Silicate to graphite ratio: 51/49, as given by the normalization of the grain size distributions in
Weingartner & Draine (2001).

### 2. $MRN77_{\rm gra}$

Same as $MRN77_{\rm gra+sil}$ but only with graphites. 

### 3. Large grains $_{\rm gra+sil}$

Same as $MRN77_{\rm gra+sil}$ but with ranges of $a = 0.1 - 1.0~\mu\rm m$ and $a = 1.0 - 10.0~\mu\rm m$.

### 4. Draine (2003)

These are the mixtures from [Draine (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1017D/abstract). The Draine (2003) mixtures have a very similar slope to that of MRN77, but with an extended grain size range and a smooth high-end cutoff.

### 5. Gaskell et al. (2004) $_{\rm gra+sil}$

This is the mixture of [Gaskell et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJ...616..147G/abstract). The dust distribution of Gaskell et al. (2004) has a flatter grain size distribution dominated by larger grains as compared to that of MRN77.

## Refined MRN77 $_{\rm gra}$ models

$\psi_{\rm Torus} = 25~\rm deg - 60~\rm deg$ in steps of 5 deg

$\psi_{\rm Cone} = 20~\rm deg - \psi_{\rm Torus}$ in steps of 5 deg

$\eta = \psi_{\rm Torus} - 90~\rm deg$ in steps of 5 deg






