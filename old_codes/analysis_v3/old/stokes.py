import numpy as np
from astropy.io import fits
from astropy.table import Table

def QU_background(x, y, STK_Par, chip, filter="R-band"):

    #Read the corefficients table.
    coeffs = Table.read("tables/{0:s}_Correction_Coefficients.dat".format(STK_Par), format='ascii')
    a  = coeffs["a_{0:s}".format(STK_Par) ][coeffs['Filter']==filter] * 1e3
    b  = coeffs["b_{0:s}".format(STK_Par) ][coeffs['Filter']==filter] * 1e3
    th = coeffs["th_{0:s}".format(STK_Par)][coeffs['Filter']==filter] * np.pi/180.
    x0 = coeffs["x0_{0:s}".format(STK_Par)][coeffs['Filter']==filter] + 1226
    y0 = coeffs["y0_{0:s}".format(STK_Par)][coeffs['Filter']==filter] + 1025

    #Since we are only using the top image, we need to add 1024 to all the y values used in order to use this correction. No need to alter x.
    x_use = x
    if chip=="1":
        y_use = y + 1024
    elif chip=="2":
        y_use = y
    else:
        print("Chip ",chip,"is not valid. Asuming chip 1.")
        y_use = y + 1024

    #Estimate the correction.
    yp = (x_use-x0)*np.sin(th) + (y_use-y0)*np.cos(th)
    xp = (x_use-x0)*np.cos(th) - (y_use-y0)*np.sin(th)
    return (yp/b)**2 - (xp/a)**2

def get_QU_with_errors(esum, eerr, osum, oerr, e_pos_ref, data_dir, chip="1", return_QU=False):

    #print("Estimating the Stokes parameters...")

    #First, add all the observed e-beam and o-beam fluxes. This is what was recommended by A. Cikotta.
    f_e = dict()
    f_o = dict()
    ef_e2 = dict()
    ef_o2 = dict()
    for fname in esum.keys():
        h = fits.open("{0:s}/{1:s}".format(data_dir, fname))
        th = "{0:.1f}".format(h[0].header['HIERARCH ESO INS RETA2 ROT'])
        h.close()
        if th in f_e.keys():
            f_e[th] += esum[fname]
            f_o[th] += osum[fname]
            ef_e2[th] += eerr[fname]**2
            ef_o2[th] += oerr[fname]**2
        else:
            f_e[th] = esum[fname]
            f_o[th] = osum[fname]
            ef_e2[th] = eerr[fname]**2
            ef_o2[th] = oerr[fname]**2

    #Now, calculate for every angle the normalized flux difference and its error through an MC.
    F = dict()
    F_resamp = dict()
    n_resamp = 1000
    for th in f_e.keys():
        F[th] = (f_o[th]-f_e[th])/(f_o[th]+f_e[th])
        f_e_resamp = np.random.normal(f_e[th], ef_e2[th]**0.5, (n_resamp, len(f_e[th])))
        f_o_resamp = np.random.normal(f_o[th], ef_o2[th]**0.5, (n_resamp, len(f_o[th])))
        F_resamp[th] = (f_o_resamp-f_e_resamp)/(f_o_resamp+f_e_resamp)

    #Finally, estimate the Stokes Q and U parameters.
    Q = 0.
    U = 0.
    Q_resamp = 0.
    U_resamp = 0.
    N = len(F)
    for th in F.keys():
        th_rad = float(th)*np.pi/180.
        Q += (2./N) * F[th] * np.cos(4.*th_rad)
        U += (2./N) * F[th] * np.sin(4.*th_rad)
        Q_resamp += (2./N) * F_resamp[th] * np.cos(4.*th_rad)
        U_resamp += (2./N) * F_resamp[th] * np.sin(4.*th_rad)

    #Subtract the background. We approximate the position to that of the reference image.
    ex = e_pos_ref[:,0]
    ey = e_pos_ref[:,1]
    Q_back = QU_background(ex, ey, "Q", chip)
    U_back = QU_background(ex, ey, "U", chip)

    Q -= Q_back
    U -= U_back
    #Q -= U_back
    #U -= Q_back
    pol_frac = (Q**2+U**2)**0.5
    pol_angle = 0.5*np.arctan2(U,Q)*180./np.pi

    #Now, we need to correct for the zero polarization angle, using Table 4.7 from the FORS2 manual. For R-band, the correction is -1.19 degrees.
    pol_angle += 1.19

    pol_angle = np.where(pol_angle<0, 180.+pol_angle, pol_angle)

    Q_resamp -= Q_back
    U_resamp -= U_back
    epol_frac = np.std((Q_resamp**2+U_resamp**2)**0.5, axis=0)
    epol_angle = np.std(0.5*np.arctan2(U_resamp,Q_resamp)*180./np.pi, axis=0)

    if return_QU:
        return pol_frac, pol_angle, epol_frac, epol_angle, Q, U
    else:
        return pol_frac, pol_angle, epol_frac, epol_angle
