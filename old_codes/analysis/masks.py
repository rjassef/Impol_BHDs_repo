import numpy as np
from astropy.io import fits
import re
import subprocess

def read_masks(fname, mask_folder, rim_folder, chip="1", force=False):

    #Mask names
    mname  = re.sub(".fits",".mask.fits",fname)
    omname = re.sub(".fits",".omask.fits",fname)
    emname = re.sub(".fits",".emask.fits",fname)

    #Open each
    try:
        if force:
            raise FileNotFoundError
        mask  = read_single_mask("{0:s}/{1:s}".format(mask_folder,mname))
        omask = read_single_mask("{0:s}/{1:s}".format(mask_folder,omname))
        emask = read_single_mask("{0:s}/{1:s}".format(mask_folder,emname))
    except FileNotFoundError:
        print("Creating masks for ",fname)
        mask, omask, emask = create_mask(fname, mask_folder, rim_folder, chip)

    return mask, omask, emask

def read_single_mask(mname):
    m_hdu = fits.open(mname)
    mask = np.zeros(m_hdu[0].data.shape, dtype=bool)
    mask[m_hdu[0].data==1] = True
    return mask

#Star masks.
def bright_star_mask(ex, ey, dx, dy, mask):

    #e-beam mask.
    mask[ey-int(dy/2):ey+int(dy/2), ex-int(dx/2):ex+int(dx/2)] = True

    #o-beam mask.
    oy = ey + 90
    ox = ex
    mask[oy-int(dy/2):oy+int(dy/2), ox-int(dx/2):ox+int(dx/2)] = True

    #e-beam gost mask.
    egy = ey - 90
    egx = ex
    mask[egy-int(dy/2):egy+int(dy/2), egx-int(dx/2):egx+int(dx/2)] = True

    #o-beam gost mask.
    ogy = oy + 90
    ogx = ox
    mask[ogy-int(dy/2):ogy+int(dy/2), ogx-int(dx/2):ogx+int(dx/2)] = True

    return mask

#Mask creation.
def create_mask(fname, mask_folder, rim_folder, chip):

    #Mask names
    mname  = re.sub(".fits",".mask.fits",fname)
    omname = re.sub(".fits",".omask.fits",fname)
    emname = re.sub(".fits",".emask.fits",fname)

    #open the file
    h = fits.open("{0:s}/{1:s}".format(rim_folder, fname))

    #Create a mask with the same shape.
    mask = np.zeros(h[0].data.shape, dtype=bool)

    #Mask the edges.
    mask[:, :187] = True
    mask[:,1858:] = True
    if chip=="1":
        mask[ 940:,:] = True
    elif chip=="2":
        mask[: 345,:] = True

    #If the image is of W0116, mask some annying bright, saturated stars.
    if h[0].header['HIERARCH ESO OBS TARG NAME']=="W0116-0505":
        if chip=="1":
            #Mask the three saturated stars.
            #Star on the center.
            mask = bright_star_mask(1232, 408, 50, 40, mask)
            #Star on the NE
            mask = bright_star_mask( 418, 600, 50, 40, mask)
            #Star on the center top.
            mask = bright_star_mask(890, 640, 30, 30, mask)

        elif chip=="2":
            mask = bright_star_mask( 260, 739,  40,  40, mask)
            mask = bright_star_mask( 410, 403,  40,  40, mask)
            mask = bright_star_mask( 573, 344,  20,  20, mask)
            mask = bright_star_mask(1051, 344,  40,  40, mask)
            mask = bright_star_mask(1823, 350,  60,  60, mask)

    #Finally, mask the gaps.
    median = np.median(h[0].data)
    mask[h[0].data<0.5*median] = True
    mname = re.sub(".fits",".mask.fits",fname)
    fits.writeto("{0:s}/{1:s}".format(mask_folder,mname), np.where(mask,1,0), overwrite=True)

    try:
        #Now, separate the mask into the o-beam and e-beam masks.
        dmask = np.zeros(mask.shape)
        imask = np.where(mask, 0, 1)
        dmask[:-1,:] = imask[:-1,:] - imask[1:,:]
        dp1 = np.argwhere(dmask[:, 215]==+1).flatten()
        dp2 = np.argwhere(dmask[:,1830]==+1).flatten()
        dm1 = np.argwhere(dmask[:, 215]==-1).flatten()
        dm2 = np.argwhere(dmask[:,1830]==-1).flatten()
        if chip=="2":
            if len(dp1)<len(dp2):
                dp1 = np.concatenate([[430],dp1])
            if len(dm1)<len(dm2):
                dm1 = np.concatenate([[430],dm1])
                dm1.sort()
        if chip=="1":
            dm1 = np.concatenate([[0],dm1])
            dm2 = np.concatenate([[0],dm2])
        elif chip=="2":
            dp1 = np.concatenate([dp1,[1023]])
            dp2 = np.concatenate([dp2,[1023]])
        x = np.arange(dmask.shape[1],dtype=np.int32)
        x = np.tile(x,(len(dp1),1))
        dp1 = np.tile(dp1,(dmask.shape[1],1)).T
        dp2 = np.tile(dp2,(dmask.shape[1],1)).T
        dm1 = np.tile(dm1,(dmask.shape[1],1)).T
        dm2 = np.tile(dm2,(dmask.shape[1],1)).T
        omask = np.copy(mask)
        emask = np.copy(mask)
        # print(dp1.shape, dp2.shape, x.shape)
        # print(dm1.shape, dm2.shape, x.shape)
        # print(imask.shape, dmask.shape)
        # input()
        jmax = (dp2-dp1)/(1830.-215.) * (x - 215) + dp1
        jmin = (dm2-dm1)/(1830.-215.) * (x - 215) + dm1
        jmax = np.int32(np.round(np.max(jmax,axis=1),0))
        jmin = np.int32(np.round(np.min(jmin,axis=1),0))
        for k in range(len(dp1)):
            if chip=="1":
                if k & 1:
                    omask[jmin[k]:jmax[k]] = True
                else:
                    emask[jmin[k]:jmax[k]] = True
            elif chip=="2":
                if k & 1:
                    emask[jmin[k]:jmax[k]] = True
                else:
                    omask[jmin[k]:jmax[k]] = True

        fits.writeto("{0:s}/{1:s}".format(mask_folder,omname),  np.where(omask,1,0),overwrite=True)
        fits.writeto("{0:s}/{1:s}".format(mask_folder,emname),  np.where(emask,1,0),overwrite=True)

    except ValueError as err:
        print(err)
        print("Could not make beam masks for file",fname)
        #print(dp1, dp2)
        #print(dm1, dm2)
        #input()
        #If the step above fails, which can happen for STD stars, just use the overall mask.
        emask = np.copy(mask)
        omask = np.copy(mask)
        fits.writeto("{0:s}/{1:s}".format(mask_folder,omname),  np.where(omask,1,0),overwrite=True)
        fits.writeto("{0:s}/{1:s}".format(mask_folder,emname),  np.where(emask,1,0),overwrite=True)

    h.close()

    return mask, omask, emask
