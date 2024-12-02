import numpy as np
import re
from pathlib import Path
from copy import deepcopy

from astropy.io import fits


class PolMasks(object):

    def __init__(self, pdata):

        #Save input. 
        self.pdata = pdata

        #If we are not using masks, skip this part. 
        if not pdata.use_masks:
            return

        #Start by checking whether the masks exist for each file. If they do not, create them.
        for i, fname in enumerate(pdata.list_of_filenames()):

            #Mask names.
            mname, omname, emname = self.get_mask_names(fname)

            #If they do not exist or if forcing to create new ones, create them.
            missing_mask = False
            for mm in [mname, omname, emname]:
                if not Path("{}/{}".format(pdata.mask_folder,mm)).exists():
                    missing_mask = True
            if pdata.force_new or missing_mask:
                fname_for_mask = None
                if self.pdata.filenames_for_mask_creation is not None:
                    fname_for_mask = self.pdata.filenames_for_mask_creation[i]
                self.create_mask(fname, fname_for_mask)

        return

    #Get the mask file names.
    def get_mask_names(self, fname):
        mname  = re.sub(".fits",".mask.fits",fname)
        omname = re.sub(".fits",".omask.fits",fname)
        emname = re.sub(".fits",".emask.fits",fname)
        return mname, omname, emname

    #Mask creation.
    def create_mask(self, fname, fname_for_mask):

        #Get the chip number from the filename. 
        m = re.search("chip(.)", fname)
        chip = m.group(1)

        #Mask names
        mname, omname, emname = self.get_mask_names(fname)

        #open the file
        if fname_for_mask is None:
            h = fits.open("{0:s}/{1:s}".format(self.pdata.rim_folder, fname))
        else:
            h = fits.open("{0:s}/{1:s}".format(self.pdata.rim_folder, fname_for_mask))

        #Create a mask with the same shape.
        mask = np.zeros(h[0].data.shape, dtype=bool)

        #Mask the edges.
        mask[:, :187] = True
        mask[:,1858:] = True
        if chip=="1":
            mask[ 940:,:] = True
        elif chip=="2":
            mask[: 345,:] = True

        #Check if there are any bright stars we should be masking. 
        bsm_name = "{}/{}.{}.{}.txt".format(self.pdata.bstar_folder, self.pdata.obj_id, self.pdata.bband, chip)
        if Path(bsm_name).exists():
            bsm = np.loadtxt(bsm_name, dtype=int)
            mask = self.bright_star_mask(bsm, mask)

        #Finally, mask the gaps.
        median = np.median(h[0].data)
        mask[h[0].data<0.5*median] = True
        mname = re.sub(".fits",".mask.fits",fname)
        fits.writeto("{0:s}/{1:s}".format(self.pdata.mask_folder,mname), np.where(mask,1,0), overwrite=True)

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

            fits.writeto("{0:s}/{1:s}".format(self.pdata.mask_folder,omname),  np.where(omask,1,0),overwrite=True)
            fits.writeto("{0:s}/{1:s}".format(self.pdata.mask_folder,emname),  np.where(emask,1,0),overwrite=True)

        except ValueError as err:
            print(err)
            print("Could not make beam masks for file",fname)
            #If the step above fails, which can happen for STD stars, just use the overall mask.
            emask = np.copy(mask)
            omask = np.copy(mask)
            fits.writeto("{0:s}/{1:s}".format(self.pdata.mask_folder,omname),  np.where(omask,1,0),overwrite=True)
            fits.writeto("{0:s}/{1:s}".format(self.pdata.mask_folder,emname),  np.where(emask,1,0),overwrite=True)

        h.close()

        return 

    #Star masks.
    def bright_star_mask(self, bsm, mask):

        for bsm_i in bsm:
            ex, ey, dx, dy = bsm_i  
            
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



    def read_masks(self, fname):

        if not self.pdata.use_masks:
            h = fits.open("{}/{}".format(self.rim_folder, fname))
            mask = np.zeros(h[0].data.shape, dtype=bool)
            omask = np.zeros(h[0].datamask.shape, dtype=bool)
            emask = np.zeros(h[0].datamask.shape, dtype=bool)
            h.close()

        else:
            #Mask names
            mname, omname, emname = self.get_mask_names(fname)

            #Open each
            mask  = self.read_single_mask("{0:s}/{1:s}".format(self.pdata.mask_folder,mname))
            omask = self.read_single_mask("{0:s}/{1:s}".format(self.pdata.mask_folder,omname))
            emask = self.read_single_mask("{0:s}/{1:s}".format(self.pdata.mask_folder,emname))

        return mask, omask, emask

    def read_single_mask(self, mname):
        m_hdu = fits.open(mname)
        mask = np.zeros(m_hdu[0].data.shape, dtype=bool)
        mask[m_hdu[0].data==1] = True
        return mask



