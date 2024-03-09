import numpy as np

#Gonzalez-Gaitan et al. points out that there is angle of 0.083 degrees between the chips, but we will just omit that.
def join_chips(im1, im2, is_mask=False):

    #First, make an array that has, in the y axis, the combined dimensions of both + the gap size of 32 pixels.
    nygap = 32
    ny1, nx1 = im1[0].data.shape
    ny2, nx2 = im2[0].data.shape

    #Copy the chips to the correct regions of the complete image.
    out_im = np.zeros((ny1+ny2+nygap, nx1))
    out_im[         :ny2,:] = im2[0].data
    out_im[ny2+nygap:   ,:] = im1[0].data

    #If it is a mask, mask out the gap.
    if is_mask:
        out_im[ny2:ny2+nygap] = 1

    return out_im
