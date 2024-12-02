import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval

def im_display(fname, data_folder, epos, opos):

    h = fits.open("{0:s}/{1:s}".format(data_folder, fname))
    fig, axs = plt.subplots(1,2)
    for k, pos in enumerate([epos[0],opos[0]]):

        y1 = int(np.round(pos[1]-10,0))
        y2 = int(np.round(pos[1]+10,0))
        x1 = int(np.round(pos[0]-10,0))
        x2 = int(np.round(pos[0]+10,0))

        im = h[0].data[y1:y2,x1:x2]
        norm = ImageNormalize(im, stretch=LinearStretch(), interval=ZScaleInterval())
        axs[k].imshow(im, norm=norm, cmap='gray')

    fig.suptitle(fname)
    plt.show()

    return 
