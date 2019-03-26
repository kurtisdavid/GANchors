from skimage.segmentation import quickshift,mark_boundaries, slic, felzenszwalb
import skimage
import matplotlib.pyplot as plt
import numpy as np

def transform_img_fast(path):
    img = skimage.io.imread(path)
    if len(img.shape) != 3:
        img = skimage.color.gray2rgb(img)
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
    return (skimage.transform.resize(crop_img, (256, 256)) - 0.5) * 2

def ShowImageNoAxis(image, boundaries=None, save=None):
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if boundaries is not None:
        ax.imshow(mark_boundaries(image / 2 + 0.5, boundaries))
    else:
        ax.imshow(image / 2 + .5)
    if save is not None:
        plt.savefig(save)
    plt.show()

def create_mask(image, segments, exp = {}):
    temp = np.ones(image.shape)
    mask = np.zeros(segments.shape)
    if 'feature' not in exp or len(exp['feature']) == 0:
#        exp['feature'] = [1, 2, 7, 8, 9, 11, 16, 17, 18,19, 20, 24, 25, 28, 29, 30, 34, 36, 39, 40, 52]
        exp['feature'] = list(range(52))
    for f in exp['feature']:
        mask[segments == f] = 1
    # this is so that we have measurement matrix A
    mask_new = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return mask_new, mask
