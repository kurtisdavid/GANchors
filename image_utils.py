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
    return (skimage.transform.resize(crop_img, (299, 299))), (skimage.transform.resize(crop_img, (256, 256)))

def ShowImageNoAxis(image, boundaries=None, save=None):
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if boundaries is not None:
        ax.imshow(np.clip(mark_boundaries(image, boundaries),0,1))
    else:
        ax.imshow(np.clip(image, 0, 1))
    if save is not None:
        plt.savefig(save)
    plt.show()

def create_mask(image, segments, exp = {}):
#    temp = np.ones(image.shape)
    mask = np.zeros(segments.shape)
    if 'feature' not in exp or len(exp['feature']) == 0:
        exp['feature'] = [1, 2, 7, 8, 9, 11, 16, 17, 18,19, 20, 24, 25, 28, 29, 30, 34, 36, 39, 40, 52]
    for f in exp['feature']:
        mask[segments == f] = 1
    # this is so that we have measurement matrix A
    mask_new = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return mask_new, mask

def extend_channels(image, nc = 3):
    return np.repeat(image[:, :, np.newaxis], 3, axis=2)

def create_segments(image, kernel_size=2, max_dist=3, ratio=0.2, target_num=15, imagenet=False, compactness=16):
    if not imagenet:
        image = extend_channels(image)
        seg =  skimage.segmentation.quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    else:
        seg = skimage.segmentation.slic(image, n_segments=target_num, compactness=compactness)
        return seg
    md = 3
    while len(np.unique(seg)) > target_num:
        print(len(np.unique(seg)), md)
#        error = len(np.unique(seg))/target_num - 1
        md += .1
        seg =  skimage.segmentation.quickshift(image, kernel_size=kernel_size, max_dist=md, ratio=ratio)

    print("found max_dist of ", md, " created ",target_num," segments")
    return seg

