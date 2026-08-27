import logging
import os
import sys

from ast import literal_eval as make_tuple

from distutils.util import strtobool
from histoqc.BaseImage import printMaskHelper

import scipy.signal

from skimage import io, img_as_ubyte
from skimage.filters import gabor_kernel, frangi, gaussian, median, laplace
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, disk, binary_opening
from skimage.feature import local_binary_pattern

from skimage.transform import rescale, resize, downscale_local_mean

from math import ceil

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from skimage import io, color


import numpy as np

import matplotlib.pyplot as plt

global_holder = {}

#WARNING: Not as robust as other modules
def roiWise(s, params):
    name = params.get("name", "classTask")
    print("\tpixelWise:\t", name, end="")

    level = int(params.get("level", 1))
    win_size = int(params.get("win_size", 2048)) #the size of the ROI which will be iteratively considered

    osh = s["os_handle"]

    dim_base = osh.level_dimensions[0]
    dims = osh.level_dimensions[level]

    ratio_x = dim_base[0] / dims[0] #figure out the difference between desi
    ratio_y = dim_base[1] / dims[1]

    frangi_scale_range = (1, 6)
    frangi_scale_step = 2
    frangi_beta1 = .5
    frangi_beta2 = 100
    frangi_black_ridges = True

    mask = []
    for x in range(0, dim_base[0], round(win_size * ratio_x)):
        row_piece = []
        print('.', end='', flush=True)
        for y in range(0, dim_base[1], round(win_size * ratio_y)):
            region = np.asarray(osh.read_region((x, y), 1, (win_size, win_size)))
            region = region[:, :, 0:3]  # remove alpha channel
            g = rgb2gray(region)
            feat = frangi(g, frangi_scale_range, frangi_scale_step, frangi_beta1, frangi_beta2, frangi_black_ridges)
            feat = feat / 8.875854409275627e-08
            region_mask = np.bitwise_and(g < .3, feat > 5)
            region_mask = remove_small_objects(region_mask, min_size=100, in_place=True)
            # region_std = region.std(axis=2)
            # region_gray = rgb2gray(region)
            # region_mask = np.bitwise_and(region_std < 20, region_gray < 100/255)
            # region_mask = scipy.ndimage.morphology.binary_dilation(region_mask, iterations=1)
            # region_mask = resize(region_mask , (region_mask.shape[0] / 2, region_mask.shape[1] / 2))
            row_piece.append(region_mask)
        row_piece = np.concatenate(row_piece, axis=0)
        mask.append(row_piece)

    mask = np.concatenate(mask, axis=1)

    if params.get("area_threshold", "") != "":
        mask = remove_small_objects(mask, min_size=int(params.get("area_threshold", "")), in_place=True)

    s.addToPrintList(name, str(mask.mean()))

    #TODO, migrate to printMaskHelper, but currently don't see how this output affects final mask
    #s.addToPrintList(name,
    #                 printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_BubbleBounds.png", img_as_ubyte(mask)) #.astype(np.uint8) * 255)

    return


def detectSmoothness(s, params):
        logging.info(f"{s['filename']} - \tBubbleRegionByRegion.detectSmoothness")
        thresh = float(params.get("threshold", ".01" ))
        kernel_size = int(params.get("kernel_size", "10"))
        min_object_size = int(params.get("min_object_size", "100"))

        img = s.getImgThumb(s["image_work_size"])
        img = color.rgb2gray(img)
        avg = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        imf = scipy.signal.convolve2d(img, avg, mode="same")
        mask_flat = abs(imf - img) < thresh

        mask_flat = remove_small_objects(mask_flat, min_size=min_object_size)
        mask_flat = ~remove_small_objects(~mask_flat, min_size=min_object_size)

        prev_mask = s["img_mask_use"]
        s["img_mask_flat"] = mask_flat

        io.imsave(s["outdir"] + os.sep + s["filename"] + "_flat.png", img_as_ubyte(mask_flat & prev_mask))

        s["img_mask_use"] = s["img_mask_use"] & ~s["img_mask_flat"]


        s.addToPrintList("flat_areas",
                         printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask,
                                         s["img_mask_use"]))

        if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
            logging.warning(f"{s['filename']} - After BubbleRegionByRegion.detectSmoothness: NO tissue "
                            f"remains detectable! Downstream modules likely to be incorrect/fail")
            s["warnings"].append(f"After BubbleRegionByRegion.detectSmoothness: NO tissue remains "
                                 f"detectable! Downstream modules likely to be incorrect/fail")

        return

