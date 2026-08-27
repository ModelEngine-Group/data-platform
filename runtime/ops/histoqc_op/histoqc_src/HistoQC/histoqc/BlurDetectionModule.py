import logging
import os

import skimage
from histoqc.BaseImage import printMaskHelper, getMaskRegionsStats
from skimage import io, img_as_ubyte, morphology, measure
from skimage.color import rgb2gray
from skimage.filters import rank
import numpy as np

import matplotlib.pyplot as plt


# Analysis of focus measure operators for shape-from-focus
# Said Pertuza,, Domenec Puiga, Miguel Angel Garciab, 2012
# https://pdfs.semanticscholar.org/8c67/5bf5b542b98bf81dcf70bd869ab52ab8aae9.pdf


def identifyBlurryRegions(s, params):
    logging.info(f"{s['filename']} - \tidentifyBlurryRegions")

    blur_radius = int(params.get("blur_radius", 7))
    blur_threshold = float(params.get("blur_threshold", .1))

    img = s.getImgThumb(params.get("image_work_size", "2.5x"))
    img = rgb2gray(img)
    img_laplace = np.abs(skimage.filters.laplace(img))
    mask = skimage.filters.gaussian(img_laplace, sigma=blur_radius) <= blur_threshold

    mask = skimage.transform.resize(mask, s.getImgThumb(s["image_work_size"]).shape, order=0)[:, :,
           1]  # for some reason resize takes a grayscale and produces a 3chan
    mask = s["img_mask_use"] & (mask > 0)

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_blurry.png", img_as_ubyte(mask))
    s["img_mask_blurry"] = (mask * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & ~s["img_mask_blurry"]

    # rps = measure.regionprops(morphology.label(mask))
    # if rps:
    #     areas = np.asarray([rp.area for rp in rps])
    #     nobj = len(rps)
    #     area_max = areas.max()
    #     area_mean = areas.mean()
    # else:
    #     nobj = area_max = area_mean = 0
    stat_rs = getMaskRegionsStats(mask)

    
    s.addToPrintList("blurry_removed_num_regions", str(stat_rs.get('num', 0)))
    s.addToPrintList("blurry_removed_mean_area", str(stat_rs.get('area_mean',0)))
    s.addToPrintList("blurry_removed_max_area", str(stat_rs.get('area_max',0)))


    s.addToPrintList("blurry_removed_percent",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(
            f"{s['filename']} - After BlurDetectionModule.identifyBlurryRegions NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(
            f"After BlurDetectionModule.identifyBlurryRegions NO tissue remains detectable! Downstream modules likely to be incorrect/fail")


    return
