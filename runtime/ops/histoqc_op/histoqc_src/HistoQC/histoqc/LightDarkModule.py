import logging
import os
import numpy as np
from histoqc.BaseImage import printMaskHelper
from skimage import io, color, img_as_ubyte
from distutils.util import strtobool
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage import exposure

import matplotlib.pyplot as plt



def getIntensityThresholdOtsu(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdOtsu")
    name = params.get("name", "classTask")    
    local = strtobool(params.get("local", "False"))
    radius = float(params.get("radius", 15))
    selem = disk(radius)

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    if local:
        thresh = rank.otsu(img, selem)
    else:
        thresh = threshold_otsu(img)

    map = img < thresh

    s["img_mask_" + name] = map > 0
    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(s["img_mask_" + name]))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_" + name]

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdOtsu:{name} NO tissue remains "
                        f"detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdOtsu:{name} NO tissue remains detectable! "
                             f"Downstream modules likely to be incorrect/fail")

    return


def getIntensityThresholdPercent(s, params):
    name = params.get("name", "classTask")
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdPercent:\t {name}")

    lower_thresh = float(params.get("lower_threshold", "-inf"))
    upper_thresh = float(params.get("upper_threshold", "inf"))

    # Prepare parameter names due to issues #213 and #219
    # set lower standard deviation
    lower_std = float(params.get("lower_var") or params.get("lower_std") or "-inf")
    # set upper standard deviation
    upper_std = float(params.get("upper_var") or params.get("upper_std") or "inf")

    img = s.getImgThumb(s["image_work_size"])
    img_std = img.std(axis=2)

    map_std = np.bitwise_and(img_std > lower_std, img_std < upper_std)

    img = color.rgb2gray(img)
    map = np.bitwise_and(img > lower_thresh, img < upper_thresh)

    map = np.bitwise_and(map, map_std)

    s["img_mask_" + name] = map > 0



    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_" + name]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(prev_mask & ~s["img_mask_" + name]))

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdPercent:{name} NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdPercent:{name} NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return






def removeBrightestPixels(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.removeBrightestPixels")

    # lower_thresh = float(params.get("lower_threshold", -float("inf")))
    # upper_thresh = float(params.get("upper_threshold", float("inf")))
    #
    # lower_var = float(params.get("lower_variance", -float("inf")))
    # upper_var = float(params.get("upper_variance", float("inf")))

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    kmeans = KMeans(n_clusters=3,  n_init=1).fit(img.reshape([-1, 1]))
    brightest_cluster = np.argmax(kmeans.cluster_centers_)
    darkest_point_in_brightest_cluster = (img.reshape([-1, 1])[kmeans.labels_ == brightest_cluster]).min()

    s["img_mask_bright"] = img > darkest_point_in_brightest_cluster



    if strtobool(params.get("invert", "False")):
        s["img_mask_bright"] = ~s["img_mask_bright"]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_bright"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_bright.png", img_as_ubyte(prev_mask & ~s["img_mask_bright"]))

    s.addToPrintList("brightestPixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.removeBrightestPixels NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.removeBrightestPixels NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return



def minimumPixelIntensityNeighborhoodFiltering(s,params):
    logging.info(f"{s['filename']} - \tLightDarkModule.minimumPixelNeighborhoodFiltering")
    disk_size = int(params.get("disk_size", 10000))
    threshold = int(params.get("upper_threshold", 200))

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)
    img = (img * 255).astype(np.uint8)
    selem = disk(disk_size)

    imgfilt = rank.minimum(img, selem)
    s["img_mask_bright"] = imgfilt > threshold


    if strtobool(params.get("invert", "True")):
        s["img_mask_bright"] = ~s["img_mask_bright"]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_bright"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_bright.png", img_as_ubyte(prev_mask & ~s["img_mask_bright"]))

    s.addToPrintList("brightestPixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.minimumPixelNeighborhoodFiltering NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.minimumPixelNeighborhoodFiltering NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return

def saveEqualisedImage(s,params):
    logging.info(f"{s['filename']} - \tLightDarkModule.saveEqualisedImage")

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    out = exposure.equalize_hist((img*255).astype(np.uint8))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_equalized_thumb.png", img_as_ubyte(out))

    return
