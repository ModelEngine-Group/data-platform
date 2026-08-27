import logging
import os
import numpy as np
from histoqc.BaseImage import printMaskHelper, getMaskRegionsStats
from skimage import io, morphology, img_as_ubyte, measure

from scipy import ndimage as ndi

import matplotlib.pyplot as plt  # these 2 are used for debugging
from histoqc.SaveModule import blend2Images #for easier debugging


def removeSmallObjects(s, params):
    logging.info(f"{s['filename']} - \tremoveSmallObjects")
    min_size = int(params.get("min_size", 64))
    img_reduced = morphology.remove_small_objects(s["img_mask_use"], min_size=min_size)
    img_small = np.invert(img_reduced) & s["img_mask_use"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_small_remove.png", img_as_ubyte(img_small))
    s["img_mask_small_filled"] = (img_small * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = img_reduced


    # rps = measure.regionprops(morphology.label(img_small))
    # if rps:
    #     areas = np.asarray([rp.area for rp in rps])
    #     nobj = len(rps)
    #     area_max = areas.max()
    #     area_mean = areas.mean()
    # else:
    #     nobj = area_max = area_mean = 0
    stats = getMaskRegionsStats(img_small)


    # s.addToPrintList("small_tissue_removed_num_regions", str(nobj))
    # s.addToPrintList("small_tissue_removed_mean_area", str(area_mean))
    # s.addToPrintList("small_tissue_removed_max_area", str(area_max))
    s.addToPrintList("small_tissue_removed_num_regions", str(stats.get('num',0)))
    s.addToPrintList("small_tissue_removed_mean_area", str(stats.get('area_mean',0)))
    s.addToPrintList("small_tissue_removed_max_area", str(stats.get('area_max',0)))




    s.addToPrintList("small_tissue_removed_percent",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))


    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After MorphologyModule.removeSmallObjects: NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After MorphologyModule.removeSmallObjects: NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return


def remove_large_objects(img, max_size):
    # code taken from morphology.remove_small_holes, except switched < with >
    selem = ndi.generate_binary_structure(img.ndim, 1)
    ccs = np.zeros_like(img, dtype=np.int32)
    ndi.label(img, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    img_out = img.copy()
    img_out[too_big_mask] = 0
    return img_out


def removeFatlikeTissue(s, params):
    logging.info(f"{s['filename']} - \tremoveFatlikeTissue")
    fat_cell_size = int(params.get("fat_cell_size", 64))
    kernel_size = int(params.get("kernel_size", 3))
    max_keep_size = int(params.get("max_keep_size", 1000))

    img_reduced = morphology.remove_small_holes(s["img_mask_use"], area_threshold=fat_cell_size)
    img_small = img_reduced & np.invert(s["img_mask_use"])
    img_small = ~morphology.remove_small_holes(~img_small, area_threshold=9)

    mask_dilate = morphology.dilation(img_small, selem=np.ones((kernel_size, kernel_size)))
    mask_dilate_removed = remove_large_objects(mask_dilate, max_keep_size)

    mask_fat = mask_dilate & ~mask_dilate_removed

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_fatlike.png", img_as_ubyte(mask_fat))
    s["img_mask_fatlike"] = (mask_fat * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = prev_mask & ~mask_fat

    # rps = measure.regionprops(morphology.label(mask_fat))
    # if rps:
    #     areas = np.asarray([rp.area for rp in rps])
    #     nobj = len(rps)
    #     area_max = areas.max()
    #     area_mean = areas.mean()
    # else:
    #     nobj = area_max = area_mean = 0
    stats = getMaskRegionsStats(mask_fat)

    # s.addToPrintList("fatlike_tissue_removed_num_regions", str(nobj))
    # s.addToPrintList("fatlike_tissue_removed_mean_area", str(area_mean))
    # s.addToPrintList("fatlike_tissue_removed_max_area", str(area_max))
    s.addToPrintList("fatlike_tissue_removed_num_regions", str(stats.get('num',0)))
    s.addToPrintList("fatlike_tissue_removed_mean_area", str(stats.get('area_mean',0)))
    s.addToPrintList("fatlike_tissue_removed_max_area", str(stats.get('area_max',0)))



    s.addToPrintList("fatlike_tissue_removed_percent",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After MorphologyModule.removeFatlikeTissue: NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After MorphologyModule.removeFatlikeTissue: NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")


def fillSmallHoles(s, params):
    logging.info(f"{s['filename']} - \tfillSmallHoles")
    min_size = int(params.get("min_size", 64))
    img_reduced = morphology.remove_small_holes(s["img_mask_use"], area_threshold=min_size)
    img_small = img_reduced & np.invert(s["img_mask_use"])


    io.imsave(s["outdir"] + os.sep + s["filename"] + "_small_fill.png", img_as_ubyte(img_small))
    s["img_mask_small_removed"] = (img_small * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = img_reduced

    # rps = measure.regionprops(morphology.label(img_small))
    # if rps:
    #     areas = np.asarray([rp.area for rp in rps])
    #     nobj = len(rps)
    #     area_max = areas.max()
    #     area_mean = areas.mean()
    # else:
    #     nobj = area_max = area_mean = 0
    stats = getMaskRegionsStats(img_small)

    # s.addToPrintList("small_tissue_filled_num_regions", str(nobj))
    # s.addToPrintList("small_tissue_filled_mean_area", str(area_mean))
    # s.addToPrintList("small_tissue_filled_max_area", str(area_max))
    s.addToPrintList("small_tissue_filled_num_regions", str(stats.get('num', 0)))
    s.addToPrintList("small_tissue_filled_mean_area", str(stats.get('area_mean', 0)))
    s.addToPrintList("small_tissue_filled_max_area", str(stats.get('area_max', 0)))

    s.addToPrintList("small_tissue_filled_percent",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After MorphologyModule.fillSmallHoles: NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After MorphologyModule.fillSmallHoles: NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")
    return
