import logging
import os
from histoqc.BaseImage import printMaskHelper, getMaskRegionsStats
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage import io, color, img_as_ubyte

import matplotlib.pyplot as plt


def getBasicStats(s, params):
    logging.info(f"{s['filename']} - \tgetBasicStats")
    osh = s["os_handle"]
    s.addToPrintList("type", osh.properties.get("openslide.vendor", "NA"))
    s.addToPrintList("levels", osh.properties.get("openslide.level-count", "NA"))
    s.addToPrintList("height", osh.properties.get("openslide.level[0].height", "NA"))
    s.addToPrintList("width", osh.properties.get("openslide.level[0].width", "NA"))
    s.addToPrintList("mpp_x", osh.properties.get("openslide.mpp-x", "NA"))
    s.addToPrintList("mpp_y", osh.properties.get("openslide.mpp-y", "NA"))
    s.addToPrintList("comment", osh.properties.get("openslide.comment", "NA").replace("\n", " ").replace("\r", " "))
    return


def finalComputations(s, params):
    mask = s["img_mask_use"]
    mask_statistics = params.get("mask_statistics", s["mask_statistics"])
    if mask_statistics == "relative2mask":
        warning_message = "relative2mask is not supported in BasicModule.finalComputations. Using absolute instead."
        logging.warning(f"{s['filename']} - {warning_message}")
        s["warnings"].append(warning_message)
        mask_statistics = "absolute"
    s.addToPrintList("pixels_to_use", printMaskHelper(mask_statistics, mask, mask))


def finalProcessingSpur(s, params):
    logging.info(f"{s['filename']} - \tfinalProcessingSpur")
    disk_radius = int(params.get("disk_radius", "25"))
    selem = disk(disk_radius)
    mask = s["img_mask_use"]
    mask_opened = binary_opening(mask, selem)
    mask_spur = ~mask_opened & mask

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_spur.png", img_as_ubyte(mask_spur))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = mask_opened

    s.addToPrintList("spur_pixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(
            f"{s['filename']} - After BasicModule.finalProcessingSpur NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(
            f"After BasicModule.finalProcessingSpur NO tissue remains detectable! Downstream modules likely to be incorrect/fail")


def finalProcessingArea(s, params):
    logging.info(f"{s['filename']} - \tfinalProcessingArea")
    area_thresh = int(params.get("area_threshold", "1000"))
    mask = s["img_mask_use"]

    mask_opened = remove_small_objects(mask, min_size=area_thresh)
    mask_removed_area = ~mask_opened & mask

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_areathresh.png", img_as_ubyte(mask_removed_area))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = mask_opened > 0

    s.addToPrintList("areaThresh",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        warning_message = "After BasicModule.finalProcessingArea NO tissue remains detectable! Downstream modules likely to be incorrect/fail"
        logging.warning(
            f"{s['filename']} - {warning_message}")
        s["warnings"].append(warning_message)


def countTissuePieces(s, params):
    mask = s["img_mask_use"]
    stats = getMaskRegionsStats(mask)
    s.addToPrintList("#pieces_of_tissue", str(stats.get('num', 0)))
