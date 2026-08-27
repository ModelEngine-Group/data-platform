import logging
import numpy as np
from skimage.filters import sobel
from skimage.color import convert_colorspace, rgb2gray
from distutils.util import strtobool


def getBrightnessGray(s, params):
    prefix = params.get("prefix", None)
    prefix = prefix+"_" if prefix else ""
    logging.info(f"{s['filename']} - \tgetContrast:{prefix}")

    limit_to_mask = strtobool(params.get("limit_to_mask", "True"))
    invert = strtobool(params.get("invert", "False"))
    mask_name = params.get("mask_name", "img_mask_use")

    img = s.getImgThumb(s["image_work_size"])

    img_g = rgb2gray(img)
    if limit_to_mask:

        mask = s[mask_name] if not invert else ~s[mask_name]

        img_g = img_g[mask]
        if img_g.size == 0:
            img_g = np.array(-100)

    s.addToPrintList(f"{prefix}grayscale_brightness", str(img_g.mean()))
    s.addToPrintList(f"{prefix}grayscale_brightness_std", str(img_g.std()))

    return


def getBrightnessByChannelinColorSpace(s, params):
    prefix = params.get("prefix", None)
    prefix = prefix + "_" if prefix else ""

    logging.info(f"{s['filename']} - \tgetContrast:{prefix}")

    to_color_space = params.get("to_color_space", "RGB")
    limit_to_mask = strtobool(params.get("limit_to_mask", "True"))
    mask_name = params.get("mask_name", "img_mask_use")

    invert = strtobool(params.get("invert", "False"))


    img = s.getImgThumb(s["image_work_size"])

    suffix = ""
    if (to_color_space != "RGB"):
        img = convert_colorspace(img, "RGB", to_color_space)
        suffix = "_" + to_color_space

    for chan in range(0, 3):
        vals = img[:, :, chan]
        if (limit_to_mask):

            mask = s[mask_name] if not invert else ~s[mask_name]
            vals = vals[mask]

            if vals.size == 0:
                vals = np.array(-100)

        s.addToPrintList(f"{prefix}chan{chan+1}_brightness{suffix}", str(vals.mean()))
        s.addToPrintList(f"{prefix}chan{chan+1}_brightness_std{suffix}", str(vals.std()))

    return


def getContrast(s, params):
    prefix = params.get("prefix", None)
    prefix = prefix + "_" if prefix else ""

    logging.info(f"{s['filename']} - \tgetContrast:{prefix}")
    limit_to_mask = strtobool(params.get("limit_to_mask", True))
    mask_name = params.get("mask_name", "img_mask_use")

    invert = strtobool(params.get("invert", "False"))


    img = s.getImgThumb(s["image_work_size"])
    img = rgb2gray(img)

    sobel_img = sobel(img) ** 2

    if limit_to_mask:

        mask = s[mask_name] if not invert else ~s[mask_name]

        sobel_img = sobel_img[mask]
        img = img[s["img_mask_use"]]

    if img.size == 0: # need a check to ensure that mask wasn't empty AND limit_to_mask is true, still want to
                      # produce metrics for completeness with warning

        s.addToPrintList(f"{prefix}tenenGrad_contrast", str(-100))
        s.addToPrintList(f"{prefix}michelson_contrast", str(-100))
        s.addToPrintList(f"{prefix}rms_contrast", str(-100))


        logging.warning(f"{s['filename']} - After BrightContrastModule.getContrast: NO tissue "
                        f"detected, statistics are impossible to compute, defaulting to -100 !")
        s["warnings"].append(f"After BrightContrastModule.getContrast: NO tissue remains "
                             f"detected, statistics are impossible to compute, defaulting to -100 !")

        return


    # tenenGrad - Note this must be performed on full image and then subsetted if limiting to mask
    tenenGrad_contrast = np.sqrt(np.sum(sobel_img)) / img.size
    s.addToPrintList(f"{prefix}tenenGrad_contrast", str(tenenGrad_contrast))

    # Michelson contrast
    max_img = img.max()
    min_img = img.min()
    contrast = (max_img - min_img) / (max_img + min_img)
    s.addToPrintList(f"{prefix}michelson_contrast", str(contrast))

    # RMS contrast
    rms_contrast = np.sqrt(pow(img - img.mean(), 2).sum() / img.size)
    s.addToPrintList(f"{prefix}rms_contrast", str(rms_contrast))

    return
