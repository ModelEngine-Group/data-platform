import logging
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from distutils.util import strtobool

global_holder = {} #this holds a local copy of the histograms of the template images so that they need only be computed once


def getHistogram(s, params):
    logging.info(f"{s['filename']} - \tgetHistogram")
    limit_to_mask = strtobool(params.get("limit_to_mask", True))
    bins = int(params.get("bins", 20))

    img = s.getImgThumb(s["image_work_size"])
    if limit_to_mask:
        img = img[s["img_mask_use"]]
    else:
        img = img.reshape(-1, 3)

    ax = plt.axes()
    ax.hist(img, bins=bins, density=True, range=(0, 255), histtype='step', color=("r", "g", "b"))

    ax.grid(True)
    ax.set_title('Color Distribution for ' + s["filename"])
    ax.set_xlabel('Pixel Val')
    ax.set_ylabel('Density')
    plt.savefig(s["outdir"] + os.sep + s["filename"] + "_hist.png")
    plt.close()
    return


def computeHistogram(img, bins, mask=-1):
    result = np.zeros(shape=(bins, 3))
    for chan in range(0, 3):
        vals = img[:, :, chan].flatten()
        if (isinstance(mask, np.ndarray)):
            vals = vals[mask.flatten()]

        result[:, chan] = np.histogram(vals, bins=bins, density=True, range=[0, 255])[0]

    return result


def compareToTemplates(s, params):
    logging.info(f"{s['filename']} - \tcompareToTemplates")
    bins = int(params.get("bins", 20))
    limit_to_mask = strtobool(params.get("limit_to_mask", True))
    if (not global_holder.get("templates", False)): #if the histograms haven't already been computed, compute and store them now
        templates = {}
        for template in params["templates"].splitlines():
            templates[os.path.splitext(os.path.basename(template))[0]] = computeHistogram(io.imread(template), bins)
            # compute each of their histograms
        global_holder["templates"] = templates

    img = s.getImgThumb(s["image_work_size"])

    if (limit_to_mask):
        mask = s["img_mask_use"]
        if len(mask.nonzero()[0]) == 0:

            logging.warning(f"{s['filename']} - HistogramModule.compareToTemplates NO tissue "
                            f"remains detectable in mask!")
            s["warnings"].append(f"HistogramModule.compareToTemplates NO tissue "
                                        f"remains detectable in mask!")

            imghst = np.zeros((bins,3))

        else:
            imghst = computeHistogram(img, bins, mask)
    else:
        imghst = computeHistogram(img, bins)

    for template in global_holder["templates"]:
        val = np.sum(pow(abs(global_holder["templates"][template] - imghst), 2))
        s.addToPrintList(template + "_MSE_hist", str(val))
    return
