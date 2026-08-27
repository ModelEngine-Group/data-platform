import logging
import numpy as np
from skimage import  color
from distutils.util import strtobool
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt



def estimateGreyComatrixFeatures(s, params):
    
    prefix = params.get("prefix", None)
    prefix = prefix+"_" if prefix else ""

    logging.info(f"{s['filename']} - \tLocalTextureEstimationModule.estimateGreyComatrixFeatures:{prefix}")
    patch_size = int(params.get("patch_size", 32))
    npatches = int(params.get("npatches", 100))
    nlevels = int(params.get("nlevels", 8))
    feats = params.get("feats","contrast:dissimilarity:homogeneity:ASM:energy:correlation").split(':')
    invert = strtobool(params.get("invert", "False"))
    mask_name = params.get("mask_name","img_mask_use")

    # set seed to random function if seed is not None
    if s["seed"] is not None:
        np.random.seed(int(s["seed"]))

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    mask = s[mask_name] if not invert else ~s[mask_name]
    if len(mask.nonzero()[0]) == 0:  # add warning in case the no tissus detected in mask
        msg = f"LocalTextureEstimationModule.estimateGreyComatrixFeatures:{prefix} Can not estimate the empty mask since NO tissue remains detectable in mask"
        logging.warning(f"{s['filename']} - {msg}")
        s["warnings"].append(msg)
        return

    maskidx = mask.nonzero()
    maskidx = np.asarray(maskidx).transpose()
    idx = np.random.choice(maskidx.shape[0], npatches)

    results = []

    for id in idx:
        r, c = maskidx[id, :]
        patch = img[r:r + patch_size, c:c + patch_size]
        glcm = graycomatrix(np.digitize(patch,np.linspace(0,1,num=nlevels),right=True), distances=[5],
                            angles=[0], levels=nlevels, symmetric=True, normed=True)

        results.append([graycoprops(glcm, prop=feat) for feat in feats])

    results = np.asarray(results).squeeze()

    for vals, feat in zip(results.transpose(), feats):
        s.addToPrintList(f"{prefix}{feat}", str(vals.mean()))
        s.addToPrintList(f"{prefix}{feat}_std", str(vals.std()))

    return
