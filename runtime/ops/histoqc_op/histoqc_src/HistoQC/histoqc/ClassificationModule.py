import logging
import os
import re
import sys

from ast import literal_eval as make_tuple

from distutils.util import strtobool

from histoqc.BaseImage import printMaskHelper
from skimage import io, img_as_ubyte, img_as_bool
from skimage.filters import gabor_kernel, frangi, gaussian, median, laplace
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, disk, dilation
from skimage.feature import local_binary_pattern

from scipy import ndimage as ndi

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import numpy as np

import matplotlib.pyplot as plt


def pixelWise(s, params):
    name = params.get("name", "classTask")
    logging.info(f"{s['filename']} - \tpixelWise:\t", name)

    thresh = float(params.get("threshold", .5))

    fname = params.get("tsv_file", "")
    if fname == "":
        logging.error(f"{s['filename']} - tsv_file not set in ClassificationModule.pixelWise for ", name)
        sys.exit(1)
        return
    model_vals = np.loadtxt(fname, delimiter="\t", skiprows=1)

    img = s.getImgThumb(s["image_work_size"])

    gnb = GaussianNB()
    gnb.fit(model_vals[:, 1:], model_vals[:, 0])
    cal = gnb.predict_proba(img.reshape(-1, 3))

    cal = cal.reshape(img.shape[0], img.shape[1], 2)
    mask = cal[:, :, 1] > thresh

    mask = s["img_mask_use"] & (mask > 0)

    s.addToPrintList(name, str(mask.mean()))

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(mask))
    s["img_mask_" + name] = (mask * 255) > 0
    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & ~s["img_mask_" + name]

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After ClassificationModule.pixelWise:{name} NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After ClassificationModule.pixelWise:{name} NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return


# extract_patches_2d(image, patch_size, max_patches=None, random_state=None

def compute_rgb(img, params):
    return img


def compute_laplace(img, params):
    laplace_ksize = int(params.get("laplace_ksize", 3))
    return laplace(rgb2gray(img), ksize=laplace_ksize)[:, :, None]


def compute_lbp(img, params):
    lbp_radius = float(params.get("lbp_radius", 3))
    lbp_points = int(params.get("lbp_points", 24))  # example sets radius * 8
    lbp_method = params.get("lbp_method", "default")

    return local_binary_pattern(rgb2gray(img), P=lbp_points, R=lbp_radius, method=lbp_method)[:, :, None]


def compute_gaussian(img, params):
    gaussian_sigma = int(params.get("gaussian_sigma", 1))
    gaussian_multichan = strtobool(params.get("gaussian_multichan", False))

    if (gaussian_multichan):
        return gaussian(img, sigma=gaussian_sigma, multichannel=gaussian_multichan)
    else:
        return gaussian(rgb2gray(img), sigma=gaussian_sigma)[:, :, None]


def compute_median(img, params):
    median_disk_size = int(params.get("median_disk_size", 3))
    return median(rgb2gray(img), selem=disk(median_disk_size))[:, :, None]


def compute_gabor(img, params):
    if not params["shared_dict"].get("gabor_kernels", False):
        gabor_theta = int(params.get("gabor_theta", 4))
        gabor_sigma = make_tuple(params.get("gabor_sigma", "(1,3)"))
        gabor_frequency = make_tuple(params.get("gabor_frequency", "(0.05, 0.25)"))

        kernels = []
        for theta in range(gabor_theta):
            theta = theta / 4. * np.pi
            for sigma in gabor_sigma:
                for frequency in gabor_frequency:
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)
        params["shared_dict"]["gabor_kernels"] = kernels

    kernels = params["shared_dict"]["gabor_kernels"]
    imgg = rgb2gray(img)
    feats = np.zeros((imgg.shape[0], imgg.shape[1], len(kernels)), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(imgg, kernel, mode='wrap')
        feats[:, :, k] = filtered
    return feats


def compute_frangi(img, params):
    frangi_scale_range = make_tuple(params.get("frangi_scale_range", "(1, 10)"))
    frangi_scale_step = float(params.get("frangi_scale_step", 2))
    frangi_beta1 = float(params.get("frangi_beta1", .5))
    frangi_beta2 = float(params.get("frangi_beta2", 15))
    frangi_black_ridges = strtobool(params.get("frangi_black_ridges", "True"))
    feat = frangi(rgb2gray(img), scale_range = frangi_scale_range, scale_step =frangi_scale_step, beta =frangi_beta1, gamma=frangi_beta2, black_ridges  =frangi_black_ridges)
    return feat[:, :, None]  # add singleton dimension


def compute_features(img, params):
    features = params.get("features", "")

    feats = []
    for feature in features.splitlines():
        func = getattr(sys.modules[__name__], f"compute_{feature}")
        feats.append(func(img, params))

    return np.concatenate(feats, axis=2)


def byExampleWithFeatures(s, params):
    name = params.get("name", "classTask")
    logging.info(f"{s['filename']} - \tClassificationModule.byExample:\t{name}")

    thresh = float(params.get("threshold", .5))
    nsamples_per_example = float(params.get("nsamples_per_example", -1))

    examples = params.get("examples", "")
    if examples == "":
        logging.error(f"{s['filename']} - No examples provided in ClassificationModule.byExample for {name} !!")
        sys.exit(1)
        return

    if params.get("features", "") == "":
        logging.error(f"{s['filename']} - No features provided in ClassificationModule.byExample for {name} !!")
        sys.exit(1)
        return

    with params["lock"]:  # this lock is shared across all threads such that only one thread needs to train the model
        # then it is shared with all other modules
        if not params["shared_dict"].get("model_" + name, False):
            logging.info(f"{s['filename']} - Training model ClassificationModule.byExample:{name}")

            model_vals = []
            model_labels = np.empty([0, 1])

            for ex in params["examples"].splitlines():
                ex = re.split(r'(?<!\W[A-Za-z]):(?!\\)', ex)  # workaround for windows: don't split on i.e. C:\
                img = io.imread(ex[0])
                eximg = compute_features(img, params)
                eximg = eximg.reshape(-1, eximg.shape[2])

                # read mask as grayscale images
                mask = io.imread(ex[1], as_gray=True)
                # convert grayscale images into binary images if images are not binary format 
                if mask.dtype.kind != 'b':
                    # warning log
                    msg = f"Mask file '{ex[1]}' is not a binary image. Automatically converting to binary..."
                    logging.warning(s['filename'] + ' - ' + msg)
                    s["warnings"].append(msg)
                    # convert to binary
                    mask = img_as_bool(mask)
                
                mask = mask.reshape(-1, 1)

                if nsamples_per_example != -1: #sub sambling required
                    nitems = nsamples_per_example if nsamples_per_example > 1 else int(mask.shape[0]*nsamples_per_example)

                    # set seed to random function if seed is not None
                    if s["seed"] is not None:
                        np.random.seed(int(s["seed"]))
                    
                    idxkeep = np.random.choice(mask.shape[0], size=int(nitems))
                    eximg = eximg[idxkeep, :]
                    mask = mask[idxkeep]


                model_vals.append(eximg)
                model_labels = np.vstack((model_labels, mask))

            # do stuff here with model_vals
            model_vals = np.vstack(model_vals)
            clf = RandomForestClassifier(n_jobs=-1)
            clf.fit(model_vals, model_labels.ravel())
            params["shared_dict"]["model_" + name] = clf
            logging.info(f"{s['filename']} - Training model ClassificationModule.byExample:{name}....done")

    clf = params["shared_dict"]["model_" + name]
    img = s.getImgThumb(s["image_work_size"])
    feats = compute_features(img, params)
    cal = clf.predict_proba(feats.reshape(-1, feats.shape[2]))
    cal = cal.reshape(img.shape[0], img.shape[1], 2)

    mask = cal[:, :, 1] > thresh

    area_thresh = int(params.get("area_threshold", "5"))
    if area_thresh > 0:
        mask = remove_small_objects(mask, min_size=area_thresh, in_place=True)

    dilate_kernel_size = int(params.get("dilate_kernel_size", "0"))
    if dilate_kernel_size > 0:
        mask = dilation(mask, selem=np.ones((dilate_kernel_size, dilate_kernel_size)))

    mask = s["img_mask_use"] & (mask > 0)

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(mask))
    s["img_mask_" + name] = (mask * 255) > 0
    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & ~s["img_mask_" + name]

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After ClassificationModule.byExampleWithFeatures:{name} NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After ClassificationModule.byExampleWithFeatures:{name} NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    s["img_mask_force"].append("img_mask_" + name)
    s["completed"].append(f"byExampleWithFeatures:{name}")

    return
