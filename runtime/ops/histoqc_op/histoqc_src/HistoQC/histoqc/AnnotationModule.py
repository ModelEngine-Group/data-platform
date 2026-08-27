import logging
from typing import List, Tuple
from histoqc.BaseImage import printMaskHelper
from skimage import io, img_as_ubyte
import os
from pathlib import PurePosixPath, Path
from shapely.geometry import Polygon
from shapely import affinity
from PIL import Image, ImageDraw
import numpy as np
from histoqc.annotations.annot_collection import AnnotCollection, PARSER_BUILDER_MAP, TYPE_SUPPORTED_PARSER, Region


def rescale_by_img_bbox(polygon: Polygon, offset_xy: Tuple[float, float], resize_factor: float) -> Polygon:
    if isinstance(offset_xy, float):
        offset_xy = (offset_xy, offset_xy)
    x_off, y_off = offset_xy
    polygon = affinity.translate(polygon, xoff=x_off, yoff=y_off)
    polygon = affinity.scale(polygon, xfact=resize_factor, yfact=resize_factor, origin=(0, 0))
    return polygon


def polygon_filled(draw_pil: ImageDraw, polygon: Polygon, offset_xy: Tuple[float, float], resize_factor: float):
    polygon = rescale_by_img_bbox(polygon, offset_xy, resize_factor)
    # outer
    exterior_coords = list(polygon.exterior.coords)
    draw_pil.polygon(exterior_coords, fill=1, outline=1, width=0)
    for component in polygon.interiors:
        interior_coord = list(component.coords)
        draw_pil.polygon(interior_coord, fill=0, outline=0, width=0)
    return draw_pil


def annotation_to_mask(width: int, height: int, annot_collection: AnnotCollection, offset_xy: Tuple[float, float],
                       resize_factor: float) -> np.ndarray:
    # binary
    mask = Image.new(mode="1", size=(width, height))
    draw_pil = ImageDraw.Draw(mask)
    all_regions: List[Region] = annot_collection.all_regions
    for region in all_regions:
        polygon: Polygon = region['polygon']
        # skip if empty ring (e.g., misclick in qupath)
        if polygon.is_empty or (not polygon.is_valid):
            continue
        draw_pil = polygon_filled(draw_pil, polygon, offset_xy, resize_factor)
    # noinspection PyTypeChecker
    return np.array(mask)


def getParams(s, params):
    # read params - format: xml, json; file_path; suffix; 
    ann_format = params.get("format", None)
    file_path = params.get("file_path", None)
    suffix = params.get("suffix", "")

    # try use default value if the params are not provided
    if not ann_format:
        # set default format
        ann_format = "xml"
        # warning msg
        msg = f"format is not provided, using xml as the default format."
        logging.warning(f"{s['filename']} - {msg}")
        s["warnings"].append(msg)

    if not file_path:
        # set default file path
        file_path = s["dir"]
        # warning msg
        msg = f"file path is not provided, using \"{s['dir']}\" as the default file path"
        logging.warning(f"{s['filename']} - {msg}")
        s["warnings"].append(msg)

    return ann_format, file_path, suffix


def saveAnnotationMask(s, params):
    logging.info(f"{s['filename']} - \tgetAnnotationMask")

    (ann_format, file_path, suffix) = getParams(s, params)

    # annotation file path
    f_path = f"{file_path}{os.sep}{PurePosixPath(s['filename']).stem}{suffix}.{ann_format}"

    if not Path(f_path).is_file():
        msg = f"Annotation file {f_path} does not exist. Skipping..."
        logging.warning(f"{s['filename']} - {msg}")
        s["warnings"].append(msg)
        return

    logging.info(f"{s['filename']} - \tusing {f_path}")

    # todo better using the Py3.10 match statement - so it will be a Literal
    # noinspection PyTypeChecker
    annotation_type: TYPE_SUPPORTED_PARSER = ann_format.lower()
    logging.info(f"{s['filename']} - \tusing {annotation_type}")
    # read points set
    if annotation_type in PARSER_BUILDER_MAP:  # xml
        annot_collection = AnnotCollection.build(parser_type=annotation_type, uri=f_path, label_map=None)
        # get_points_from_geojson(s, f_path)
    else:  # unsupported format
        msg = f"unsupported file format '{ann_format}'. Skipping..."
        logging.warning(f"{s['filename']} - {msg}")
        s["warnings"].append(msg)
        return

    (off_x, off_y, ncol, nrow) = s["img_bbox"]
    resize_factor = np.shape(s["img_mask_use"])[1] / ncol
    height, width = s["img_mask_use"].shape
    annotationMask = annotation_to_mask(width, height, annot_collection, (off_x, off_y), resize_factor) > 0

    mask_file_name = f"{s['outdir']}{os.sep}{s['filename']}_annot_{ann_format.lower()}.png"
    io.imsave(mask_file_name, img_as_ubyte(annotationMask))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = prev_mask & annotationMask
    s.addToPrintList("getAnnotationMask",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(
            f"{s['filename']} - After AnnotationModule.getAnnotationMask "
            f"NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(
            f"After AnnotationModule.getAnnotationMask NO tissue remains detectable!"
            f" Downstream modules likely to be incorrect/fail")
    return
