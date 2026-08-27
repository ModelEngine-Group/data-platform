"""
A standalone tile extraction module to locate tile bounding boxes in usable tissue region obtained by previous steps.
Coordinates are saved in the half-open 4-tuple convention of (left, top, right, bottom), where `right` and `bottom`
are open.
"""
import os
import openslide
import json
from histoqc.BaseImage import BaseImage
from typing import Callable, Dict, Any, List, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import regionprops
from contextlib import contextmanager
from distutils.util import strtobool
import logging
from histoqc.import_wrapper.typing import Literal, get_args
# from histoqc.import_wrapper.helper import dynamic_import
# __TYPE_GET_ARGS = Callable[[Type, ], Tuple[Any, ...]]
# Literal: TypeVar = dynamic_import("typing", "Literal", "typing_extensions")
# get_args: __TYPE_GET_ARGS = dynamic_import("typing", "get_args", "typing_extensions")

TYPE_TILE_SIZE = Literal['tile_size']
TYPE_TILE_STRIDE = Literal['tile_stride']
TYPE_TISSUE_RATIO = Literal['tissue_ratio']
TYPE_OUTPUT_ROOT = Literal['tile_output']
TYPE_LOCK = Literal['lock']
TYPE_SUFFIX = Literal['suffix']
TYPE_OUTLINE = Literal['outline']
TYPE_WIDTH = Literal['width']
TYPE_SAVE_FLAG = Literal['save_image']
PARAMS = Literal[TYPE_TILE_SIZE, TYPE_TILE_STRIDE, TYPE_TISSUE_RATIO,
                 TYPE_OUTPUT_ROOT, TYPE_SUFFIX,
                 TYPE_LOCK, TYPE_OUTLINE, TYPE_WIDTH, TYPE_SAVE_FLAG]


TYPE_BBOX_FLOAT = Tuple[float, float, float, float]
TYPE_BBOX_INT = Tuple[int, int, int, int]


def default_screen_identity(img: np.ndarray):
    return True


class MaskTileWindows:
    """
    Locate the window of tiles in the given downsampled mask. Output Convention: (left, top, right, bottom).
    Coordinates are half-open as [left, right) and [top, bottom).
    """

    __rp_list: List
    __mask: np.ndarray
    __mask_pil: Image.Image
    __tissue_thresh: float
    # note that the tile size on the corresponding downsampled masks may no longer be integer, therefore cause
    # loss of precision when convert back to original tile size after working on mask
    __windows_on_mask: List[List[TYPE_BBOX_FLOAT]]
    __windows_on_original_image: List[List[Tuple[int, int, int, int]]]

    @property
    def mask_pil(self) -> Image.Image:
        return Image.fromarray(self._mask)

    @property
    def _mask(self) -> np.ndarray:
        return self.__mask

    @property
    def _rp_list(self) -> List:
        """

        Returns:
            List of regionprops objects (see sklearn.regionprops)
        """
        attr_name = '__rp_list'
        if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
            setattr(self, attr_name, regionprops(self.__mask))
        return getattr(self, attr_name)

    def __init_mask(self, mask: np.ndarray):
        self.__mask = mask

    def __init__(self, mask: np.ndarray, *, work_tile_size: int, work_stride: int,
                 size_factor: float,
                 tissue_thresh: float):
        """
        Args:
            mask:
            work_tile_size:
            work_stride:
            size_factor: size ratio of image to mask
        """
        self.__init_mask(mask.astype(np.int32))
        self.work_tile_size = work_tile_size
        self.work_stride = work_stride
        self.__size_factor = size_factor
        self.__tissue_thresh = tissue_thresh

    @staticmethod
    def validate_tile_mask_area_thresh(mask_pil: Image.Image,
                                       tile_window_on_mask: TYPE_BBOX_FLOAT,
                                       tissue_thresh: float) -> bool:
        """ Validate whether the given tile window (left, top, right, bottom) contains sufficient tissue. This is
        computed by calculating the tissue % in the corresponding mask region.
        Note that if the coordinates are not int the actual area of region may be different
        Args:
            mask_pil:
            tile_window_on_mask: List of (left, top, right, bottom). Open on right and bottom
            tissue_thresh: minimum requirement of tissue percentage

        Returns:
            True if the window has sufficient tissue.
        """
        left, top, right, bottom = tile_window_on_mask
        # window_on_mask_work = tuple(round(x) for x in tile_window_on_mask)
        window_on_mask_work = round(left), round(top), round(right), round(bottom)
        window_pil = mask_pil.crop(window_on_mask_work)
        # noinspection PyTypeChecker
        window_np = np.array(window_pil, copy=False)
        window_bool = window_np > 0
        return window_bool.mean() >= tissue_thresh

    @staticmethod
    def _valid_tile_windows_on_mask_helper(mask_pil: Image.Image,
                                           tile_cand_pil_window_on_mask: List[TYPE_BBOX_FLOAT],
                                           tissue_thresh: float) -> List[TYPE_BBOX_FLOAT]:
        """ All tile windows with sufficient usable tissue from the grid of window candidates
        Args:
            mask_pil:
            tile_cand_pil_window_on_mask: left, top, right, bottom. Potential candidates of windows
            tissue_thresh: minimum requirement of tissue %
        Returns:
            List of validated windows (left, top, right, bottom)  from the given candidates.
        """
        # output = []
        # for window in tile_cand_pil_window:
        #     has_enough_usable_tissue = _MaskTileLocator.validate_tile_mask_area_thresh(mask_pil, window,
        #                                                                                tissue_thresh)
        #     if not has_enough_usable_tissue:
        #         continue
        return [window for window in tile_cand_pil_window_on_mask
                if MaskTileWindows.validate_tile_mask_area_thresh(mask_pil, window, tissue_thresh)]

    @staticmethod
    def region_tile_cand_pil_window_on_mask(rp_bbox: TYPE_BBOX_INT,
                                            work_tile_size: float,
                                            work_stride: float) -> List[TYPE_BBOX_FLOAT]:
        """ Split the region given by the region property bounding box into a grid of tile windows. Support overlapping.
        This computes the all possible window given by the rp regardless of the tissue condition. Refinement can be
        performed in further steps.
        Args:
            rp_bbox: sklearn region property style: [top, left, bottom, right]. Half-open
                -- [Left, Right) and [Top, Bottom)
            work_tile_size:
            work_stride:

        Returns:
            List of (left, top, right, bottom) tuples. Half-open
        """
        top_rp, left_rp, bottom_rp, right_rp = rp_bbox
        # top/left of the right/bottom most tile
        tile_max_top, tile_max_left = MaskTileWindows.max_tile_bbox_top_left_coord(rp_bbox,
                                                                                   work_tile_size,
                                                                                   work_stride)
        # obtain the top/left coord of all tile bboxes
        all_tops = np.arange(top_rp, tile_max_top + 1, work_stride, dtype=int)
        all_lefts = np.arange(left_rp, tile_max_left + 1, work_stride, dtype=int)
        # since it's open on right and bottom, right = left + size and bottom = top + size, wherein the right-1
        # is the actual right most pixel and likewise bottom-1 is the actual bottom-most pixel.
        def window(left, top, size): return left, top, (left + size), (top + size)
        # get full tile bbox representation
        all_tile_pil_window = [window(left, top, work_tile_size) for left in all_lefts for top in all_tops]
        return all_tile_pil_window

    @staticmethod
    def rp_tile_windows_on_mask(mask_pil,
                                rp_bbox: TYPE_BBOX_INT,
                                work_tile_size: float,
                                work_stride: float,
                                tissue_thresh: float) -> List[TYPE_BBOX_FLOAT]:
        """Wrapper. Obtain the valid window with sufficient tissue from a list of region property objects based on
        a given mask. For each individual region, a list of window in format (left, top, right, bottom) is obtained.
        Resulting lists of windows of all regions are nested into a list as the object.
        Args:
            mask_pil: PIL handle of the downsampled mask
            rp_bbox: bounding box of sklearn region properties. Note that its convention is [top, left, bottom, right],
                which is different to the (left, top, right, bottom) in PIL and OpenSlide. Int coords.
            work_tile_size: Working tile size on the downsampled mask
            work_stride:    Working stride size on the downsampled mask
            tissue_thresh:  Minimum requirement of tissue % in each window

        Returns:
            List of (left, top, right, bottom)
        """
        candidates = MaskTileWindows.region_tile_cand_pil_window_on_mask(rp_bbox, work_tile_size, work_stride)
        return MaskTileWindows._valid_tile_windows_on_mask_helper(mask_pil, candidates, tissue_thresh)

    def _tile_windows_on_mask(self) -> List[List[TYPE_BBOX_FLOAT]]:
        """Helper function to locate the windows of each region in format of (left, top, right, bottom)
            Note that to retain the precision the coordinates are in the float form, rather than cast to int.
            (Noticeably the right/bottom coords due to that size may not be integer)
            THe actual validation of corresponding bbox regions on mask on the other hand should convert the coords
            correspondingly.
        Returns:
            List of List of (left, top, right, bottom), nested by connected regions in the mask
        """
        result_list: List[List[TYPE_BBOX_FLOAT]] = []
        # loop the regionprop list
        for region in self._rp_list:
            # get bounding box of the individual region
            rp_bbox = region.bbox
            # get list of possible tile bounding boxes within the region bounding box, computed from
            # tile size, stride, and tissue thresh
            windows: List[TYPE_BBOX_FLOAT] = MaskTileWindows.rp_tile_windows_on_mask(self.mask_pil,
                                                                                     rp_bbox,
                                                                                     self.work_tile_size,
                                                                                     self.work_stride,
                                                                                     self.__tissue_thresh)
            # result_list += windows
            result_list.append(windows)
        return result_list

    @property
    def windows_on_mask(self) -> List[List[TYPE_BBOX_FLOAT]]:
        """
        Returns:
            Obtain the cached tile windows on the given mask. Results are cached.
        """
        if not hasattr(self, '__windows_on_mask') or self.__windows_on_mask is None:
            self.__windows_on_mask = self._tile_windows_on_mask()
        return self.__windows_on_mask

    @property
    def windows_on_original_image(self) -> List[List[TYPE_BBOX_INT]]:
        """Zoom the windows from the mask (which is often downsampled) to the original image, using the defined
        size factor
        Returns:
            Zoomed windows on the original image (left, top, right, bottom)
        """
        if not hasattr(self, '__windows_on_original_image') or self.__windows_on_original_image is None:
            self.__windows_on_original_image = MaskTileWindows.__window_list_resize(self.windows_on_mask,
                                                                                    self.__size_factor)
        return self.__windows_on_original_image

    @staticmethod
    def max_tile_bbox_top_left_coord(rp_bbox: TYPE_BBOX_INT, work_tile_size: float, work_stride: float) \
            -> Tuple[int, int]:
        """ find the coords of the top/left corner of the most right / bottom tile ever possible given the current
        size and stride.
        Args:
            rp_bbox: [top, left, bottom, right]. Half-open -- [Left, Right) and [Top, Bottom). Note that this is the
                convention of sklearn's region properties, which is different to the (left, top, right, bottom) used by
                PIL or OpenSlide. The bbox of connected tissue regions on mask. Int coords.
            work_tile_size: Tile size on the working mask, which might be downsampled.
            work_stride: Stride size on the working mask, which might be downsampled.
        Returns:
            Tuple[int, int]
        """
        assert work_stride > 0, f"work stride must be greater than 0 - got {work_stride}"
        assert work_tile_size > 0, f"work tile size must be greater than 0 - got {work_tile_size}"

        # not for skimage regionprops, the bbox is half-open at the bottom / right coordinates.
        # [left, right) and [top, bottom). Hence, the "+1" operation below for coord computation
        # is already priced-in
        top_rp, left_rp, bottom_rp, right_rp = rp_bbox
        # start + n_step * stride + tile_size = bottom/rightmost -->  (rp_limit - tile_size) // stride = max step
        max_step_horiz = (right_rp - left_rp - work_tile_size) / work_stride
        max_step_vert = (bottom_rp - top_rp - work_tile_size) / work_stride
        tile_max_left = left_rp + max_step_horiz * work_stride
        tile_max_top = top_rp + max_step_vert * work_stride

        assert round(tile_max_left + work_tile_size) <= right_rp,\
            f"left + size check" \
            f" {tile_max_left + work_tile_size} = {tile_max_left} + {work_tile_size} <= {right_rp} fail"
        assert round(tile_max_top + work_tile_size) <= bottom_rp,\
            f"top + size check" \
            f" {tile_max_top + work_tile_size} = {tile_max_top} + {work_tile_size} <= {bottom_rp} fail"
        return int(tile_max_top), int(tile_max_left)

    @staticmethod
    def __window_resize_helper(window_on_mask: Union[TYPE_BBOX_FLOAT, TYPE_BBOX_INT], size_factor) -> TYPE_BBOX_INT:
        """Helper function to zoom the window coordinates on downsampled mask to the original sized image.
        Convert back to int.
        Args:
            window_on_mask:  (left, top, right, bottom)
            size_factor:  size_factor = img_size / mask_size

        Returns:
            Resized (left, top, right, bottom)
        """
        # work around of the type check of IDE. Otherwise, I will return directly
        left, top, right, bottom, *rest = tuple(int(r * size_factor) for r in window_on_mask)
        return left, top, right, bottom

    @staticmethod
    def __window_list_resize(window_on_mask: Union[List[List[TYPE_BBOX_FLOAT]], List[List[TYPE_BBOX_INT]]],
                             size_factor: float) -> List[List[Tuple[int, int, int, int]]]:
        """
        Args:
            window_on_mask:  list of list of (left, top, right, bottom), nested at region-level.
            size_factor:  size_factor = img_size / mask_size

        Returns:
            Resized (left, top, right, bottom)
        """
        return [
                    [
                        MaskTileWindows.__window_resize_helper(win, size_factor)
                        for win in win_list_region
                    ]
                    for win_list_region in window_on_mask
              ]


DRAW_TARGET_IMG_THUMB = Literal['img_thumb']
DRAW_TARGET_MASK = Literal['mask_thumb']


class TileExtractor:

    __tile_window_cache: Dict[str, MaskTileWindows]

    def __init__(self, s: BaseImage):
        self.__tile_window_cache = dict()
        self.baseimage = s
        self._filename = s['filename']

    @staticmethod
    def _tile_windows_helper(mask_use_for_tiles,
                             img_w, img_h,
                             tile_size: int = 256, tile_stride: int = 256,
                             tissue_thresh: float = 0.5,
                             filename: str = '') -> MaskTileWindows:
        """
        Initiated the mask_tile_window using a given tile_size, tile_stride, and tissue_thresh.
        Args:
            mask_use_for_tiles:
            img_w: width of base image in original mag
            img_h: height of base image in original mag
            tile_size:
            tile_stride:
            tissue_thresh:

        Returns:

        """
        mask = mask_use_for_tiles
        # image_handle: openslide.OpenSlide = s["os_handle"]
        # img_w, img_h = image_handle.dimensions
        assert mask is not None, f"{filename}: mask is not initialized"
        assert isinstance(mask, np.ndarray), f"The mask is expected to be a Numpy NDArray"

        # shape --> row column --> height width
        mask_w, mask_h = mask.shape[1], mask.shape[0]
        size_factor = img_w / mask_w
        size_factor_ref = img_h / mask_h
        assert size_factor > 0, f"{size_factor} negative"
        if round(size_factor) != round(size_factor_ref):
            logging.warning(f"{filename}: Aspect Ratio Mismatch: {img_w, img_h} vs. "
                            f"{mask_w, mask_h}")
        logging.debug(f"{filename}: size ratio between img and thumb: {size_factor}")
        work_tile_size = tile_size / size_factor
        work_stride = tile_stride / size_factor
        mask_tile_windows = MaskTileWindows(mask, work_tile_size=work_tile_size,
                                            work_stride=work_stride,
                                            size_factor=size_factor, tissue_thresh=tissue_thresh)

        return mask_tile_windows

    # @staticmethod
    # def _img_use_for_tiles(s: BaseImage):
    #     return s.getImgThumb(s["image_work_size"])
    #
    # @staticmethod
    # def _mask_use_for_tiles(s: BaseImage):
    #     """
    #     For now, we just use the img_mask_use field in BaseImage
    #     Args:
    #         s:
    #
    #     Returns:
    #
    #     """
    #     return s['img_mask_use']

    def tile_windows(self,
                     mask_use_for_tiles: np.ndarray,
                     img_w, img_h,
                     tile_size: int = 256, tile_stride: int = 256,
                     tissue_thresh: float = 0.5, *, force_rewrite: bool = False) -> MaskTileWindows:
        # s = self.baseimage
        key = f"{tile_size}_{tile_stride}_{tissue_thresh}"
        root_dict = self.__tile_window_cache
        entry = root_dict.get(key, None)
        if entry is None or force_rewrite:
            # img_w, img_h = s['os_handle'].dimensions
            root_dict[key] = TileExtractor._tile_windows_helper(mask_use_for_tiles, img_w, img_h,
                                                                tile_size,
                                                                tile_stride, tissue_thresh)
        return root_dict[key]

    def clear_tile_window(self, tile_size: int = 256, tile_stride: int = 256,
                          tissue_thresh: float = 0.5):
        key = f"{tile_size}_{tile_stride}_{tissue_thresh}"
        root_dict = self.__tile_window_cache
        root_dict.pop(key, None)

    @staticmethod
    def __bbox_overlay_helper(img: np.ndarray,
                              windows_grouped_by_region: Union[List[List[TYPE_BBOX_INT]], List[List[TYPE_BBOX_FLOAT]]],
                              outline: str = 'green', width: int = 2) -> Image.Image:
        """
        Helper function to draw bbox and overlay to the img thumbnail
        Args:
            img: image to overlay the bboxes
            windows_grouped_by_region:
            outline: outline color of the bboxes
            width: width of bboxes outlines

        Returns:
            the PIL object after drawing the bboxes over the img
        """
        # avoid inplace change
        copy = np.array(img, copy=True)
        draw_pil = Image.fromarray(copy, mode="RGB")
        draw_context = ImageDraw.Draw(draw_pil, mode="RGB")
        for window_list in windows_grouped_by_region:
            for window in window_list:
                # ImageDraw accepts x0 y0 x1 y1
                window = tuple(round(x) for x in window)
                left, top, right, bottom = window
                draw_context.rectangle((left, top, right, bottom), outline=outline, width=width)

        return draw_pil

    def bbox_overlay(self,
                     img_use_for_tiles,
                     mask_use_for_tiles: np.ndarray,
                     img_w, img_h,
                     target: Literal[DRAW_TARGET_IMG_THUMB, DRAW_TARGET_MASK] = 'img_thumb',
                     tile_size_on_img: int = 256, tile_stride_on_img: int = 256,
                     tissue_thresh: float = 0.5, *, force_rewrite: bool = False, outline='green', width=2
                     ) -> Image.Image:
        # note that the window_on_image corresponding to the original image at base-mag.
        # the thumb image is at the same size of masks
        tile_windows: MaskTileWindows = self.tile_windows(mask_use_for_tiles, img_w, img_h,
                                                          tile_size_on_img, tile_stride_on_img, tissue_thresh,
                                                          force_rewrite=force_rewrite)
        windows_on_mask: List[List[TYPE_BBOX_FLOAT]] = tile_windows.windows_on_mask
        # all properties below are cached
        mapping: Dict[Literal[DRAW_TARGET_IMG_THUMB, DRAW_TARGET_MASK], np.ndarray] = {
            get_args(DRAW_TARGET_IMG_THUMB)[0]: img_use_for_tiles,
            get_args(DRAW_TARGET_MASK)[0]: mask_use_for_tiles,
        }
        target_img_arr = mapping[target]
        return self.__bbox_overlay_helper(target_img_arr, windows_grouped_by_region=windows_on_mask, outline=outline,
                                          width=width)

    @staticmethod
    def __window_convert(window: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """TO OpenSlide style (left, top) + (width, height)
        Args:
            window:

        Returns:

        """
        left, top, right, bottom = window

        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        location = (left, top)
        width = right - left
        height = bottom - top
        size = (width, height)
        return location, size

    @contextmanager
    def mp_tile_window_manager(self,
                               mask_use_for_tiles,
                               img_w,
                               img_h,
                               tile_size: int = 256,
                               tile_stride: int = 256, tissue_thresh: float = 0.5, force_rewrite: bool = False):
        """Avoid pickling the MaskTileWindow in MP. Clear cached objects on exit
        Args:
            mask_use_for_tiles: tissue mask for tile extraction
            img_w: width of the image under original magnification
            img_h: height of the image under original magnification
            tile_size:
            tile_stride:
            tissue_thresh:
            force_rewrite:

        Returns:

        """
        self.tile_windows(mask_use_for_tiles, img_w, img_h,
                          tile_size, tile_stride, tissue_thresh, force_rewrite=force_rewrite)
        yield
        # clearn cache
        self.clear_tile_window(tile_size, tile_stride, tissue_thresh)

    @staticmethod
    def __save_on_flag(path: str, prefix: str, suffix: str,
                       window: Tuple[int, int, int, int],
                       region: Image.Image,
                       save_image: bool):

        if save_image:
            full_export_dest = os.path.join(path, f"{prefix}_{window}{suffix}")
            actual_path, actual_base = os.path.split(full_export_dest)
            os.makedirs(actual_path, exist_ok=True)
            region.save(full_export_dest)

    def valid_tile_extraction(self,
                              s: BaseImage,
                              mask_use_for_tiles,
                              img_w,
                              img_h,
                              path, *, prefix='', suffix='.png',
                              screen_callbacks: Callable = default_screen_identity,
                              tile_size: int = 256,
                              tile_stride: int = 256,
                              tissue_thresh: float = 0.5,
                              save_image: bool = True,
                              force_rewrite: bool = False) -> List[List[Tuple[int, int, int, int]]]:

        tw: MaskTileWindows = self.tile_windows(mask_use_for_tiles, img_w, img_h,
                                                tile_size, tile_stride, tissue_thresh, force_rewrite=force_rewrite)
        window_list_of_regions = tw.windows_on_original_image
        image_handle: openslide.OpenSlide = s["os_handle"]
        valid_window_list_all_regions: List[List[Tuple[int, int, int, int]]] = []
        for region_windows in window_list_of_regions:
            region_windows: List[Tuple[int, int, int, int]]
            valid_windows_curr:  List[Tuple[int, int, int, int]] = []
            for window in region_windows:
                window: Tuple[int, int, int, int]
                # just to make the convention clear
                location, size = TileExtractor.__window_convert(window)
                region = image_handle.read_region(location, 0, size)
                tile_np = np.array(region, copy=False)
                valid_flag = screen_callbacks(tile_np)
                if not valid_flag:
                    continue
                valid_windows_curr.append(window)
                # prefix can be the slide name
                TileExtractor.__save_on_flag(path=path, prefix=prefix, suffix=suffix, window=window,
                                             region=region, save_image=save_image)
            if len(valid_windows_curr) > 0:
                valid_window_list_all_regions.append(valid_windows_curr)
        return valid_window_list_all_regions


def extract(s: BaseImage, params: Dict[PARAMS, Any]):
    logging.info(f"{s['filename']} - \textract")
    with params['lock']:
        slide_out = s['outdir']
        tile_output_dir = params.get('tile_output', os.path.join(slide_out, 'tiles'))
        os.makedirs(tile_output_dir, exist_ok=True)
        prefix, _ = os.path.splitext(os.path.basename(s['filename']))

        suffix: str = params.get('suffix', '.png')
        outline: str = params.get('outline', "green")
        width: int = int(params.get('width', 2))
        save_image: bool = bool(strtobool(params.get("save_image", "False")))
        tile_size = int(params.get('tile_size', 256))
        tile_stride = int(params.get('tile_stride', 256))
        tissue_thresh = float(params.get('tissue_ratio', 0.5))

        img_use_for_tiles = s.getImgThumb(s["image_work_size"])
        mask_use_for_tiles = s['img_mask_use']
        image_handle: openslide.OpenSlide = s['os_handle']
        img_w, img_h = image_handle.dimensions

        tile_extractor = TileExtractor(s)
        # binding MaskTileWindow into the BaseImage might not be the best idea in multiprocessing
        # clear the cached MaskTileWindow to prevent hanging of the pool
        with tile_extractor.mp_tile_window_manager(mask_use_for_tiles, img_w, img_h,
                                                   tile_size, tile_stride, tissue_thresh, False):
            window_list_of_regions: List[List[Tuple[int, int, int, int]]]\
                = tile_extractor.valid_tile_extraction(s,
                                                       mask_use_for_tiles,
                                                       img_w, img_h,
                                                       tile_output_dir,
                                                       prefix=prefix,
                                                       suffix=suffix,
                                                       tile_size=tile_size,
                                                       tile_stride=tile_stride,
                                                       tissue_thresh=tissue_thresh,
                                                       save_image=save_image,
                                                       force_rewrite=False)
            # #
            overlay_export = os.path.join(s["outdir"], f"{s['filename']}_tile_bbox.png")
            bbox_overlay = tile_extractor.bbox_overlay(img_use_for_tiles, mask_use_for_tiles,
                                                       img_w, img_h,
                                                       target="img_thumb",
                                                       tile_size_on_img=tile_size,
                                                       tile_stride_on_img=tile_stride,
                                                       tissue_thresh=tissue_thresh,
                                                       force_rewrite=False, outline=outline, width=width)

            bbox_overlay.save(overlay_export)
            bbox_json_loc = os.path.join(slide_out, f'{s["filename"]}_bbox.json')
            with open(bbox_json_loc, 'w') as root:
                json.dump(window_list_of_regions, root, indent=4)
        return
