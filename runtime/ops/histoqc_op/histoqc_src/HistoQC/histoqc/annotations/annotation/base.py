from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Union, List, Tuple, TypedDict
from lazy_property import LazyProperty
from shapely.geometry import Polygon, MultiPolygon
import logging

T = TypeVar("T")
TYPE_POINT = Tuple[int, int]
TYPE_POINT_SET = List[TYPE_POINT]
TYPE_HOLED_SET = Tuple[TYPE_POINT_SET, Union[List[TYPE_POINT_SET], None]]
# TYPE_HOLED_SET_COLLECTION = List[TYPE_HOLED_SET]

TYPE_LABEL = Union[int, None]
TYPE_RAW_LABEL = Union[str, None, TYPE_LABEL]

WARNING_NOT_SIMPLE_POLYGON = f"Not a Simple Polygon: buffer of the polygon " \
                             f"with 0-distance resulting multiple polygons." \
                             f"The shape of these polygons may not be identical to" \
                             f"input annotations"


class Region(TypedDict):
    polygon: Polygon
    point_set: TYPE_HOLED_SET  # TYPE_POINT_SET
    label: TYPE_RAW_LABEL
    raw_label: TYPE_RAW_LABEL
    uri: str


class Annotation(ABC, Generic[T]):
    """
    Annotation --> an atomic annotation that may contain one or multiple regions.
     One label is assigned to one annotation.
    """

    _label_map: Union[Dict[TYPE_RAW_LABEL, TYPE_LABEL], None]
    _ann_data: T
    _uri: str

    @staticmethod
    def point_to_int(point_xy: Tuple[Union[float, str], Union[float, str]]) -> TYPE_POINT:
        raw_x, raw_y = point_xy
        return int(float(raw_x)), int(float(raw_y))

    @abstractmethod
    def point_set_list(self) -> List[TYPE_HOLED_SET]:
        return NotImplemented

    @abstractmethod
    def label_from_annotation(self) -> TYPE_RAW_LABEL:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def annotation_list_from_uri(uri) -> List[T]:
        return NotImplemented

    @staticmethod
    def _enough_points(point_set: TYPE_POINT_SET):
        return len(point_set) >= 3

    @staticmethod
    def _sanitized_points_helper(point_set: TYPE_HOLED_SET) -> Union[TYPE_HOLED_SET, None]:
        outer, inner = point_set
        # if shell has less than 3 --> discard directly
        if not Annotation._enough_points(outer):
            return None
        inner = [hole for hole in inner if Annotation._enough_points(hole)] if inner is not None else None
        return outer, inner

    @staticmethod
    def _sanitized_points(point_set_list: List[TYPE_HOLED_SET]) -> List[TYPE_HOLED_SET]:
        out_list = []
        for point_set in point_set_list:
            sanitized = Annotation._sanitized_points_helper(point_set)
            if sanitized is None:
                continue
            out_list.append(sanitized)
        return out_list

    @staticmethod
    def valid_polygon_helper(polygon: Polygon, point_set: TYPE_HOLED_SET) -> Tuple[List[Polygon], List[TYPE_HOLED_SET]]:
        """
        In case
        Returns:

        """
        if polygon.is_valid:
            return [polygon, ], [point_set, ]

        valid_poly = polygon.buffer(0)
        if isinstance(valid_poly, Polygon):
            return [valid_poly, ], [point_set, ]
        # if not simple polygon but multiple polygons
        assert isinstance(valid_poly, MultiPolygon)
        logging.warning(WARNING_NOT_SIMPLE_POLYGON)
        # warning
        polygon_list: List[Polygon] = list(valid_poly.geoms)
        exterior_list: List[List[TYPE_POINT_SET]] = [list(x.exterior.coords) for x in polygon_list]
        interior_list: List[List[List[TYPE_POINT_SET]]] = [[list(interior.coords) for interior in x.interiors]
                                                           for x in polygon_list]
        point_set_list: List[TYPE_HOLED_SET] = [(outer, inner)
                                                for outer, inner in zip(exterior_list, interior_list)]
        return polygon_list, point_set_list

    @staticmethod
    def valid_polygon(point_set: TYPE_HOLED_SET) -> Tuple[List[Polygon], List[TYPE_HOLED_SET]]:
        outer, inner = point_set
        polygon = Polygon(outer, holes=inner)
        # if not polygon.is_valid:
        #     return polygon.buffer(0)
        # assert not isinstance(polygon, MultiPolygon)
        # return [polygon, ], [point_set, ]
        return Annotation.valid_polygon_helper(polygon, point_set)

    @staticmethod
    def regions_accumulate_helper(polygon_list: List[Polygon],
                                  point_set_list: List[TYPE_HOLED_SET], label, raw_label, uri) -> List[Region]:
        return [Region(polygon=polygon, point_set=point_set, label=label, raw_label=raw_label, uri=uri)
                for polygon, point_set in zip(polygon_list, point_set_list)]

    @LazyProperty
    def regions(self) -> List[Region]:
        point_set_list: List[TYPE_HOLED_SET] = self.point_set_list()
        clean_list = Annotation._sanitized_points(point_set_list)
        region_list: List[Region] = []
        for point_set in clean_list:
            point_set: TYPE_HOLED_SET
            # polygon: Polygon = Annotation.valid_polygon(point_set)
            polygon_list, point_set_list = Annotation.valid_polygon(point_set)
            label = self.label
            raw_label = self.raw_label
            uri = self._uri
            # curr_region = Region(polygon=polygon, point_set=point_set,
            # label=label, raw_label=raw_label, uri=self._uri)
            # region_list.append(curr_region)
            curr_region_list = Annotation.regions_accumulate_helper(polygon_list,
                                                                    point_set_list, label, raw_label, uri)
            region_list += curr_region_list
        return region_list

    @staticmethod
    def _mapped_label(label_map: Dict[TYPE_RAW_LABEL, TYPE_LABEL],
                      label_var: TYPE_RAW_LABEL) -> Union[TYPE_RAW_LABEL, TYPE_LABEL]:
        if label_map is None or len(label_map) == 0:
            return label_var
        assert label_var in label_map
        return label_map[label_var]

    @LazyProperty
    def raw_label(self) -> TYPE_RAW_LABEL:
        return self.label_from_annotation()

    @LazyProperty
    def label(self) -> Union[TYPE_RAW_LABEL, TYPE_LABEL]:
        raw_label = self.raw_label
        label = Annotation._mapped_label(self._label_map, raw_label)
        return label

    @property
    def ann_data(self) -> T:
        return self._ann_data

    def __init__(self, uri: str, ann_data: T, label_map: Dict[Union[str, int], int]):
        self._uri = uri
        self._ann_data = ann_data
        self._label_map = label_map

    @classmethod
    def build(cls, uri: str, ann_data: T, label_map: Dict[Union[str, int], int]) -> "Annotation":
        return cls(uri=uri, ann_data=ann_data, label_map=label_map)

    @classmethod
    def build_from_uri(cls, uri: str, label_map: Union[Dict[TYPE_RAW_LABEL, TYPE_LABEL], None]) -> List["Annotation"]:
        ann_data_list: List[T] = cls.annotation_list_from_uri(uri)
        return [cls.build(uri=uri, ann_data=ann_data, label_map=label_map) for ann_data in ann_data_list]
