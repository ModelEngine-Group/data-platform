from typing import List, Dict, Union, Type, Tuple, Mapping  # Literal, get_args,
from types import MappingProxyType
# from shapely.strtree import STRtree
# from shapely.geometry import box as shapely_box
from histoqc.import_wrapper.typing import Literal, get_args
from lazy_property import LazyProperty
from .annotation.base import Annotation, Region, TYPE_RAW_LABEL
from .annotation.imagescope import ImageScopeAnnotation
from .annotation.geojson import GEOJsonAnnotation


TYPE_BBOX = Tuple[int, int, int, int]

TYPE_GEO = Literal["geojson"]
TYPE_IMAGESCOPE = Literal["imagescope"]
TYPE_JSON = Literal["json"]
TYPE_XML = Literal["xml"]

TYPE_SUPPORTED_PARSER = Literal[TYPE_GEO, TYPE_IMAGESCOPE, TYPE_JSON, TYPE_XML]

PARSER_BUILDER_MAP: Dict[str, Type[Annotation]] = {
    get_args(TYPE_GEO)[0]: GEOJsonAnnotation,
    get_args(TYPE_IMAGESCOPE)[0]: ImageScopeAnnotation,
    # for HistoQC
    get_args(TYPE_JSON)[0]: GEOJsonAnnotation,  # duplicate
    get_args(TYPE_XML)[0]: ImageScopeAnnotation,
}


class AnnotCollection:
    _annotation_list: List[Annotation]
    _label_to_regions_map: Mapping[TYPE_RAW_LABEL, List[Region]]

    @LazyProperty
    def all_regions(self) -> List[Region]:
        region_list = []
        for annotation in self._annotation_list:
            region_list += annotation.regions
        return region_list

    # @LazyProperty
    # def multipolygons(self) -> MultiPolygon:
    #     for annotation in self._annotation_list:

    def __init__(self, annotation_list: List[Annotation]):
        self._annotation_list = annotation_list
        self._label_to_regions_map = self._new_label_to_regions_map()

    @classmethod
    def build(cls, parser_type: TYPE_SUPPORTED_PARSER, uri: str, label_map: Union[Dict[Union[str, int], int], None]):
        construct = PARSER_BUILDER_MAP[parser_type]
        annotation_list = construct.build_from_uri(uri=uri, label_map=label_map)
        return cls(annotation_list)

    def _new_label_to_regions_map(self) -> Mapping[TYPE_RAW_LABEL, List[Region]]:
        out_dict: Dict[TYPE_RAW_LABEL, List[Region]] = dict()
        for region in self.all_regions:
            region: Region
            label: TYPE_RAW_LABEL = region['label']
            out_dict[label] = out_dict.get(label, [])
            out_dict[label].append(region)
        return MappingProxyType(out_dict)

    @property
    def label_to_regions_map(self) -> Mapping[TYPE_RAW_LABEL, List[Region]]:
        return self._label_to_regions_map
