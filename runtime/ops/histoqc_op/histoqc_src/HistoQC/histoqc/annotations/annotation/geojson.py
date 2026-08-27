from typing import List, Dict, Callable, Any  # Literal, get_args
from histoqc.import_wrapper.typing import Literal, get_args
from ..io_utils.json import load_json
from .base import Annotation, TYPE_POINT_SET, TYPE_RAW_LABEL, TYPE_POINT, TYPE_HOLED_SET

TYPE_GEO_MULTIPOLYGON = Literal['MultiPolygon']
TYPE_GEO_POLYGON = Literal['Polygon']
TYPE_GEO_LINE_STRING = Literal['LineString']

TYPE_GEOTYPE = Literal[TYPE_GEO_MULTIPOLYGON, TYPE_GEO_POLYGON, TYPE_GEO_LINE_STRING]


class GEOJsonAnnotation(Annotation[Dict]):
    PROP: str = "properties"
    CLASS: str = "classification"
    NAME: str = "name"

    """
    Parses a typical GeoJSON file containing one or more Polygon or MultiPolygon features.
    These JSON files are the preferred way to serialize QuPath annotations, for example.
    See https://qupath.readthedocs.io/en/latest/docs/scripting/overview.html#serialization-json
    """

    @staticmethod
    def point_set_helper_multipolygon(coordinates: List[List[TYPE_POINT_SET]]) -> List[TYPE_HOLED_SET]:
        out_list = []
        for roi in coordinates:
            out_list += GEOJsonAnnotation.point_set_helper_polygon(roi)
        return out_list

    @staticmethod
    def point_set_helper_polygon(coordinates: List[TYPE_POINT_SET]) -> List[TYPE_HOLED_SET]:
        inner_list = []
        outer = GEOJsonAnnotation._point_set_single(coordinates[0])
        inner_source = coordinates[1:]
        for points in inner_source:
            points: List[TYPE_POINT]
            inner_list.append(GEOJsonAnnotation._point_set_single(points))
        # if len(inner_list) == 0:
        #     inner_list = None
        holed: TYPE_HOLED_SET = outer, inner_list
        return [holed]

    @staticmethod
    def point_set_helper_lines(coordinates: List[TYPE_POINT]) -> List[TYPE_HOLED_SET]:
        holed_set: TYPE_HOLED_SET = GEOJsonAnnotation._point_set_single(coordinates), None
        return [holed_set]

    @staticmethod
    def _point_set_single(coordinates: List[TYPE_POINT]) -> List[TYPE_POINT]:
        return [Annotation.point_to_int((coord[0], coord[1])) for coord in coordinates]

    @staticmethod
    def _func_from_geom_type(geom_type: TYPE_GEOTYPE) -> Callable[[Any], List[TYPE_HOLED_SET]]:
        GEOMETRY_MAP: Dict[str, Callable[[Any], List[TYPE_HOLED_SET]]] = {
            get_args(TYPE_GEO_MULTIPOLYGON)[0]: GEOJsonAnnotation.point_set_helper_multipolygon,
            get_args(TYPE_GEO_POLYGON)[0]: GEOJsonAnnotation.point_set_helper_polygon,
            get_args(TYPE_GEO_LINE_STRING)[0]: GEOJsonAnnotation.point_set_helper_lines,
        }
        assert geom_type in GEOMETRY_MAP, f"Unsupported Geometry Type: {geom_type}"
        return GEOMETRY_MAP[geom_type]

    def point_set_list(self) -> List[TYPE_HOLED_SET]:
        geometry = self.ann_data['geometry']
        geom_type = geometry['type']
        coordinates = geometry['coordinates']
        func = GEOJsonAnnotation._func_from_geom_type(geom_type)
        return func(coordinates)

    def label_from_annotation(self) -> TYPE_RAW_LABEL:
        prop = self.ann_data.get(GEOJsonAnnotation.PROP)
        classification = prop.get(GEOJsonAnnotation.CLASS)
        if classification is not None:
            return classification.get(GEOJsonAnnotation.NAME)
        return None

    @staticmethod
    def annotation_list_from_uri(uri) -> List[Dict]:
        data = load_json(uri)
        if isinstance(data, Dict):
            return [data]
        assert isinstance(data, List)
        return data
