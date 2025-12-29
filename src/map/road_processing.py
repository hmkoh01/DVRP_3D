"""Road layer parsing for visualization.

Loads the national spatial data shapefiles (road surface, centerline, boundary)
and converts them into a simple `RoadNetwork` structure that can be rendered by
the Panda3D visualizer. Coordinates are scaled into simulation space using the
same bounds/offsets as the building generator to keep everything aligned.
"""

from __future__ import annotations

import math
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure project root is on sys.path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import transform, unary_union, substring

import config

# Silence recurring EUC-KR -> UTF-8 conversion warnings from pyogrio when attributes contain
# non-decodable strings; geometry loading is unaffected.
warnings.filterwarnings(
    "ignore",
    message="One or several characters couldn't be converted correctly from EUC-KR to UTF-8.*",
    category=RuntimeWarning,
)

Point2D = Tuple[float, float]
Segment2D = Tuple[Point2D, Point2D]


@dataclass
class LaneMarking:
    """Lane line segments with style metadata."""
    segments: List[Segment2D]
    dashed: bool = False
    width: float = 0.2  # meters


@dataclass
class RoadPaintPatch:
    """Rectangular paint patches such as crosswalks or stop lines."""
    footprint: List[Point2D]
    style: str  # "crosswalk" or "stopline"
    tile_size: float  # UV repeat control for texturing
    height: float = 0.03  # small lift to avoid z-fighting


@dataclass
class RoadNetwork:
    """Collection of pre-scaled road geometries ready for rendering."""
    surfaces: List[List[Point2D]]
    sidewalks: List[List[Point2D]]
    boundaries: List[List[Point2D]]
    lane_markings: List[LaneMarking]
    paint_patches: List[RoadPaintPatch]

    def is_empty(self) -> bool:
        return not (self.surfaces or self.sidewalks or self.boundaries)

    def to_dict(self) -> Dict:
        """Serialize to a GeoJSON-like dictionary."""
        features = []
        for poly in self.surfaces:
            features.append({
                "type": "Feature",
                "properties": {"layer": "surface"},
                "geometry": {"type": "Polygon", "coordinates": [poly + [poly[0]]]},
            })
        for poly in self.sidewalks:
            features.append({
                "type": "Feature",
                "properties": {"layer": "sidewalk"},
                "geometry": {"type": "Polygon", "coordinates": [poly + [poly[0]]]},
            })
        for line in self.boundaries:
            features.append({
                "type": "Feature",
                "properties": {"layer": "boundary"},
                "geometry": {"type": "LineString", "coordinates": line},
            })
        for marking in self.lane_markings:
            for seg in marking.segments:
                features.append({
                    "type": "Feature",
                    "properties": {"layer": "lane_marking", "dashed": marking.dashed, "width": marking.width},
                    "geometry": {"type": "LineString", "coordinates": list(seg)},
                })
        for patch in self.paint_patches:
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "paint_patch",
                    "style": patch.style,
                    "tile_size": patch.tile_size,
                    "height": patch.height,
                },
                "geometry": {"type": "Polygon", "coordinates": [patch.footprint + [patch.footprint[0]]]},
            })
        return {"type": "FeatureCollection", "features": features}

    @staticmethod
    def from_dict(data: Dict) -> "RoadNetwork":
        """Deserialize from a GeoJSON-like dictionary."""
        features = data.get("features", [])
        surfaces: List[List[Point2D]] = []
        sidewalks: List[List[Point2D]] = []
        boundaries: List[List[Point2D]] = []
        lane_markings: List[LaneMarking] = []
        patches: List[RoadPaintPatch] = []

        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            layer = props.get("layer")

            if gtype == "Polygon" and coords:
                ring = coords[0]
                if not ring:
                    continue
                footprint = [(float(x), float(y)) for x, y in ring[:-1]]
                if layer == "surface":
                    surfaces.append(footprint)
                elif layer == "sidewalk":
                    sidewalks.append(footprint)
                elif layer == "paint_patch":
                    patches.append(
                        RoadPaintPatch(
                            footprint=footprint,
                            style=str(props.get("style", "crosswalk")),
                            tile_size=float(props.get("tile_size", 1.0)),
                            height=float(props.get("height", 0.03)),
                        )
                    )
            elif gtype == "LineString" and coords:
                line = [(float(x), float(y)) for x, y in coords]
                if layer == "boundary":
                    boundaries.append(line)
                elif layer == "lane_marking":
                    lane_markings.append(
                        LaneMarking(
                            segments=[(line[0], line[-1])],
                            dashed=bool(props.get("dashed", False)),
                            width=float(props.get("width", 0.2)),
                        )
                    )

        return RoadNetwork(
            surfaces=surfaces,
            sidewalks=sidewalks,
            boundaries=boundaries,
            lane_markings=lane_markings,
            paint_patches=patches,
        )


def _ensure_path(path_like: Path | str) -> Path:
    path = path_like if isinstance(path_like, Path) else Path(path_like)
    if not path.is_absolute():
        return config.PROJECT_ROOT / path
    return path


def _ensure_ccw(points: List[Point2D]) -> List[Point2D]:
    """Enforce CCW orientation for stable triangulation."""
    area2 = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    if area2 < 0:
        points = list(reversed(points))
    return points


def _flatten_polygons(geom) -> List[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    return []


def _flatten_lines(geom) -> List[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    return []


def _scale_geometry(
    geom, bounds: Tuple[float, float, float, float], scale_x: float, scale_z: float, offset_x: float, offset_z: float
):
    """Apply the same scaling/offset as building footprints."""
    min_x, _, min_z, _ = bounds

    def _transform(x, y, z=None):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        sx = offset_x + (x_arr - min_x) * scale_x
        sz = offset_z + (y_arr - min_z) * scale_z
        return (sx, sz)

    return transform(_transform, geom)


def _coords_from_polygon(poly: Polygon) -> Optional[List[Point2D]]:
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return None
    coords = coords[:-1]  # drop closing duplicate
    return _ensure_ccw([(float(x), float(y)) for x, y in coords])


def _line_to_segments(
    geom, dash_length: Optional[float] = None, gap_length: Optional[float] = None
) -> List[Segment2D]:
    """Break a line (or multilines) into drawable segments."""
    segments: List[Segment2D] = []
    for line in _flatten_lines(geom):
        if line.length <= 0:
            continue
        if dash_length and gap_length:
            step = max(dash_length + gap_length, 0.1)
            for start in np.arange(0, line.length, step):
                end = min(start + dash_length, line.length)
                if end <= start:
                    continue
                p1 = line.interpolate(start)
                p2 = line.interpolate(end)
                segments.append(((float(p1.x), float(p1.y)), (float(p2.x), float(p2.y))))
        else:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                segments.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return segments


def _line_end_segments(geom, end_length: float) -> List[LineString]:
    """Return short LineStrings representing the start and end portions of a line/multiline."""
    lines = _flatten_lines(geom)
    parts: List[LineString] = []
    if end_length <= 0:
        return parts
    for line in lines:
        if line.length <= 0:
            continue
        start_len = min(end_length, line.length)
        start_part = substring(line, 0, start_len, normalized=False)
        if start_part and not start_part.is_empty and start_part.length > 0:
            parts.append(start_part)
        if line.length > end_length:
            end_start = max(line.length - end_length, 0)
            end_part = substring(line, end_start, line.length, normalized=False)
            if end_part and not end_part.is_empty and end_part.length > 0:
                parts.append(end_part)
    return parts


def _estimate_half_width(line: LineString, boundaries: Sequence[LineString], fallback: float) -> float:
    """Approximate half width using distance to closest boundary polylines."""
    if not boundaries:
        return fallback
    distances: List[float] = []
    samples = max(3, int(line.length / 10))
    for t in np.linspace(0.1, 0.9, samples):
        pt = line.interpolate(line.length * t)
        d = min((b.distance(pt) for b in boundaries), default=0.0)
        if d > 0:
            distances.append(d)
    if not distances:
        return fallback
    return float(np.median(distances))


def _rect_along_line(
    line: LineString, width: float, depth: float, at_start: bool, offset: float
) -> Optional[List[Point2D]]:
    """Create a rectangle perpendicular to the line at its start/end."""
    if line.length <= 0:
        return None
    center_dist = offset + depth * 0.5 if at_start else max(line.length - offset - depth * 0.5, 0)
    center_dist = max(0.0, min(line.length, center_dist))
    center = line.interpolate(center_dist)
    back = line.interpolate(max(center_dist - 0.5, 0.0))
    fwd = line.interpolate(min(center_dist + 0.5, line.length))
    dx = fwd.x - back.x
    dy = fwd.y - back.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    tx, ty = dx / norm, dy / norm
    nx, ny = -ty, tx
    hw = width * 0.5
    hd = depth * 0.5
    cx, cy = center.x, center.y
    return [
        (cx + tx * hd + nx * hw, cy + ty * hd + ny * hw),
        (cx + tx * hd - nx * hw, cy + ty * hd - ny * hw),
        (cx - tx * hd - nx * hw, cy - ty * hd - ny * hw),
        (cx - tx * hd + nx * hw, cy - ty * hd + ny * hw),
    ]


def _has_neighbor(line: LineString, others: Sequence[LineString], at_start: bool, radius: float) -> bool:
    """Heuristic intersection detection based on endpoint proximity."""
    anchor = Point(line.coords[0] if at_start else line.coords[-1])
    for other in others:
        if other is line:
            continue
        if other.distance(anchor) < radius:
            return True
    return False


def _build_lane_markings(
    centerlines: Sequence[LineString],
    boundaries: Sequence[LineString],
    lane_width: float,
    dash_len: float,
    dash_gap: float,
    endpoint_len: float,
) -> List[LaneMarking]:
    markings: List[LaneMarking] = []
    for line in centerlines:
        if line.length < 1.0:
            continue
        half_width = _estimate_half_width(line, boundaries, fallback=lane_width)
        total_width = max(lane_width, half_width * 2)
        lane_count = max(1, int(round(total_width / lane_width)))
        lane_count = min(lane_count, 5)
        spacing = total_width / lane_count if lane_count else total_width
        offsets = [(-total_width * 0.5 + spacing * i) for i in range(1, lane_count)]
        if not offsets:
            offsets = [0.0]
        for offset in offsets:
            side = "left" if offset >= 0 else "right"
            dashed = abs(offset) < 0.05  # treat the true centerline as dashed/painted
            offset_line = line.parallel_offset(abs(offset), side, join_style=2)
            end_lines = _line_end_segments(offset_line, endpoint_len)
            segments: List[Segment2D] = []
            for seg_line in end_lines:
                segments.extend(
                    _line_to_segments(seg_line, dash_len if dashed else None, dash_gap if dashed else None)
                )
            if segments:
                markings.append(LaneMarking(segments=segments, dashed=dashed, width=0.18))
    return markings


def _build_paint_patches(
    centerlines: Sequence[LineString],
    boundaries: Sequence[LineString],
    lane_width: float,
    crosswalk_depth: float,
    stopline_depth: float,
) -> List[RoadPaintPatch]:
    # Crosswalk/stopline generation disabled per request.
    return []


def _load_layer(path: Path, encoding: str) -> gpd.GeoDataFrame:
    path = _ensure_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Road layer not found: {path}")
    return gpd.read_file(path, encoding=encoding)


def _load_bounds_from_buildings_geojson(path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Return bounds (min_x, max_x, min_z, max_z) from the building geojson if available."""
    path = _ensure_path(path)
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
        min_x, min_y, max_x, max_y = gdf.total_bounds
        return (float(min_x), float(max_x), float(min_y), float(max_y))
    except Exception:
        return None


def export_road_geojson(road_net: RoadNetwork, path: Path) -> Path:
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = road_net.to_dict()
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)
    return path


def load_road_geojson(path: Path) -> RoadNetwork:
    path = _ensure_path(path)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return RoadNetwork.from_dict(data)


def _load_terrain_bounds(paths: Sequence[Path]) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[str]]:
    """Load terrain contour shapefiles and return (bounds, crs)."""
    if not paths:
        return None, None
    try:
        gdf_list = []
        for p in paths:
            p = _ensure_path(p)
            if not p.exists():
                continue
            gdf_list.append(gpd.read_file(p, encoding="UTF-8"))
        if not gdf_list:
            return None, None
        terrain_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        min_x, min_y, max_x, max_y = terrain_gdf.total_bounds
        return (float(min_x), float(max_x), float(min_y), float(max_y)), terrain_gdf.crs
    except Exception as e:
        print(f"[road] 지형 데이터 로드 실패, 필터링 건너뜀: {e}")
        return None, None


def load_road_network(
    config_overrides: Optional[dict] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    scale_x: float = 1.0,
    scale_z: float = 1.0,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
    use_cache: Optional[bool] = None,
    write_cache: Optional[bool] = None,
) -> RoadNetwork:
    """Load road data, optionally using a cached GeoJSON for speed."""
    cfg = config.get_road_data_config(config_overrides)
    data_dir = _ensure_path(cfg["data_dir"])
    centerline_path = data_dir / cfg["centerline_filename"]
    boundary_path = data_dir / cfg["boundary_filename"]
    encoding = cfg.get("encoding", "EUC-KR")
    cache_path = cfg.get("cache_geojson_path")
    cache_path = _ensure_path(cache_path) if cache_path else None
    use_cache = cfg.get("use_cache", False) if use_cache is None else use_cache
    write_cache = cfg.get("write_cache", False) if write_cache is None else write_cache
    terrain_bounds, terrain_crs = _load_terrain_bounds(cfg.get("terrain_paths", []))

    if use_cache and cache_path and cache_path.exists():
        try:
            return load_road_geojson(cache_path)
        except Exception as e:
            print(f"[road] 캐시 GeoJSON 로드 실패, SHP 재생성 시도: {e}")

    centerline_gdf = _load_layer(centerline_path, encoding)
    boundary_gdf = _load_layer(boundary_path, encoding)

    target_crs = terrain_crs or centerline_gdf.crs or boundary_gdf.crs
    if target_crs:
        if centerline_gdf.crs != target_crs:
            centerline_gdf = centerline_gdf.to_crs(target_crs)
        if boundary_gdf.crs != target_crs:
            boundary_gdf = boundary_gdf.to_crs(target_crs)

    # Spatial clip to terrain bounds if available
    if terrain_bounds:
        min_x, max_x, min_y, max_y = terrain_bounds
        centerline_gdf = centerline_gdf.cx[min_x:max_x, min_y:max_y].copy()
        boundary_gdf = boundary_gdf.cx[min_x:max_x, min_y:max_y].copy()

    # Use surface bounds when not provided by caller
    if bounds is None:
        bounds = terrain_bounds or _load_bounds_from_buildings_geojson(cfg.get("bounds_from_buildings_geojson"))
    if bounds is None:
        if not centerline_gdf.empty:
            min_x, min_y, max_x, max_y = centerline_gdf.total_bounds
        elif not boundary_gdf.empty:
            min_x, min_y, max_x, max_y = boundary_gdf.total_bounds
        else:
            raise ValueError("No road geometries available to determine bounds.")
        bounds = (float(min_x), float(max_x), float(min_y), float(max_y))

    # Apply default margin if caller did not set offsets
    if offset_x == 0.0 and offset_z == 0.0:
        margin = cfg.get("margin_m", 0.0)
        offset_x = margin
        offset_z = margin

    # Scale to simulation coordinates
    centerline_gdf = centerline_gdf.copy()
    boundary_gdf = boundary_gdf.copy()
    centerline_gdf.geometry = centerline_gdf.geometry.apply(
        lambda g: _scale_geometry(g, bounds, scale_x, scale_z, offset_x, offset_z)
    )
    boundary_gdf.geometry = boundary_gdf.geometry.apply(
        lambda g: _scale_geometry(g, bounds, scale_x, scale_z, offset_x, offset_z)
    )

    boundaries: List[LineString] = []
    for geom in boundary_gdf.geometry:
        boundaries.extend(_flatten_lines(geom))
    centerlines: List[LineString] = []
    for geom in centerline_gdf.geometry:
        centerlines.extend(_flatten_lines(geom))

    # 면적 레이어(아스팔트, 보도) 제거: 선형 정보만 유지
    surfaces: List[List[Point2D]] = []
    sidewalks: List[List[Point2D]] = []

    lane_markings = _build_lane_markings(
        centerlines=centerlines,
        boundaries=boundaries,
        lane_width=cfg.get("lane_width_m", 3.25),
        dash_len=cfg.get("dash_length_m", 6.0),
        dash_gap=cfg.get("dash_gap_m", 6.0),
        endpoint_len=cfg.get("lane_endpoint_length_m", 12.0),
    )

    paint_patches = _build_paint_patches(
        centerlines=centerlines,
        boundaries=boundaries,
        lane_width=cfg.get("lane_width_m", 3.25),
        crosswalk_depth=cfg.get("crosswalk_depth_m", 3.5),
        stopline_depth=cfg.get("stopline_depth_m", 0.6),
    )

    boundary_coords: List[List[Point2D]] = []
    for line in boundaries:
        coords = [(float(x), float(y)) for x, y in line.coords]
        if len(coords) >= 2:
            boundary_coords.append(coords)

    road_net = RoadNetwork(
        surfaces=surfaces,
        sidewalks=sidewalks,
        boundaries=boundary_coords,
        lane_markings=lane_markings,
        paint_patches=paint_patches,
    )

    # Remove any paint patches (crosswalk/stopline) to satisfy current requirements
    road_net.paint_patches = []

    if write_cache and cache_path:
        try:
            export_road_geojson(road_net, cache_path)
            print(f"[road] 캐시 GeoJSON 저장: {cache_path}")
        except Exception as e:
            print(f"[road] 캐시 저장 실패: {e}")

    return road_net


def _network_bounds(net: RoadNetwork) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    zs: List[float] = []
    for line in net.boundaries:
        for x, z in line:
            xs.append(x)
            zs.append(z)
    for lm in net.lane_markings:
        for seg in lm.segments:
            (x1, z1), (x2, z2) = seg
            xs.extend([x1, x2])
            zs.extend([z1, z2])
    for patch in net.paint_patches:
        for x, z in patch.footprint:
            xs.append(x)
            zs.append(z)
    if not xs or not zs:
        return None
    return (min(xs), max(xs), min(zs), max(zs))


def visualize_road_2d(road_net: RoadNetwork, output_path: Path, dpi: int = 300):
    """Save a simple 2D PNG of the road network."""
    if road_net is None or road_net.is_empty():
        print("[road] 시각화 건너뜀: 데이터 없음")
        return
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    output_path = _ensure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bounds = _network_bounds(road_net)
    if bounds:
        min_x, max_x, min_z, max_z = bounds
        span_x = max_x - min_x
        span_z = max_z - min_z
    else:
        min_x = min_z = 0.0
        span_x = span_z = 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f7f7")

    # Boundaries/curbs
    for line in road_net.boundaries:
        if len(line) < 2:
            continue
        xs, zs = zip(*line)
        ax.plot(xs, zs, color="#b4b4b4", linewidth=1.0, alpha=0.9)

    # Lane markings
    for marking in road_net.lane_markings:
        if not marking.segments:
            continue
        for seg in marking.segments:
            (x1, z1), (x2, z2) = seg
            ax.plot([x1, x2], [z1, z2], color="#ffffff", linewidth=max(1.0, marking.width * 6), alpha=0.95)

    # Paint patches (crosswalks/stoplines)
    for patch in road_net.paint_patches:
        poly = MplPolygon(patch.footprint, closed=True, facecolor="#ffffff", edgecolor="#ffffff", alpha=0.9)
        ax.add_patch(poly)

    margin_x = span_x * 0.05 if span_x > 0 else 10
    margin_z = span_z * 0.05 if span_z > 0 else 10
    ax.set_xlim(min_x - margin_x, min_x + span_x + margin_x)
    ax.set_ylim(min_z - margin_z, min_z + span_z + margin_z)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Road Network (scaled to simulation coordinates)")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def generate_road_assets(config_overrides: Optional[dict] = None) -> RoadNetwork:
    """Preprocess road shapefiles into a cached GeoJSON for fast reuse."""
    cfg = config.get_road_data_config(config_overrides)
    cache_path = cfg.get("cache_geojson_path")
    output_2d = cfg.get("output_2d_filename")
    dpi_2d = cfg.get("dpi_2d", 300)
    margin = cfg.get("margin_m", 10.0)
    bounds = _load_bounds_from_buildings_geojson(cfg.get("bounds_from_buildings_geojson"))

    # Use same 1:1 scale + margin used by RealMapGenerator
    road_net = load_road_network(
        config_overrides={**(config_overrides or {}), "write_cache": False},
        bounds=bounds,
        scale_x=1.0,
        scale_z=1.0,
        offset_x=margin,
        offset_z=margin,
        use_cache=False,
        write_cache=False,
    )

    if cache_path:
        export_road_geojson(road_net, cache_path)
        print(f"[road] 전처리 완료 → {cache_path}")

    if output_2d:
        try:
            visualize_road_2d(road_net, output_2d, dpi=dpi_2d)
            print(f"[road] 2D 도로 맵 저장: {output_2d}")
        except Exception as e:
            print(f"[road] 2D 시각화 실패: {e}")

    print(
        f"[road] surfaces={len(road_net.surfaces)}, sidewalks={len(road_net.sidewalks)}, "
        f"lanes={len(road_net.lane_markings)}, patches={len(road_net.paint_patches)}"
    )
    return road_net


if __name__ == "__main__":
    generate_road_assets()
