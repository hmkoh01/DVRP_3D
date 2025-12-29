"""
Panda3D-based 3D Visualizer for DVRP Simulation

Coordinate System Note:
- Simulation/Ursina uses Y-up: (X=horizontal, Y=height, Z=depth)
- Panda3D uses Z-up: (X=horizontal, Y=depth, Z=height)
- This visualizer converts coordinates: sim(x, y, z) -> panda3d(x, z, y)
"""

import math
from typing import Dict, List, Optional, Tuple
from typing import Sequence as TypingSequence

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import ClockObject
from panda3d.core import (
    NodePath, GeomNode, Geom, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, GeomTriangles, GeomLines, Vec3, Vec4, Point3, LVector3f,
    AmbientLight, DirectionalLight, TextNode,
    CardMaker, Texture, TextureStage, PNMImage, WindowProperties, AntialiasAttrib,
    TransparencyAttrib, CollisionNode, CollisionBox, CollisionSphere, CollisionRay,
    CollisionTraverser, CollisionHandlerQueue, BitMask32,
    loadPrcFileData, LineSegs
)
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton, DirectFrame

import config
from src.models.entities import Map, Building, Depot, Drone, Motorbike, DroneStatus, EntityType, Position

# Type alias for vehicle
from typing import Union
Vehicle = Union[Drone, Motorbike]


# Configure Panda3D before ShowBase initialization
loadPrcFileData('', 'window-title DVRP 3D Simulation')
loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'sync-video false')


def _sim_to_panda3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert simulation coordinates (Y-up) to Panda3D coordinates (Z-up)
    
    Simulation: (X=horizontal, Y=height, Z=depth)
    Panda3D:    (X=horizontal, Y=depth, Z=height)
    """
    return (x, z, y)


def _sim_pos_to_panda3d(pos: Position) -> Tuple[float, float, float]:
    """Convert Position object to Panda3D coordinates"""
    return _sim_to_panda3d(pos.x, pos.y, pos.z)


def _polygon_area(points: TypingSequence[Tuple[float, float]]) -> float:
    """Calculate signed area of a 2D polygon (in X-Z plane of simulation)"""
    area = 0.0
    for i in range(len(points)):
        x1, z1 = points[i]
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return area / 2.0


def _ensure_ccw(points: TypingSequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ensure points are in counter-clockwise order"""
    pts = list(points)
    if _polygon_area(pts) < 0:
        pts.reverse()
    return pts


def _is_convex(a, b, c) -> bool:
    """Check if three points form a convex angle"""
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0


def _point_in_triangle(p, a, b, c) -> bool:
    """Check if point p is inside triangle abc using barycentric coordinates"""
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-8:
        return False
    w1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    w2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    w3 = 1 - w1 - w2
    return (0 < w1 < 1) and (0 < w2 < 1) and (0 < w3 < 1)


def _triangulate_polygon(points: TypingSequence[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    """Triangulate a polygon using ear clipping algorithm"""
    pts = _ensure_ccw(points)
    if len(pts) < 3:
        return []

    indices = list(range(len(pts)))
    triangles: List[Tuple[int, int, int]] = []
    guard = 0

    while len(indices) > 3 and guard < 2000:
        ear_found = False
        for i in range(len(indices)):
            prev_idx = indices[i - 1]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % len(indices)]

            a, b, c = pts[prev_idx], pts[curr_idx], pts[next_idx]
            if not _is_convex(a, b, c):
                continue

            if any(
                _point_in_triangle(pts[k], a, b, c)
                for k in indices
                if k not in (prev_idx, curr_idx, next_idx)
            ):
                continue

            triangles.append((prev_idx, curr_idx, next_idx))
            del indices[i]
            ear_found = True
            break

        if not ear_found:
            break
        guard += 1

    if len(indices) == 3:
        triangles.append(tuple(indices))

    if not triangles:
        for i in range(1, len(pts) - 1):
            triangles.append((0, i, i + 1))

    return triangles


def _build_prism_geom(points: TypingSequence[Tuple[float, float]], height: float, color: Vec4) -> Optional[GeomNode]:
    """Build a prism geometry from polygon base points
    
    Points are in simulation X-Z plane (horizontal ground plane).
    Height is along simulation Y axis (vertical).
    Output geometry is in Panda3D coordinates (Z-up).
    """
    if len(points) < 3 or height <= 0:
        return None

    pts = _ensure_ccw(points)
    n = len(pts)
    half_height = height / 2

    # Create vertex data format
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('prism', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Points are in simulation (x, z) which maps to Panda3D (x, y)
    # Height (simulation y) maps to Panda3D z
    
    # Bottom vertices (z = -half_height in Panda3D)
    for sim_x, sim_z in pts:
        # Panda3D: (x, y, z) = (sim_x, sim_z, -half_height)
        vertex_writer.add_data3(sim_x, sim_z, -half_height)
        color_writer.add_data4(color)
    
    # Top vertices (z = half_height in Panda3D)
    for sim_x, sim_z in pts:
        vertex_writer.add_data3(sim_x, sim_z, half_height)
        color_writer.add_data4(color)

    # Create triangles
    prim = GeomTriangles(Geom.UH_static)
    
    # Triangulate top and bottom faces
    tri_indices = _triangulate_polygon(pts)
    
    for a, b, c in tri_indices:
        # Top face (in Panda3D, looking down -Z)
        prim.add_vertices(a + n, c + n, b + n)
        # Bottom face
        prim.add_vertices(a, b, c)

    # Side faces
    for i in range(n):
        j = (i + 1) % n
        prim.add_vertices(i, j + n, j)
        prim.add_vertices(i, i + n, j + n)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('prism')
    node.add_geom(geom)
    
    return node


def _create_box_geom(width: float, height: float, depth: float, color: Vec4) -> GeomNode:
    """Create a simple box geometry in Panda3D coordinates
    
    Args:
        width: Size along X axis (simulation and Panda3D)
        height: Size along simulation Y axis (Panda3D Z)
        depth: Size along simulation Z axis (Panda3D Y)
    """
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('box', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Panda3D coordinates: width=X, depth=Y, height=Z
    hw = width / 2   # half width (X)
    hd = depth / 2   # half depth (Y in Panda3D, Z in simulation)
    hh = height / 2  # half height (Z in Panda3D, Y in simulation)
    
    # 8 vertices of a box in Panda3D coords (X, Y, Z)
    vertices = [
        (-hw, -hd, -hh), (hw, -hd, -hh), (hw, hd, -hh), (-hw, hd, -hh),  # bottom (Z-)
        (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh)       # top (Z+)
    ]
    
    for v in vertices:
        vertex_writer.add_data3(*v)
        color_writer.add_data4(color)

    prim = GeomTriangles(Geom.UH_static)
    
    # 6 faces (2 triangles each)
    faces = [
        (0, 2, 1), (0, 3, 2),  # bottom (Z-)
        (4, 5, 6), (4, 6, 7),  # top (Z+)
        (0, 1, 5), (0, 5, 4),  # front (Y-)
        (2, 3, 7), (2, 7, 6),  # back (Y+)
        (0, 4, 7), (0, 7, 3),  # left (X-)
        (1, 2, 6), (1, 6, 5),  # right (X+)
    ]
    
    for f in faces:
        prim.add_vertices(*f)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('box')
    node.add_geom(geom)
    
    return node


def _create_cylinder_geom(radius: float, height: float, segments: int, color: Vec4) -> GeomNode:
    """Create a cylinder geometry aligned with Panda3D Z axis (simulation Y)"""
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('cylinder', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    hh = height / 2  # half height along Z (Panda3D)
    
    # Bottom center
    vertex_writer.add_data3(0, 0, -hh)
    color_writer.add_data4(color)
    
    # Bottom ring
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertex_writer.add_data3(x, y, -hh)
        color_writer.add_data4(color)
    
    # Top center
    vertex_writer.add_data3(0, 0, hh)
    color_writer.add_data4(color)
    
    # Top ring
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertex_writer.add_data3(x, y, hh)
        color_writer.add_data4(color)

    prim = GeomTriangles(Geom.UH_static)
    
    # Bottom face
    for i in range(segments):
        next_i = (i + 1) % segments + 1
        prim.add_vertices(0, next_i, i + 1)
    
    # Top face
    top_center = segments + 1
    for i in range(segments):
        curr = top_center + 1 + i
        next_i = top_center + 1 + (i + 1) % segments
        prim.add_vertices(top_center, curr, next_i)
    
    # Side faces
    for i in range(segments):
        b1 = i + 1
        b2 = (i + 1) % segments + 1
        t1 = top_center + 1 + i
        t2 = top_center + 1 + (i + 1) % segments
        prim.add_vertices(b1, b2, t2)
        prim.add_vertices(b1, t2, t1)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('cylinder')
    node.add_geom(geom)
    
    return node


def _create_sphere_geom(radius: float, segments: int, rings: int, color: Vec4) -> GeomNode:
    """Create a sphere geometry"""
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('sphere', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Generate vertices - sphere aligned with Z-up
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vertex_writer.add_data3(x, y, z)
            color_writer.add_data4(color)

    prim = GeomTriangles(Geom.UH_static)
    
    # Generate triangles
    for i in range(rings):
        for j in range(segments):
            curr = i * segments + j
            next_j = i * segments + (j + 1) % segments
            below = (i + 1) * segments + j
            below_next = (i + 1) * segments + (j + 1) % segments
            
            if i != 0:
                prim.add_vertices(curr, below, next_j)
            if i != rings - 1:
                prim.add_vertices(next_j, below, below_next)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('sphere')
    node.add_geom(geom)
    
    return node


def _hash_noise(x: float, y: float) -> float:
    """Deterministic pseudo-random noise in [0, 1) for procedural textures"""
    return (math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0


def _make_window_texture(name: str, wall_color: Tuple[float, float, float],
                         window_color: Tuple[float, float, float],
                         frame_color: Tuple[float, float, float]) -> Texture:
    """Procedurally generate a simple windowed facade texture"""
    size = 128
    rows, cols = 6, 4
    img = PNMImage(size, size, 4)
    
    # Fill base wall with slight vertical gradient and noise
    for y in range(size):
        for x in range(size):
            grain = (_hash_noise(x * 0.7, y * 0.9) - 0.5) * 0.04
            vertical = (y / size) * 0.05
            r = max(0, min(1, wall_color[0] + grain + vertical))
            g = max(0, min(1, wall_color[1] + grain + vertical))
            b = max(0, min(1, wall_color[2] + grain + vertical))
            img.setXelA(x, y, r, g, b, 1.0)
    
    margin_x = size * 0.08
    margin_y = size * 0.12
    cell_w = (size - 2 * margin_x) / cols
    cell_h = (size - 2 * margin_y) / rows
    frame = max(1, int(min(cell_w, cell_h) * 0.08))
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            x0 = int(margin_x + c_idx * cell_w)
            x1 = int(x0 + cell_w)
            y0 = int(margin_y + r_idx * cell_h)
            y1 = int(y0 + cell_h)
            
            for y in range(y0, min(y1, size)):
                for x in range(x0, min(x1, size)):
                    is_frame = (
                        x - x0 < frame or x1 - x <= frame or
                        y - y0 < frame or y1 - y <= frame
                    )
                    if is_frame:
                        img.setXelA(x, y, frame_color[0], frame_color[1], frame_color[2], 1.0)
                    else:
                        sparkle = (_hash_noise(x * 1.7, y * 1.3) - 0.5) * 0.08
                        wr = max(0, min(1, window_color[0] + sparkle))
                        wg = max(0, min(1, window_color[1] + sparkle))
                        wb = max(0, min(1, window_color[2] + sparkle))
                        img.setXelA(x, y, wr, wg, wb, 1.0)
    
    tex = Texture(name)
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex




def _make_roof_texture() -> Texture:
    """Create a simple gravel/metal roof texture"""
    size = 128
    img = PNMImage(size, size, 3)
    
    base = (0.68, 0.68, 0.66)
    speck1 = (0.58, 0.58, 0.57)
    speck2 = (0.76, 0.76, 0.74)
    
    for y in range(size):
        for x in range(size):
            n1 = _hash_noise(x * 0.9, y * 1.1)
            n2 = _hash_noise((x + 17) * 1.4, (y + 3) * 0.6)
            blend = 0.65 + 0.35 * n1
            r = base[0] * blend + speck1[0] * (1 - blend)
            g = base[1] * blend + speck1[1] * (1 - blend)
            b = base[2] * blend + speck1[2] * (1 - blend)
            
            # Occasional lighter flecks
            if n2 > 0.82:
                r, g, b = speck2
            img.setXel(x, y, r, g, b)
    
    tex = Texture("roof_texture")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex


def _make_asphalt_texture() -> Texture:
    """Procedural dark asphalt with subtle speckling."""
    size = 128
    img = PNMImage(size, size, 3)
    base = (0.16, 0.16, 0.17)
    speck1 = (0.20, 0.20, 0.21)
    speck2 = (0.12, 0.12, 0.13)

    for y in range(size):
        for x in range(size):
            n1 = _hash_noise(x * 1.3, y * 1.1)
            n2 = _hash_noise((x + 7) * 0.6, (y + 5) * 0.9)
            r = base[0] + (speck1[0] - base[0]) * n1 + (speck2[0] - base[0]) * n2 * 0.4
            g = base[1] + (speck1[1] - base[1]) * n1 + (speck2[1] - base[1]) * n2 * 0.4
            b = base[2] + (speck1[2] - base[2]) * n1 + (speck2[2] - base[2]) * n2 * 0.4
            img.setXel(x, y, r, g, b)

    tex = Texture("asphalt_texture")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex


def _make_sidewalk_texture() -> Texture:
    """Simple tiled sidewalk pattern."""
    size = 128
    img = PNMImage(size, size, 3)
    grout = (0.70, 0.70, 0.70)
    tile = (0.78, 0.78, 0.79)
    tile_size = 16

    for y in range(size):
        for x in range(size):
            noise = (_hash_noise(x * 0.8, y * 0.8) - 0.5) * 0.03
            if x % tile_size < 2 or y % tile_size < 2:
                r, g, b = grout
            else:
                r = tile[0] + noise
                g = tile[1] + noise
                b = tile[2] + noise
            img.setXel(x, y, r, g, b)

    tex = Texture("sidewalk_texture")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex


def _make_grid_texture() -> Texture:
    """Create a subtle dark grid for synthetic maps."""
    size = 128
    img = PNMImage(size, size, 3)
    base = 0.1
    grid = 1
    gap = 20
    for y in range(size):
        for x in range(size):
            if x % gap == 0 or y % gap == 0:
                val = grid
            else:
                val = base
            img.setXel(x, y, val, val, val)
    tex = Texture("grid_texture")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex


def _build_facade_geom(
    start: Tuple[float, float],
    end: Tuple[float, float],
    height: float,
    color: Vec4,
    u_repeat: float,
    v_repeat: float,
    outward_offset: Tuple[float, float] = (0.0, 0.0)
) -> Optional[GeomNode]:
    """Build a textured quad for a building facade segment"""
    width = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    if width < 0.1 or height <= 0:
        return None
    
    vformat = GeomVertexFormat.get_v3c4t2()
    vdata = GeomVertexData('facade', vformat, Geom.UH_static)
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')
    tex_writer = GeomVertexWriter(vdata, 'texcoord')
    
    half_h = height / 2.0
    x1, z1 = start
    x2, z2 = end
    ox, oz = outward_offset
    x1 += ox
    x2 += ox
    z1 += oz
    z2 += oz
    
    vertices = [
        (x1, z1, -half_h),  # bottom start
        (x2, z2, -half_h),  # bottom end
        (x2, z2, half_h),   # top end
        (x1, z1, half_h),   # top start
    ]
    uvs = [
        (0.0, 0.0),
        (u_repeat, 0.0),
        (u_repeat, v_repeat),
        (0.0, v_repeat),
    ]
    
    for (vx, vy, vz), (u, v) in zip(vertices, uvs):
        vertex_writer.add_data3(vx, vy, vz)
        color_writer.add_data4(color)
        tex_writer.add_data2(u, v)
    
    prim = GeomTriangles(Geom.UH_static)
    prim.add_vertices(0, 1, 2)
    prim.add_vertices(0, 2, 3)
    
    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('facade')
    node.add_geom(geom)
    return node


def _build_roof_geom(
    points: TypingSequence[Tuple[float, float]],
    color: Vec4,
    tile_size: float
) -> Optional[GeomNode]:
    """Build a textured top surface for a footprint"""
    pts = _ensure_ccw(points)
    if len(pts) < 3:
        return None
    
    vformat = GeomVertexFormat.get_v3c4t2()
    vdata = GeomVertexData('roof', vformat, Geom.UH_static)
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')
    tex_writer = GeomVertexWriter(vdata, 'texcoord')
    
    for x, z in pts:
        u = x / tile_size
        v = z / tile_size
        vertex_writer.add_data3(x, z, 0.0)
        color_writer.add_data4(color)
        tex_writer.add_data2(u, v)
    
    prim = GeomTriangles(Geom.UH_static)
    for a, b, c in _triangulate_polygon(pts):
        prim.add_vertices(a, b, c)
    
    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('roof')
    node.add_geom(geom)
    return node


class Panda3DVisualizer(ShowBase):
    """3D visualization for DVRP simulation using Panda3D engine
    
    Handles coordinate conversion from simulation (Y-up) to Panda3D (Z-up).
    """
    
    def __init__(self, map_width: float = 1000, map_depth: float = 1000):
        """Initialize Panda3D visualizer
        
        Args:
            map_width: Width of the map (X-axis, same in both coordinate systems)
            map_depth: Depth of the map (simulation Z-axis, Panda3D Y-axis)
        """
        ShowBase.__init__(self)
        
        # Store map dimensions (in simulation coordinates - actual meters)
        self.map_width = map_width
        self.map_depth = map_depth
        self.map_size = max(map_width, map_depth)  # Reference size for scaling
        self.height_scale = getattr(config, "VISUALIZATION_HEIGHT_SCALE", 1.0)
        
        # Global visualization scale - makes everything bigger/smaller uniformly
        self.vis_scale = getattr(config, "VISUALIZATION_SCALE", 1.0)
        
        # Auto-calculate camera speed and distances based on map size
        # Base reference: 2000m map with speed 500
        self.map_scale_factor = self.map_size / 2000.0
        
        # Entity scale will be calculated based on average building size
        # This ensures depot/drone sizes are proportional to actual buildings
        self.entity_scale = 1.0  # Default, will be updated in create_map_entities
        
        # Fallback size_scale based on map dimensions (used for initial setup)
        self.size_scale = min(map_width, map_depth) / 2000.0
        
        # Enable antialiasing
        self.render.setAntialias(AntialiasAttrib.MAuto)
        
        # Create scene root node for global scaling
        self.scene_root = self.render.attachNewNode("scene_root")
        self.scene_root.setScale(self.vis_scale)
        self.ground = None
        self.terrain_node = None
        self.terrain_wireframe = None
        self.road_nodes: List[NodePath] = []
        
        # Pre-built textures/materials for the scene
        self.window_tile_size = 6.0  # meters per tile repeat for facades
        self.facade_textures = self._build_facade_textures()
        self.ground_texture = None
        self.ground_tile_size = None
        self.roof_texture = _make_roof_texture()
        self.roof_tile_size = 5.0  # meters per roof tile repeat
        self.asphalt_texture = _make_asphalt_texture()
        self.sidewalk_texture = _make_sidewalk_texture()
        self.grid_texture = _make_grid_texture()
        self.asphalt_tile_size = 5.0
        self.sidewalk_tile_size = 2.5
        
        # Vehicle visual tracking
        self.vehicle_color_targets: Dict[int, List[NodePath]] = {}
        self._drone_last_pos: Dict[int, Point3] = {}
        self.drone_buttons: Dict[int, DirectButton] = {}
        self.drone_button_frame: Optional[DirectFrame] = None
        self.vehicle_state_cache: Dict[int, Vehicle] = {}
        
        # Picking setup (mouse selection)
        self.pick_mask = BitMask32.bit(1)
        self.picker = CollisionTraverser('picker')
        self.pq = CollisionHandlerQueue()
        picker_node = CollisionNode('mouseRay')
        picker_node.setFromCollideMask(self.pick_mask)
        self.picker_ray = CollisionRay()
        picker_node.addSolid(self.picker_ray)
        self.picker_np = self.camera.attachNewNode(picker_node)
        self.picker.addCollider(self.picker_np, self.pq)
        
        # Tracking (third-person follow) state
        self.tracked_drone_id: Optional[int] = None
        self.tracked_heading: float = 0.0
        # Default offsets tuned for close chase view; adjust if needed
        self.follow_distance = 300.0   # meters behind the drone
        self.follow_height = 300.0     # meters above the drone
        self.follow_look_ahead = 8.0  # meters ahead along flight direction for look-at
        self.follow_lerp_speed = 4.0
        self.heading_lerp_speed = 3.5
        self.follow_yaw_offset = 0.0   # User-controlled yaw offset while tracking (degrees)
        self.follow_pitch_offset = -15.0  # User-controlled pitch offset while tracking (degrees)
        # Selection + monitor UI state
        self.selected_vehicle_id: Optional[int] = None
        self.vehicle_order_history: Dict[int, Dict[int, str]] = {}
        self.detail_frame: Optional[DirectFrame] = None
        self.detail_text: Optional[OnscreenText] = None
        
        print(f"Map size: {map_width:.0f}m x {map_depth:.0f}m (scale factor: {self.map_scale_factor:.2f})")
        
        # Setup camera
        self._setup_camera()
        
        # Create ground plane
        self._create_ground()
        
        # Setup lighting
        self._setup_lighting()
        
        # Create sky (background color)
        self.setBackgroundColor(0.53, 0.81, 0.92, 1)  # Sky blue
        
        # Entity storage
        self.building_nodes: List[NodePath] = []
        self.depot_nodes: List[NodePath] = []
        self.drone_nodes: Dict[int, NodePath] = {}
        self.drone_labels: Dict[int, NodePath] = {}
        self.drone_routes: Dict[int, NodePath] = {}  # Route visualization lines
        self.drone_colors: Dict[int, Vec4] = {}  # Track current drone colors for updates
        self.failure_markers: Dict[int, NodePath] = {}
        
        # Route visualization settings
        self.show_routes = True  # Toggle route visualization
        
        # Camera control state (in Panda3D coordinates, auto-scaled to map size)
        self._mouse_pressed = False
        self._last_mouse_pos = None
        self._camera_heading = 45
        self._camera_pitch = -30
        # Camera distance - start close for detail view
        self._camera_distance = self.map_size * 0.15 * self.vis_scale
        # Camera target in Panda3D coords: center of map at ground level
        self._camera_target = Point3(
            map_width/2 * self.vis_scale, 
            map_depth/2 * self.vis_scale, 
            0
        )
        
        # Setup input handling
        self._setup_input()
        
        # Update camera position
        self._update_camera_position()
        
        # Key state tracking for input handling
        self._key_states = {}
        
        # UI for drone selection
        self._setup_drone_button_panel()
        self._setup_monitor_panel()
        self.accept("aspectRatioChanged", self._layout_ui)
        self._layout_ui()
    
    def _build_facade_textures(self) -> Dict[str, Texture]:
        """Create reusable facade textures for each building type"""
        return {
            "store": _make_window_texture(
                "facade_store",
                wall_color=(0.75, 0.86, 0.78),
                window_color=(0.50, 0.68, 0.82),
                frame_color=(0.18, 0.35, 0.22),
            ),
            "customer": _make_window_texture(
                "facade_customer",
                wall_color=(0.86, 0.80, 0.80),
                window_color=(0.52, 0.64, 0.80),
                frame_color=(0.35, 0.20, 0.20),
            ),
            "neutral": _make_window_texture(
                "facade_neutral",
                wall_color=(0.82, 0.82, 0.82),
                window_color=(0.56, 0.70, 0.86),
                frame_color=(0.30, 0.30, 0.30),
            ),
        }
        
    def _setup_camera(self):
        """Setup camera for 3D view"""
        # Disable default mouse camera control
        self.disableMouse()
        
        # Set initial camera position (Panda3D coords)
        self.camera.setPos(self.map_width/2, -500, 500)
        self.camera.lookAt(self.map_width/2, self.map_depth/2, 0)
        
    def _setup_input(self):
        """Setup keyboard and mouse input handling"""
        # Mouse button events
        self.accept('mouse1', self._on_left_click)
        self.accept('mouse2', self._on_mouse_press, ['middle'])
        self.accept('mouse2-up', self._on_mouse_release, ['middle'])
        self.accept('mouse3', self._on_mouse_press, ['right'])
        self.accept('mouse3-up', self._on_mouse_release, ['right'])
        
        # Scroll wheel
        self.accept('wheel_up', self._on_scroll, [-50])
        self.accept('wheel_down', self._on_scroll, [50])
        
        # Keyboard
        self.accept('escape', self._on_escape)
        self.accept('h', self._toggle_help)
        self.accept('v', self.toggle_route_visualization)  # Toggle route lines
        
        # Camera movement keys
        for key in ['w', 'a', 's', 'd', 'q', 'e']:
            self.accept(key, self._set_key_state, [key, True])
            self.accept(f'{key}-up', self._set_key_state, [key, False])
        
        # Add task for continuous camera movement
        self.taskMgr.add(self._camera_move_task, 'camera_move_task')
        self.taskMgr.add(self._mouse_look_task, 'mouse_look_task')
        self.taskMgr.add(self._camera_follow_task, 'camera_follow_task')
    
    def _setup_drone_button_panel(self):
        """Create a small UI panel with buttons to track active drones."""
        self.drone_button_frame = DirectFrame(
            frameColor=(0, 0, 0, 0.35),
            frameSize=(-0.3, 0.3, -0.55, 0.55),
            pos=(1.15, 0, 0.0),
        )
        self._refresh_drone_buttons([])

    def _setup_monitor_panel(self):
        """Side panel that shows selected vehicle status (orders + battery)."""
        self.detail_frame = DirectFrame(
            frameColor=(0, 0, 0, 0.42),
            frameSize=(-0.5, 0.5, -0.6, 0.6),
            pos=(0.0, 0, 0.0)
        )
        self.detail_text = OnscreenText(
            parent=self.detail_frame,
            text="Select a vehicle",
            pos=(0, 0.48),
            scale=0.045,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter,
            mayChange=True
        )
        self._hide_detail_panel()
    
    def _clear_drone_buttons(self):
        for btn in self.drone_buttons.values():
            try:
                btn.destroy()
            except Exception:
                pass
        self.drone_buttons.clear()
    
    def _refresh_drone_buttons(self, drones: List[Vehicle]):
        """Refresh the vehicle selection buttons based on active drones/motorbikes."""
        if self.drone_button_frame is None:
            return
        
        # Show vehicles (drones or motorbikes) that are active (not idle/loading)
        active = [
            d for d in drones
            if isinstance(d, (Drone, Motorbike)) and d.status not in (DroneStatus.IDLE, DroneStatus.LOADING)
        ]
        active_ids = {d.id for d in active}
        if active_ids == set(self.drone_buttons.keys()):
            # Update labels if status changed
            for d in active:
                btn = self.drone_buttons.get(d.id)
                if btn:
                    prefix = "D" if isinstance(d, Drone) else "M"
                    btn["text"] = f"{prefix}{d.id} {d.status.name}"
            return
        
        # Rebuild buttons
        self._clear_drone_buttons()
        max_buttons = 12
        for i, drone in enumerate(sorted(active, key=lambda d: d.id)[:max_buttons]):
            y = 0.48 - i * 0.085
            prefix = "D" if isinstance(drone, Drone) else "M"
            btn = DirectButton(
                parent=self.drone_button_frame,
                text=f"{prefix}{drone.id} {drone.status.name}",
                text_scale=0.04,
                text_fg=(1, 1, 1, 1),
                frameColor=(0.18, 0.18, 0.2, 0.75),
                relief=1,
                pos=(0, 0, y),
                command=self._on_vehicle_button,
                extraArgs=[drone.id],
                pressEffect=True
            )
            self.drone_buttons[drone.id] = btn

    def _layout_ui(self):
        """Keep UI panels inside the viewport when aspect ratio changes."""
        aspect = self.getAspectRatio()
        margin = 0.07
        if self.drone_button_frame:
            frame = self.drone_button_frame["frameSize"]
            width = abs(frame[1])
            self.drone_button_frame.setPos(aspect - width - margin, 0, 0.0)
        if self.detail_frame:
            self.detail_frame.setPos(0.0, 0.0, 0.0)

    def _update_selected_vehicle_panel(self):
        """Update side panel with selected vehicle info."""
        if self.detail_text is None or self.detail_frame is None:
            return
        if self.detail_frame.isHidden():
            return
        vid = self.selected_vehicle_id
        if vid is None:
            self.detail_text.setText("Select a vehicle")
            return

        vehicle = self.vehicle_state_cache.get(vid)
        if vehicle is None:
            self.detail_text.setText("Vehicle inactive")
            return

        is_motorbike = isinstance(vehicle, Motorbike)
        v_prefix = "Motorbike" if is_motorbike else "Drone"
        battery_pct = vehicle.battery_level * 100 if not is_motorbike else None

        # Track order history to keep completed items visible briefly
        history = self.vehicle_order_history.setdefault(vid, {})
        active_orders = []
        if getattr(vehicle, "current_orders", None):
            active_orders.extend(vehicle.current_orders)
        if getattr(vehicle, "current_order", None):
            active_orders.append(vehicle.current_order)

        for order in active_orders:
            history[order.id] = order

        # Keep only recent 8 entries
        if len(history) > 8:
            for key in sorted(history.keys())[:-8]:
                history.pop(key, None)

        if history:
            order_lines = "\n".join(
                [
                    f"- #{oid} "
                    f"(C{getattr(order, 'customer_id', '?')} - "
                    f"S{getattr(order, 'store_id', '?')}): "
                    f"{getattr(order.status, 'name', getattr(order, 'status', 'UNKNOWN'))}"
                    for oid, order in sorted(history.items(), reverse=True)
                ]
            )
        else:
            order_lines = "- None"

        if is_motorbike:
            battery_line = "Battery: N/A (motorbike)"
        else:
            battery_line = f"Battery: {battery_pct:.0f}%"

        text = (
            f"{v_prefix} {vid}\n"
            f"Status: {vehicle.status.name}\n"
            f"{battery_line}\n\n"
            f"Orders:\n{order_lines}"
        )
        self.detail_text.setText(text)

    def _hide_detail_panel(self):
        if self.detail_frame:
            self.detail_frame.hide()
        if self.detail_text:
            self.detail_text.setText("Select a vehicle")

    def _show_detail_panel(self):
        if self.detail_frame:
            self.detail_frame.show()
        
    def _set_key_state(self, key, state):
        """Track key press state"""
        self._key_states[key] = state
        
    def _camera_move_task(self, task):
        """Task to handle continuous camera movement"""
        if self.tracked_drone_id is not None:
            return Task.cont  # Disable manual move while following
        
        dt = ClockObject.getGlobalClock().getDt()
        # Speed scales with map size - reduced for finer control
        base_speed = getattr(config, "CAMERA_MOVE_SPEED", 500.0)
        speed = base_speed * self.map_scale_factor * self.vis_scale * dt * 0.3
        
        # Calculate forward and right vectors based on camera heading
        heading_rad = math.radians(self._camera_heading)
        
        # Forward vector (direction camera is looking in X-Y plane)
        forward_x = -math.sin(heading_rad)
        forward_y = math.cos(heading_rad)
        
        # Right vector (perpendicular to forward)
        right_x = math.cos(heading_rad)
        right_y = math.sin(heading_rad)
        
        # W/S: Move forward/backward relative to camera view
        if self._key_states.get('w'):
            self._camera_target.x += forward_x * speed
            self._camera_target.y += forward_y * speed
        if self._key_states.get('s'):
            self._camera_target.x -= forward_x * speed
            self._camera_target.y -= forward_y * speed
            
        # A/D: Move left/right relative to camera view (swapped direction)
        if self._key_states.get('a'):
            self._camera_target.x -= right_x * speed
            self._camera_target.y -= right_y * speed
        if self._key_states.get('d'):
            self._camera_target.x += right_x * speed
            self._camera_target.y += right_y * speed
            
        # Q/E: Move down/up (along Panda3D Z axis)
        if self._key_states.get('q'):
            self._camera_target.z -= speed
        if self._key_states.get('e'):
            self._camera_target.z += speed
            
        if any(self._key_states.get(k) for k in ['w', 'a', 's', 'd', 'q', 'e']):
            self._update_camera_position()
            
        return Task.cont
    
    def _camera_follow_task(self, task):
        """Smooth third-person follow camera for selected drone"""
        if self.tracked_drone_id is None:
            return Task.cont
        
        node = self.drone_nodes.get(self.tracked_drone_id)
        if node is None or node.isEmpty():
            self._stop_tracking()
            return Task.cont
        
        dt = ClockObject.getGlobalClock().getDt()
        current_pos = node.getPos()
        prev_pos = self._drone_last_pos.get(self.tracked_drone_id, current_pos)
        move_vec = current_pos - prev_pos
        flat_len = math.hypot(move_vec.x, move_vec.y)
        
        # Determine heading: prefer route direction, fallback to movement, else keep previous
        heading_deg = self.tracked_heading
        next_dir = self._get_next_waypoint_direction(self.tracked_drone_id, current_pos)
        if next_dir is not None:
            heading_deg = next_dir
        elif flat_len > 0.5:
            heading_deg = math.degrees(math.atan2(move_vec.y, move_vec.x))
        # Smooth heading
        heading_alpha = min(1.0, self.heading_lerp_speed * dt)
        self.tracked_heading = (1 - heading_alpha) * self.tracked_heading + heading_alpha * heading_deg
        
        heading_rad = math.radians(self.tracked_heading)
        forward = Vec3(math.cos(heading_rad + math.radians(self.follow_yaw_offset)), math.sin(heading_rad + math.radians(self.follow_yaw_offset)), 0)
        pitch_rad = math.radians(self.follow_pitch_offset)
        horiz_dist = self.follow_distance * self.entity_scale * math.cos(pitch_rad)
        vertical = self.follow_height * self.entity_scale + self.follow_distance * self.entity_scale * math.sin(pitch_rad)
        
        # Desired camera and target positions
        desired_target = self._get_forward_look_target(current_pos, forward)
        desired_cam = current_pos - forward * horiz_dist
        desired_cam.z += vertical
        
        # Smoothly interpolate camera position and look-at target
        lerp_alpha = min(1.0, self.follow_lerp_speed * dt)
        cam_pos = self.camera.getPos()
        new_cam_pos = cam_pos + (desired_cam - cam_pos) * lerp_alpha
        new_target = Point3(
            self._camera_target.x + (desired_target.x - self._camera_target.x) * lerp_alpha,
            self._camera_target.y + (desired_target.y - self._camera_target.y) * lerp_alpha,
            self._camera_target.z + (desired_target.z - self._camera_target.z) * lerp_alpha,
        )
        
        self._camera_target = new_target
        self.camera.setPos(new_cam_pos)
        self.camera.lookAt(new_target)
        self._drone_last_pos[self.tracked_drone_id] = Point3(current_pos)
        
        return Task.cont
    
    def _get_next_waypoint_direction(self, drone_id: int, current_pos: Point3) -> Optional[float]:
        """Return heading (degrees) toward the next waypoint if available."""
        vehicle = self.vehicle_state_cache.get(drone_id)
        if vehicle and getattr(vehicle, "route", None):
            wp = vehicle.route[0]
            wp_panda = _sim_pos_to_panda3d(wp)
            dx = wp_panda[0] - current_pos.x
            dy = wp_panda[1] - current_pos.y
            if abs(dx) + abs(dy) > 1e-3:
                return math.degrees(math.atan2(dy, dx))
        return None
    
    def _get_forward_look_target(self, current_pos: Point3, forward: Vec3) -> Point3:
        """Look ahead along flight direction so camera aim follows the path."""
        ahead = current_pos + forward * (self.follow_look_ahead * self.entity_scale)
        ahead.z = max(ahead.z, current_pos.z + 0.5 * self.entity_scale)
        return Point3(ahead)
    
    def _mouse_look_task(self, task):
        """Task to handle mouse look (camera rotation)"""
        if not self._mouse_pressed:
            self._last_mouse_pos = None
            return Task.cont
            
        if not self.mouseWatcherNode.hasMouse():
            return Task.cont
            
        mouse_x = self.mouseWatcherNode.getMouseX()
        mouse_y = self.mouseWatcherNode.getMouseY()
        
        if self._last_mouse_pos is not None:
            dx = mouse_x - self._last_mouse_pos[0]
            dy = mouse_y - self._last_mouse_pos[1]
            if self.tracked_drone_id is not None:
                # While tracking, adjust camera yaw/pitch offset around the tracked target
                self.follow_yaw_offset -= dx * 120  # drag right -> rotate right
                self.follow_pitch_offset = max(-75, min(45, self.follow_pitch_offset - dy * 90))
            else:
                # Rotate camera - adjusted sensitivity and corrected direction
                # Horizontal mouse movement rotates around Z axis (heading)
                # Vertical mouse movement changes pitch (looking up/down)
                self._camera_heading -= dx * 150  # Drag right = view moves right
                self._camera_pitch = max(-85, min(-5, self._camera_pitch - dy * 100))  # Positive dy = look up
                
                self._update_camera_position()
            
        self._last_mouse_pos = (mouse_x, mouse_y)
        
        return Task.cont
        
    def _on_mouse_press(self, button):
        """Handle mouse button press"""
        if button in ['middle', 'right']:
            self._mouse_pressed = True
            
    def _on_mouse_release(self, button):
        """Handle mouse button release"""
        if button in ['middle', 'right']:
            self._mouse_pressed = False
            self._last_mouse_pos = None
    
    def _on_left_click(self):
        """Handle left click for selecting a drone (start/stop tracking)"""
        self._pick_drone_at_cursor()
            
    def _on_scroll(self, delta):
        """Handle mouse scroll wheel"""
        # Zoom speed and range scale with map size
        scaled_delta = delta * self.map_scale_factor * self.vis_scale * 0.5  # Slower zoom for precision
        if self.tracked_drone_id is None:
            min_dist = 5 * self.vis_scale  # Can zoom in extremely close (5m)
            max_dist = self.map_size * 1.5 * self.vis_scale  # Can zoom out to see full map
            self._camera_distance = max(min_dist, min(max_dist, self._camera_distance + scaled_delta))
            self._update_camera_position()
        else:
            # While tracking, zoom by adjusting follow distance/height together
            min_follow = 5 * self.entity_scale
            max_follow = self.map_size * 1.0 * self.vis_scale
            new_dist = max(min_follow, min(max_follow, self.follow_distance + scaled_delta))
            if new_dist != self.follow_distance:
                ratio = new_dist / max(self.follow_distance, 1e-3)
                self.follow_distance = new_dist
                # Keep height roughly proportional to distance to maintain angle
                self.follow_height = max(2 * self.entity_scale, self.follow_height * ratio)
        
    def _on_escape(self):
        """Handle escape key"""
        if self.tracked_drone_id is not None:
            self._stop_tracking()
        else:
            self.userExit()
        
    def _toggle_help(self):
        """Toggle help text visibility"""
        if hasattr(self, 'help_text') and self.help_text:
            if self.help_text.isHidden():
                self.help_text.show()
            else:
                self.help_text.hide()
                
    def _update_camera_position(self):
        """Update camera position based on orbit parameters (Panda3D Z-up)"""
        heading_rad = math.radians(self._camera_heading)
        pitch_rad = math.radians(self._camera_pitch)
        
        # Calculate camera position in Panda3D coords
        # Orbit around target in X-Y plane, with Z as height
        x = self._camera_target.x + self._camera_distance * math.sin(heading_rad) * math.cos(pitch_rad)
        y = self._camera_target.y - self._camera_distance * math.cos(heading_rad) * math.cos(pitch_rad)
        z = self._camera_target.z + self._camera_distance * math.sin(-pitch_rad)
        
        self.camera.setPos(x, y, z)
        self.camera.lookAt(self._camera_target)
    
    def _pick_drone_at_cursor(self):
        """Cast a ray from the mouse cursor to select a drone; clicking empty space clears tracking."""
        if not self.mouseWatcherNode.hasMouse():
            return
        
        mpos = self.mouseWatcherNode.getMouse()
        self.picker_ray.setFromLens(self.camNode, mpos.getX(), mpos.getY())
        self.picker.traverse(self.scene_root)
        
        if self.pq.getNumEntries() == 0:
            self._stop_tracking()
            return
        
        self.pq.sortEntries()
        for entry in self.pq.getEntries():
            np = entry.getIntoNodePath()
            tagged = np.findNetTag('drone_id')
            if not tagged.isEmpty():
                try:
                    drone_id = int(tagged.getNetTag('drone_id'))
                except ValueError:
                    continue
                self._start_tracking(drone_id)
                return
        
        # No drone hit
        self._stop_tracking()
    
    def _on_vehicle_button(self, drone_id: int):
        """Toggle selection: same id hides, new id shows panel and tracking."""
        if self.selected_vehicle_id == drone_id:
            self._hide_detail_panel()
            self.selected_vehicle_id = None
            self.tracked_drone_id = None
            print("Selection cleared")
            return
        if drone_id not in self.drone_nodes:
            return
        self.selected_vehicle_id = drone_id
        self.tracked_drone_id = drone_id
        self.tracked_heading = self._camera_heading
        self._show_detail_panel()
        print(f"Tracking drone {drone_id}")
    
    def _stop_tracking(self):
        """Return to free camera mode"""
        if self.tracked_drone_id is not None:
            print("Tracking cleared")
        self.tracked_drone_id = None
        self.follow_yaw_offset = 0.0
        self.follow_pitch_offset = -15.0
        
    def _create_ground(self):
        """Create ground plane (in Panda3D X-Y plane at Z=0)"""
        if self.ground is not None:
            self.ground.removeNode()
        # Create a large flat plane using CardMaker
        cm = CardMaker('ground')
        cm.setFrame(0, self.map_width, 0, self.map_depth)
        
        ground_node = self.scene_root.attachNewNode(cm.generate())
        # Card is in X-Y plane by default in Panda3D, which is correct for Z-up
        ground_node.setP(-90)  # Rotate to be horizontal (X-Y plane)
        ground_node.setPos(0, 0, 0)
        ground_node.setColor(0.15, 0.15, 0.16, 1)  # Neutral dark asphalt tone
        ground_node.setTextureOff(1)
        
        self.ground = ground_node

    def _clear_roads(self):
        for node in self.road_nodes:
            try:
                node.removeNode()
            except Exception:
                pass
        self.road_nodes.clear()

    def _refresh_terrain(self, map_data: Map):
        """Terrain mesh rendering disabled; always use flat ground."""
        for node in [self.terrain_node, self.terrain_wireframe]:
            if node:
                node.removeNode()
        self.terrain_node = None
        self.terrain_wireframe = None
        self._create_ground_from_map(map_data)

    def _create_ground_from_map(self, map_data: Map):
        """Create ground using map boundary polygon when available, else rectangle."""
        if self.ground is not None:
            self.ground.removeNode()

        boundary = getattr(map_data, "boundary_polygon", None)
        if boundary:
            geom = _build_roof_geom(boundary, Vec4(0.15, 0.15, 0.16, 1), self.asphalt_tile_size)
            if geom:
                node = self.scene_root.attachNewNode(geom)
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setDepthOffset(-5)
                self.ground = node
                return

        # Fallback rectangle
        self._create_ground()

    def _render_roads(self, map_data: Map):
        """Render textured road surfaces, sidewalks, lane lines, and crosswalks."""
        self._clear_roads()
        road_net = getattr(map_data, "road_network", None)
        if not road_net:
            return
        if hasattr(road_net, "is_empty") and road_net.is_empty():
            return

        base_height = 0.02

        # Surfaces (asphalt)
        for poly in road_net.surfaces:
            geom = _build_roof_geom(poly, Vec4(0.22, 0.22, 0.23, 1), self.asphalt_tile_size)
            if geom is None:
                continue
            node = self.scene_root.attachNewNode(geom)
            node.setTexture(self.asphalt_texture)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthOffset(1)
            node.setZ(base_height)
            self.road_nodes.append(node)

        # Sidewalk bands buffered from road surfaces
        for poly in road_net.sidewalks:
            geom = _build_roof_geom(poly, Vec4(0.82, 0.82, 0.82, 1), self.sidewalk_tile_size)
            if geom is None:
                continue
            node = self.scene_root.attachNewNode(geom)
            node.setTexture(self.sidewalk_texture)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthOffset(0)
            node.setZ(base_height + 0.01)
            self.road_nodes.append(node)

        # Curb/boundary lines
        for boundary in road_net.boundaries:
            if len(boundary) < 2:
                continue
            segs = LineSegs()
            segs.setThickness(2.0)
            segs.setColor(0.7, 0.7, 0.7, 0.9)
            for a, b in zip(boundary[:-1], boundary[1:]):
                segs.moveTo(a[0], a[1], base_height + 0.03)
                segs.drawTo(b[0], b[1], base_height + 0.03)
            node = self.scene_root.attachNewNode(segs.create())
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthOffset(3)
            self.road_nodes.append(node)

        # Lane markings (solid + dashed)
        for marking in road_net.lane_markings:
            if not marking.segments:
                continue
            segs = LineSegs()
            segs.setThickness(max(1.5, marking.width * 10.0))
            segs.setColor(1, 1, 1, 0.95)
            for seg in marking.segments:
                (x1, z1), (x2, z2) = seg
                segs.moveTo(x1, z1, base_height + 0.05)
                segs.drawTo(x2, z2, base_height + 0.05)
            node = self.scene_root.attachNewNode(segs.create())
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthOffset(4)
            self.road_nodes.append(node)

        def _patch_axes(footprint: TypingSequence[Tuple[float, float]]):
            if len(footprint) < 4:
                return None
            p0, p1, p2, p3 = footprint[:4]
            center = (np.array(p0) + np.array(p2)) * 0.5
            normal_vec = np.array(p1) - np.array(p0)
            tangent_vec = np.array(p0) - np.array(p3)
            width = np.linalg.norm(normal_vec)
            depth = np.linalg.norm(tangent_vec)
            if width < 1e-3 or depth < 1e-3:
                return None
            return center, normal_vec / width, tangent_vec / depth, width, depth

        # Crosswalks and stop lines
        for patch in road_net.paint_patches:
            axes = _patch_axes(patch.footprint)
            if not axes:
                continue
            center, n_hat, t_hat, width, depth = axes
            if patch.style == "crosswalk":
                stripe_w = patch.tile_size
                gap = stripe_w * 0.8
                step = max(0.2, stripe_w + gap)
                count = max(1, int(width // step))
                start = -width * 0.5 + stripe_w * 0.5
                for i in range(count):
                    offset = start + i * step
                    stripe_center = center + n_hat * offset
                    hw = stripe_w * 0.5
                    hd = depth * 0.5
                    corners = [
                        stripe_center + t_hat * hd + n_hat * hw,
                        stripe_center + t_hat * hd - n_hat * hw,
                        stripe_center - t_hat * hd - n_hat * hw,
                        stripe_center - t_hat * hd + n_hat * hw,
                    ]
                    geom = _build_roof_geom(
                        [(float(c[0]), float(c[1])) for c in corners],
                        Vec4(1, 1, 1, 0.92),
                        max(stripe_w, depth)
                    )
                    if geom:
                        node = self.scene_root.attachNewNode(geom)
                        node.setTransparency(TransparencyAttrib.MAlpha)
                        node.setDepthOffset(5)
                        node.setZ(base_height + patch.height)
                        self.road_nodes.append(node)
            else:
                geom = _build_roof_geom(
                    [(float(x), float(z)) for x, z in patch.footprint],
                    Vec4(1, 1, 1, 0.95),
                    patch.tile_size
                )
                if geom:
                    node = self.scene_root.attachNewNode(geom)
                    node.setTransparency(TransparencyAttrib.MAlpha)
                    node.setDepthOffset(5)
                    node.setZ(base_height + patch.height)
                    self.road_nodes.append(node)

    def _apply_ground_style(self, map_data: Map):
        """Apply base ground styling. For random maps (no road network), use a grid texture."""
        if self.ground is None:
            return
        road_net = getattr(map_data, "road_network", None)
        if road_net:
            # Use plain ground when real road data exists
            self.ground.setTextureOff(1)
            self.ground.setColor(0.15, 0.15, 0.16, 1)
        else:
            # Random map: apply subtle grid to suggest roads
            self.ground.setTexture(self.grid_texture, 1)
            tile = max(1.0, self.map_size / 800.0)  # fewer repeats => larger grid spacing
            self.ground.setTexScale(TextureStage.getDefault(), tile, tile)
            self.ground.setColor(1, 1, 1, 1)

    def _setup_lighting(self):
        """Setup scene lighting"""
        # Ambient light
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.4, 0.4, 0.4, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        
        # Directional light (sun) - shining from above and side
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(0.8, 0.8, 0.8, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self.render.setLight(sun_np)
        
        self.ambient_light = ambient_np
        self.sun_light = sun_np
        
    def _format_drone_label(self, vehicle: Vehicle) -> str:
        """Format vehicle (drone or motorbike) label text with status and service time info"""
        is_motorbike = isinstance(vehicle, Motorbike)
        prefix = "M" if is_motorbike else "D"  # M for Motorbike, D for Drone
        
        # Show service status if applicable (using ASCII text instead of emojis for Panda3D compatibility)
        service_time = getattr(vehicle, '_service_time_remaining', 0.0)
        
        # For drones, show battery. For motorbikes, show floor info during service
        if is_motorbike:
            floor_num = getattr(vehicle, '_current_target_floor', 0)
            
            if vehicle.status == DroneStatus.PICKING_UP and service_time > 0:
                return f"{prefix}{vehicle.id}\n[PICKUP F{floor_num}] {service_time:.0f}s"
            elif vehicle.status == DroneStatus.DROPPING_OFF and service_time > 0:
                return f"{prefix}{vehicle.id}\n[DELIVER F{floor_num}] {service_time:.0f}s"
            elif vehicle.status == DroneStatus.FLYING:
                return f"{prefix}{vehicle.id}\n[DRIVING]"
            elif vehicle.status == DroneStatus.DELIVERING:
                return f"{prefix}{vehicle.id}\n[DELIVERY]"
            elif vehicle.status == DroneStatus.RETURNING:
                return f"{prefix}{vehicle.id}\n[RETURN]"
            else:
                return f"{prefix}{vehicle.id}"
        else:
            # Drone - show battery percentage
            battery_pct = max(0.0, min(1.0, getattr(vehicle, 'battery_level', 0.0))) * 100
            
            if vehicle.status == DroneStatus.PICKING_UP and service_time > 0:
                return f"{prefix}{vehicle.id}: {battery_pct:.0f}%\n[PICKUP] {service_time:.0f}s"
            elif vehicle.status == DroneStatus.DROPPING_OFF and service_time > 0:
                return f"{prefix}{vehicle.id}: {battery_pct:.0f}%\n[DELIVER] {service_time:.0f}s"
            elif vehicle.status == DroneStatus.FLYING:
                return f"{prefix}{vehicle.id}: {battery_pct:.0f}%\n[FLYING]"
            elif vehicle.status == DroneStatus.DELIVERING:
                return f"{prefix}{vehicle.id}: {battery_pct:.0f}%\n[DELIVERY]"
            elif vehicle.status == DroneStatus.RETURNING:
                return f"{prefix}{vehicle.id}: {battery_pct:.0f}%\n[RETURN]"
            else:
                return f"{prefix}{vehicle.id}: {battery_pct:.0f}%"
    
    def _create_route_line(self, drone: Drone, color: Vec4 = None) -> Optional[NodePath]:
        """Create a line visualization for drone's route
        
        Args:
            drone: Drone with route to visualize
            color: Line color (default: orange)
            
        Returns:
            NodePath of the route line, or None if no route
        """
        if not drone.route or len(drone.route) < 1:
            return None
            
        if color is None:
            color = Vec4(1.0, 0.5, 0.0, 1.0)  # Orange
        
        # Create line segments
        lines = LineSegs()
        lines.setThickness(3.0)  # Line thickness
        lines.setColor(color)
        
        # Start from drone's current position
        start_pos = _sim_pos_to_panda3d(drone.position)
        lines.moveTo(start_pos[0], start_pos[1], start_pos[2])
        
        # Draw lines to each waypoint in the route
        for waypoint in drone.route:
            panda_pos = _sim_pos_to_panda3d(waypoint)
            lines.drawTo(panda_pos[0], panda_pos[1], panda_pos[2])
        
        # Create the node
        line_node = lines.create()
        line_np = self.scene_root.attachNewNode(line_node)
        
        return line_np
    
    def toggle_route_visualization(self):
        """Toggle route visualization on/off"""
        self.show_routes = not self.show_routes
        print(f"Route visualization: {'ON' if self.show_routes else 'OFF'}")
        
        # Show/hide existing route lines
        for route_np in self.drone_routes.values():
            if self.show_routes:
                route_np.show()
            else:
                route_np.hide()
        
    def _get_scaled_height_and_center(self, building: Building) -> Tuple[float, float]:
        """Get scaled height and center Y position for a building (in simulation coords)"""
        scaled_height = building.height * self.height_scale
        base_y = building.position.y - (building.height / 2)
        scaled_center_y = base_y + (scaled_height / 2)
        return scaled_height, scaled_center_y
        
    def _create_building_node(
        self,
        building: Building,
        bldg_color: Vec4,
        scaled_height: float,
        scaled_center_y: float
    ) -> Optional[NodePath]:
        """Create a building node with proper coordinate conversion"""
        if scaled_height <= 0:
            return None
            
        footprints = getattr(building, "footprints", None)
        
        if footprints:
            # Create prism from footprints
            center_x = building.position.x
            center_z = building.position.z  # simulation Z = depth
            
            parent_node = self.scene_root.attachNewNode(f"building_{building.id}")
            
            for footprint in footprints:
                # footprint points are in simulation (x, z) coords
                local_points = [(x - center_x, z - center_z) for x, z in footprint]
                geom_node = _build_prism_geom(local_points, scaled_height, bldg_color)
                if geom_node:
                    np = parent_node.attachNewNode(geom_node)
                    np.setTransparency(TransparencyAttrib.MAlpha)
            
            # Position in Panda3D coords: (sim_x, sim_z, sim_y)
            parent_node.setPos(center_x, center_z, scaled_center_y)
            return parent_node
        
        # Fallback to box
        geom_node = _create_box_geom(building.width, scaled_height, building.depth, bldg_color)
        np = self.scene_root.attachNewNode(geom_node)
        # Position in Panda3D coords
        np.setPos(building.position.x, building.position.z, scaled_center_y)
        np.setTransparency(TransparencyAttrib.MAlpha)
        return np
    
    def _get_local_footprints(self, building: Building) -> List[List[Tuple[float, float]]]:
        """Return building footprints expressed in local space around the center"""
        cx = building.position.x
        cz = building.position.z
        if building.footprints:
            return [[(x - cx, z - cz) for x, z in footprint] for footprint in building.footprints]
        
        # Simple rectangle footprint
        half_w = building.width / 2
        half_d = building.depth / 2
        return [[
            (-half_w, -half_d),
            (half_w, -half_d),
            (half_w, half_d),
            (-half_w, half_d),
        ]]
    
    def _apply_building_facade(
        self,
        building: Building,
        parent_node: NodePath,
        scaled_height: float,
        facade_tint: Vec4
    ):
        """Overlay textured facade panels to give buildings more detail"""
        if not parent_node:
            return
        
        if building.entity_type == EntityType.STORE:
            texture = self.facade_textures.get("store")
        elif building.entity_type == EntityType.CUSTOMER:
            texture = self.facade_textures.get("customer")
        else:
            texture = self.facade_textures.get("neutral")
        
        if texture is None:
            return
        
        footprints = self._get_local_footprints(building)
        if not footprints:
            return
        
        for footprint in footprints:
            loop = _ensure_ccw(footprint)
            for i in range(len(loop)):
                start = loop[i]
                end = loop[(i + 1) % len(loop)]
                edge_len = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                u_repeat = max(1.0, edge_len / self.window_tile_size)
                v_repeat = max(1.0, scaled_height / (self.window_tile_size * 0.8))
                
                # Offset facade slightly outward to avoid z-fighting with base geometry
                dx = end[0] - start[0]
                dz = end[1] - start[1]
                length = math.sqrt(dx * dx + dz * dz) or 1.0
                nx = -dz / length
                nz = dx / length
                offset = 0.08  # meters
                outward = (nx * offset, nz * offset)
                
                geom_node = _build_facade_geom(start, end, scaled_height, Vec4(1, 1, 1, 1), u_repeat, v_repeat, outward_offset=outward)
                if geom_node:
                    facade_np = parent_node.attachNewNode(geom_node)
                    facade_np.setTexture(texture)
                    facade_np.setTransparency(TransparencyAttrib.MAlpha)
                    facade_np.setDepthOffset(1)
                    tint = Vec4(
                        0.25 + facade_tint[0] * 0.75,
                        0.25 + facade_tint[1] * 0.75,
                        0.25 + facade_tint[2] * 0.75,
                        1.0
                    )
                    facade_np.setColor(tint)
    
    def _apply_roof(
        self,
        building: Building,
        parent_node: NodePath,
        scaled_height: float
    ):
        """Add a textured roof to the building"""
        if not parent_node:
            return
        
        local_footprints = self._get_local_footprints(building)
        if not local_footprints:
            return
        
        roof_color = Vec4(0.9, 0.9, 0.9, 1)
        for footprint in local_footprints:
            roof_geom = _build_roof_geom(footprint, roof_color, self.roof_tile_size)
            if roof_geom is None:
                continue
            roof_np = parent_node.attachNewNode(roof_geom)
            roof_np.setTexture(self.roof_texture)
            roof_np.setTransparency(TransparencyAttrib.MAlpha)
            roof_np.setDepthOffset(1)
            # Lift to top surface
            roof_np.setZ(scaled_height / 2 + 0.05)
        
    def _create_3d_text(self, text: str, sim_pos: Tuple[float, float, float], 
                        scale: float = 2, color: Vec4 = Vec4(1, 1, 1, 1)) -> NodePath:
        """Create 3D text that billboards toward camera
        
        Args:
            text: Text to display
            sim_pos: Position in simulation coordinates (x, y, z) where y=height
            scale: Text scale
            color: Text color
        """
        text_node = TextNode(f'text_{id(text)}')
        text_node.setText(text)
        text_node.setAlign(TextNode.ACenter)
        text_node.setTextColor(color)
        
        text_np = self.scene_root.attachNewNode(text_node)
        # Convert to Panda3D coords
        panda_pos = _sim_to_panda3d(sim_pos[0], sim_pos[1], sim_pos[2])
        text_np.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
        text_np.setScale(scale)
        text_np.setBillboardPointEye()  # Always face camera
        
        return text_np
        
    def create_map_entities(self, map_data: Map):
        """Render map entities (buildings, stores, customers, depots) in 3D
        
        Args:
            map_data: Map object containing buildings and depots
        """
        # Clear existing entities
        self.clear_map_entities()
        self._create_ground_from_map(map_data)
        self._render_roads(map_data)
        self._apply_ground_style(map_data)
        
        # Calculate entity scale based on average building size
        if map_data.buildings:
            avg_building_size = sum(
                (b.width + b.depth) / 2 for b in map_data.buildings
            ) / len(map_data.buildings)
            # Scale relative to a "standard" building size of 50 units
            self.entity_scale = avg_building_size / 50.0
        else:
            self.entity_scale = self.size_scale
        
        # Color definitions (RGBA 0-1)
        color_store = Vec4(0, 1, 0, 0.8)      # Green
        color_customer = Vec4(1, 0, 0, 0.8)   # Red
        color_building = Vec4(1, 1, 1, 0.9)   # White
        color_depot = Vec4(0, 0, 1, 0.9)      # Blue
        
        # Render buildings
        for building in map_data.buildings:
            # Determine color based on entity type
            if building.entity_type == EntityType.STORE:
                bldg_color = color_store
            elif building.entity_type == EntityType.CUSTOMER:
                bldg_color = color_customer
            else:
                bldg_color = color_building
                
            scaled_height, scaled_center_y = self._get_scaled_height_and_center(building)
            if scaled_height <= 0:
                continue
                
            building_node = self._create_building_node(
                building,
                bldg_color,
                scaled_height,
                scaled_center_y
            )
            if building_node is None:
                continue
            
            # Add window facades for extra realism
            self._apply_building_facade(building, building_node, scaled_height, bldg_color)
            # Add roof texture
            self._apply_roof(building, building_node, scaled_height)
            self.building_nodes.append(building_node)
            
            # Add label for stores and customers
            if building.entity_type in [EntityType.STORE, EntityType.CUSTOMER]:
                label_text = "STORE" if building.entity_type == EntityType.STORE else "CUSTOMER"
                # Label position in simulation coords
                label_offset = 5 * self.entity_scale
                label_sim_pos = (
                    building.position.x, 
                    scaled_center_y + scaled_height/2 + label_offset,  # y = height
                    building.position.z
                )
                label_scale = 8.0 * self.entity_scale  # Larger labels for visibility
                label = self._create_3d_text(
                    f"{label_text}",
                    label_sim_pos,
                    scale=label_scale,
                    color=bldg_color
                )
                self.building_nodes.append(label)
                
        # Render depots
        for depot in map_data.depots:
            # Depot size scales with average building size
            depot_radius = 2.0 * self.entity_scale
            depot_height = 1.0 * self.entity_scale
            
            geom_node = _create_cylinder_geom(depot_radius, depot_height, 16, color_depot)
            depot_node = self.scene_root.attachNewNode(geom_node)
            # Position in Panda3D coords
            panda_pos = _sim_pos_to_panda3d(depot.position)
            depot_node.setPos(panda_pos[0], panda_pos[1], panda_pos[2] + depot_height/2)
            depot_node.setTransparency(TransparencyAttrib.MAlpha)
            
            self.depot_nodes.append(depot_node)
            
            # Add depot label (simulation coords)
            label_scale = 6.0 * self.entity_scale  # Larger labels for visibility
            depot_label = self._create_3d_text(
                f"DEPOT\n{depot.id}",
                (depot.position.x, depot.position.y + depot_height + 3, depot.position.z),
                scale=label_scale,
                color=color_depot
            )
            self.depot_nodes.append(depot_label)
            
    def clear_map_entities(self):
        """Clear all map entities from the scene"""
        self._clear_roads()
        for node in self.building_nodes:
            node.removeNode()
        for node in self.depot_nodes:
            node.removeNode()
            
        self.building_nodes.clear()
        self.depot_nodes.clear()
        self._drone_last_pos.clear()
    
    def _get_vehicle_color(self, vehicle: Vehicle) -> Vec4:
        """Always render drones/motorbikes in yellow for consistent visibility."""
        return Vec4(1, 1, 0, 1)
    
    def _color_changed(self, vehicle_id: int, new_color: Vec4) -> bool:
        """Check if a vehicle's representative color has changed"""
        current_color = self.drone_colors.get(vehicle_id)
        if current_color is None:
            return True
        current_tuple = (current_color[0], current_color[1], current_color[2])
        new_tuple = (new_color[0], new_color[1], new_color[2])
        return current_tuple != new_tuple
    
    def _update_vehicle_label(self, vehicle: Vehicle):
        """Create or update a floating label for the vehicle"""
        label_offset = 10 * self.entity_scale
        label_scale = 5.0 * self.entity_scale
        panda_pos = _sim_to_panda3d(
            vehicle.position.x,
            vehicle.position.y + label_offset,
            vehicle.position.z
        )
        label = self.drone_labels.get(vehicle.id)
        if label is None:
            label = self._create_3d_text(
                self._format_drone_label(vehicle),
                (vehicle.position.x, vehicle.position.y + label_offset, vehicle.position.z),
                scale=label_scale,
                color=Vec4(0, 0, 0, 1)
            )
            self.drone_labels[vehicle.id] = label
        else:
            label.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
            text_node = label.node()
            if isinstance(text_node, TextNode):
                text_node.setText(self._format_drone_label(vehicle))
        label.show()
    
    def _create_drone_model(self, color: Vec4) -> Tuple[NodePath, List[NodePath]]:
        """Build a lightweight X-frame drone model (no spinning props)"""
        root = self.scene_root.attachNewNode("drone_model")
        color_targets: List[NodePath] = []
        
        scale = max(0.8, 1.0 * self.entity_scale)
        arm_length = 5.5 * scale
        arm_thickness = 0.22 * scale
        
        # Base arms (plus shape), then rotate root 45° to form X
        arm_x_geom = _create_box_geom(arm_length, arm_thickness, arm_thickness, Vec4(1, 1, 1, 1))
        # Lay the second arm along Y (depth) axis, keeping it flat (thickness on Z)
        arm_y_geom = _create_box_geom(arm_thickness, arm_thickness, arm_length, Vec4(1, 1, 1, 1))
        
        arm_x = root.attachNewNode(arm_x_geom)
        arm_y = root.attachNewNode(arm_y_geom)
        arm_x.setTransparency(TransparencyAttrib.MAlpha)
        arm_y.setTransparency(TransparencyAttrib.MAlpha)
        arm_x.setZ(0.05 * scale)
        arm_y.setZ(0.05 * scale)
        color_targets.extend([arm_x, arm_y])
        
        # Rotate the whole frame to get a clear X silhouette
        root.setH(90)
        
        # Central body
        body_geom = _create_cylinder_geom(0.65 * scale, 0.45 * scale, 16, Vec4(1, 1, 1, 1))
        body = root.attachNewNode(body_geom)
        body.setColor(color)
        color_targets.append(body)
        
        # Apply status color to frame elements
        for node in color_targets:
            node.setColor(color)
        
        return root, color_targets
    
    def _create_motorbike_model(self, color: Vec4) -> Tuple[NodePath, List[NodePath]]:
        """Build a motorbike: rectangular body with vertical disc wheels at both ends"""
        root = self.scene_root.attachNewNode("motorbike_model")
        color_targets: List[NodePath] = []
        
        scale = max(1.0, 1.2 * self.entity_scale)
        body_geom = _create_box_geom(1.8 * scale, 0.6 * scale, 0.6 * scale, Vec4(1, 1, 1, 1))
        body = root.attachNewNode(body_geom)
        # Raise body so it does not intersect the ground
        body_height = 0.6 * scale
        wheel_radius = 0.5 * scale
        body.setZ(wheel_radius + body_height * 0.5 + 0.05 * scale)
        color_targets.append(body)
        
        # Wheels: thin cylinders rotated to stand vertically (axis along Y)
        wheel_radius = 0.5 * scale
        wheel_width = 0.25 * scale
        wheel_geom_1 = _create_cylinder_geom(wheel_radius, wheel_width, 18, Vec4(0.12, 0.12, 0.12, 1))
        front_wheel = root.attachNewNode(wheel_geom_1)
        front_wheel.setP(90)
        front_wheel.setPos(1.0 * scale, 0, wheel_radius + 0.02 * scale)
        wheel_geom_2 = _create_cylinder_geom(wheel_radius, wheel_width, 18, Vec4(0.12, 0.12, 0.12, 1))
        rear_wheel = root.attachNewNode(wheel_geom_2)
        rear_wheel.setP(90)
        rear_wheel.setPos(-1.0 * scale, 0, wheel_radius + 0.02 * scale)
        
        # Apply status color
        for node in color_targets:
            node.setColor(color)
        
        return root, color_targets
    
    def update_drone_visuals(self, drones: List[Vehicle]):
        """Update vehicle (drone or motorbike) positions and create new entities if needed
        
        Args:
            drones: List of active vehicles (drones or motorbikes) to visualize
        """
        if not drones:
            return
            
        active_drone_ids = set()
        
        for vehicle in drones:
            active_drone_ids.add(vehicle.id)
            vehicle_color = self._get_vehicle_color(vehicle)
            panda_pos = _sim_pos_to_panda3d(vehicle.position)
            is_motorbike = isinstance(vehicle, Motorbike)
            # Cache state for camera/waypoint lookups
            self.vehicle_state_cache[vehicle.id] = vehicle
            
            node = self.drone_nodes.get(vehicle.id)
            color_targets = self.vehicle_color_targets.get(vehicle.id, [])
            
            if node is None:
                if is_motorbike:
                    node, color_targets = self._create_motorbike_model(vehicle_color)
                else:
                    node, color_targets = self._create_drone_model(vehicle_color)
                node.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
                node.setTag('drone_id', str(vehicle.id))
                node.setCollideMask(self.pick_mask)
                self._drone_last_pos[vehicle.id] = Point3(*panda_pos)
                self.drone_nodes[vehicle.id] = node
                self.vehicle_color_targets[vehicle.id] = color_targets
                self.drone_colors[vehicle.id] = Vec4(vehicle_color[0], vehicle_color[1], vehicle_color[2], vehicle_color[3])
            else:
                node.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
                if self._color_changed(vehicle.id, vehicle_color):
                    for target in color_targets or [node]:
                        target.setColor(vehicle_color)
                    self.drone_colors[vehicle.id] = Vec4(vehicle_color[0], vehicle_color[1], vehicle_color[2], vehicle_color[3])
            
            node.show()
            self._update_vehicle_label(vehicle)
            
            # Update route visualization
            if self.show_routes:
                if vehicle.id in self.drone_routes:
                    self.drone_routes[vehicle.id].removeNode()
                    del self.drone_routes[vehicle.id]
                
                if vehicle.route and len(vehicle.route) > 0:
                    route_np = self._create_route_line(vehicle)  # type: ignore[arg-type]
                    if route_np:
                        self.drone_routes[vehicle.id] = route_np
        
        # Update UI buttons to reflect current active drones
        self._refresh_drone_buttons(drones)
        # Update detail panel for selected vehicle
        self._update_selected_vehicle_panel()
        
        # Remove inactive vehicles
        inactive_drone_ids = set(self.drone_nodes.keys()) - active_drone_ids
        for drone_id in inactive_drone_ids:
            node = self.drone_nodes.pop(drone_id, None)
            if node:
                node.removeNode()
            
            label = self.drone_labels.pop(drone_id, None)
            if label:
                label.removeNode()
            
            route = self.drone_routes.pop(drone_id, None)
            if route:
                route.removeNode()
            
            if drone_id in self.drone_colors:
                del self.drone_colors[drone_id]
            if drone_id in self.vehicle_color_targets:
                del self.vehicle_color_targets[drone_id]
            if drone_id in self._drone_last_pos:
                del self._drone_last_pos[drone_id]
            if drone_id in self.vehicle_state_cache:
                del self.vehicle_state_cache[drone_id]
            if self.selected_vehicle_id == drone_id:
                self.selected_vehicle_id = None
                self.tracked_drone_id = None
                self._hide_detail_panel()
                
    def clear_drones(self):
        """Clear all drone entities from the scene"""
        for node in self.drone_nodes.values():
            node.removeNode()
        for label in self.drone_labels.values():
            label.removeNode()
        for route in self.drone_routes.values():
            route.removeNode()
            
        self.drone_nodes.clear()
        self.drone_labels.clear()
        self.drone_routes.clear()
        self.drone_colors.clear()
        self.vehicle_color_targets.clear()
        self.vehicle_state_cache.clear()
        self.clear_failure_markers()
        self._clear_drone_buttons()
        # Keep button frame alive so it can be repopulated after reset
        if self.detail_frame:
            self._hide_detail_panel()
        self.selected_vehicle_id = None
        self.tracked_drone_id = None
        
    def update_failure_markers(self, failure_events: List[Dict]):
        """Update failure markers on the map"""
        if failure_events is None:
            return
            
        seen_ids = set()
        for event in failure_events:
            order_id = event.get('order_id')
            if order_id is None:
                continue
            seen_ids.add(order_id)
            if order_id in self.failure_markers:
                continue
            position = event.get('customer_position') or event.get('store_position')
            if position is None:
                continue
            
            # Create marker at simulation position (with height offset)
            marker_offset = 8 * self.entity_scale
            marker_scale = 8.0 * self.entity_scale  # Larger for visibility
            marker = self._create_3d_text(
                f"FAIL #{order_id}",
                (position.x, max(position.y, 5) + marker_offset, position.z),
                scale=marker_scale,
                color=Vec4(1, 0, 0, 1)
            )
            self.failure_markers[order_id] = marker
            
        stale_ids = set(self.failure_markers.keys()) - seen_ids
        for order_id in stale_ids:
            node = self.failure_markers.pop(order_id, None)
            if node:
                node.removeNode()
                
    def clear_failure_markers(self):
        """Clear all failure markers"""
        for marker in self.failure_markers.values():
            marker.removeNode()
        self.failure_markers.clear()
        
    def update(self):
        """Update function called every frame (compatibility method)"""
        # Panda3D uses tasks instead, but this can be called for compatibility
        pass
        
    def run(self):
        """Start the Panda3D application loop"""
        # This calls ShowBase.run() which starts the main loop
        ShowBase.run(self)
        
    def cleanup(self):
        """Cleanup resources"""
        self.clear_map_entities()
        self.clear_drones()
        
        if hasattr(self, 'ground') and self.ground:
            self.ground.removeNode()


if __name__ == '__main__':
    # Test visualization
    visualizer = Panda3DVisualizer(map_width=1000, map_depth=1000)
    
    # Create test map
    from src.models.entities import Map, Building, Depot, EntityType
    
    test_map = Map(width=1000, depth=1000, max_height=100)
    
    # Add test buildings (simulation coords: x=horizontal, y=height, z=depth)
    test_buildings = [
        Building(1, Position(200, 15, 200), 30, 30, 30, EntityType.STORE),
        Building(2, Position(500, 20, 300), 40, 40, 40, EntityType.CUSTOMER),
        Building(3, Position(700, 25, 600), 50, 50, 50, EntityType.STORE),
        Building(4, Position(300, 10, 700), 20, 20, 20, None),
    ]
    
    for building in test_buildings:
        test_map.add_building(building)
    
    # Add test depot
    test_depot = Depot(1, Position(100, 0, 100), [])
    test_map.add_depot(test_depot)
    
    # Create map entities
    visualizer.create_map_entities(test_map)
    
    # Create test drone
    test_drone = Drone(
        id=1,
        position=Position(150, 50, 150),
        depot=test_depot,
        speed=50
    )
    
    visualizer.update_drone_visuals([test_drone])
    
    # Run visualization
    visualizer.run()
