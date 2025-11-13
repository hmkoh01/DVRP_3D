"""
Ursina-based 3D Visualizer for DVRP Simulation
"""

import math
from typing import Dict, List, Optional, Tuple
from typing import Sequence as TypingSequence

from ursina import *
from ursina import lights as ursina_lights
from src.models.entities import Map, Building, Depot, Drone, EntityType, Position


def _ensure_directional_light_shadow_setter():
    """Patch Ursina 5.2 DirectionalLight which lacks a setter for 'shadows'."""
    shadow_prop = getattr(ursina_lights.DirectionalLight, "shadows", None)
    if isinstance(shadow_prop, property) and shadow_prop.fset is None:

        def _set_shadows(self, value):
            cast = bool(value)
            self._shadows = cast
            self._light.setShadowCaster(cast)
            if cast:
                self.update_bounds()

        ursina_lights.DirectionalLight.shadows = property(
            shadow_prop.fget,
            _set_shadows,
            shadow_prop.fdel,
            shadow_prop.__doc__,
        )


_ensure_directional_light_shadow_setter()


def _polygon_area(points: TypingSequence[Tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, z1 = points[i]
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return area / 2.0


def _ensure_ccw(points: TypingSequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = list(points)
    if _polygon_area(pts) < 0:
        pts.reverse()
    return pts


def _is_convex(a, b, c) -> bool:
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0


def _point_in_triangle(p, a, b, c) -> bool:
    # Barycentric technique
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-8:
        return False
    w1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    w2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    w3 = 1 - w1 - w2
    return (0 < w1 < 1) and (0 < w2 < 1) and (0 < w3 < 1)


def _triangulate_polygon(points: TypingSequence[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
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


def _build_prism_mesh(points: TypingSequence[Tuple[float, float]], height: float) -> Optional[Mesh]:
    if len(points) < 3 or height <= 0:
        return None

    pts = _ensure_ccw(points)
    n = len(pts)
    half_height = height / 2

    vertices: List[Vec3] = []
    for x, z in pts:
        vertices.append(Vec3(x, -half_height, z))
    for x, z in pts:
        vertices.append(Vec3(x, half_height, z))

    triangles: List[int] = []
    tri_indices = _triangulate_polygon(pts)

    for a, b, c in tri_indices:
        triangles.extend([a + n, b + n, c + n])  # Top
        triangles.extend([c, b, a])  # Bottom (flipped)

    for i in range(n):
        j = (i + 1) % n
        triangles.extend([i, j, j + n])
        triangles.extend([i, j + n, i + n])

    return Mesh(vertices=vertices, triangles=triangles, mode='triangle')


class UrsinaVisualizer:
    """3D visualization for DVRP simulation using Ursina engine"""
    
    def __init__(self, map_width: float = 1000, map_depth: float = 1000):
        """Initialize Ursina visualizer
        
        Args:
            map_width: Width of the map (x-axis)
            map_depth: Depth of the map (z-axis)
        """
        # Initialize Ursina app
        self.app = Ursina()
        
        # Set window properties
        window.title = 'DVRP 3D Simulation'
        window.borderless = False
        window.fullscreen = False
        window.exit_button.visible = False
        window.fps_counter.enabled = True
        
        # Setup camera (EditorCamera for free mouse control)
        self.camera = EditorCamera()
        self.camera.position = (map_width/2, 1200, -1000)
        self.camera.rotation_x = 40
        
        # Create ground plane
        self.ground = Entity(
            model='plane',
            scale=(map_width, 0, map_depth),
            position=(map_width/2, 0, map_depth/2),
            texture= 'grass',
            collider='box'
        )
        '''
        # Add grid lines for better depth perception
        self.grid = Entity(
            model=Grid(map_width, map_depth),
            position=(map_width/2, 0.1, map_depth/2),
            color=color.dark_gray,
            alpha=0.3
        )
        '''
        # Store map dimensions
        self.map_width = map_width
        self.map_depth = map_depth
        
        # Entity storage
        self.building_entities: List[Entity] = []
        self.depot_entities: List[Entity] = []
        self.drone_entities: Dict[int, Entity] = {}  # drone_id -> Entity
        self.drone_labels: Dict[int, Text] = {}  # drone_id -> Text label
        
        # Add lighting
        self.setup_lighting()

        # Add sky
        self.sky = Sky()
        
        
    def setup_lighting(self):
        """Setup scene lighting"""
        # Ambient light
        self.ambient_light = AmbientLight(color=color.rgba(100, 100, 100, 100))
        
        # Directional light (sun)
        self.sun = DirectionalLight(
            position=(100, 100, -100),
            rotation=(45, -45, 0),
            color=color.white
        )
        # Explicitly disable heavy shadow rendering for performance
        self.sun.shadows = False
    
    def _create_building_entity(self, building: Building, bldg_color, alpha: float) -> Entity:
        footprint = getattr(building, "footprint", None)
        if footprint and len(footprint) >= 3:
            center_x = building.position.x
            center_z = building.position.z
            local_points = [(x - center_x, z - center_z) for x, z in footprint]
            mesh = _build_prism_mesh(local_points, building.height)
            if mesh:
                return Entity(
                    model=mesh,
                    position=(center_x, building.position.y, center_z),
                    color=bldg_color,
                    alpha=alpha,
                    collider='mesh'
                )
        
        return Entity(
            model='cube',
            position=(building.position.x, building.position.y, building.position.z),
            scale=(building.width, building.height, building.depth),
            color=bldg_color,
            alpha=alpha,
            collider='box'
        )
        
    def create_map_entities(self, map_data: Map):
        """Render map entities (buildings, stores, customers, depots) in 3D
        
        Args:
            map_data: Map object containing buildings and depots
        """
        # Clear existing entities
        self.clear_map_entities()
        
        # Render buildings
        for building in map_data.buildings:
            # Determine color based on entity type
            if building.entity_type == EntityType.STORE:
                bldg_color = color.green
                alpha = 0.8
            elif building.entity_type == EntityType.CUSTOMER:
                bldg_color = color.red
                alpha = 0.8
            else:
                bldg_color = color.white
                alpha = 0.9
            
            building_entity = self._create_building_entity(building, bldg_color, alpha)
            self.building_entities.append(building_entity)
            
            # Add label for stores and customers
            if building.entity_type in [EntityType.STORE, EntityType.CUSTOMER]:
                label_text = "STORE" if building.entity_type == EntityType.STORE else "CUSTOMER"
                label = Text(
                    text=f"{label_text}\n{building.id}",
                    position=(building.position.x, building.position.y + building.height/2 + 5, building.position.z),
                    scale=2,
                    color=color.white,
                    billboard=True,
                    origin=(0, 0)
                )
                self.building_entities.append(label)
        
        # Render depots
        for depot in map_data.depots:
            # Create depot entity (cylinder for visual distinction)
            # Depot position is at ground level (y should be small, around 1)
            depot_entity = Entity(
                model=Cylinder(),
                position=(depot.position.x, depot.position.y, depot.position.z),
                scale=(10, 2, 10),
                color=color.blue,
                alpha=0.9,
                collider='box'
            )
            
            self.depot_entities.append(depot_entity)
            
            # Add depot label (slightly above depot)
            depot_label = Text(
                text=f"DEPOT\n{depot.id}",
                position=(depot.position.x, depot.position.y + 2, depot.position.z),
                scale=2.5,
                color=color.white,
                billboard=True,
                origin=(0, 0)
            )
            
            self.depot_entities.append(depot_label)
    
    def clear_map_entities(self):
        """Clear all map entities from the scene"""
        for entity in self.building_entities:
            destroy(entity)
        for entity in self.depot_entities:
            destroy(entity)
        
        self.building_entities.clear()
        self.depot_entities.clear()
    
    def update_drone_visuals(self, drones: List[Drone]):
        """Update drone positions and create new drone entities if needed
        
        Args:
            drones: List of active drones to visualize
        """
        if not drones:
            # No active drones to display - don't clear immediately to avoid flickering
            return
        
        active_drone_ids = set()
        
        for drone in drones:
            active_drone_ids.add(drone.id)
            
            # Create new drone entity if it doesn't exist
            if drone.id not in self.drone_entities:
                # Check collision status for new drone
                collision_status = getattr(drone, 'collision_status', 'none')
                
                # Set initial color based on collision status
                if collision_status == 'accidental':
                    drone_color = color.red
                else:
                    drone_color = color.yellow
                
                # Set initial visibility
                is_visible = (collision_status != 'destination_entry')
                
                drone_entity = Entity(
                    model='sphere',
                    scale=5,
                    color=drone_color,
                    position=(drone.position.x, drone.position.y, drone.position.z),
                    enabled=is_visible,
                    visible=is_visible
                )
                self.drone_entities[drone.id] = drone_entity
                
                # Create label for drone
                drone_label = Text(
                    text=f"D{drone.id}",
                    position=(drone.position.x, drone.position.y + 5, drone.position.z),
                    scale=1.5,
                    color=color.black,
                    billboard=True,
                    origin=(0, 0),
                    enabled=is_visible
                )
                self.drone_labels[drone.id] = drone_label
            
            # Update position for existing drone
            else:
                new_pos = (drone.position.x, drone.position.y, drone.position.z)
                self.drone_entities[drone.id].position = new_pos
                
                # Update visibility and color based on collision status
                collision_status = getattr(drone, 'collision_status', 'none')
                
                if collision_status == 'destination_entry':
                    # Hide drone when entering destination building
                    self.drone_entities[drone.id].visible = False
                    self.drone_entities[drone.id].enabled = False
                    
                    # Hide label as well
                    if drone.id in self.drone_labels:
                        self.drone_labels[drone.id].enabled = False
                
                elif collision_status == 'accidental':
                    # Show drone in red when accidentally colliding
                    self.drone_entities[drone.id].visible = True
                    self.drone_entities[drone.id].enabled = True
                    self.drone_entities[drone.id].color = color.red
                    
                    # Show label
                    if drone.id in self.drone_labels:
                        self.drone_labels[drone.id].enabled = True
                
                else:  # collision_status == 'none'
                    # Show drone in yellow (normal color)
                    self.drone_entities[drone.id].visible = True
                    self.drone_entities[drone.id].enabled = True
                    self.drone_entities[drone.id].color = color.yellow
                    
                    # Show label
                    if drone.id in self.drone_labels:
                        self.drone_labels[drone.id].enabled = True
                
                # Update label position (if visible)
                if drone.id in self.drone_labels and collision_status != 'destination_entry':
                    self.drone_labels[drone.id].position = (
                        drone.position.x,
                        drone.position.y + 5,
                        drone.position.z
                    )
        
        # Remove drones that are no longer active
        inactive_drone_ids = set(self.drone_entities.keys()) - active_drone_ids
        for drone_id in inactive_drone_ids:
            if drone_id in self.drone_entities:
                destroy(self.drone_entities[drone_id])
                del self.drone_entities[drone_id]
            
            if drone_id in self.drone_labels:
                destroy(self.drone_labels[drone_id])
                del self.drone_labels[drone_id]
    
    def clear_drones(self):
        """Clear all drone entities from the scene"""
        for drone_entity in self.drone_entities.values():
            destroy(drone_entity)
        for drone_label in self.drone_labels.values():
            destroy(drone_label)
        
        self.drone_entities.clear()
        self.drone_labels.clear()
    
    def update(self):
        """Update function called every frame by Ursina"""
        # Handle keyboard input
        if held_keys['escape']:
            application.quit()
        
        # Camera controls info (can be displayed as help text)
        if held_keys['h']:
            print("Camera Controls:")
            print("  - Middle Mouse: Rotate camera")
            print("  - Right Mouse: Pan camera")
            print("  - Scroll: Zoom in/out")
            print("  - WASD: Move camera")
            print("  - ESC: Quit application")
    
    def run(self):
        """Start the Ursina application loop"""
        while self.app.win and not self.app.win.is_closed():
            self.app.step()
    
    def cleanup(self):
        """Cleanup resources"""
        self.clear_map_entities()
        self.clear_drones()
        
        if hasattr(self, 'ground'):
            destroy(self.ground)
        if hasattr(self, 'grid'):
            destroy(self.grid)
        if hasattr(self, 'sky'):
            destroy(self.sky)

'''
# Utility function to create a grid model
def Grid(width: float, depth: float, spacing: float = 50) -> Mesh:
    """Create a grid mesh for better depth perception
    
    Args:
        width: Grid width
        depth: Grid depth
        spacing: Distance between grid lines
    """
    vertices = []
    
    # Create vertical lines (along z-axis)
    for x in range(0, int(width) + 1, int(spacing)):
        vertices.extend([
            Vec3(x - width/2, 0, -depth/2),
            Vec3(x - width/2, 0, depth/2)
        ])
    
    # Create horizontal lines (along x-axis)
    for z in range(0, int(depth) + 1, int(spacing)):
        vertices.extend([
            Vec3(-width/2, 0, z - depth/2),
            Vec3(width/2, 0, z - depth/2)
        ])
    
    return Mesh(vertices=vertices, mode='line')
'''

if __name__ == '__main__':
    # Test visualization
    visualizer = UrsinaVisualizer(map_width=1000, map_depth=1000)
    
    # Create test map
    from src.models.entities import Map, Building, Depot, EntityType
    
    test_map = Map(width=1000, depth=1000, max_height=100)
    
    # Add test buildings
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
