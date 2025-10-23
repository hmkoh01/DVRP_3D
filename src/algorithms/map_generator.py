"""
Map generation algorithms for creating urban environments (3D)
"""

import random
import math
from typing import List, Tuple, Optional
from ..models.entities import Building, Position, EntityType, Map, Depot, Drone, DroneStatus, Store, Customer
import config

# Import floor height from config
FLOOR_HEIGHT = config.FLOOR_HEIGHT  # Height of each floor in meters


class MapGenerator:
    
    def __init__(self, map_width: float = config.MAP_WIDTH, map_depth: float = config.MAP_DEPTH, 
                 max_height: float = config.MAX_MAP_HEIGHT, seed: Optional[int] = None):
        """Initialize 3D Map Generator
        
        Args:
            map_width: Width of map (x-axis)
            map_depth: Depth of map (z-axis)
            max_height: Maximum height for the map (y-axis)
            seed: Random seed for reproducibility
        """
        self.map_width = map_width
        self.map_depth = map_depth
        self.max_height = max_height
        self.map = Map(map_width, map_depth, max_height)
        self.seed = seed

    
    def generate_buildings(self, num_buildings: int = config.TOTAL_BUILDINGS) -> List[Building]:
        """Generate random 3D buildings in the map"""
        buildings = []
        
        for i in range(num_buildings):
            # Random building dimensions
            # width and depth: horizontal dimensions (x and z axes)
            width = random.uniform(config.BUILDING_MIN_SIZE, config.BUILDING_MAX_SIZE)
            depth = random.uniform(config.BUILDING_MIN_SIZE, config.BUILDING_MAX_SIZE)
            # height: vertical dimension (y-axis), buildings can be taller
            height = random.uniform(config.BUILDING_MIN_HEIGHT, config.BUILDING_MAX_HEIGHT)
            
            # Try to find a valid position on the ground
            # get_random_valid_position returns Position(x, height/2, z) - centered vertically
            position = self.map.get_random_valid_position(width, height, depth)
            
            if position:
                building = Building(
                    id=i,
                    position=position,
                    width=width,
                    height=height,
                    depth=depth
                )
                buildings.append(building)
                self.map.add_building(building)
        
        return buildings
    
    def assign_entities_to_buildings(self, buildings: List[Building], 
                                   store_ratio: float = 0.3) -> Tuple[List[Store], List[Customer]]:
        """Assign stores and customers to buildings and create floor-by-floor entities
        
        For each building designated as store/customer building:
        - Creates Store/Customer objects for each floor
        - One building can only be either all stores OR all customers (not mixed)
        
        Returns:
            Tuple of (list of Store objects, list of Customer objects)
        """
        # Shuffle buildings to randomize assignment
        available_buildings = buildings.copy()
        random.shuffle(available_buildings)
        
        num_store_buildings = int(len(buildings) * store_ratio)
        num_customer_buildings = len(buildings) - num_store_buildings
        
        all_stores = []
        all_customers = []
        
        store_id_counter = 0
        customer_id_counter = 0
        
        # Assign store buildings and create store entities per floor
        print(f"  Creating stores in {num_store_buildings} buildings...")
        for i in range(num_store_buildings):
            if available_buildings:
                building = available_buildings.pop()
                building.entity_type = EntityType.STORE
                
                # Calculate number of floors in this building
                num_floors = max(1, int(building.height / FLOOR_HEIGHT))
                
                # Create a Store entity for each floor
                for floor in range(num_floors):
                    floor_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2  # Center of floor
                    
                    store = Store(
                        id=store_id_counter,
                        position=Position(building.position.x, floor_y, building.position.z),
                        building_id=building.id,
                        floor_number=floor
                    )
                    
                    all_stores.append(store)
                    self.map.add_store(store)
                    store_id_counter += 1
        
        print(f"    - Created {len(all_stores)} store entities across floors")
        
        # Assign customer buildings and create customer entities per floor
        print(f"  Creating customers in {num_customer_buildings} buildings...")
        for i in range(num_customer_buildings):
            if available_buildings:
                building = available_buildings.pop()
                building.entity_type = EntityType.CUSTOMER
                
                # Calculate number of floors in this building
                num_floors = max(1, int(building.height / FLOOR_HEIGHT))
                
                # Create a Customer entity for each floor
                for floor in range(num_floors):
                    floor_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2  # Center of floor
                    
                    customer = Customer(
                        id=customer_id_counter,
                        position=Position(building.position.x, floor_y, building.position.z),
                        building_id=building.id,
                        floor_number=floor
                    )
                    
                    all_customers.append(customer)
                    self.map.add_customer(customer)
                    customer_id_counter += 1
        
        print(f"    - Created {len(all_customers)} customer entities across floors")
        
        return all_stores, all_customers
    
    def generate_map(self, num_buildings: int = config.TOTAL_BUILDINGS, 
                     store_ratio: float = 0.3) -> Map:
        """Generate a complete 3D map with buildings, stores, and customers
        
        Creates:
        - 3D building structures
        - Store entities on each floor of store buildings
        - Customer entities on each floor of customer buildings
        """
        if self.seed is not None:
            print(f"Generating map with fixed seed: {self.seed}")
            random.seed(self.seed)

        print("Generating 3D buildings...")
        buildings = self.generate_buildings(num_buildings)
        print(f"Generated {len(buildings)} 3D buildings")
        
        print("Assigning stores and customers to building floors...")
        stores, customers = self.assign_entities_to_buildings(buildings, store_ratio)
        print(f"Total entities: {len(stores)} stores + {len(customers)} customers")
        
        return self.map
    
    def get_building_positions_for_clustering(self) -> List[Tuple[float, float, float, EntityType]]:
        """Get 3D positions of buildings with their entity types for clustering"""
        positions = []
        for building in self.map.buildings:
            if building.entity_type:
                center = building.get_center()
                positions.append((center.x, center.y, center.z, building.entity_type))
        return positions


class DepotPlacer:
    """Places depots in optimal locations based on clustering"""
    
    def __init__(self, map_obj: Map):
        self.map = map_obj
    
    def find_optimal_depot_positions(self, num_depots: int = config.TOTAL_DEPOTS) -> List[Position]:
        """Find optimal ground-level positions for depots based on building distribution
        
        Depots are always placed at ground level (y=0) and must not overlap with building footprints.
        """
        depot_positions = []
        
        # Get all building floor centers (on the ground)
        building_centers = []
        for building in self.map.buildings:
            if building.entity_type:  # Only consider buildings with entities
                building_centers.append(building.get_floor_center())
        
        if not building_centers:
            return depot_positions
        
        # Simple approach: divide map into grid (on x-z plane) and place depots in areas with most buildings
        grid_size = 3  # 3x3 grid
        cell_width = self.map.width / grid_size
        cell_depth = self.map.depth / grid_size
        
        # Count buildings in each grid cell (using x and z coordinates)
        cell_counts = {}
        for center in building_centers:
            grid_x = int(center.x // cell_width)
            grid_z = int(center.z // cell_depth)
            grid_x = min(grid_x, grid_size - 1)
            grid_z = min(grid_z, grid_size - 1)
            
            cell_key = (grid_x, grid_z)
            cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
        
        # Sort cells by building count and select top cells for depot placement
        sorted_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Depot dimensions (ground-level platform)
        depot_width = config.DEPOT_SIZE
        depot_height = 2  # Small height for depot platform
        depot_depth = config.DEPOT_SIZE
        
        print(f"  Placing {num_depots} depots at ground level...")
        for i in range(min(num_depots, len(sorted_cells))):
            grid_x, grid_z = sorted_cells[i][0]
            
            # Calculate center of the grid cell on ground plane (y=0)
            cell_center_x = grid_x * cell_width + cell_width / 2
            cell_center_z = grid_z * cell_depth + cell_depth / 2
            
            # IMPORTANT: Depot is always at ground level (y = height/2 for centered position)
            cell_center_y = depot_height / 2  # Centered at ground level
            
            # Find a valid position that doesn't overlap with building footprints
            depot_pos = self._find_valid_ground_depot_position(
                Position(cell_center_x, cell_center_y, cell_center_z), 
                depot_width, depot_depth
            )
            
            if depot_pos:
                depot_positions.append(depot_pos)
                print(f"    - Depot {i} placed at ({depot_pos.x:.1f}, 0, {depot_pos.z:.1f})")
        
        # If we need more depots, place them randomly (still at ground level)
        while len(depot_positions) < num_depots:
            depot_pos = self._find_random_ground_depot_position(depot_width, depot_depth)
            if depot_pos and depot_pos not in depot_positions:
                depot_positions.append(depot_pos)
                print(f"    - Depot {len(depot_positions)-1} placed at ({depot_pos.x:.1f}, 0, {depot_pos.z:.1f})")
        
        return depot_positions
    
    def _check_ground_footprint_overlap(self, pos: Position, width: float, depth: float) -> bool:
        """Check if a ground-level footprint overlaps with any building footprint
        
        Args:
            pos: Center position on ground (x, y, z)
            width: Width along x-axis
            depth: Depth along z-axis
            
        Returns:
            True if overlaps with any building, False otherwise
        """
        half_width = width / 2
        half_depth = depth / 2
        
        for building in self.map.buildings:
            # Check 2D footprint overlap on x-z plane
            building_half_width = building.width / 2
            building_half_depth = building.depth / 2
            
            # Check if rectangles overlap in 2D
            x_overlap = abs(pos.x - building.position.x) < (half_width + building_half_width)
            z_overlap = abs(pos.z - building.position.z) < (half_depth + building_half_depth)
            
            if x_overlap and z_overlap:
                return True  # Overlaps with this building
        
        return False  # No overlap
    
    def _find_valid_ground_depot_position(self, preferred_pos: Position, width: float, depth: float) -> Position:
        """Find a valid ground-level position for depot near preferred position
        
        Ensures depot footprint doesn't overlap with building footprints.
        
        Args:
            preferred_pos: Preferred position to place depot
            width: Depot width (x-axis)
            depth: Depot depth (z-axis)
        """
        depot_height = 2  # Fixed height for depot
        half_width = width / 2
        half_depth = depth / 2
        
        # Try the preferred position first
        if not self._check_ground_footprint_overlap(preferred_pos, width, depth):
            # Check map bounds
            if (preferred_pos.x - half_width >= 0 and preferred_pos.x + half_width <= self.map.width and
                preferred_pos.z - half_depth >= 0 and preferred_pos.z + half_depth <= self.map.depth):
                return Position(preferred_pos.x, depot_height / 2, preferred_pos.z)
        
        # Try positions in expanding circles around preferred position (on x-z plane)
        max_radius = min(self.map.width, self.map.depth) / 4
        radius_step = max(width, depth) * 1.5
        
        for radius in range(int(radius_step), int(max_radius), int(radius_step)):
            # Try positions at different angles on horizontal plane
            for angle in range(0, 360, 30):  # Every 30 degrees
                angle_rad = math.radians(angle)
                offset_x = radius * math.cos(angle_rad)
                offset_z = radius * math.sin(angle_rad)
                
                test_pos = Position(
                    preferred_pos.x + offset_x,
                    depot_height / 2,  # Ground level
                    preferred_pos.z + offset_z
                )
                
                # Check bounds
                if (test_pos.x - half_width >= 0 and test_pos.x + half_width <= self.map.width and
                    test_pos.z - half_depth >= 0 and test_pos.z + half_depth <= self.map.depth):
                    
                    # Check if footprint overlaps with buildings
                    if not self._check_ground_footprint_overlap(test_pos, width, depth):
                        return test_pos
        
        return None
    
    def _find_random_ground_depot_position(self, width: float, depth: float, max_attempts: int = 100) -> Optional[Position]:
        """Find a random valid ground-level position for depot
        
        Args:
            width: Depot width (x-axis)
            depth: Depot depth (z-axis)
            max_attempts: Maximum number of random attempts
        """
        depot_height = 2
        half_width = width / 2
        half_depth = depth / 2
        
        for _ in range(max_attempts):
            # Random position on ground plane
            x = random.uniform(half_width, self.map.width - half_width)
            z = random.uniform(half_depth, self.map.depth - half_depth)
            
            test_pos = Position(x, depot_height / 2, z)
            
            # Check if footprint overlaps with buildings
            if not self._check_ground_footprint_overlap(test_pos, width, depth):
                return test_pos
        
        return None
    
    def create_depots_with_drones(self, depot_positions: List[Position]) -> List[Depot]:
        """Create depot objects with drones at given positions"""
        depots = []
        
        for i, pos in enumerate(depot_positions):
            
            # 1. 드론 리스트가 비어있는 Depot 객체를 먼저 생성합니다.
            depot = Depot(
                id=i,
                position=pos,
                drones=[], # 우선 빈 리스트로 생성
                capacity=config.DRONES_PER_DEPOT
            )
            
            # 2. 생성된 Depot의 중심점(get_center())을 이용해 드론들을 생성합니다.
            drones = []
            for j in range(config.DRONES_PER_DEPOT):
                drone = Drone(
                    id=f"drone_{i}_{j}",
                    # Depot의 중심점을 드론의 초기 위치로 설정
                    position=depot.get_center().copy(), 
                    depot=depot, # 생성 시점에 depot을 바로 할당
                    status=DroneStatus.IDLE
                )
                drones.append(drone)
            
            # 3. 생성된 드론 리스트를 Depot 객체에 다시 할당합니다.
            depot.drones = drones
            
            depots.append(depot)
            self.map.add_depot(depot)
        
        return depots
