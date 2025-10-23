"""
Map generation algorithms for creating urban environments
"""

import random
import math
from typing import List, Tuple, Optional
from ..models.entities import Building, Position, EntityType, Map, Depot, Drone, DroneStatus
import config


class MapGenerator:
    
    def __init__(self, map_width: float = config.MAP_WIDTH, map_height: float = config.MAP_HEIGHT, seed: Optional[int] = None):
        self.map_width = map_width
        self.map_height = map_height
        self.map = Map(map_width, map_height)
        self.seed = seed

    
    def generate_buildings(self, num_buildings: int = config.TOTAL_BUILDINGS) -> List[Building]:
        """Generate random buildings in the map"""
        buildings = []
        
        for i in range(num_buildings):
            # Random building size
            width = random.uniform(config.BUILDING_MIN_SIZE, config.BUILDING_MAX_SIZE)
            height = random.uniform(config.BUILDING_MIN_SIZE, config.BUILDING_MAX_SIZE)
            
            # Try to find a valid position
            position = self.map.get_random_valid_position(max(width, height))
            
            if position:
                building = Building(
                    id=i,
                    position=position,
                    width=width,
                    height=height
                )
                buildings.append(building)
                self.map.add_building(building)
        
        return buildings
    
    def assign_entities_to_buildings(self, buildings: List[Building], 
                                   store_ratio: float = 0.3) -> Tuple[List[Building], List[Building]]:
        """Assign stores and customers to buildings randomly"""
        # Shuffle buildings to randomize assignment
        available_buildings = buildings.copy()
        random.shuffle(available_buildings)
        
        num_stores = int(len(buildings) * store_ratio)
        num_customers = len(buildings) - num_stores
        
        stores = []
        customers = []
        
        # Assign stores
        for i in range(num_stores):
            if available_buildings:
                building = available_buildings.pop()
                building.entity_type = EntityType.STORE
                stores.append(building)
                # Update map stores list
                if building not in self.map.stores:
                    self.map.stores.append(building)
        
        # Assign customers
        for i in range(num_customers):
            if available_buildings:
                building = available_buildings.pop()
                building.entity_type = EntityType.CUSTOMER
                customers.append(building)
                # Update map customers list
                if building not in self.map.customers:
                    self.map.customers.append(building)
        
        return stores, customers
    
    def generate_map(self, num_buildings: int = config.TOTAL_BUILDINGS, 
                     store_ratio: float = 0.3) -> Map:
        """Generate a complete map with buildings, stores, and customers"""
        if self.seed is not None:
            print(f"Generating map with fixed seed: {self.seed}")
            random.seed(self.seed)

        print("Generating buildings...")
        buildings = self.generate_buildings(num_buildings)
        print(f"Generated {len(buildings)} buildings")
        
        print("Assigning stores and customers...")
        stores, customers = self.assign_entities_to_buildings(buildings, store_ratio)
        print(f"Assigned {len(stores)} stores and {len(customers)} customers")
        
        return self.map
    
    def get_building_positions_for_clustering(self) -> List[Tuple[float, float, EntityType]]:
        """Get positions of buildings with their entity types for clustering"""
        positions = []
        for building in self.map.buildings:
            if building.entity_type:
                center = building.get_center()
                positions.append((center.x, center.y, building.entity_type))
        return positions


class DepotPlacer:
    """Places depots in optimal locations based on clustering"""
    
    def __init__(self, map_obj: Map):
        self.map = map_obj
    
    def find_optimal_depot_positions(self, num_depots: int = config.TOTAL_DEPOTS) -> List[Position]:
        """Find optimal positions for depots based on building distribution"""
        depot_positions = []
        
        # Get all building centers
        building_centers = []
        for building in self.map.buildings:
            if building.entity_type:  # Only consider buildings with entities
                building_centers.append(building.get_center())
        
        if not building_centers:
            return depot_positions
        
        # Simple approach: divide map into grid and place depots in areas with most buildings
        grid_size = 3  # 3x3 grid
        cell_width = self.map.width / grid_size
        cell_height = self.map.height / grid_size
        
        # Count buildings in each grid cell
        cell_counts = {}
        for center in building_centers:
            grid_x = int(center.x // cell_width)
            grid_y = int(center.y // cell_height)
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)
            
            cell_key = (grid_x, grid_y)
            cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
        
        # Sort cells by building count and select top cells for depot placement
        sorted_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(min(num_depots, len(sorted_cells))):
            grid_x, grid_y = sorted_cells[i][0]
            
            # Calculate center of the grid cell
            cell_center_x = grid_x * cell_width + cell_width / 2
            cell_center_y = grid_y * cell_height + cell_height / 2
            
            # Try to find a valid position near the cell center
            depot_pos = self._find_valid_depot_position(
                Position(cell_center_x, cell_center_y), 
                config.DEPOT_SIZE
            )
            
            if depot_pos:
                depot_positions.append(depot_pos)
        
        # If we need more depots, place them randomly
        while len(depot_positions) < num_depots:
            depot_pos = self.map.get_random_valid_position(config.DEPOT_SIZE)
            if depot_pos and depot_pos not in depot_positions:
                depot_positions.append(depot_pos)
        
        return depot_positions
    
    def _find_valid_depot_position(self, preferred_pos: Position, depot_size: float) -> Position:
        """Find a valid position for depot near preferred position"""
        # Try the preferred position first
        if self.map.is_position_valid(preferred_pos, depot_size):
            return preferred_pos
        
        # Try positions in expanding circles around preferred position
        max_radius = min(self.map.width, self.map.height) / 4
        radius_step = depot_size * 2
        
        for radius in range(int(radius_step), int(max_radius), int(radius_step)):
            # Try positions at different angles
            for angle in range(0, 360, 30):  # Every 30 degrees
                angle_rad = math.radians(angle)
                offset_x = radius * math.cos(angle_rad)
                offset_y = radius * math.sin(angle_rad)
                
                test_pos = Position(
                    preferred_pos.x + offset_x,
                    preferred_pos.y + offset_y
                )
                
                # Check bounds
                if (test_pos.x >= 0 and test_pos.x + depot_size <= self.map.width and
                    test_pos.y >= 0 and test_pos.y + depot_size <= self.map.height):
                    
                    # Check if position is valid
                    if self.map.is_position_valid(test_pos, depot_size):
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
