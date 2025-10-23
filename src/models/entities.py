"""
Entity classes for the DVRP simulation
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import config

class EntityType(Enum):
    STORE = "store"
    CUSTOMER = "customer"
    DEPOT = "depot"


class OrderStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DroneStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    FLYING = "flying"
    DELIVERING = "delivering"
    RETURNING = "returning"


@dataclass
class Position:
    """Represents a 3D position"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate 3D Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.z
        else:
            raise IndexError(f"Position index out of range: {idx}")

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __lt__(self, other: 'Position') -> bool:
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def copy(self) -> 'Position':
        return Position(self.x, self.y, self.z)


@dataclass
class Building:
    """Represents a building in the 3D urban environment
    
    In Ursina, y-axis is up (height). Building position represents the center point:
    - x, z: horizontal center of the building
    - y: vertical center (height/2 from ground)
    - width: size along x-axis
    - depth: size along z-axis
    - height: size along y-axis (vertical)
    """
    id: int
    position: Position  # Center position (x, height/2, z)
    width: float  # Size along x-axis
    height: float  # Size along y-axis (vertical)
    depth: float  # Size along z-axis
    entity_type: Optional[EntityType] = None
    
    def contains_point(self, pos: Position) -> bool:
        """Check if a 3D point is inside this building"""
        half_width = self.width / 2
        half_height = self.height / 2
        half_depth = self.depth / 2
        
        return (self.position.x - half_width <= pos.x <= self.position.x + half_width and
                self.position.y - half_height <= pos.y <= self.position.y + half_height and
                self.position.z - half_depth <= pos.z <= self.position.z + half_depth)
    
    def get_center(self) -> Position:
        """Get the center position of the building"""
        return self.position.copy()
    
    def get_floor_center(self) -> Position:
        """Get the center position at ground level (y=0)"""
        return Position(self.position.x, 0, self.position.z)
    
    def collides_with(self, other: 'Building') -> bool:
        """Check if this building collides with another building in 3D space"""
        half_width_self = self.width / 2
        half_height_self = self.height / 2
        half_depth_self = self.depth / 2
        
        half_width_other = other.width / 2
        half_height_other = other.height / 2
        half_depth_other = other.depth / 2
        
        # Check overlap on all three axes
        x_overlap = abs(self.position.x - other.position.x) < (half_width_self + half_width_other)
        y_overlap = abs(self.position.y - other.position.y) < (half_height_self + half_height_other)
        z_overlap = abs(self.position.z - other.position.z) < (half_depth_self + half_depth_other)
        
        return x_overlap and y_overlap and z_overlap


@dataclass
class Store:
    """Represents a store location (can be on any floor of a building)"""
    id: int
    position: Position  # 3D position (x, floor_height, z)
    building_id: int  # Reference to parent building
    floor_number: int  # Which floor (0-indexed)
    
    def get_center(self) -> Position:
        """Get the center position of the store"""
        return self.position.copy()


@dataclass
class Customer:
    """Represents a customer location (can be on any floor of a building)"""
    id: int
    position: Position  # 3D position (x, floor_height, z)
    building_id: int  # Reference to parent building
    floor_number: int  # Which floor (0-indexed)
    
    def get_center(self) -> Position:
        """Get the center position of the customer"""
        return self.position.copy()


@dataclass
class Depot:
    """Represents a drone depot"""
    id: int
    position: Position
    drones: List['Drone']
    capacity: int = 5
    
    def __post_init__(self):
        if not hasattr(self, 'drones') or self.drones is None:
            self.drones = []

    def get_center(self) -> Position:
        """Get the center position of the depot in 3D space."""
        # For 3D, depot position is already at center
        # If depot has a size, we can offset appropriately
        return self.position.copy()

    def get_available_drones(self) -> List['Drone']:
        """Get list of available (idle) drones"""
        return [drone for drone in self.drones if drone.status == DroneStatus.IDLE]
    
    def assign_drone(self, order: 'Order') -> Optional['Drone']:
        """Assign an available drone to an order"""
        available_drones = self.get_available_drones()
        if available_drones:
            drone = available_drones[0]
            drone.assign_order(order)
            return drone
        return None


@dataclass
class Drone:
    """Represents a delivery drone in 3D space"""
    id: int
    position: Position  # 3D position (x, y, z)
    depot: Depot
    status: DroneStatus = DroneStatus.IDLE
    current_order: Optional['Order'] = None
    route: List[Position] = None  # List of 3D waypoints
    battery_level: float = 1.0  # 0.0 to 1.0
    speed: float = config.DRONE_SPEED  # horizontal speed (units per second)
    vertical_speed: float = config.DRONE_SPEED * 0.5  # vertical speed (units per second)
    
    def assign_order(self, order: 'Order'):
        """Assign an order to this drone"""
        self.current_order = order
        self.status = DroneStatus.LOADING
        order.status = OrderStatus.ASSIGNED
    
    def start_delivery(self, route: List[Position]):
        """Start delivery with given route"""
        self.route = route
        self.status = DroneStatus.FLYING
    
    def update_position(self, dt: float):
        """
        경로에 따라 드론 위치를 업데이트하고, 각 경유지에 도달할 때마다
        스스로 상태를 올바르게 변경합니다. (3D 이동 지원)
        """
        # 경로가 없거나 비어있으면 아무것도 하지 않습니다.
        if not self.route:
            return

        target = self.route[0]
        direction = Position(
            target.x - self.position.x,
            target.y - self.position.y,
            target.z - self.position.z
        )
        distance = self.position.distance_to(target)

        # 목표 지점에 도달할 만큼 가까워졌는지 확인합니다.
        # 수평/수직 속도를 고려한 효과적인 이동 속도 계산
        horizontal_distance = math.sqrt(direction.x**2 + direction.z**2)
        vertical_distance = abs(direction.y)
        
        # 수평 및 수직 이동 속도 계산
        effective_speed = self.speed
        if distance > 0:
            # 전체 이동 거리 기준으로 이동
            move_distance = effective_speed * dt
            
            if distance < move_distance:
                # 목표 지점에 도착
                self.position = target.copy()
                self.route.pop(0)  # 경로에서 현재 위치 제거

                if self.status == DroneStatus.FLYING:
                    # '가게'에 도착했으므로, 이제 '배달 중' 상태로 변경합니다.
                    self.status = DroneStatus.DELIVERING
                
                elif self.status == DroneStatus.DELIVERING:
                    # '고객'에게 도착했으므로, 이제 '복귀 중' 상태로 변경합니다.
                    self.status = DroneStatus.RETURNING

                # 경로의 마지막 목적지에 도착했는지 확인합니다.
                if not self.route:
                    if self.status == DroneStatus.RETURNING:
                        # 'Depot'에 도착했으므로, '대기' 상태로 전환되어 화면에서 사라집니다.
                        self.status = DroneStatus.IDLE
            else:
                # 목표 지점을 향해 이동합니다.
                ratio = move_distance / distance
                self.position.x += direction.x * ratio
                self.position.y += direction.y * ratio
                self.position.z += direction.z * ratio

@dataclass
class Order:
    """Represents a food delivery order"""
    id: int
    customer_id: int
    store_id: int
    customer_position: Position
    store_position: Position
    created_time: float
    status: OrderStatus = OrderStatus.PENDING
    assigned_drone: Optional[Drone] = None
    estimated_delivery_time: Optional[float] = None
    
    def get_distance(self) -> float:
        """Calculate distance between store and customer"""
        return self.store_position.distance_to(self.customer_position)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if order has expired based on maximum wait time"""
        return current_time - self.created_time > 300  # 5 minutes max wait


class Map:
    """Represents the 3D urban environment map"""
    
    def __init__(self, width: float, depth: float, max_height: float = 100):
        """
        Initialize 3D map
        Args:
            width: Size along x-axis
            depth: Size along z-axis  
            max_height: Maximum height along y-axis
        """
        self.width = width
        self.depth = depth
        self.max_height = max_height
        self.buildings: List[Building] = []
        self.depots: List[Depot] = []
        self.stores: List['Store'] = []  # Store objects on various floors
        self.customers: List['Customer'] = []  # Customer objects on various floors
    
    def add_building(self, building: Building):
        """Add a building to the map"""
        self.buildings.append(building)
    
    def add_store(self, store: 'Store'):
        """Add a store to the map"""
        self.stores.append(store)
    
    def add_customer(self, customer: 'Customer'):
        """Add a customer to the map"""
        self.customers.append(customer)
    
    def add_depot(self, depot: Depot):
        """Add a depot to the map"""
        self.depots.append(depot)
    
    def get_building_at_position(self, pos: Position) -> Optional[Building]:
        """Get building at a specific 3D position"""
        for building in self.buildings:
            if building.contains_point(pos):
                return building
        return None
    
    def is_position_valid(self, pos: Position, width: float = 0, height: float = 0, depth: float = 0) -> bool:
        """Check if a 3D position is valid (not overlapping with buildings)
        
        Args:
            pos: Center position of the object
            width: Size along x-axis
            height: Size along y-axis (vertical)
            depth: Size along z-axis
        """
        # Check bounds (assuming pos is center)
        half_width = width / 2
        half_depth = depth / 2
        
        if (pos.x - half_width < 0 or pos.x + half_width > self.width or 
            pos.z - half_depth < 0 or pos.z + half_depth > self.depth or
            pos.y < 0 or pos.y > self.max_height):
            return False
        
        # Check building collisions
        if width > 0 and height > 0 and depth > 0:
            test_building = Building(0, pos, width, height, depth)
            for building in self.buildings:
                if test_building.collides_with(building):
                    return False
        
        return True
    
    def get_random_valid_position(self, width: float, height: float, depth: float, 
                                  max_attempts: int = 100) -> Optional[Position]:
        """Get a random valid 3D position for placing entities
        
        Args:
            width: Size along x-axis
            height: Size along y-axis (vertical)
            depth: Size along z-axis
            max_attempts: Maximum number of placement attempts
        """
        for _ in range(max_attempts):
            # Random position on ground plane (y = height/2 to center the building)
            x = random.uniform(width/2, self.width - width/2)
            z = random.uniform(depth/2, self.depth - depth/2)
            y = height / 2  # Center vertically, ground at y=0
            
            pos = Position(x, y, z)
            
            if self.is_position_valid(pos, width, height, depth):
                return pos
        
        return None
