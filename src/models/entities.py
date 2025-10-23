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
    """Represents a 2D position"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)

    def __getitem__(self, x):
        if x == 0:
            return self.x
        elif x == 1:
            return self.y
        else:
            assert IndexError

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, other: 'Position') -> bool:
        return (self.x, self.y) < (other.x, other.y)

    def copy(self) -> 'Position':
        return Position(self.x, self.y)


@dataclass
class Building:
    """Represents a building in the urban environment"""
    id: int
    position: Position
    width: float
    height: float
    entity_type: Optional[EntityType] = None
    
    def contains_point(self, pos: Position) -> bool:
        """Check if a point is inside this building"""
        return (self.position.x <= pos.x <= self.position.x + self.width and
                self.position.y <= pos.y <= self.position.y + self.height)
    
    def get_center(self) -> Position:
        """Get the center position of the building"""
        return Position(
            self.position.x + self.width / 2,
            self.position.y + self.height / 2
        )
    
    def collides_with(self, other: 'Building') -> bool:
        """Check if this building collides with another building"""
        return not (self.position.x + self.width <= other.position.x or
                   other.position.x + other.width <= self.position.x or
                   self.position.y + self.height <= other.position.y or
                   other.position.y + other.height <= self.position.y)


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
        """Get the center position of the depot."""
        # Depot의 위치(좌측 상단)에 크기의 절반을 더해 중심점을 계산합니다.
        offset = config.DEPOT_SIZE / 2
        return Position(self.position.x + offset, self.position.y + offset)

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
    """Represents a delivery drone"""
    id: int
    position: Position
    depot: Depot
    status: DroneStatus = DroneStatus.IDLE
    current_order: Optional['Order'] = None
    route: List[Position] = None
    battery_level: float = 1.0  # 0.0 to 1.0
    speed: float = config.DRONE_SPEED  # units per second
    
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
        스스로 상태를 올바르게 변경합니다.
        """
        # 경로가 없거나 비어있으면 아무것도 하지 않습니다.
        if not self.route:
            return

        target = self.route[0]
        direction = Position(target.x - self.position.x, target.y - self.position.y)
        distance = self.position.distance_to(target)

        # 목표 지점에 도달할 만큼 가까워졌는지 확인합니다.
        if distance < self.speed * dt:
            # 목표 지점에 도착
            self.position = target
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
            if distance > 0:
                move_distance = self.speed * dt
                ratio = move_distance / distance
                self.position.x += direction.x * ratio
                self.position.y += direction.y * ratio

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
    """Represents the urban environment map"""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.buildings: List[Building] = []
        self.depots: List[Depot] = []
        self.stores: List[Building] = []
        self.customers: List[Building] = []
    
    def add_building(self, building: Building):
        """Add a building to the map"""
        self.buildings.append(building)
        
        if building.entity_type == EntityType.STORE:
            self.stores.append(building)
        elif building.entity_type == EntityType.CUSTOMER:
            self.customers.append(building)
    
    def add_depot(self, depot: Depot):
        """Add a depot to the map"""
        self.depots.append(depot)
    
    def get_building_at_position(self, pos: Position) -> Optional[Building]:
        """Get building at a specific position"""
        for building in self.buildings:
            if building.contains_point(pos):
                return building
        return None
    
    def is_position_valid(self, pos: Position, size: float = 0) -> bool:
        """Check if a position is valid (not overlapping with buildings)"""
        # Check bounds
        if (pos.x < 0 or pos.x + size > self.width or 
            pos.y < 0 or pos.y + size > self.height):
            return False
        
        # Check building collisions
        test_building = Building(0, pos, size, size)
        for building in self.buildings:
            if test_building.collides_with(building):
                return False
        
        return True
    
    def get_random_valid_position(self, size: float, max_attempts: int = 100) -> Optional[Position]:
        """Get a random valid position for placing entities"""
        for _ in range(max_attempts):
            x = random.uniform(0, self.width - size)
            y = random.uniform(0, self.height - size)
            pos = Position(x, y)
            
            if self.is_position_valid(pos, size):
                return pos
        
        return None
