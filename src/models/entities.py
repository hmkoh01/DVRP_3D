"""
Entity classes for the DVRP simulation
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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

    def normalize(self) -> 'Position':
        """벡터를 정규화하여 단위 벡터(길이 1)를 반환합니다."""
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Position(0, 0, 0)
        return Position(self.x / mag, self.y / mag, self.z / mag)

    def __mul__(self, scalar: float) -> 'Position':
        """벡터에 스칼라 값을 곱합니다."""
        return Position(self.x * scalar, self.y * scalar, self.z * scalar)

    # distance_to_2d 함수도 routing.py에서 사용하므로 추가합니다.
    def distance_to_2d(self, other: 'Position') -> float:
        """Calculate Euclidean distance in 2D (x, z plane)"""
        return math.sqrt((self.x - other.x)**2 + (self.z - other.z)**2)

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
    - footprint: optional list of (x, z) tuples describing polygonal base in world coords
    """
    id: int
    position: Position  # Center position (x, height/2, z)
    width: float  # Size along x-axis
    height: float  # Size along y-axis (vertical)
    depth: float  # Size along z-axis
    entity_type: Optional[EntityType] = None
    footprint: Optional[List[Tuple[float, float]]] = None
    
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
    
    def collides_with(self, other: 'Building', safety_margin: float = 0.0) -> bool:
        """Check if this building collides with another building in 3D space
        
        Args:
            other: Another building to check collision with
            safety_margin: Additional safety distance to maintain between buildings
        
        Returns:
            True if buildings overlap (including safety margin), False otherwise
        """
        half_width_self = self.width / 2
        half_height_self = self.height / 2
        half_depth_self = self.depth / 2
        
        half_width_other = other.width / 2
        half_height_other = other.height / 2
        half_depth_other = other.depth / 2
        
        # Check overlap on all three axes (including safety margin)
        x_overlap = abs(self.position.x - other.position.x) < (half_width_self + half_width_other + safety_margin)
        y_overlap = abs(self.position.y - other.position.y) < (half_height_self + half_height_other)
        z_overlap = abs(self.position.z - other.position.z) < (half_depth_self + half_depth_other + safety_margin)
        
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
        """Assign an available drone with sufficient battery to an order"""
        available_drones = self.get_available_drones()
        if not available_drones:
            return None

        # Prefer drones with higher battery levels
        sorted_drones = sorted(available_drones, key=lambda d: d.battery_level, reverse=True)
        for drone in sorted_drones:
            if not order or drone.can_complete_order(order):
                drone.assign_order(order)
                depot_pos = self.get_center()
                print(
                    f"🚚 [Depot.assign_drone] Depot {self.id} -> {drone.id} "
                    f"(drone_pos=({drone.position.x:.1f}, {drone.position.y:.1f}, {drone.position.z:.1f}), "
                    f"depot_center=({depot_pos.x:.1f}, {depot_pos.y:.1f}, {depot_pos.z:.1f})) "
                    f"Order={order.id if order else 'N/A'}"
                )
                return drone
        return None


@dataclass
class Drone:
    """Represents a delivery drone in 3D space"""
    id: int
    position: Position  # 3D position (x, y, z)
    depot: Depot
    status: DroneStatus = DroneStatus.IDLE
    current_order: Optional['Order'] = None  # legacy single-order reference
    current_orders: List['Order'] = field(default_factory=list)
    route_waypoint_order_map: Dict[int, 'Order'] = field(default_factory=dict)
    route: List[Position] = None  # List of 3D waypoints
    battery_level: float = 1.0  # 0.0 to 1.0
    speed: float = config.DRONE_SPEED  # horizontal speed (units per second)
    vertical_speed: float = config.DRONE_SPEED * 0.5  # vertical speed (units per second)
    collision_status: str = 'none'  # 'none', 'accidental', 'destination_entry'
    service_wait_remaining: float = 0.0
    service_wait_type: Optional[str] = None
    picked_up_orders: List['Order'] = field(default_factory=list)  # 픽업 완료된 주문 (실제 적재 중인 음식)
    
    def __hash__(self):
        """Make Drone hashable based on its id"""
        return hash(self.id)
    
    def __eq__(self, other):
        """Compare drones based on their id"""
        if not isinstance(other, Drone):
            return False
        return self.id == other.id
    
    def assign_order(self, order: 'Order'):
        """Assign an order to this drone"""
        self.current_order = order
        self.status = DroneStatus.LOADING
        if order:
            order.status = OrderStatus.ASSIGNED
    
    def start_delivery(self, route: List[Position]):
        """Start delivery with given route"""
        if route and len(route) > 1:
            # 🔧 수정: 첫 waypoint가 현재 위치와 동일하면 제거
            filtered_route = route.copy()
            skip_count = 0  # route_waypoint_order_map 인덱스 조정용
            if len(filtered_route) > 0:
                first_waypoint = filtered_route[0]
                distance_to_first = self.position.distance_to(first_waypoint)
                if distance_to_first < 0.1:
                    print(f"🔧 [Drone.start_delivery] Drone {self.id}: Removing first waypoint (at current position, distance={distance_to_first:.4f}m)")
                    filtered_route = filtered_route[1:]
                    skip_count = 1
            
            if len(filtered_route) > 0:
                self.route = filtered_route
                self.status = DroneStatus.FLYING
                self._popped_waypoint_count = skip_count  # 건너뛴 waypoint 수로 초기화
                self.picked_up_orders = []  # 새 배달 시작 시 적재 목록 초기화
                depot_center = self.depot.get_center() if self.depot else None
                depot_distance = self.position.distance_to(depot_center) if depot_center else float('nan')
                print(
                    f"🚁 [Drone.start_delivery] {self.id} launching with {len(filtered_route)} waypoints | "
                    f"start_pos=({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f}) "
                    f"| depot_center=({depot_center.x:.1f}, {depot_center.y:.1f}, {depot_center.z:.1f}) "
                    f"| distance_from_depot={depot_distance:.2f}m"
                )
                print(f"🚁 Drone {self.id}: Starting delivery with {len(filtered_route)} waypoints")
            else:
                print(f"❌ ERROR: Drone {self.id} received invalid route (all waypoints removed)")
                self.route = None
                self.status = DroneStatus.IDLE
                self._popped_waypoint_count = 0
        else:
            print(f"❌ ERROR: Drone {self.id} received invalid route (length: {len(route) if route else 0})")
            self.route = None
            self.status = DroneStatus.IDLE
            self._popped_waypoint_count = 0
    
    def update_position(self, dt: float):
        """
        경로에 따라 드론 위치를 업데이트하고, 각 경유지에 도달할 때마다
        스스로 상태를 올바르게 변경합니다. (3D 이동 지원)
        """
        # 🔍 로그 추가: update_position 호출 확인 (처음 몇 번만)
        if not hasattr(self, '_update_call_count'):
            self._update_call_count = 0
        self._update_call_count += 1
        
        # 🔍 로그 추가: service_wait 상태 확인
        if self.service_wait_remaining > 0:
            self.service_wait_remaining = max(0.0, self.service_wait_remaining - dt)
            if self.service_wait_remaining == 0:
                # service_wait 완료 시 항상 로그 출력 (중요한 이벤트)
                print(f"✅ [update_position] {self.status.value} Drone {self.id}: service_wait completed ({self.service_wait_type}), continuing movement")
                
                # 매장에서 픽업 완료 시 picked_up_orders에 추가
                if self.service_wait_type == "store":
                    # 방금 도착한 매장의 주문을 찾아서 picked_up_orders에 추가
                    popped_count = getattr(self, '_popped_waypoint_count', 0)
                    # 직전에 pop된 waypoint의 메타데이터 확인 (popped_count - 1)
                    waypoint_meta = self.route_waypoint_order_map.get(popped_count - 1, None) if hasattr(self, 'route_waypoint_order_map') else None
                    if waypoint_meta is not None:
                        order, visit_type = waypoint_meta
                        if visit_type == "store" and order not in self.picked_up_orders:
                            self.picked_up_orders.append(order)
                            print(f"📦 Drone {self.id}: Picked up Order {order.id} (now carrying {len(self.picked_up_orders)} orders)")
                
                self.service_wait_type = None
            else:
                return

        if not self.route:
            if self.status != DroneStatus.IDLE:
                if not hasattr(self, '_no_route_warned'):
                    self._no_route_warned = set()
                if self.id not in self._no_route_warned:
                    print(f"⚠️  WARNING: Drone {self.id} has no route but status is {self.status.value}")
                    self._no_route_warned.add(self.id)
            return
        
        target = self.route[0]
        direction = Position(
            target.x - self.position.x,
            target.y - self.position.y,
            target.z - self.position.z
        )
        distance = self.position.distance_to(target)

        # 🔍 로그 추가: DELIVERING 상태일 때 상세 로그
        if self.status == DroneStatus.DELIVERING:
            if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                print(f"🔍 [update_position] DELIVERING Drone {self.id} (call #{self._update_call_count}):")
                print(f"   service_wait_remaining: {self.service_wait_remaining:.2f}s")
                print(f"   service_wait_type: {self.service_wait_type}")
                print(f"   route length: {len(self.route) if self.route else 0}")
                if self.route:
                    print(f"   first waypoint: ({self.route[0].x:.1f}, {self.route[0].y:.1f}, {self.route[0].z:.1f})")
                print(f"   current position: ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f})")
                print(f"   target: ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
                print(f"   distance: {distance:.4f}m")
        
        # 🔍 로그 추가: RETURNING 상태일 때 상세 로그
        if self.status == DroneStatus.RETURNING:
            if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                print(f"🔍 [update_position] RETURNING Drone {self.id} (call #{self._update_call_count}):")
                print(f"   service_wait_remaining: {self.service_wait_remaining:.2f}s")
                print(f"   service_wait_type: {self.service_wait_type}")
                print(f"   route length: {len(self.route) if self.route else 0}")
                if self.route:
                    print(f"   first waypoint: ({self.route[0].x:.1f}, {self.route[0].y:.1f}, {self.route[0].z:.1f})")
                print(f"   current position: ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f})")
                print(f"   target: ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
                print(f"   distance: {distance:.4f}m")
        
        # 🔍 로그 추가: 거리 확인 (처음 몇 번만)
        if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
            if self.status not in [DroneStatus.DELIVERING, DroneStatus.RETURNING]:  # DELIVERING과 RETURNING은 위에서 이미 로그 출력
                print(f"🔍 [update_position] Drone {self.id} (call #{self._update_call_count}):")
                print(f"   Status: {self.status.value}")
                print(f"   Target: ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
                print(f"   Current: ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f})")
                print(f"   Distance: {distance:.4f}m")
                print(f"   Route length: {len(self.route)}")

        if distance < 0.1:
            if self._update_call_count <= 3:
                print(f"🔍 [update_position] Drone {self.id}: Already at target (distance={distance:.4f}m < 0.1m), popping waypoint")
            
            # 현재 waypoint의 메타데이터 확인 (route_waypoint_order_map 사용)
            current_waypoint_idx = 0  # 항상 route[0]을 처리 중
            waypoint_meta = self.route_waypoint_order_map.get(self._popped_waypoint_count, None) if hasattr(self, 'route_waypoint_order_map') else None
            
            self.route.pop(0)
            if not hasattr(self, '_popped_waypoint_count'):
                self._popped_waypoint_count = 0
            self._popped_waypoint_count += 1
            
            # waypoint 메타데이터에 따라 상태 전환 결정
            if waypoint_meta is not None:
                order, visit_type = waypoint_meta
                if visit_type == "store":
                    # Store에 도착 - FLYING -> DELIVERING로 전환하고 service_wait 시작
                    if self.status == DroneStatus.FLYING:
                        self.status = DroneStatus.DELIVERING
                        print(f"✈️  Drone {self.id}: Arrived at STORE (Order {order.id})")
                        self.service_wait_remaining = config.SERVICE_TIME_PER_STOP
                        self.service_wait_type = "store"
                        return
                elif visit_type == "customer":
                    # Customer에 도착 - 주문 완료 처리
                    if self.status == DroneStatus.DELIVERING:
                        order.status = OrderStatus.COMPLETED
                        print(f"✅ Drone {self.id}: Order {order.id} COMPLETED (delivered to customer)")
                        # current_orders 리스트에서 완료된 주문 제거
                        if order in self.current_orders:
                            self.current_orders.remove(order)
                        # picked_up_orders에서도 제거 (배달 완료)
                        if order in self.picked_up_orders:
                            self.picked_up_orders.remove(order)
                            print(f"📦 Drone {self.id}: Delivered Order {order.id} (now carrying {len(self.picked_up_orders)} orders)")
                        if self.current_order == order:
                            self.current_order = None
                        
                        # 다음 주문이 있으면 current_order 갱신
                        if self.current_orders:
                            self.current_order = self.current_orders[0]
                            print(f"📦 Drone {self.id}: Next order is {self.current_order.id}, continuing delivery")
                            self.service_wait_remaining = config.SERVICE_TIME_PER_STOP
                            self.service_wait_type = "customer"
                            return
                        else:
                            self.status = DroneStatus.RETURNING
                            print(f"📦 Drone {self.id}: All deliveries completed, returning to depot")
                            self.service_wait_remaining = config.SERVICE_TIME_PER_STOP
                            self.service_wait_type = "customer"
                            return
            # waypoint_meta가 None이면 일반 경유지 - 상태 전환 없이 계속 이동
            
            if not self.route:
                if self.status == DroneStatus.RETURNING:
                    # 모든 주문이 이미 완료된 상태로 depot에 도착
                    # 혹시 남은 주문이 있다면 완료 처리
                    for order in self.current_orders:
                        if order.status != OrderStatus.COMPLETED:
                            order.status = OrderStatus.COMPLETED
                            print(f"✅ Drone {self.id}: Order {order.id} COMPLETED (on depot return)")
                    self.current_orders.clear()
                    self.current_order = None
                    self.picked_up_orders.clear()  # 적재 목록 초기화
                    self.status = DroneStatus.IDLE
                    self._popped_waypoint_count = 0  # 리셋
                    print(f"🏠 Drone {self.id}: Returned to depot")
            return
        
        # 목표 지점에 도달할 만큼 가까워졌는지 확인합니다.
        # 수평/수직 속도를 고려한 효과적인 이동 속도 계산
        horizontal_distance = math.sqrt(direction.x**2 + direction.z**2)
        vertical_distance = abs(direction.y)
        
        # 수평 및 수직 이동 속도 계산
        effective_speed = self.speed
        
        # 🔍 로그 추가: RETURNING 상태일 때 distance > 0 조건 확인
        if self.status == DroneStatus.RETURNING:
            if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                print(f"🔍 [update_position] RETURNING Drone {self.id}: distance > 0 check")
                print(f"   distance: {distance:.4f}m")
                print(f"   Will enter movement logic: {distance > 0}")
        
        if distance > 0:
            # 전체 이동 거리 기준으로 이동
            move_distance = effective_speed * dt
            self.battery_level -= move_distance / config.DRONE_BATTERY_LIFE
            
            # 🔍 로그 추가: RETURNING 상태일 때 상세 로그
            if self.status == DroneStatus.RETURNING:
                if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                    print(f"🔍 [update_position] RETURNING Drone {self.id}: Entering movement logic")
                    print(f"   distance: {distance:.4f}m")
                    print(f"   move_distance: {move_distance:.4f}m")
                    print(f"   effective_speed: {effective_speed:.2f}, dt: {dt:.4f}")
            
            # 🔍 로그 추가: 이동 계산 (처음 몇 번만)
            if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                if self.status != DroneStatus.RETURNING:  # RETURNING은 위에서 이미 로그 출력
                    print(f"🔍 [update_position] Drone {self.id}: move_distance={move_distance:.4f}m, effective_speed={effective_speed:.2f}, dt={dt:.4f}")
            
            if distance < move_distance:
                if self._update_call_count <= 3:
                    print(f"🔍 [update_position] Drone {self.id}: Reached target (distance={distance:.4f}m < move_distance={move_distance:.4f}m)")
                self.position = target.copy()
                
                # 현재 waypoint의 메타데이터 확인 (route_waypoint_order_map 사용)
                waypoint_meta = self.route_waypoint_order_map.get(self._popped_waypoint_count, None) if hasattr(self, 'route_waypoint_order_map') else None
                
                self.route.pop(0)
                if not hasattr(self, '_popped_waypoint_count'):
                    self._popped_waypoint_count = 0
                self._popped_waypoint_count += 1
                
                # waypoint 메타데이터에 따라 상태 전환 결정
                if waypoint_meta is not None:
                    order, visit_type = waypoint_meta
                    if visit_type == "store":
                        # Store에 도착 - FLYING -> DELIVERING로 전환하고 service_wait 시작
                        if self.status == DroneStatus.FLYING:
                            self.status = DroneStatus.DELIVERING
                            print(f"✈️  Drone {self.id}: Arrived at STORE (Order {order.id})")
                            self.service_wait_remaining = config.SERVICE_TIME_PER_STOP
                            self.service_wait_type = "store"
                            return
                    elif visit_type == "customer":
                        # Customer에 도착 - 주문 완료 처리
                        if self.status == DroneStatus.DELIVERING:
                            order.status = OrderStatus.COMPLETED
                            print(f"✅ Drone {self.id}: Order {order.id} COMPLETED (delivered to customer)")
                            # current_orders 리스트에서 완료된 주문 제거
                            if order in self.current_orders:
                                self.current_orders.remove(order)
                            # picked_up_orders에서도 제거 (배달 완료)
                            if order in self.picked_up_orders:
                                self.picked_up_orders.remove(order)
                                print(f"📦 Drone {self.id}: Delivered Order {order.id} (now carrying {len(self.picked_up_orders)} orders)")
                            if self.current_order == order:
                                self.current_order = None
                            
                            # 다음 주문이 있으면 current_order 갱신
                            if self.current_orders:
                                self.current_order = self.current_orders[0]
                                print(f"📦 Drone {self.id}: Next order is {self.current_order.id}, continuing delivery")
                                self.service_wait_remaining = config.SERVICE_TIME_PER_STOP
                                self.service_wait_type = "customer"
                                return
                            else:
                                self.status = DroneStatus.RETURNING
                                print(f"📦 Drone {self.id}: All deliveries completed, returning to depot")
                                self.service_wait_remaining = config.SERVICE_TIME_PER_STOP
                                self.service_wait_type = "customer"
                                return
                # waypoint_meta가 None이면 일반 경유지 - 상태 전환 없이 계속 이동

                if not self.route:
                    if self.status == DroneStatus.RETURNING:
                        # 모든 주문이 이미 완료된 상태로 depot에 도착
                        for order in self.current_orders:
                            if order.status != OrderStatus.COMPLETED:
                                order.status = OrderStatus.COMPLETED
                                print(f"✅ Drone {self.id}: Order {order.id} COMPLETED (on depot return)")
                        self.current_orders.clear()
                        self.current_order = None
                        self.picked_up_orders.clear()  # 적재 목록 초기화
                        self.status = DroneStatus.IDLE
                        self._popped_waypoint_count = 0  # 리셋
                        print(f"🏠 Drone {self.id}: Returned to depot")
            else:
                ratio = move_distance / distance
                old_position = self.position.copy()
                self.position.x += direction.x * ratio
                self.position.y += direction.y * ratio
                self.position.z += direction.z * ratio
                
                # 🔍 로그 추가: RETURNING 상태일 때 이동 확인
                if self.status == DroneStatus.RETURNING:
                    if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                        moved_distance = old_position.distance_to(self.position)
                        print(f"🔍 [update_position] RETURNING Drone {self.id}: Moved {moved_distance:.4f}m")
                        print(f"   Old position: ({old_position.x:.1f}, {old_position.y:.1f}, {old_position.z:.1f})")
                        print(f"   New position: ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f})")
                        print(f"   New distance to target: {self.position.distance_to(target):.4f}m")
                
                # 🔍 로그 추가: 실제 이동 확인 (처음 몇 번만)
                if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                    if self.status != DroneStatus.RETURNING:  # RETURNING은 위에서 이미 로그 출력
                        moved_distance = old_position.distance_to(self.position)
                        print(f"🔍 [update_position] Drone {self.id}: Moved {moved_distance:.4f}m")
                        print(f"   Old position: ({old_position.x:.1f}, {old_position.y:.1f}, {old_position.z:.1f})")
                        print(f"   New position: ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f})")
                        print(f"   New distance to target: {self.position.distance_to(target):.4f}m")
        else:
            # 🔍 로그 추가: distance == 0인 경우 (RETURNING 상태일 때)
            if self.status == DroneStatus.RETURNING:
                if self._update_call_count <= 3 or self._update_call_count % 50 == 0:
                    print(f"🔍 [update_position] RETURNING Drone {self.id}: distance == 0, skipping movement")
                    print(f"   This means target is at current position, should have been popped earlier")

    def can_complete_order(self, order: 'Order') -> bool:
        """Return True if current battery can finish depot->store->customer->depot trip."""
        if not order:
            return True

        required_distance = (
            self.position.distance_to(order.store_position) +
            order.store_position.distance_to(order.customer_position) +
            order.customer_position.distance_to(self.depot.get_center())
        )
        max_distance = self.battery_level * config.DRONE_BATTERY_LIFE
        return required_distance <= max_distance

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
    store_building_id: Optional[int] = None  # ID of building containing the store
    customer_building_id: Optional[int] = None  # ID of building containing the customer
    
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

    def get_building_containing_point(self, point: Position) -> Optional[Building]:
        """주어진 3D 좌표(point)가 포함된 건물을 반환합니다. 없으면 None을 반환합니다."""
        for building in self.buildings:
            half_w = building.width / 2
            half_d = building.depth / 2
            
            # 건물의 X, Z 경계 확인 (건물 중심 기준)
            within_xz = (
                (building.position.x - half_w <= point.x <= building.position.x + half_w) and
                (building.position.z - half_d <= point.z <= building.position.z + half_d)
            )
            
            # 건물의 Y(높이) 경계 확인 (바닥은 0)
            within_y = (0 <= point.y <= building.height)
            
            if within_xz and within_y:
                return building # 점이 건물 내부에 있음

        return None # 어떤 건물에도 포함되지 않음

    def get_building_at_position(self, pos: Position) -> Optional[Building]:
        """Get building at a specific 3D position"""
        for building in self.buildings:
            if building.contains_point(pos):
                return building
        return None
    
    def is_position_valid(self, pos: Position, width: float = 0, height: float = 0, depth: float = 0,
                          safety_margin: float = 0.0) -> bool:
        """Check if a 3D position is valid (not overlapping with buildings)
        
        Args:
            pos: Center position of the object
            width: Size along x-axis
            height: Size along y-axis (vertical)
            depth: Size along z-axis
            safety_margin: Additional safety distance to maintain from other buildings
        """
        # Check bounds (assuming pos is center)
        half_width = width / 2
        half_depth = depth / 2
        
        if (pos.x - half_width < 0 or pos.x + half_width > self.width or 
            pos.z - half_depth < 0 or pos.z + half_depth > self.depth or
            pos.y < 0 or pos.y > self.max_height):
            return False
        
        # Check building collisions (with safety margin)
        if width > 0 and height > 0 and depth > 0:
            test_building = Building(0, pos, width, height, depth)
            for building in self.buildings:
                if test_building.collides_with(building, safety_margin):
                    return False
        
        return True
    
    def get_random_valid_position(self, width: float, height: float, depth: float, 
                                  max_attempts: int = 100, safety_margin: float = 0.0) -> Optional[Position]:
        """Get a random valid 3D position for placing entities
        
        Args:
            width: Size along x-axis
            height: Size along y-axis (vertical)
            depth: Size along z-axis
            max_attempts: Maximum number of placement attempts
            safety_margin: Additional safety distance to maintain from other buildings
        """
        for _ in range(max_attempts):
            # Random position on ground plane (y = height/2 to center the building)
            x = random.uniform(width/2, self.width - width/2)
            z = random.uniform(depth/2, self.depth - depth/2)
            y = height / 2  # Center vertically, ground at y=0
            
            pos = Position(x, y, z)
            
            if self.is_position_valid(pos, width, height, depth, safety_margin):
                return pos
        
        return None
