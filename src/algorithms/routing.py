"""
Routing algorithms and path optimization interfaces (3D)
"""

import math
import heapq
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from ..models.entities import Position, Order, Drone, Map, Building
import config

class RoutingAlgorithm(ABC):
    
    @abstractmethod
    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        pass
    
    @abstractmethod
    def calculate_distance(self, route: List[Position]) -> float:
        pass


class SimpleRouting(RoutingAlgorithm):
    
    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        route = [start]
        for waypoint in waypoints:
            route.append(waypoint)
        route.append(end)
        return route
    
    def calculate_distance(self, route: List[Position]) -> float:
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        return total_distance


class MultiLevelAStarRouting(RoutingAlgorithm):
    
    def __init__(self, map_obj: Map, k_levels: int = 10):
        self.map = map_obj
        self.k_levels = k_levels
        self.height_levels = self._get_height_levels()
        
        print(f"  MultiLevelAStarRouting initialized with {len(self.height_levels)} height levels")
        print(f"    Levels: {[f'{h:.1f}' for h in self.height_levels[:5]]}{'...' if len(self.height_levels) > 5 else ''}")

    def _get_height_levels(self) -> List[float]:
        if not self.map.buildings:
            return [0.0]
        
        building_heights = [0.0]
        for building in self.map.buildings:
            building_heights.append(building.height)
        
        max_height = max(building_heights)
        
        levels = np.linspace(0, max_height, self.k_levels).tolist()
        
        if 0.0 not in levels:
            levels.insert(0, 0.0)
        if max_height not in levels:
            levels.append(max_height)
        
        return sorted(set(levels))
    
    def _get_building_vertices_3d(self, building: Building) -> List[Position]:
        half_w = building.width / 2
        half_d = building.depth / 2
        cx, cy, cz = building.position.x, building.position.y, building.position.z
        
        vertices = []
        for dy in [0, building.height]:
            for dx in [-half_w, half_w]:
                for dz in [-half_d, half_d]:
                    # 건물의 기준 y가 0이므로 dy를 더하지 않고, y좌표를 dy로 설정합니다.
                    # 건물의 position.y는 Ursina 좌표계 (중심)이므로 0으로 가정합니다.
                    # 아, entities에서 building.position.y가 (x, height/2, z)였던 것을
                    # 맵 생성기에서 (x, y=0, z) 기준으로 생성했다면 y=0이 맞습니다.
                    # 여기서는 building.position.y가 0이라고 가정하고, dy (0 or height)를 y좌표로 씁니다.
                    vertices.append(Position(cx + dx, dy, cz + dz))
        
        return vertices
    
    def _project_vertices_to_levels(self, building: Building) -> List[Position]:
        half_w = building.width / 2
        half_d = building.depth / 2
        cx, cz = building.position.x, building.position.z
        
        corners_2d = [
            (cx - half_w, cz - half_d),
            (cx + half_w, cz - half_d),
            (cx + half_w, cz + half_d),
            (cx - half_w, cz + half_d)
        ]
        
        projected = []
        for level in self.height_levels:
            for x, z in corners_2d:
                projected.append(Position(x, level, z))
        
        return projected
    
    def _segment_collides_3d(self, p1: Position, p2: Position) -> bool:
        for building in self.map.buildings:
            half_w = building.width / 2
            half_d = building.depth / 2
            bx, bz = building.position.x, building.position.z
            
            rect_x_min = bx - half_w
            rect_z_min = bz - half_d
            rect_x_max = bx + half_w
            rect_z_max = bz + half_d

            if not self._segment_intersects_rect_2d(
                p1.x, p1.z, p2.x, p2.z,
                rect_x_min, rect_z_min, rect_x_max, rect_z_max
            ):
                continue
            
            # 2D에서 교차하거나 포함될 경우, 3D 충돌 검사
            
            # (수정) 2D 선분이 사각형 내부에 완전히 포함되는 경우도 검사해야 합니다.
            p1_inside = (rect_x_min <= p1.x <= rect_x_max) and (rect_z_min <= p1.z <= rect_z_max)
            p2_inside = (rect_x_min <= p2.x <= rect_x_max) and (rect_z_min <= p2.z <= rect_z_max)

            if not self._segment_intersects_rect_2d(
                p1.x, p1.z, p2.x, p2.z,
                rect_x_min, rect_z_min, rect_x_max, rect_z_max
            ) and not p1_inside and not p2_inside:
                 continue


            seg_y_min = min(p1.y, p2.y)
            seg_y_max = max(p1.y, p2.y)
            building_y_min = 0 
            building_y_max = building.height
            
            if seg_y_max >= building_y_min and seg_y_min <= building_y_max:
                # (수정) 더 정확한 3D 교차 검사
                # 2D에서 교차하는 지점의 Y값이 건물 높이 범위에 있는지 확인
                
                # 2D 선분이 건물 중심을 통과하는지 등 단순화된 검사
                # 여기서는 AABB (Axis-Aligned Bounding Box) 충돌로 간주합니다.
                
                # 2D에서 겹치고, Y축(높이)에서도 겹치면 충돌로 간주
                return True
        
        return False
    
    def _segment_intersects_rect_2d(self, x1: float, z1: float, x2: float, z2: float,
                                      rect_x_min: float, rect_z_min: float,
                                      rect_x_max: float, rect_z_max: float) -> bool:
        
        # 선분의 AABB가 사각형과 겹치는지 빠른 검사
        if max(x1, x2) < rect_x_min or min(x1, x2) > rect_x_max or \
           max(z1, z2) < rect_z_min or min(z1, z2) > rect_z_max:
            return False

        # Liang-Barsky 알고리즘 (또는 Cohen-Sutherland)
        dx = x2 - x1
        dz = z2 - z1
        
        t_min = 0.0
        t_max = 1.0
        
        edges = [
            (-dx, x1 - rect_x_min),  # Left
            ( dx, rect_x_max - x1),  # Right
            (-dz, z1 - rect_z_min),  # Bottom
            ( dz, rect_z_max - z1)   # Top
        ]
        
        for p, q in edges:
            if p == 0:
                if q < 0:
                    return False
            else:
                t = q / p
                if p < 0:
                    if t > t_max: return False
                    t_min = max(t_min, t)
                else:
                    if t < t_min: return False
                    t_max = min(t_max, t)

        return t_min <= t_max

    
    def _euclidean_distance_3d(self, pos1: Tuple[float, float, float], 
                               pos2: Tuple[float, float, float]) -> float:
        return math.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )
    
    def _find_3d_path(self, start_pos: Position, end_pos: Position) -> List[Position]:
        if start_pos == end_pos:
            return [start_pos]
        
        G = nx.Graph() # (수정) 방향성이 없는 가시성 그래프이므로 DiGraph -> Graph
        
        start_node = (start_pos.x, start_pos.y, start_pos.z)
        end_node = (end_pos.x, end_pos.y, end_pos.z)
        
        G.add_node(start_node, pos=start_node)
        G.add_node(end_node, pos=end_node)
        
        nodes = {start_node, end_node}
        
        for building in self.map.buildings:
            vertices = self._get_building_vertices_3d(building)
            for vertex in vertices:
                node = (vertex.x, vertex.y, vertex.z)
                if node not in nodes:
                    G.add_node(node, pos=node)
                    nodes.add(node)
        
        for building in self.map.buildings:
            projected = self._project_vertices_to_levels(building)
            for vertex in projected:
                node = (vertex.x, vertex.y, vertex.z)
                if node not in nodes:
                    G.add_node(node, pos=node)
                    nodes.add(node)
        
        print(f"    Graph has {len(nodes)} nodes")
        
        node_list = list(nodes)
        
        # (수정) O(n^2)으로 모든 노드 쌍 간의 가시성 검사
        # 기존의 수평/수직/대각선 분리 로직 대신 통합
        edges_added = 0
        for i, n1 in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                n2 = node_list[j]
                
                p1 = Position(n1[0], n1[1], n1[2])
                p2 = Position(n2[0], n2[1], n2[2])
                
                if not self._segment_collides_3d(p1, p2):
                    weight = self._euclidean_distance_3d(n1, n2)
                    G.add_edge(n1, n2, weight=weight)
                    edges_added += 1

        print(f"    Graph has {edges_added} edges")
        
        try:
            # A* 휴리스틱 함수 정의 (튜플 직접 사용)
            def heuristic(u_node, v_node):
                return self._euclidean_distance_3d(u_node, v_node)

            path_nodes = nx.astar_path(
                G, start_node, end_node,
                heuristic=heuristic,
                weight='weight'
            )
            
            path = [Position(n[0], n[1], n[2]) for n in path_nodes]
            # print(f"    Path found with {len(path)} waypoints") # 너무 많은 로그 삭제
            return path
            
        except nx.NetworkXNoPath:
            print(f"    No path found from {start_node} to {end_node}")
            return [] # (수정) 실패 시 빈 리스트 반환

    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        positions = [start] + waypoints + [end]
        full_route = []
        
        for i in range(len(positions) - 1):
            segment_route = self._find_3d_path(positions[i], positions[i + 1])
            
            # (수정) 세그먼트 경로 찾기 실패 시 전체 경로 실패 처리
            if not segment_route:
                print(f"    [Routing Error] Failed to find path segment from {positions[i]} to {positions[i+1]}")
                return []
            
            if i == 0:
                full_route.extend(segment_route)
            else:
                full_route.extend(segment_route[1:])
        
        return full_route
    
    def calculate_distance(self, route: List[Position]) -> float:
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        return total_distance


class DroneRouteOptimizer:
    
    def __init__(self, routing_algorithm: RoutingAlgorithm = None):
        self.routing_algorithm = routing_algorithm or SimpleRouting()
    
    def optimize_delivery_route(self, drone: Drone, order: Order) -> List[Position]:
        if not drone.current_order or drone.current_order.id != order.id:
            raise ValueError("Drone is not assigned to this order")
        
        start = drone.position.copy()
        waypoints = [order.store_position.copy()]
        end = order.customer_position.copy()
        
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        
        # (수정) 경로 탐색 실패 시 빈 리스트 반환
        if not route:
            return []

        depot_pos = drone.depot.get_center().copy()
        return_route = self.routing_algorithm.calculate_route(
            order.customer_position, [], depot_pos
        )
        
        # (수정) 복귀 경로 탐색 실패 시 빈 리스트 반환
        if not return_route:
            return []
        
        full_route = route + return_route[1:]
        
        return [position.copy() for position in full_route]

    def calculate_delivery_time(self, route: List[Position], drone_speed: float = None) -> float:
        if drone_speed is None:
            drone_speed = config.DRONE_SPEED
        
        total_distance = self.routing_algorithm.calculate_distance(route)
        delivery_time = total_distance / drone_speed
        
        return delivery_time
    
    def optimize_multiple_deliveries(self, drone: Drone, orders: List[Order]) -> List[Position]:
        if not orders:
            return []
        
        if len(orders) == 1:
            return self.optimize_delivery_route(drone, orders[0])
        
        start = drone.position
        waypoints = []
        
        for order in orders:
            waypoints.append(order.store_position)
        
        for order in orders:
            waypoints.append(order.customer_position)
        
        end = drone.depot.get_center()
        
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        return route


class RouteValidator:
    
    @staticmethod
    def validate_route_feasibility(route: List[Position], drone: Drone) -> Tuple[bool, str]:
        if not route:
            return False, "Empty route"
        
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        max_distance = drone.battery_level * drone.speed * config.DRONE_BATTERY_LIFE
        
        if total_distance > max_distance:
            return False, f"Route distance {total_distance:.2f} exceeds battery range {max_distance:.2f}"
        
        estimated_time = total_distance / drone.speed
        max_delivery_time = config.MAX_ORDER_DELAY
        
        if estimated_time > max_delivery_time:
            return False, f"Estimated delivery time {estimated_time:.2f}s exceeds maximum {max_delivery_time}s"
        
        return True, "Route is feasible"
    
    @staticmethod
    def validate_route_safety(route: List[Position], map_obj) -> Tuple[bool, str]:
        for position in route:
            if (position.x < 0 or position.x > map_obj.width or
                position.z < 0 or position.z > map_obj.depth or
                position.y < 0 or position.y > map_obj.max_height):
                return False, f"Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f}) is outside map bounds"
            
            building_at_pos = map_obj.get_building_at_position(position)
            if building_at_pos:
                return False, f"Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f}) collides with building {building_at_pos.id}"
        
        return True, "Route is safe"


class RouteAnalyzer:
    
    @staticmethod
    def analyze_route_efficiency(route: List[Position]) -> dict:
        if len(route) < 2:
            return {
                'total_distance': 0,
                'straight_line_distance': 0,
                'efficiency_ratio': 1.0,
                'number_of_segments': 0
            }
        
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        straight_line_distance = route[0].distance_to(route[-1])
        
        efficiency_ratio = straight_line_distance / max(total_distance, 0.001)
        
        return {
            'total_distance': total_distance,
            'straight_line_distance': straight_line_distance,
            'efficiency_ratio': efficiency_ratio,
            'number_of_segments': len(route) - 1
        }
    
    @staticmethod
    def compare_routes(routes: List[List[Position]]) -> dict:
        if not routes:
            return {'best_route_index': -1, 'comparison': {}}
        
        route_analyses = []
        for i, route in enumerate(routes):
            analysis = RouteAnalyzer.analyze_route_efficiency(route)
            analysis['route_index'] = i
            route_analyses.append(analysis)
        
        best_route = min(route_analyses, key=lambda x: x['total_distance'])
        
        return {
            'best_route_index': best_route['route_index'],
            'comparison': route_analyses,
            'best_route': best_route
        }


class DynamicRouteUpdater:
    
    def __init__(self, routing_algorithm: RoutingAlgorithm):
        self.routing_algorithm = routing_algorithm
    
    def update_route_for_traffic(self, original_route: List[Position], 
                               traffic_conditions: dict) -> List[Position]:
        return original_route
    
    def update_route_for_weather(self, original_route: List[Position], 
                               weather_conditions: dict) -> List[Position]:
        return original_route
    
    def reroute_for_emergency(self, current_position: Position, 
                            emergency_location: Position,
                            original_destination: Position) -> List[Position]:
        waypoints = []
        return self.routing_algorithm.calculate_route(current_position, waypoints, original_destination)