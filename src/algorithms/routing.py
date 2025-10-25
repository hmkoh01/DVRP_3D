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

# Import floor height constant
FLOOR_HEIGHT = config.FLOOR_HEIGHT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NODE_OFFSET = getattr(config, 'NODE_OFFSET', 1.0)
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
        """Generate height levels based on FLOOR_HEIGHT to match Store/Customer positions"""
        if not self.map.buildings:
            return [0.0]
        
        # Get maximum building height
        max_height = max(building.height for building in self.map.buildings)
        
        # Generate levels at each floor height (matching Store/Customer positions)
        # floor_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2
        levels = []
        
        # Add ground level
        levels.append(0.0)
        
        # Add center of each floor level
        floor = 0
        while True:
            floor_center_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2
            if floor_center_y > max_height:
                break
            levels.append(floor_center_y)
            floor += 1
        
        # Also add the maximum building height for top coverage
        if max_height not in levels:
            levels.append(max_height)
        
        return sorted(set(levels))
    
    def _get_building_vertices_3d(self, building: Building) -> List[Position]:
        """건물의 꼭짓점에서 약간 바깥쪽으로 오프셋된 노드 위치를 반환합니다.
        
        지면(0)과 건물의 실제 꼭대기 높이에 노드를 생성합니다.
        층별 노드는 _project_vertices_to_levels에서 생성됩니다.
        오프셋은 XZ 평면(수평)으로만 적용됩니다.
        """
        vertices = []
        cx, cz = building.position.x, building.position.z # 건물 중심 (지면 기준)

        # 지면과 건물 꼭대기에만 노드 생성
        for dy in [0, building.height]:
            for sign_x in [-1, 1]:
                for sign_z in [-1, 1]:
                    # 원래 꼭짓점 위치
                    corner_x = cx + sign_x * (building.width / 2)
                    corner_z = cz + sign_z * (building.depth / 2)
                    corner_pos = Position(corner_x, dy, corner_z)

                    # 건물 중심에서 꼭짓점으로 향하는 방향 벡터 (XZ 평면, 수평만)
                    # 같은 높이 레벨에서의 방향 벡터 계산
                    direction_vector = (corner_pos - Position(cx, dy, cz)).normalize()

                    # 방향 벡터로 NODE_OFFSET만큼 이동 (수평 방향으로만)
                    offset_pos = corner_pos + direction_vector * NODE_OFFSET
                    vertices.append(offset_pos)

        return vertices
    
    def _project_vertices_to_levels(self, building: Building) -> List[Position]:
        """건물 바닥면 꼭짓점을 각 높이 레벨로 투영하고 오프셋을 적용합니다.
        
        각 건물의 높이 범위 내에서만 노드를 생성합니다.
        """
        projected_offset = []
        cx, cz = building.position.x, building.position.z # 건물 중심 (지면 기준)

        for level in self.height_levels:
            # 건물 높이를 초과하는 레벨은 무시 (건물 위 허공은 불필요)
            if level > building.height:
                continue

            for sign_x in [-1, 1]:
                for sign_z in [-1, 1]:
                    # 원래 꼭짓점 위치 (해당 레벨 높이)
                    corner_x = cx + sign_x * (building.width / 2)
                    corner_z = cz + sign_z * (building.depth / 2)
                    corner_pos = Position(corner_x, level, corner_z)

                    # 건물 중심에서 꼭짓점으로 향하는 방향 벡터 (수평)
                    direction_vector = (corner_pos - Position(cx, level, cz)).normalize()

                    # 방향 벡터로 NODE_OFFSET만큼 이동
                    offset_pos = corner_pos + direction_vector * NODE_OFFSET
                    projected_offset.append(offset_pos)

        return projected_offset

    def _filter_relevant_buildings(self, p1: Position, p2: Position,
                                   start_building_id: Optional[int] = None,
                                   end_building_id: Optional[int] = None) -> List[Building]:
        """주어진 직선 경로(p1-p2)와 교차하는 건물 중 시작/종료 건물을 제외하고 필터링합니다."""
        # (이전 답변의 코드와 동일)
        relevant_buildings = []
        baseline_2d_min_x = min(p1.x, p2.x)
        baseline_2d_max_x = max(p1.x, p2.x)
        baseline_2d_min_z = min(p1.z, p2.z)
        baseline_2d_max_z = max(p1.z, p2.z)

        for building in self.map.buildings:
            if building.id == start_building_id or building.id == end_building_id:
                continue

            half_w = building.width / 2
            half_d = building.depth / 2
            b_min_x = building.position.x - half_w
            b_max_x = building.position.x + half_w
            b_min_z = building.position.z - half_d
            b_max_z = building.position.z + half_d

            if b_max_x < baseline_2d_min_x or b_min_x > baseline_2d_max_x or \
               b_max_z < baseline_2d_min_z or b_min_z > baseline_2d_max_z:
                continue

            # 기준선이 건물을 통과하는지 확인
            # destination_building_id는 관련 없으므로 None 전달
            if self._segment_collides_3d(p1, p2, destination_building_id=None, only_building=building):
                 relevant_buildings.append(building)
        return relevant_buildings
            
    def _segment_collides_3d(self, p1: Position, p2: Position,
                               # start_building_id: Optional[int] = None, # 더 이상 사용 안 함
                               destination_building_id: Optional[int] = None, # 도착 건물 ID는 유지
                               only_building: Optional[Building] = None) -> bool:
        """3D 선분 p1-p2가 건물과 충돌하는지 검사합니다.
           선분의 양 끝점이 속한 건물은 충돌 검사에서 제외합니다.
        """
        buildings_to_check = [only_building] if only_building else self.map.buildings

        # 선분 끝점이 속한 건물 ID 찾기 (매번 계산)
        p1_building = self.map.get_building_containing_point(p1)
        p2_building = self.map.get_building_containing_point(p2)
        p1_building_id = p1_building.id if p1_building else None
        p2_building_id = p2_building.id if p2_building else None

        for building in buildings_to_check:
            # 도착지 건물이거나, 선분의 끝점이 속한 건물이면 충돌 검사 무시
            if (destination_building_id is not None and building.id == destination_building_id) or \
               (p1_building_id is not None and building.id == p1_building_id) or \
               (p2_building_id is not None and building.id == p2_building_id):
                continue

            # (기존 충돌 검사 로직)
            half_w = building.width / 2
            half_d = building.depth / 2
            bx, bz = building.position.x, building.position.z
            rect_x_min, rect_z_min = bx - half_w, bz - half_d
            rect_x_max, rect_z_max = bx + half_w, bz + half_d
            p1_inside = (rect_x_min <= p1.x <= rect_x_max) and (rect_z_min <= p1.z <= rect_z_max)
            p2_inside = (rect_x_min <= p2.x <= rect_x_max) and (rect_z_min <= p2.z <= rect_z_max)
            intersects_2d = self._segment_intersects_rect_2d(p1.x, p1.z, p2.x, p2.z, rect_x_min, rect_z_min, rect_x_max, rect_z_max)

            if not intersects_2d and not p1_inside and not p2_inside: continue
            seg_y_min, seg_y_max = min(p1.y, p2.y), max(p1.y, p2.y)
            building_y_min, building_y_max = 0, building.height
            if seg_y_max >= building_y_min and seg_y_min <= building_y_max:
                return True # 충돌 발생

        return False # 충돌 없음

    def _find_path_core(self, start_pos: Position, end_pos: Position) -> List[Tuple[float, float, float]]:
        """기준선 기반 필터링된 그래프에서 A* 경로를 찾아 노드 리스트(튜플)를 반환합니다."""
        if start_pos == end_pos:
            return [(start_pos.x, start_pos.y, start_pos.z)]

        G = nx.Graph()
        start_node = (start_pos.x, start_pos.y, start_pos.z)
        end_node = (end_pos.x, end_pos.y, end_pos.z)

        G.add_node(start_node, pos=start_node)
        G.add_node(end_node, pos=end_node)
        nodes = {start_node, end_node}

        # 도착점이 속한 건물 ID 찾기 (충돌 예외 처리용)
        end_building = self.map.get_building_containing_point(end_pos)
        dest_id = end_building.id if end_building else None

        # 관련 건물 필터링 (시작 건물은 여기서 필터링 안 함, 어차피 충돌 무시됨)
        relevant_buildings = self._filter_relevant_buildings(start_pos, end_pos, end_building_id=dest_id)
        # print(f"       CORE: Found {len(relevant_buildings)} relevant buildings.")

        # 관련 건물의 노드 추가 (오프셋 적용됨)
        for building in relevant_buildings:
            # 꼭짓점 노드 추가
            vertices = self._get_building_vertices_3d(building) # 오프셋 적용된 노드
            for vertex in vertices:
                node = (vertex.x, vertex.y, vertex.z)
                if node not in nodes: G.add_node(node, pos=node); nodes.add(node)
            # 투영 노드 추가
            projected = self._project_vertices_to_levels(building) # 오프셋 적용된 노드
            for vertex in projected:
                node = (vertex.x, vertex.y, vertex.z)
                if node not in nodes: G.add_node(node, pos=node); nodes.add(node)

        node_list = list(nodes)

        # 간선 추가: 생성된 노드들 사이, 모든 건물과 충돌 검사 (선분 끝점 건물 제외)
        edges_added = 0
        for i, n1_tuple in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                n2_tuple = node_list[j]
                p1 = Position(*n1_tuple)
                p2 = Position(*n2_tuple)

                # 도착 건물 ID는 dest_id 사용, 선분 끝점 건물은 함수 내부에서 자동으로 제외됨
                if not self._segment_collides_3d(p1, p2, destination_building_id=dest_id):
                    weight = self._euclidean_distance_3d(n1_tuple, n2_tuple)
                    G.add_edge(n1_tuple, n2_tuple, weight=weight)
                    edges_added += 1

        # A* 경로 탐색
        try:
            def heuristic(u, v): return self._euclidean_distance_3d(u, v)
            path_nodes = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')

            return path_nodes
        except nx.NetworkXNoPath:
            print(f"⚠️  Routing: No path found")
            return []

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

    def _visualize_full_route(self, route: List[Position], title: str = "Full Delivery Route", 
                              store_pos: Position = None, customer_pos: Position = None):
        """전체 배달 경로를 3D로 시각화합니다.
        
        Args:
            route: 전체 경로 리스트
            title: 그래프 제목
            store_pos: Store 위치 (정확한 표시를 위해)
            customer_pos: Customer 위치 (정확한 표시를 위해)
        """
        if not route or len(route) < 2:
            print("No route to visualize")
            return
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # --- Draw Building Outlines ---
        for building in self.map.buildings:
            half_w = building.width / 2
            half_d = building.depth / 2
            cx = building.position.x
            cz = building.position.z
            h = building.height
            x_coords = [cx - half_w, cx + half_w, cx + half_w, cx - half_w, cx - half_w]
            z_coords = [cz - half_d, cz - half_d, cz + half_d, cz + half_d, cz - half_d]
            ax.plot(x_coords, z_coords, zs=0, color='black', alpha=0.3, linewidth=1)
            ax.plot(x_coords, z_coords, zs=h, color='black', alpha=0.3, linewidth=1)
            for i in range(4):
                ax.plot([x_coords[i], x_coords[i]], [z_coords[i], z_coords[i]], [0, h], color='black', alpha=0.3, linewidth=1)
        
        # --- Draw Route Path ---
        route_xs = [pos.x for pos in route]
        route_ys = [pos.z for pos in route]  # Z -> Y in matplotlib
        route_zs = [pos.y for pos in route]  # Y (height) -> Z in matplotlib
        
        ax.plot(route_xs, route_ys, route_zs, color='red', linewidth=3, marker='o', markersize=6, label='Delivery Route', alpha=0.9)
        
        # --- Highlight Key Points ---
        # Start (Depot)
        ax.scatter(route[0].x, route[0].z, route[0].y, s=200, c='green', marker='o', label='Start (Depot)', depthshade=True, edgecolors='black', linewidths=2)
        # End (back to Depot)
        ax.scatter(route[-1].x, route[-1].z, route[-1].y, s=200, c='blue', marker='s', label='End (Depot)', depthshade=True, edgecolors='black', linewidths=2)
        
        # Key Waypoints and Intermediate Points
        if len(route) > 2:
            store_drawn = False
            customer_drawn = False
            
            for pos in route[1:-1]:  # Start와 End 제외
                # Store 위치 확인 (정확한 위치가 주어진 경우)
                is_store = False
                is_customer = False
                
                if store_pos and abs(pos.x - store_pos.x) < 0.1 and abs(pos.y - store_pos.y) < 0.1 and abs(pos.z - store_pos.z) < 0.1:
                    is_store = True
                elif customer_pos and abs(pos.x - customer_pos.x) < 0.1 and abs(pos.y - customer_pos.y) < 0.1 and abs(pos.z - customer_pos.z) < 0.1:
                    is_customer = True
                
                if is_store and not store_drawn:
                    # Store (주황색 별)
                    ax.scatter(pos.x, pos.z, pos.y, s=300, c='orange', marker='*', 
                              label='Store (Pickup)', depthshade=True, edgecolors='darkred', linewidths=2.5)
                    store_drawn = True
                elif is_customer and not customer_drawn:
                    # Customer (보라색 다이아몬드)
                    ax.scatter(pos.x, pos.z, pos.y, s=250, c='purple', marker='D', 
                              label='Customer (Delivery)', depthshade=True, edgecolors='darkviolet', linewidths=2)
                    customer_drawn = True
                else:
                    # 일반 경유지 (작은 회색 다이아몬드)
                    ax.scatter(pos.x, pos.z, pos.y, s=50, c='lightgray', marker='d', 
                              alpha=0.5, depthshade=True, edgecolors='gray', linewidths=0.5)
        
        # --- Setup Plot ---
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Z (Depth)', fontsize=12)
        ax.set_zlabel('Y (Height)', fontsize=12)
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.depth)
        ax.set_zlim(0, self.map.max_height * 1.1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        plt.show()

    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """점진적 경로 탐색(Incremental Pathfinding)을 사용하여 전체 경로를 계산합니다."""
        full_route = [start]
        segment_targets = waypoints + [end] # 거쳐갈 목표 지점들

        current_segment_start = start

        for segment_end in segment_targets:
            working_path_segment = [current_segment_start] # 현재 세그먼트의 경로
            u_prime = current_segment_start
            v_final_segment = segment_end # 이번 세그먼트의 최종 목표

            MAX_ITERATIONS = 100 # 무한 루프 방지
            iterations = 0

            while u_prime != v_final_segment and iterations < MAX_ITERATIONS:
                iterations += 1

                # 1. 현재 위치에서 세그먼트 끝까지 직접 갈 수 있는지 확인 (모든 건물 대상)
                dest_building = self.map.get_building_containing_point(v_final_segment)
                dest_id = dest_building.id if dest_building else None
                is_direct_path_safe = not self._segment_collides_3d(u_prime, v_final_segment, destination_building_id=dest_id)

                if is_direct_path_safe:
                    # 직접 경로가 안전하면 바로 이동하고 이 세그먼트 종료
                    working_path_segment.append(v_final_segment)
                    u_prime = v_final_segment # 루프 종료 조건 만족
                    # print(f"   Direct path found from {u_prime} to {v_final_segment}")
                    break # while 루프 종료

                # 2. 직접 경로가 막혔으면, CORE 알고리즘으로 다음 스텝 찾기
                # CORE는 u_prime에서 v_final_segment까지의 기준선 장애물만 고려
                partial_path_nodes = self._find_path_core(u_prime, v_final_segment)

                if not partial_path_nodes or len(partial_path_nodes) < 2:
                    print(f"❌ Routing Error: Path calculation failed")
                    return []

                # 3. CORE가 제안한 다음 스텝(p2) 추출
                p1_node = partial_path_nodes[0] # u_prime과 동일
                p2_node = partial_path_nodes[1]
                p1_pos = Position(*p1_node)
                p2_pos = Position(*p2_node)

                # 4. 제안된 첫 스텝(p1 -> p2)이 실제로 안전한지 *모든* 건물 기준으로 재확인
                # (CORE는 필터링된 그래프만 봤으므로 재확인 필요)
                # p1, p2가 속한 건물은 충돌 검사에서 제외됨
                is_first_step_safe = not self._segment_collides_3d(p1_pos, p2_pos, destination_building_id=dest_id) # 도착지 건물 ID 전달 유지

                if is_first_step_safe:
                    # 첫 스텝이 안전하면 그곳으로 이동하고, 다음 목표는 다시 세그먼트 끝(v_final_segment)
                    working_path_segment.append(p2_pos)
                    u_prime = p2_pos
                else:
                    print(f"❌ Routing Error: Unsafe path segment detected")
                    return []

            if iterations >= MAX_ITERATIONS:
                print(f"❌ Routing Error: Max iterations reached")
                return []

            # 현재 세그먼트 경로를 전체 경로에 추가 (시작점 제외)
            full_route.extend(working_path_segment[1:])
            current_segment_start = segment_end # 다음 세그먼트 시작점 업데이트

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
    
    def optimize_delivery_route(self, drone: Drone, order: Order, visualize: bool = False) -> List[Position]:
        """드론 배달 경로를 최적화합니다.
        
        Args:
            drone: 배달을 수행할 드론
            order: 배달할 주문
            visualize: True일 경우 전체 경로를 3D로 시각화 (기본값: False)
        
        Returns:
            전체 배달 경로 (depot → store → customer → depot)
        """
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
        
        # 전체 경로 시각화 (옵션)
        if visualize and isinstance(self.routing_algorithm, MultiLevelAStarRouting):
            self.routing_algorithm._visualize_full_route(
                full_route, 
                title=f"Full Delivery Route - Order {order.id} (Drone {drone.id})",
                store_pos=order.store_position,
                customer_pos=order.customer_position
            )
        
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