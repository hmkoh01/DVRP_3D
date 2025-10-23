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
    """Abstract base class for routing algorithms"""
    
    @abstractmethod
    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """Calculate optimal route through waypoints"""
        pass
    
    @abstractmethod
    def calculate_distance(self, route: List[Position]) -> float:
        """Calculate total distance of a route"""
        pass


class SimpleRouting(RoutingAlgorithm):
    """Simple straight-line routing algorithm"""
    
    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """Calculate route as straight lines through waypoints"""
        route = [start]
        
        # Add waypoints in order
        for waypoint in waypoints:
            route.append(waypoint)
        
        # Add end point
        route.append(end)
        
        return route
    
    def calculate_distance(self, route: List[Position]) -> float:
        """Calculate total distance of route"""
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        return total_distance


class MultiLevelAStarRouting(RoutingAlgorithm):
    """Multi-level A* pathfinding algorithm for 3D obstacle avoidance
    
    Creates a 3D graph with:
    - Multiple height levels based on building heights
    - Building vertices at ground and top
    - Projected vertices at each height level
    - Visibility-based edges (horizontal, vertical, diagonal)
    """
    
    def __init__(self, map_obj: Map, k_levels: int = 10):
        """Initialize 3D routing with map and height levels
        
        Args:
            map_obj: 3D Map object containing buildings
            k_levels: Number of discrete height levels to use
        """
        self.map = map_obj
        self.k_levels = k_levels
        self.height_levels = self._get_height_levels()
        
        print(f"  MultiLevelAStarRouting initialized with {len(self.height_levels)} height levels")
        print(f"    Levels: {[f'{h:.1f}' for h in self.height_levels[:5]]}{'...' if len(self.height_levels) > 5 else ''}")

    def _get_height_levels(self) -> List[float]:
        """Generate k discrete height levels based on building heights
        
        Returns:
            List of height levels from 0 to max building height
        """
        if not self.map.buildings:
            return [0.0]
        
        # Get all building heights
        building_heights = [0.0]  # Ground level
        for building in self.map.buildings:
            building_heights.append(building.height)
        
        max_height = max(building_heights)
        
        # Create k evenly spaced levels from 0 to max_height
        levels = np.linspace(0, max_height, self.k_levels).tolist()
        
        # Ensure 0 and max_height are included
        if 0.0 not in levels:
            levels.insert(0, 0.0)
        if max_height not in levels:
            levels.append(max_height)
        
        return sorted(set(levels))  # Remove duplicates and sort
    
    def _get_building_vertices_3d(self, building: Building) -> List[Position]:
        """Get 3D vertices of a building (8 corners of the box)
        
        Args:
            building: Building object
            
        Returns:
            List of 8 Position objects representing corners
        """
        half_w = building.width / 2
        half_d = building.depth / 2
        cx, cy, cz = building.position.x, building.position.y, building.position.z
        
        # 8 corners of the box
        vertices = []
        for dy in [0, building.height]:  # Ground and top
            for dx in [-half_w, half_w]:
                for dz in [-half_d, half_d]:
                    vertices.append(Position(cx + dx, dy, cz + dz))
        
        return vertices
    
    def _project_vertices_to_levels(self, building: Building) -> List[Position]:
        """Project building's 2D footprint vertices to all height levels
        
        Args:
            building: Building object
            
        Returns:
            List of Position objects at various heights
        """
        half_w = building.width / 2
        half_d = building.depth / 2
        cx, cz = building.position.x, building.position.z
        
        # 4 corners of 2D footprint
        corners_2d = [
            (cx - half_w, cz - half_d),
            (cx + half_w, cz - half_d),
            (cx + half_w, cz + half_d),
            (cx - half_w, cz + half_d)
        ]
        
        # Project to each height level
        projected = []
        for level in self.height_levels:
            for x, z in corners_2d:
                projected.append(Position(x, level, z))
        
        return projected
    
    def _segment_collides_3d(self, p1: Position, p2: Position) -> bool:
        """Check if 3D line segment p1-p2 collides with any building
        
        Args:
            p1: Start position
            p2: End position
            
        Returns:
            True if collision detected, False otherwise
        """
        # Check against each building
        for building in self.map.buildings:
            # First check 2D footprint intersection
            # Line segment in 2D: (p1.x, p1.z) to (p2.x, p2.z)
            # Building footprint: centered at (bx, bz) with half_w, half_d
            
            half_w = building.width / 2
            half_d = building.depth / 2
            bx, bz = building.position.x, building.position.z
            
            # Check if 2D segment intersects building's 2D AABB
            if not self._segment_intersects_rect_2d(
                p1.x, p1.z, p2.x, p2.z,
                bx - half_w, bz - half_d, bx + half_w, bz + half_d
            ):
                continue  # No 2D intersection, skip this building
            
            # Check vertical overlap
            seg_y_min = min(p1.y, p2.y)
            seg_y_max = max(p1.y, p2.y)
            building_y_min = 0
            building_y_max = building.height
            
            # Check if vertical ranges overlap
            if seg_y_max >= building_y_min and seg_y_min <= building_y_max:
                return True  # Collision detected
        
        return False  # No collision
    
    def _segment_intersects_rect_2d(self, x1: float, z1: float, x2: float, z2: float,
                                     rect_x_min: float, rect_z_min: float,
                                     rect_x_max: float, rect_z_max: float) -> bool:
        """Check if 2D line segment intersects with 2D axis-aligned rectangle
        
        Uses Liang-Barsky algorithm for line-rectangle intersection
        """
        # Direction vector
        dx = x2 - x1
        dz = z2 - z1
        
        # Parameters for line segment (t in [0, 1])
        t_min = 0.0
        t_max = 1.0
        
        # Check against each edge of rectangle
        # Left edge (x = rect_x_min)
        if dx != 0:
            t1 = (rect_x_min - x1) / dx
            t2 = (rect_x_max - x1) / dx
            if dx < 0:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        elif x1 < rect_x_min or x1 > rect_x_max:
            return False
        
        # Bottom edge (z = rect_z_min)
        if dz != 0:
            t1 = (rect_z_min - z1) / dz
            t2 = (rect_z_max - z1) / dz
            if dz < 0:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        elif z1 < rect_z_min or z1 > rect_z_max:
            return False
        
        return t_min <= t_max
    
    def _euclidean_distance_3d(self, pos1: Tuple[float, float, float], 
                              pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance for A* heuristic
        
        Args:
            pos1: (x, y, z) tuple
            pos2: (x, y, z) tuple
            
        Returns:
            3D Euclidean distance
        """
        return math.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )
    
    def _find_3d_path(self, start_pos: Position, end_pos: Position) -> List[Position]:
        """Find 3D path from start to end using multi-level A*
        
        Args:
            start_pos: Start position (3D)
            end_pos: End position (3D)
            
        Returns:
            List of Position waypoints forming the path
        """
        if start_pos == end_pos:
            return [start_pos]
        
        # Create networkx directed graph
        G = nx.DiGraph()
        
        # Node format: (x, y, z) tuples (hashable)
        start_node = (start_pos.x, start_pos.y, start_pos.z)
        end_node = (end_pos.x, end_pos.y, end_pos.z)
        
        # Add start and end nodes
        G.add_node(start_node)
        G.add_node(end_node)
        
        nodes = {start_node, end_node}
        
        # Add building vertices (ground and top corners)
        for building in self.map.buildings:
            vertices = self._get_building_vertices_3d(building)
            for vertex in vertices:
                node = (vertex.x, vertex.y, vertex.z)
                G.add_node(node)
                nodes.add(node)
        
        # Add projected vertices at each height level
        for building in self.map.buildings:
            projected = self._project_vertices_to_levels(building)
            for vertex in projected:
                node = (vertex.x, vertex.y, vertex.z)
                G.add_node(node)
                nodes.add(node)
        
        print(f"    Graph has {len(nodes)} nodes")
        
        # Convert nodes to list for edge checking
        node_list = list(nodes)
        
        # Add edges with visibility checks
        edges_added = 0
        
        # Horizontal edges (same height level)
        for i, n1 in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                n2 = node_list[j]
                
                # Check if same height
                if abs(n1[1] - n2[1]) < 0.01:  # Same level
                    # Check 2D visibility at this height
                    p1 = Position(n1[0], n1[1], n1[2])
                    p2 = Position(n2[0], n2[1], n2[2])
                    
                    if not self._segment_collides_3d(p1, p2):
                        weight = self._euclidean_distance_3d(n1, n2)
                        G.add_edge(n1, n2, weight=weight)
                        G.add_edge(n2, n1, weight=weight)
                        edges_added += 2
        
        # Vertical edges (same x, z, different y)
        for i, n1 in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                n2 = node_list[j]
                
                # Check if same x, z
                if abs(n1[0] - n2[0]) < 0.01 and abs(n1[2] - n2[2]) < 0.01:
                    p1 = Position(n1[0], n1[1], n1[2])
                    p2 = Position(n2[0], n2[1], n2[2])
                    
                    if not self._segment_collides_3d(p1, p2):
                        weight = abs(n1[1] - n2[1])
                        G.add_edge(n1, n2, weight=weight)
                        G.add_edge(n2, n1, weight=weight)
                        edges_added += 2
        
        # Diagonal edges (general 3D connections)
        # Sample to avoid O(n^2) - only check nearby nodes
        for n1 in node_list:
            nearby = [n2 for n2 in node_list if self._euclidean_distance_3d(n1, n2) < 100]
            for n2 in nearby:
                if n1 == n2:
                    continue
                
                # Skip if already checked horizontal or vertical
                same_level = abs(n1[1] - n2[1]) < 0.01
                same_xz = abs(n1[0] - n2[0]) < 0.01 and abs(n1[2] - n2[2]) < 0.01
                
                if not same_level and not same_xz:
                    p1 = Position(n1[0], n1[1], n1[2])
                    p2 = Position(n2[0], n2[1], n2[2])
                    
                    if not self._segment_collides_3d(p1, p2):
                        weight = self._euclidean_distance_3d(n1, n2)
                        if not G.has_edge(n1, n2):
                            G.add_edge(n1, n2, weight=weight)
                            edges_added += 1
        
        print(f"    Graph has {edges_added} edges")
        
        # Find path using A*
        try:
            path_nodes = nx.astar_path(
                G, start_node, end_node,
                heuristic=self._euclidean_distance_3d,
                weight='weight'
            )
            
            # Convert back to Position objects
            path = [Position(n[0], n[1], n[2]) for n in path_nodes]
            print(f"    Path found with {len(path)} waypoints")
            return path
            
        except nx.NetworkXNoPath:
            print(f"    No path found from {start_node} to {end_node}")
            # Return direct path as fallback
            return [start_pos, end_pos]

    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """Calculate 3D route through waypoints using multi-level A*
        
        Args:
            start: Starting position (3D)
            waypoints: List of waypoints to visit (3D)
            end: End position (3D)
            
        Returns:
            Complete route as list of 3D Position waypoints
        """
        positions = [start] + waypoints + [end]
        full_route = []
        
        # Find path between consecutive positions
        for i in range(len(positions) - 1):
            segment_route = self._find_3d_path(positions[i], positions[i + 1])
            
            # Add segment to route (avoid duplicating waypoints)
            if i == 0:
                full_route.extend(segment_route)
            else:
                full_route.extend(segment_route[1:])  # Skip first point (duplicate)
        
        return full_route
    
    def calculate_distance(self, route: List[Position]) -> float:
        """Calculate total 3D distance of route"""
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        return total_distance


class DroneRouteOptimizer:
    """Optimizes drone routes for delivery orders (3D)"""
    
    def __init__(self, routing_algorithm: RoutingAlgorithm = None):
        """Initialize optimizer with routing algorithm
        
        Args:
            routing_algorithm: Algorithm to use (default: SimpleRouting)
                             Use MultiLevelAStarRouting for 3D obstacle avoidance
        """
        self.routing_algorithm = routing_algorithm or SimpleRouting()
    
    def optimize_delivery_route(self, drone: Drone, order: Order) -> List[Position]:
        """Optimize 3D route for a drone delivering an order
        
        Route: Depot(z=0) -> Store(z=floor) -> Customer(z=floor) -> Depot(z=0)
        
        Args:
            drone: Drone to plan route for
            order: Order to deliver
            
        Returns:
            List of 3D Position waypoints
        """
        if not drone.current_order or drone.current_order.id != order.id:
            raise ValueError("Drone is not assigned to this order")
        
        # Define route waypoints (all 3D positions)
        start = drone.position.copy()  # Current drone position (3D)
        waypoints = [order.store_position.copy()]  # Pick up from store (3D, floor level)
        end = order.customer_position.copy()  # Deliver to customer (3D, floor level)
        
        # Calculate optimized 3D route to customer
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        
        # Add return to depot (ground level)
        depot_pos = drone.depot.get_center().copy()
        return_route = self.routing_algorithm.calculate_route(
            order.customer_position, [], depot_pos
        )
        
        # Combine routes (excluding duplicate customer position)
        full_route = route + return_route[1:]
        
        return [position.copy() for position in full_route]

    def calculate_delivery_time(self, route: List[Position], drone_speed: float = None) -> float:
        """Calculate estimated delivery time for a route"""
        if drone_speed is None:
            drone_speed = config.DRONE_SPEED
        
        total_distance = self.routing_algorithm.calculate_distance(route)
        delivery_time = total_distance / drone_speed
        
        return delivery_time
    
    def optimize_multiple_deliveries(self, drone: Drone, orders: List[Order]) -> List[Position]:
        """Optimize route for multiple deliveries (if drone capacity > 1)"""
        if not orders:
            return []
        
        if len(orders) == 1:
            return self.optimize_delivery_route(drone, orders[0])
        
        # For multiple orders, use a simple approach
        # In a full implementation, this would use TSP-like algorithms
        start = drone.position
        waypoints = []
        
        # Add all store positions as pickup points
        for order in orders:
            waypoints.append(order.store_position)
        
        # Add all customer positions as delivery points
        for order in orders:
            waypoints.append(order.customer_position)
        
        end = drone.depot.get_center()
        
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        return route


class RouteValidator:
    """Validates routes for feasibility and constraints"""
    
    @staticmethod
    def validate_route_feasibility(route: List[Position], drone: Drone) -> Tuple[bool, str]:
        """Validate if a route is feasible for a drone"""
        if not route:
            return False, "Empty route"
        
        # Check battery constraints
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        max_distance = drone.battery_level * drone.speed * config.DRONE_BATTERY_LIFE
        
        if total_distance > max_distance:
            return False, f"Route distance {total_distance:.2f} exceeds battery range {max_distance:.2f}"
        
        # Check time constraints (simplified)
        estimated_time = total_distance / drone.speed
        max_delivery_time = config.MAX_ORDER_DELAY
        
        if estimated_time > max_delivery_time:
            return False, f"Estimated delivery time {estimated_time:.2f}s exceeds maximum {max_delivery_time}s"
        
        return True, "Route is feasible"
    
    @staticmethod
    def validate_route_safety(route: List[Position], map_obj) -> Tuple[bool, str]:
        """Validate route safety in 3D (avoid obstacles)"""
        for position in route:
            # Check if position is within 3D map bounds
            if (position.x < 0 or position.x > map_obj.width or
                position.z < 0 or position.z > map_obj.depth or
                position.y < 0 or position.y > map_obj.max_height):
                return False, f"Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f}) is outside map bounds"
            
            # Check for building collisions (3D)
            building_at_pos = map_obj.get_building_at_position(position)
            if building_at_pos:
                return False, f"Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f}) collides with building {building_at_pos.id}"
        
        return True, "Route is safe"


class RouteAnalyzer:
    """Analyzes route performance and provides insights"""
    
    @staticmethod
    def analyze_route_efficiency(route: List[Position]) -> dict:
        """Analyze route efficiency metrics"""
        if len(route) < 2:
            return {
                'total_distance': 0,
                'straight_line_distance': 0,
                'efficiency_ratio': 1.0,
                'number_of_segments': 0
            }
        
        # Calculate actual route distance
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        # Calculate straight-line distance from start to end
        straight_line_distance = route[0].distance_to(route[-1])
        
        # Calculate efficiency ratio
        efficiency_ratio = straight_line_distance / max(total_distance, 0.001)
        
        return {
            'total_distance': total_distance,
            'straight_line_distance': straight_line_distance,
            'efficiency_ratio': efficiency_ratio,
            'number_of_segments': len(route) - 1
        }
    
    @staticmethod
    def compare_routes(routes: List[List[Position]]) -> dict:
        """Compare multiple routes and find the best one"""
        if not routes:
            return {'best_route_index': -1, 'comparison': {}}
        
        route_analyses = []
        for i, route in enumerate(routes):
            analysis = RouteAnalyzer.analyze_route_efficiency(route)
            analysis['route_index'] = i
            route_analyses.append(analysis)
        
        # Find route with shortest distance
        best_route = min(route_analyses, key=lambda x: x['total_distance'])
        
        return {
            'best_route_index': best_route['route_index'],
            'comparison': route_analyses,
            'best_route': best_route
        }


class DynamicRouteUpdater:
    """Updates routes dynamically based on changing conditions"""
    
    def __init__(self, routing_algorithm: RoutingAlgorithm):
        self.routing_algorithm = routing_algorithm
    
    def update_route_for_traffic(self, original_route: List[Position], 
                               traffic_conditions: dict) -> List[Position]:
        """Update route to avoid traffic congestion"""
        # Simple implementation - just return original route
        # In a full implementation, this would recalculate routes based on traffic data
        return original_route
    
    def update_route_for_weather(self, original_route: List[Position], 
                               weather_conditions: dict) -> List[Position]:
        """Update route to avoid adverse weather"""
        # Simple implementation - just return original route
        # In a full implementation, this would adjust routes for wind, rain, etc.
        return original_route
    
    def reroute_for_emergency(self, current_position: Position, 
                            emergency_location: Position,
                            original_destination: Position) -> List[Position]:
        """Reroute drone away from emergency area"""
        # Calculate alternative route that avoids emergency area
        # For now, just return a simple route
        waypoints = []
        return self.routing_algorithm.calculate_route(current_position, waypoints, original_destination)
