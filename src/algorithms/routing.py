"""
Routing algorithms and path optimization interfaces
"""

import math
import heapq
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..models.entities import Position, Order, Drone, Map
import config
from .geometry_utils import segments_intersect, segment_intersects_polygon, dist

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


class AStarRouting(RoutingAlgorithm):
    """A* pathfinding algorithm for obstacle avoidance"""
    
    def __init__(self, map_obj: Map):
        self.map = map_obj
        self.buildings = []
        for building in self.map.buildings:
            x, y = building.position.x, building.position.y
            w, h = building.width, building.height
            self.buildings.append([Position(x, y), Position(x+w, y), Position(x+w, y+h), Position(x, y+h)])

    def routing(self, start, end, depth=0):
        assert start != end
        if depth > 20:
            print('recursion error', start, end, self.buildings)
            return [start, end]

        point_to_index = dict()
        index_to_point = []
        point_index_to_polygon = []
        polygons = []

        point_to_index[start] = 0
        index_to_point.append(start)
        point_index_to_polygon.append(set())
        point_to_index[end] = 1
        index_to_point.append(end)
        point_index_to_polygon.append(set())
        for i, polygon in enumerate(self.buildings):
            if segment_intersects_polygon((start, end), polygon):
                polygons.append(polygon)
                for point in polygon:
                    if point not in point_to_index:
                        point_to_index[point] = len(point_to_index)
                        index_to_point.append(point)
                        point_index_to_polygon.append(set())
                    index = point_to_index[point]
                    point_index_to_polygon[index].add(i)

        graph = [[] for _ in range(len(point_to_index))]
        for polygon in polygons:
            for i in range(len(polygon)):
                a, b = point_to_index[polygon[i-1]], point_to_index[polygon[i]]
                graph[a].append(b)
                graph[b].append(a)
        
        n = len(index_to_point)
        for i in range(n):
            for j in range(i):
                if point_index_to_polygon[i] & point_index_to_polygon[j]: continue
                p1 = index_to_point[i]
                p2 = index_to_point[j]
                for other_polygon in polygons:
                    if segment_intersects_polygon((p1, p2), other_polygon) == 3: break
                else:
                    graph[i].append(j)
                    graph[j].append(i)

        dist_table = [-1] * n
        connection = [None] * n
        dist_table[point_to_index[start]] = 0
        queue = [(0, 0, point_to_index[start])]
        while queue:
            _, d, x = heapq.heappop(queue)
            if x == point_to_index[end]: break
            if dist_table[x] != d: continue

            for y in graph[x]:
                d_ = dist(index_to_point[x], index_to_point[y]) + dist_table[x]
                if dist_table[y] != -1 and d_ >= dist_table[y]: continue
                dist_table[y] = d_
                connection[y] = x
                heapq.heappush(queue, (d_ + dist(index_to_point[1], index_to_point[y]), d_, y))

        x = point_to_index[end]
        if connection[x] == None:
            return []

        route = [x]
        while x != 0:
            x = connection[x]
            route.append(x)
        route.reverse()

        if len(route) == 2:
            return [index_to_point[x] for x in route]
        else:
            new_route = []
            for i in range(len(route) - 1):
                new_route.extend(self.routing(index_to_point[route[i]], index_to_point[route[i+1]], depth+1))
                new_route.pop()
            new_route.append(index_to_point[route[-1]])
            return new_route

    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """Calculate route using A* algorithm"""
        positions = [start] + waypoints + [end]
        route = []
        for i in range(len(positions) - 1):
            route.extend(self.routing(positions[i], positions[i+1]))
            route.pop()
        route.append(end)

        return route
    
    def calculate_distance(self, route: List[Position]) -> float:
        """Calculate distance with obstacle avoidance"""
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            segment_distance = self._calculate_path_distance(route[i], route[i + 1])
            total_distance += segment_distance
        
        return total_distance
    
    def _calculate_path_distance(self, start: Position, end: Position) -> float:
        """Calculate distance between two points, avoiding obstacles"""
        # Simple implementation - just Euclidean distance
        # In a full implementation, this would check for obstacles and find detours
        return start.distance_to(end)


class DroneRouteOptimizer:
    """Optimizes drone routes for delivery orders"""
    
    def __init__(self, routing_algorithm: RoutingAlgorithm = None):
        self.routing_algorithm = routing_algorithm or SimpleRouting()
    
    def optimize_delivery_route(self, drone: Drone, order: Order) -> List[Position]:
        """Optimize route for a drone delivering an order"""
        if not drone.current_order or drone.current_order.id != order.id:
            raise ValueError("Drone is not assigned to this order")
        
        # Define route waypoints
        start = drone.position  # Current drone position
        waypoints = [order.store_position]  # Pick up from store
        end = order.customer_position  # Deliver to customer
        
        # Calculate optimized route
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        
        # Add return to depot
        return_route = self.routing_algorithm.calculate_route(
            order.customer_position, [], drone.depot.get_center()
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
        """Validate route safety (avoid obstacles)"""
        for position in route:
            # Check if position is within map bounds
            if (position.x < 0 or position.x > map_obj.width or
                position.y < 0 or position.y > map_obj.height):
                return False, f"Position {position} is outside map bounds"
            
            # Check for building collisions
            building_at_pos = map_obj.get_building_at_position(position)
            if building_at_pos:
                return False, f"Position {position} collides with building {building_at_pos.id}"
        
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
