"""
Order management system for dynamic order generation and depot assignment
"""

import random
import time
import math
from typing import List, Dict, Optional, Tuple
from ..models.entities import Order, Building, Depot, Drone, OrderStatus, Position, EntityType
from .clustering import MixedClustering
from .routing import DroneRouteOptimizer, SimpleRouting, AStarRouting
import config


class OrderGenerator:

    def __init__(self, map_obj, generation_rate: float = config.ORDER_GENERATION_RATE, seed: Optional[int] = None):
        self.map = map_obj
        self.generation_rate = generation_rate
        self.order_counter = 0
        self.last_generation_time = 0.0
        self.seed = seed
        if self.seed is not None:
            print(f"Initializing order generator with fixed seed: {self.seed}")
            random.seed(self.seed)
    
    def generate_random_order(self, current_time: float) -> Optional[Order]:
        """Generate a random order if enough time has passed"""
        time_since_last = current_time - self.last_generation_time
        
        # Calculate probability of generating an order
        probability = self.generation_rate * time_since_last
        
        if random.random() < probability:
            self.last_generation_time = current_time
            return self._create_random_order(current_time)
        
        return None
    
    def _create_random_order(self, current_time: float) -> Order:
        """Create a random order between a customer and a store"""
        # Get available customers and stores
        customers = [b for b in self.map.customers if b.entity_type == EntityType.CUSTOMER]
        stores = [b for b in self.map.stores if b.entity_type == EntityType.STORE]
        
        if not customers or not stores:
            return None
        
        # Randomly select customer and store
        customer_building = random.choice(customers)
        store_building = random.choice(stores)
        
        # Create order
        order = Order(
            id=self.order_counter,
            customer_id=customer_building.id,
            store_id=store_building.id,
            customer_position=customer_building.get_center(),
            store_position=store_building.get_center(),
            created_time=current_time
        )
        
        self.order_counter += 1
        return order
    
    def generate_batch_orders(self, current_time: float, batch_size: int = 5) -> List[Order]:
        """Generate multiple orders at once for testing"""
        orders = []
        for _ in range(batch_size):
            order = self._create_random_order(current_time)
            if order:
                orders.append(order)
        return orders


class DepotSelector:
    """Selects the best depot for a given order"""
    
    def __init__(self, map_obj):
        self.map = map_obj
        self.clustering_algorithm = MixedClustering()
    
    def select_best_depot(self, order: Order) -> Optional[Depot]:
        """Select the best depot for an order based on multiple criteria"""
        if not self.map.depots:
            return None
        
        # Calculate scores for each depot
        depot_scores = {}
        
        for depot in self.map.depots:
            score = self._calculate_depot_score(depot, order)
            depot_scores[depot.id] = score
        
        # Select depot with highest score
        if depot_scores:
            best_depot_id = max(depot_scores.keys(), key=lambda k: depot_scores[k])
            return next(depot for depot in self.map.depots if depot.id == best_depot_id)
        
        return None
    
    def _calculate_depot_score(self, depot: Depot, order: Order) -> float:
        """Calculate a score for depot-order assignment"""
        # Check if depot has available drones
        available_drones = depot.get_available_drones()
        if not available_drones:
            return 0.0  # No available drones
        
        # Distance from depot to store
        depot_to_store_distance = depot.get_center().distance_to(order.store_position)
        
        # Distance from store to customer
        store_to_customer_distance = order.get_distance()
        
        # Total distance
        total_distance = depot_to_store_distance + store_to_customer_distance
        
        # Distance from depot back to depot (return trip)
        customer_to_depot_distance = order.customer_position.distance_to(depot.get_center())
        
        # Complete trip distance
        complete_trip_distance = total_distance + customer_to_depot_distance
        
        # Calculate score (lower distance = higher score)
        # Add bonus for having available drones
        drone_bonus = len(available_drones) * 10
        
        # Distance penalty (inverse relationship)
        distance_score = max(0, 1000 - complete_trip_distance)
        
        # Battery consideration (simplified)
        battery_score = 100  # Assume all drones have good battery
        
        total_score = distance_score + drone_bonus + battery_score
        
        return total_score
    
    def get_depot_assignment_metrics(self, orders: List[Order]) -> Dict[int, Dict]:
        """Analyze depot assignment patterns"""
        depot_stats = {}
        
        for depot in self.map.depots:
            depot_stats[depot.id] = {
                'total_orders': 0,
                'available_drones': len(depot.get_available_drones()),
                'total_drones': len(depot.drones),
                'avg_distance': 0.0,
                'total_distance': 0.0
            }
        
        # Analyze orders
        for order in orders:
            if order.assigned_drone and order.assigned_drone.depot:
                depot_id = order.assigned_drone.depot.id
                if depot_id in depot_stats:
                    depot_stats[depot_id]['total_orders'] += 1
                    
                    # Calculate distance for this order
                    depot = order.assigned_drone.depot
                    distance = (depot.position.distance_to(order.store_position) + 
                              order.store_position.distance_to(order.customer_position) +
                              order.customer_position.distance_to(depot.position))
                    
                    depot_stats[depot_id]['total_distance'] += distance
        
        # Calculate averages
        for depot_id, stats in depot_stats.items():
            if stats['total_orders'] > 0:
                stats['avg_distance'] = stats['total_distance'] / stats['total_orders']
        
        return depot_stats


class OrderManager:
    """Manages the complete order lifecycle"""
    
    def __init__(self, map_obj, seed: Optional[int] = None):
        self.map = map_obj
        self.orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.order_generator = OrderGenerator(map_obj, seed=seed)
        self.depot_selector = DepotSelector(map_obj)
        # self.route_optimizer = DroneRouteOptimizer(SimpleRouting())
        self.route_optimizer = DroneRouteOptimizer(AStarRouting(map_obj))
    
    def process_orders(self, current_time: float) -> List[Order]:
        """Process pending orders and generate new ones"""
        # Generate new orders
        new_order = self.order_generator.generate_random_order(current_time)
        if new_order:
            self.orders.append(new_order)
            print(f"New order generated: Customer {new_order.customer_id} -> Store {new_order.store_id}")
        
        # Process pending orders
        new_completed = []
        for order in self.orders[:]:  # Copy to avoid modification during iteration
            if order.status == OrderStatus.PENDING:
                self._assign_order_to_depot(order)
            elif order.status == OrderStatus.COMPLETED:
                self.orders.remove(order)
                self.completed_orders.append(order)
                new_completed.append(order)
            elif order.is_expired(current_time):
                order.status = OrderStatus.CANCELLED
                self.orders.remove(order)
                print(f"Order {order.id} expired and cancelled")
        
        return new_completed
    
    def _assign_order_to_depot(self, order: Order):
        """Assign an order to the best available depot and drone"""
        best_depot = self.depot_selector.select_best_depot(order)
        
        if best_depot:
            assigned_drone = best_depot.assign_drone(order)
            if assigned_drone:
                order.assigned_drone = assigned_drone
                order.status = OrderStatus.ASSIGNED
                
                # Calculate and set route for the drone
                try:
                    route = self.route_optimizer.optimize_delivery_route(assigned_drone, order)
                    assigned_drone.start_delivery(route)
                    print(f"Order {order.id} assigned to depot {best_depot.id}, drone {assigned_drone.id} with route")
                except Exception as e:
                    print(f"Failed to set route for order {order.id}: {e}")
                    # Still assign the order even if route setting fails
                
            else:
                print(f"No available drones at depot {best_depot.id} for order {order.id}")
        else:
            print(f"No suitable depot found for order {order.id}")
    
    def get_order_statistics(self) -> Dict:
        """Get statistics about orders"""
        total_orders = len(self.orders) + len(self.completed_orders)
        pending_orders = len([o for o in self.orders if o.status == OrderStatus.PENDING])
        assigned_orders = len([o for o in self.orders if o.status == OrderStatus.ASSIGNED])
        in_progress_orders = len([o for o in self.orders if o.status == OrderStatus.IN_PROGRESS])
        completed_orders = len(self.completed_orders)
        
        avg_delivery_distance = 0
        if completed_orders > 0:
            total_distance = sum(o.get_distance() for o in self.completed_orders)
            avg_delivery_distance = total_distance / completed_orders
        
        return {
            'total_orders': total_orders,
            'pending_orders': pending_orders,
            'assigned_orders': assigned_orders,
            'in_progress_orders': in_progress_orders,
            'completed_orders': completed_orders,
            'avg_delivery_distance': avg_delivery_distance
        }
    
    def generate_test_orders(self, num_orders: int = 10, current_time: float = None) -> List[Order]:
        """Generate test orders for simulation"""
        if current_time is None:
            current_time = time.time()
        
        test_orders = self.order_generator.generate_batch_orders(current_time, num_orders)
        self.orders.extend(test_orders)
        
        print(f"Generated {len(test_orders)} test orders")
        return test_orders
    
    def assign_all_pending_orders(self):
        """Assign all pending orders to depots (for testing)"""
        pending_orders = [o for o in self.orders if o.status == OrderStatus.PENDING]
        
        for order in pending_orders:
            self._assign_order_to_depot(order)
        
        print(f"Assigned {len(pending_orders)} pending orders with routes")
    
    def get_depot_load_balancing_info(self) -> Dict:
        """Get information about depot load balancing"""
        depot_info = {}
        
        for depot in self.map.depots:
            total_drones = len(depot.drones)
            available_drones = len(depot.get_available_drones())
            busy_drones = total_drones - available_drones
            
            # Avoid division by zero
            utilization_rate = 0.0
            if total_drones > 0:
                utilization_rate = (busy_drones / total_drones) * 100
            
            depot_info[depot.id] = {
                'total_drones': total_drones,
                'available_drones': available_drones,
                'busy_drones': busy_drones,
                'utilization_rate': utilization_rate
            }
        
        return depot_info


class OrderValidator:
    """Validates orders for feasibility and constraints"""
    
    @staticmethod
    def validate_order_feasibility(order: Order, drone: Drone) -> bool:
        """Check if an order can be feasibly delivered by a drone"""
        # Check distance constraints
        total_distance = (drone.position.distance_to(order.store_position) +
                          order.store_position.distance_to(order.customer_position) +
                          order.customer_position.distance_to(drone.depot.get_center()))
        
        # Check battery life (simplified calculation)
        max_distance = drone.battery_level * drone.speed * config.DRONE_BATTERY_LIFE
        
        return total_distance <= max_distance
    
    @staticmethod
    def validate_order_constraints(order: Order) -> bool:
        """Validate basic order constraints"""
        # Check if customer and store are different
        if order.customer_id == order.store_id:
            return False
        
        # Check if positions are valid
        if (order.customer_position.x < 0 or order.customer_position.x > config.MAP_WIDTH or
            order.customer_position.y < 0 or order.customer_position.y > config.MAP_HEIGHT or
            order.store_position.x < 0 or order.store_position.x > config.MAP_WIDTH or
            order.store_position.y < 0 or order.store_position.y > config.MAP_HEIGHT):
            return False
        
        return True
