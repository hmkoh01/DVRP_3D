"""
Order management system for dynamic order generation and depot assignment (3D)
"""

import random
import time
import math
from typing import List, Dict, Optional, Tuple
# (수정) DroneStatus 임포트
from ..models.entities import (
    Order, Depot, Drone, OrderStatus, DroneStatus, Position, Store, Customer
)
from .clustering import MixedClustering
from .routing import DroneRouteOptimizer, SimpleRouting, MultiLevelAStarRouting
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
        time_since_last = current_time - self.last_generation_time
        
        probability = self.generation_rate * time_since_last
        
        if random.random() < probability:
            self.last_generation_time = current_time
            return self._create_random_order(current_time)
        
        return None
    
    def _create_random_order(self, current_time: float) -> Order:
        customers = self.map.customers
        stores = self.map.stores
        
        if not customers or not stores:
            return None
        
        customer_entity = random.choice(customers)
        store_entity = random.choice(stores)
        
        order = Order(
            id=self.order_counter,
            customer_id=customer_entity.id,
            store_id=store_entity.id,
            customer_position=customer_entity.get_center(),
            store_position=store_entity.get_center(),
            created_time=current_time
        )
        
        self.order_counter += 1
        return order
    
    def generate_batch_orders(self, current_time: float, batch_size: int = 5) -> List[Order]:
        orders = []
        for _ in range(batch_size):
            order = self._create_random_order(current_time)
            if order:
                orders.append(order)
        return orders


class DepotSelector:
    
    def __init__(self, map_obj):
        self.map = map_obj
        self.clustering_algorithm = MixedClustering()
    
    def select_best_depot(self, order: Order) -> Optional[Depot]:
        if not self.map.depots:
            return None
        
        depot_scores = {}
        
        for depot in self.map.depots:
            score = self._calculate_depot_score(depot, order)
            depot_scores[depot.id] = score
        
        if depot_scores:
            best_depot_id = max(depot_scores.keys(), key=lambda k: depot_scores[k])
            return next(depot for depot in self.map.depots if depot.id == best_depot_id)
        
        return None
    
    def _calculate_depot_score(self, depot: Depot, order: Order) -> float:
        available_drones = depot.get_available_drones()
        if not available_drones:
            return 0.0
        
        depot_to_store_distance = depot.get_center().distance_to(order.store_position)
        
        store_to_customer_distance = order.get_distance()
        
        total_distance = depot_to_store_distance + store_to_customer_distance
        
        customer_to_depot_distance = order.customer_position.distance_to(depot.get_center())
        
        complete_trip_distance = total_distance + customer_to_depot_distance
        
        drone_bonus = len(available_drones) * 10
        
        distance_score = max(0, 1000 - complete_trip_distance)
        
        battery_score = 100
        
        total_score = distance_score + drone_bonus + battery_score
        
        return total_score
    
    def get_depot_assignment_metrics(self, orders: List[Order]) -> Dict[int, Dict]:
        depot_stats = {}
        
        for depot in self.map.depots:
            depot_stats[depot.id] = {
                'total_orders': 0,
                'available_drones': len(depot.get_available_drones()),
                'total_drones': len(depot.drones),
                'avg_distance': 0.0,
                'total_distance': 0.0
            }
        
        for order in orders:
            if order.assigned_drone and order.assigned_drone.depot:
                depot_id = order.assigned_drone.depot.id
                if depot_id in depot_stats:
                    depot_stats[depot_id]['total_orders'] += 1
                    
                    depot = order.assigned_drone.depot
                    distance = (depot.position.distance_to(order.store_position) + 
                                order.store_position.distance_to(order.customer_position) +
                                order.customer_position.distance_to(depot.position))
                    
                    depot_stats[depot_id]['total_distance'] += distance
        
        for depot_id, stats in depot_stats.items():
            if stats['total_orders'] > 0:
                stats['avg_distance'] = stats['total_distance'] / stats['total_orders']
        
        return depot_stats


class OrderManager:
    
    def __init__(self, map_obj, seed: Optional[int] = None):
        self.map = map_obj
        self.orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.order_generator = OrderGenerator(map_obj, seed=seed)
        self.depot_selector = DepotSelector(map_obj)
        
        print("  Initializing 3D routing...")
        self.route_optimizer = DroneRouteOptimizer(MultiLevelAStarRouting(map_obj, k_levels=3))
        
        # 시각화 설정
        self.first_route_visualized = False
    
    def process_orders(self, current_time: float) -> List[Order]:
        new_order = self.order_generator.generate_random_order(current_time)
        if new_order:
            self.orders.append(new_order)
            print(f"📝 New order #{new_order.id}: Customer {new_order.customer_id} -> Store {new_order.store_id}")
        
        new_completed = []
        for order in self.orders[:]:
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
        best_depot = self.depot_selector.select_best_depot(order)
        
        if best_depot:
            assigned_drone = best_depot.assign_drone(order)
            if assigned_drone:
                order.assigned_drone = assigned_drone
                order.status = OrderStatus.ASSIGNED
                
                try:
                    # 시각화 옵션 결정
                    should_visualize = False
                    if config.VISUALIZE_ALL_ROUTES:
                        should_visualize = True
                    elif config.VISUALIZE_FIRST_ROUTE and not self.first_route_visualized:
                        should_visualize = True
                        self.first_route_visualized = True
                        print(f"\n{'='*60}\n📊 Visualizing first delivery route\n{'='*60}")
                    
                    route = self.route_optimizer.optimize_delivery_route(assigned_drone, order, visualize=should_visualize)
                    
                    # (수정) 경로 탐색 실패 시 할당 해제
                    if not route or len(route) < 2:
                        print(f"❌ Order {order.id}: Route calculation FAILED")
                        order.status = OrderStatus.PENDING
                        order.assigned_drone = None
                        assigned_drone.current_order = None
                        assigned_drone.status = DroneStatus.IDLE
                    else:
                        assigned_drone.start_delivery(route)
                        print(f"✓ Order {order.id} assigned to Depot {best_depot.id}, Drone {assigned_drone.id}")
                
                except Exception as e:
                    print(f"Failed to set route for order {order.id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # (수정) 예외 발생 시에도 할당 해제
                    order.status = OrderStatus.PENDING
                    order.assigned_drone = None
                    assigned_drone.current_order = None
                    assigned_drone.status = DroneStatus.IDLE
                
            else:
                # print(f"No available drones at depot {best_depot.id} for order {order.id}")
                pass
        else:
            # print(f"No suitable depot found for order {order.id}")
            pass
    
    def get_order_statistics(self) -> Dict:
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
        if current_time is None:
            current_time = time.time()
        
        test_orders = self.order_generator.generate_batch_orders(current_time, num_orders)
        self.orders.extend(test_orders)
        
        print(f"Generated {len(test_orders)} test orders")
        return test_orders
    
    def assign_all_pending_orders(self):
        pending_orders = [o for o in self.orders if o.status == OrderStatus.PENDING]
        
        for order in pending_orders:
            self._assign_order_to_depot(order)
        
        print(f"Assigned {len(pending_orders)} pending orders with routes")
    
    def get_depot_load_balancing_info(self) -> Dict:
        depot_info = {}
        
        for depot in self.map.depots:
            total_drones = len(depot.drones)
            available_drones = len(depot.get_available_drones())
            busy_drones = total_drones - available_drones
            
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
    
    @staticmethod
    def validate_order_feasibility(order: Order, drone: Drone) -> bool:
        total_distance = (drone.position.distance_to(order.store_position) +
                          order.store_position.distance_to(order.customer_position) +
                          order.customer_position.distance_to(drone.depot.get_center()))
        
        max_distance = drone.battery_level * drone.speed * config.DRONE_BATTERY_LIFE
        
        return total_distance <= max_distance
    
    @staticmethod
    def validate_order_constraints(order: Order) -> bool:
        if order.customer_id == order.store_id:
            return False
        
        if (order.customer_position.x < 0 or order.customer_position.x > config.MAP_WIDTH or
            order.customer_position.y < 0 or order.customer_position.y > config.MAP_HEIGHT or
            order.store_position.x < 0 or order.store_position.x > config.MAP_WIDTH or
            order.store_position.y < 0 or order.store_position.y > config.MAP_HEIGHT):
            return False
        
        return True