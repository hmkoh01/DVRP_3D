"""
Real-time simulation engine for drone delivery system
"""

import time
import threading
import queue
from typing import List, Dict, Optional, Callable
from ..models.entities import Drone, Order, OrderStatus, DroneStatus
from ..algorithms.order_manager import OrderManager
import config


class SimulationEngine:
    """Main simulation engine for the drone delivery system"""
    
    def __init__(self, map_obj, order_manager: OrderManager = None):
        self.map = map_obj
        self.order_manager = order_manager or OrderManager(map_obj)
        
        # Simulation state
        self.is_running = False
        self.simulation_time = 0.0
        self.speed_multiplier = config.SIMULATION_SPEED
        self.last_update_time = time.time()
        
        # Event system
        self.event_queue = queue.Queue()
        self.event_handlers = {}
        
        # Statistics
        self.stats = {
            'total_orders_processed': 0,
            'total_deliveries_completed': 0,
            'average_delivery_time': 0.0,
            'total_drone_distance': 0.0,
            'simulation_duration': 0.0
        }
        
        # Threading
        self.simulation_thread = None
        self.update_lock = threading.Lock()
    
    def start_simulation(self):
        """Start the simulation"""
        if self.is_running:
            print("Simulation is already running")
            return
        
        print("Starting simulation...")
        self.is_running = True
        self.paused = False
        self.simulation_time = 0.0
        self.last_update_time = time.time()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self._emit_event('simulation_started', {'time': self.simulation_time})
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.is_running:
            print("Simulation is not running")
            return
        
        print("Stopping simulation...")
        self.is_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join()
        
        self._emit_event('simulation_stopped', {'time': self.simulation_time})
    
    def pause_simulation(self):
        """Pause the simulation"""
        if not self.paused:
            self.paused = True
            self._emit_event('simulation_paused', {'time': self.simulation_time})
    
    def resume_simulation(self):
        """Resume the simulation"""
        if self.paused:
            self.paused = False
            self.last_update_time = time.time()
            self._emit_event('simulation_resumed', {'time': self.simulation_time})
    
    def set_speed(self, speed_multiplier: float):
        """Set simulation speed multiplier"""
        self.speed_multiplier = max(0.1, min(10.0, speed_multiplier))
        print(f"Simulation speed set to {self.speed_multiplier}x")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        while self.is_running:
            try:
                if self.paused: continue
                current_time = time.time()
                real_delta_time = current_time - self.last_update_time
                
                # Apply speed multiplier
                simulation_delta_time = real_delta_time * self.speed_multiplier
                self.simulation_time += simulation_delta_time
                self.last_update_time = current_time
                
                # Update simulation
                self._update_simulation(simulation_delta_time)
                
                # Process events
                self._process_events()
                
                # Sleep to control update frequency
                time.sleep(0.016)
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                break
        
        print("Simulation loop ended")
    
    def _update_simulation(self, delta_time: float):
        """Update simulation state"""
        with self.update_lock:
            # Update order manager
            completed_orders = self.order_manager.process_orders(self.simulation_time)
            
            # Update drones
            self._update_drones(delta_time)
            
            # Update statistics
            self._update_statistics(completed_orders)
            
            # Emit update event
            self._emit_event('simulation_update', {
                'time': self.simulation_time,
                'delta_time': delta_time,
                'completed_orders': len(completed_orders)
            })
    
    def _update_drones(self, delta_time: float):
        """Update all drones in the simulation"""
        for depot in self.map.depots:
            for drone in depot.drones:
                if drone.status != DroneStatus.IDLE:
                    distance_moved = drone.speed * delta_time
                    self.stats['total_drone_distance'] += distance_moved
                
                self._update_drone(drone, delta_time)
    
    def _update_drone(self, drone: Drone, delta_time: float):
        """
        개별 드론을 업데이트합니다.
        이제 드론이 스스로 위치와 상태를 관리하도록 update_position만 호출합니다.
        """
        if drone.status == DroneStatus.IDLE:
            return

        # 드론의 위치 업데이트 (상태 변경은 이 함수 안에서 자동으로 이루어짐)
        drone.update_position(delta_time)
    
    def _update_statistics(self, completed_orders: List[Order]):
        """Update simulation statistics"""
        self.stats['total_orders_processed'] += len(completed_orders)
        
        if completed_orders:
            # Calculate average delivery time
            total_delivery_time = 0
            valid_orders = 0
            for order in completed_orders:
                if hasattr(order, 'created_time') and order.created_time is not None:
                    delivery_time = self.simulation_time - order.created_time
                    if delivery_time >= 0:  # Ensure non-negative delivery time
                        total_delivery_time += delivery_time
                        valid_orders += 1
            
            if valid_orders > 0:
                avg_delivery_time = total_delivery_time / valid_orders
                
                # Update running average
                if self.stats['total_deliveries_completed'] > 0:
                    total_processed = self.stats['total_deliveries_completed'] + valid_orders
                    if total_processed > 0:
                        self.stats['average_delivery_time'] = (
                            (self.stats['average_delivery_time'] * self.stats['total_deliveries_completed'] + 
                             avg_delivery_time * valid_orders) / total_processed
                        )
                else:
                    self.stats['average_delivery_time'] = avg_delivery_time
                
                self.stats['total_deliveries_completed'] += valid_orders
        
        self.stats['simulation_duration'] = self.simulation_time
    
    def assign_order_to_drone(self, order: Order) -> bool:
        """Manually assign an order to a drone using OrderManager"""
        if order.status != OrderStatus.PENDING:
            return False
        
        # Use OrderManager to assign the order (which will handle route setting)
        self.order_manager._assign_order_to_depot(order)
        
        if order.status == OrderStatus.ASSIGNED and order.assigned_drone:
            self._emit_event('order_assigned', {
                'order_id': order.id,
                'drone_id': order.assigned_drone.id,
                'depot_id': order.assigned_drone.depot.id if order.assigned_drone.depot else None
            })
            return True
        
        return False
    
    def get_simulation_state(self) -> Dict:
        """Get current simulation state"""
        with self.update_lock:
            return {
                'is_running': not self.paused,
                'simulation_time': self.simulation_time,
                'speed_multiplier': self.speed_multiplier,
                'stats': self.stats.copy(),
                'active_orders': len(self.order_manager.orders),
                'completed_orders': len(self.order_manager.completed_orders),
                'depot_info': self.order_manager.get_depot_load_balancing_info()
            }
    
    def get_drone_positions(self) -> Dict:
        """Get current positions of all drones"""
        drone_positions = {}
        
        for depot in self.map.depots:
            for drone in depot.drones:
                drone_positions[drone.id] = {
                    'position': drone.position,
                    'status': drone.status.value,
                    'depot_id': depot.id,
                    'has_route': drone.route is not None and len(drone.route) > 0
                }
        
        return drone_positions
    
    # Event system
    def _emit_event(self, event_type: str, data: Dict):
        """Emit an event"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': self.simulation_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass  # Drop event if queue is full
    
    def _process_events(self):
        """Process pending events"""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                
                # Call registered handlers
                if event['type'] in self.event_handlers:
                    for handler in self.event_handlers[event['type']]:
                        try:
                            handler(event)
                        except Exception as e:
                            print(f"Error in event handler: {e}")
                
                self.event_queue.task_done()
                
            except queue.Empty:
                break
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def unregister_event_handler(self, event_type: str, handler: Callable):
        """Unregister an event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.stop_simulation()
        
        # Reset state
        self.simulation_time = 0.0
        self.stats = {
            'total_orders_processed': 0,
            'total_deliveries_completed': 0,
            'average_delivery_time': 0.0,
            'drone_utilization': 0.0,
            'simulation_duration': 0.0
        }
        
        # Reset order manager
        self.order_manager.orders = []
        self.order_manager.completed_orders = []
        
        # Reset all drones to idle state
        for depot in self.map.depots:
            for drone in depot.drones:
                drone.status = DroneStatus.IDLE
                drone.current_order = None
                drone.route = None
                drone.position = depot.position
        
        print("Simulation reset to initial state")
