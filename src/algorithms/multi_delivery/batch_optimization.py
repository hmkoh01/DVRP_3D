"""
Batch-based multi-delivery optimizer (Approach B).
"""

from __future__ import annotations

from typing import Dict, List

import config
from ...models.entities import Drone, Order, DroneStatus, OrderStatus
from .lns_solver import LNSSolver
from .route_evaluator import RouteEvaluator


class BatchOptimizationStrategy:
    """Collects orders in short batches and solves them via LNS."""

    def __init__(self, route_optimizer, map_obj):
        self.route_optimizer = route_optimizer
        self.map = map_obj
        self.pending_batch: List[Order] = []
        self.last_batch_time: float = 0.0
        self.processing_batch: bool = False  # 배치 처리 중 플래그

        self.route_evaluator = RouteEvaluator(route_optimizer.routing_algorithm)
        self.lns_solver = LNSSolver(
            route_evaluator=self.route_evaluator,
            max_iterations=config.LNS_ITERATIONS,
            destroy_ratio=config.LNS_DESTROY_RATIO,
        )

    # ----------------------------------------------------------
    # Batch management
    # ----------------------------------------------------------
    def add_order_to_batch(self, order: Order) -> None:
        if order.status == OrderStatus.PENDING and order not in self.pending_batch:
            self.pending_batch.append(order)

    def should_process_batch(self, current_time: float) -> bool:
        if not self.pending_batch:
            return False
        if self.processing_batch:  # 이미 처리 중이면 False
            return False
        elapsed = current_time - self.last_batch_time
        return elapsed >= config.BATCH_WINDOW_SIZE

    # ----------------------------------------------------------
    # Batch execution
    # ----------------------------------------------------------
    def process_batch(self, current_time: float) -> Dict[Drone, List[Order]]:
        if not self.pending_batch:
            return {}
        if self.processing_batch:
            return {}

        self.processing_batch = True
        try:
            available_drones = self._collect_available_drones()
            if not available_drones:
                return {}

            solution = self.lns_solver.solve(
                orders=list(self.pending_batch),
                available_drones=available_drones,
                current_time=current_time,
            )

            assignments = self._apply_solution(solution)
            
            remaining_orders = solution.unassigned_orders.copy()
            self.pending_batch = remaining_orders
            self.last_batch_time = current_time
            return assignments
        finally:
            self.processing_batch = False  # 처리 완료

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _collect_available_drones(self) -> List[Drone]:
        """Return idle drones that can accept new routes."""
        drones: List[Drone] = []
        for depot in self.map.depots:
            for drone in depot.drones:
                if drone.status == DroneStatus.IDLE:
                    drones.append(drone)
        return drones

    def _apply_solution(self, solution) -> Dict[Drone, List[Order]]:
        """Convert solution routes to concrete drone assignments."""
        assignments: Dict[Drone, List[Order]] = {}
        failed_orders: List[Order] = []

        for route_idx, route in enumerate(solution.routes):
            orders = route.get_orders()
            if not orders:
                continue

            # 🔧 수정: route.drone은 copy.deepcopy()로 인해 복사본이므로, 
            # depot에서 실제 드론 객체를 찾아서 사용해야 함
            route_drone_id = route.drone.id
            actual_drone = None
            for depot in self.map.depots:
                for drone in depot.drones:
                    if drone.id == route_drone_id:
                        actual_drone = drone
                        break
                if actual_drone:
                    break
            
            if not actual_drone:
                print(f"      ❌ Drone {route_drone_id} not found in depots, skipping")
                failed_orders.extend(orders)
                continue
            
            # 🔧 수정: route.drone 대신 actual_drone 사용
            drone = actual_drone
            
            if drone.status != DroneStatus.IDLE:
                print(f"      ⚠️ Drone {drone.id} is not idle (status: {drone.status.value}), skipping assignment")
                failed_orders.extend(orders)
                continue

            exact_route = self.route_evaluator.calculate_exact_route(route)
            if not exact_route or len(exact_route) < 2:
                print(f"      ❌ Failed to calculate route or route too short (length: {len(exact_route) if exact_route else 0})")
                failed_orders.extend(orders)
                continue

            drone.current_orders = orders.copy()
            drone.current_order = orders[0] if orders else None
            drone.route_waypoint_order_map = self._build_waypoint_order_map(
                route, exact_route
            )

            for order in orders:
                order.assigned_drone = drone
                order.status = OrderStatus.ASSIGNED

            drone.start_delivery(exact_route)
            assignments[drone] = orders

        # Keep failed orders pending for next batch
        for order in failed_orders:
            if order not in solution.unassigned_orders:
                solution.unassigned_orders.append(order)
        
        return assignments

    def _build_waypoint_order_map(self, route, exact_route):
        """Map key waypoints to their respective orders and visit types.
        
        Returns:
            Dict[int, Tuple[Order, str]]: waypoint 인덱스 -> (Order, visit_type) 매핑
        """
        mapping = {}
        visit_index = 0
        last_matched_idx = 0
        threshold = max(config.NODE_OFFSET, 5.0)

        for idx, waypoint in enumerate(exact_route):
            if visit_index >= len(route.visits):
                break

            visit = route.visits[visit_index]
            if waypoint.distance_to(visit.position) <= threshold:
                # (order, visit_type) 튜플로 저장
                mapping[idx] = (visit.order, visit.visit_type)
                visit_index += 1
                last_matched_idx = idx

        # If some visits were not matched (due to numerical issues), map them sequentially
        while visit_index < len(route.visits):
            visit = route.visits[visit_index]
            mapping[last_matched_idx] = (visit.order, visit.visit_type)
            visit_index += 1
            last_matched_idx += 1

        return mapping

