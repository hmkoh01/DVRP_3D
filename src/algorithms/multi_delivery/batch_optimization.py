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
        if self.processing_batch:  # 중복 실행 방지
            return {}

        self.processing_batch = True  # 처리 시작
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

        for route in solution.routes:
            orders = route.get_orders()
            if not orders:
                continue

            drone = route.drone
            
            # 이미 할당된 드론은 건너뛰기
            if drone.status != DroneStatus.IDLE:
                print(f"⚠️ Drone {drone.id} is not idle (status: {drone.status}), skipping assignment")
                failed_orders.extend(orders)
                continue

            exact_route = self.route_evaluator.calculate_exact_route(route)
            if not exact_route or len(exact_route) < 2:
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
        """Map key waypoints to their respective orders."""
        mapping = {}
        visit_index = 0
        last_matched_idx = 0
        threshold = max(config.NODE_OFFSET, 5.0)

        for idx, waypoint in enumerate(exact_route):
            if visit_index >= len(route.visits):
                break

            visit = route.visits[visit_index]
            if waypoint.distance_to(visit.position) <= threshold:
                mapping[idx] = visit.order
                visit_index += 1
                last_matched_idx = idx

        # If some visits were not matched (due to numerical issues), map them sequentially
        while visit_index < len(route.visits):
            mapping[last_matched_idx] = route.visits[visit_index].order
            visit_index += 1
            last_matched_idx += 1

        return mapping

