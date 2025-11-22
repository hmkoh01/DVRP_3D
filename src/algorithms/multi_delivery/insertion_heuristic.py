"""
Real-time insertion heuristic (Approach A) for multi-delivery routing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import config
from ...models.entities import (
    Drone,
    DroneStatus,
    Order,
    OrderStatus,
    Position,
)
from ..routing import DroneRouteOptimizer
from .route_evaluator import RouteEvaluator
from .solution_representation import Route, Visit


@dataclass
class InsertionCandidate:
    drone: Drone
    store_index: int
    customer_index: int
    cost_delta: float
    route: Route


class InsertionHeuristicStrategy:
    """Assigns incoming orders by inserting them into existing routes."""

    def __init__(self, route_optimizer: DroneRouteOptimizer, map_obj):
        self.route_optimizer = route_optimizer
        self.map = map_obj
        self.route_evaluator = RouteEvaluator(route_optimizer.routing_algorithm)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def assign_order(self, order: Order, current_time: float) -> bool:
        candidates = self._generate_candidates(order, current_time)
        if not candidates:
            return False

        best = min(candidates, key=lambda c: c.cost_delta)
        return self._apply_insertion(best, order, current_time)

    # ------------------------------------------------------------------ #
    # Candidate generation
    # ------------------------------------------------------------------ #
    def _generate_candidates(self, order: Order, current_time: float) -> List[InsertionCandidate]:
        store_visit = Visit(
            order_id=order.id,
            position=order.store_position.copy(),
            visit_type="store",
            order=order,
        )
        customer_visit = Visit(
            order_id=order.id,
            position=order.customer_position.copy(),
            visit_type="customer",
            order=order,
        )

        drones = self._collect_candidate_drones()
        candidates: List[InsertionCandidate] = []

        for drone in drones:
            base_route = self._build_route_from_drone(drone, current_time)
            base_cost = self.route_evaluator.estimate_route_cost(base_route)

            for store_idx in range(len(base_route.visits) + 1):
                for customer_idx in range(store_idx + 1, len(base_route.visits) + 2):
                    test_route = copy.deepcopy(base_route)
                    test_route.visits.insert(store_idx, store_visit)
                    test_route.visits.insert(customer_idx, customer_visit)

                    is_valid, reason = test_route.is_valid(current_time)
                    if not is_valid:
                        continue

                    new_cost = self.route_evaluator.estimate_route_cost(test_route)
                    candidates.append(
                        InsertionCandidate(
                            drone=drone,
                            store_index=store_idx,
                            customer_index=customer_idx,
                            cost_delta=new_cost - base_cost,
                            route=test_route,
                        )
                    )

        return candidates

    def _collect_candidate_drones(self) -> List[Drone]:
        drones: List[Drone] = []
        for depot in self.map.depots:
            for drone in depot.drones:
                if drone.status == DroneStatus.IDLE or self._has_capacity(drone):
                    drones.append(drone)
        return drones

    def _has_capacity(self, drone: Drone) -> bool:
        if hasattr(drone, "current_orders"):
            return len(drone.current_orders) < config.DRONE_CAPACITY
        return drone.current_order is None

    def _build_route_from_drone(self, drone: Drone, current_time: float) -> Route:
        route = Route(
            drone_id=drone.id,
            drone=drone,
            visits=[],
            start_position=drone.position.copy(),
            depot_position=drone.depot.get_center().copy(),
            start_time=current_time,
        )

        if drone.status == DroneStatus.IDLE or not drone.route:
            return route

        # Use stored current_orders to rebuild visit list
        if hasattr(drone, "current_orders") and drone.current_orders:
            for order in drone.current_orders:
                route.visits.append(
                    Visit(
                        order_id=order.id,
                        position=order.store_position.copy(),
                        visit_type="store",
                        order=order,
                    )
                )
                route.visits.append(
                    Visit(
                        order_id=order.id,
                        position=order.customer_position.copy(),
                        visit_type="customer",
                        order=order,
                    )
                )

        return route

    # ------------------------------------------------------------------ #
    # Apply insertion
    # ------------------------------------------------------------------ #
    def _apply_insertion(
        self,
        candidate: InsertionCandidate,
        order: Order,
        current_time: float,
    ) -> bool:
        drone = candidate.drone
        route = candidate.route

        exact_route = self.route_evaluator.calculate_exact_route(route)
        if not exact_route or len(exact_route) < 2:
            return False

        if not hasattr(drone, "current_orders"):
            drone.current_orders = []
        if order not in drone.current_orders:
            drone.current_orders.append(order)

        drone.current_order = order
        order.assigned_drone = drone
        order.status = OrderStatus.ASSIGNED

        if drone.status == DroneStatus.IDLE:
            drone.route_waypoint_order_map = self._build_waypoint_order_map(route, exact_route)
            drone.start_delivery(exact_route)
        else:
            drone.route_waypoint_order_map = self._build_waypoint_order_map(route, exact_route)
            self._update_drone_route(drone, exact_route)

        return True

    def _build_waypoint_order_map(self, route: Route, exact_route: List[Position]):
        mapping = {}
        visit_idx = 0
        threshold = max(config.NODE_OFFSET, 5.0)

        for idx, waypoint in enumerate(exact_route):
            if visit_idx >= len(route.visits):
                break

            visit = route.visits[visit_idx]
            if waypoint.distance_to(visit.position) <= threshold:
                mapping[idx] = (visit.order, visit.visit_type)
                visit_idx += 1

        # remaining visits
        while visit_idx < len(route.visits):
            mapping[idx] = (route.visits[visit_idx].order, route.visits[visit_idx].visit_type)
            visit_idx += 1
            idx += 1

        return mapping

    def _update_drone_route(self, drone: Drone, new_route: List[Position]) -> None:
        current_pos = drone.position
        best_idx = 0
        best_dist = float("inf")

        for idx, waypoint in enumerate(new_route):
            dist = current_pos.distance_to(waypoint)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist < config.NODE_OFFSET:
            drone.route = new_route[best_idx + 1 :]
        else:
            drone.route = new_route[best_idx:]

