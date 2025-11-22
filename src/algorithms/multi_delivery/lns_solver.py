"""
Large Neighborhood Search (LNS) solver for batch optimization.
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

import config
from ...models.entities import Drone
from .route_evaluator import RouteEvaluator
from .solution_representation import Route, Visit, Solution


class LNSSolver:
    """Improves multi-delivery routes using Large Neighborhood Search."""

    def __init__(
        self,
        route_evaluator: RouteEvaluator,
        max_iterations: Optional[int] = None,
        destroy_ratio: Optional[float] = None,
    ):
        self.route_evaluator = route_evaluator
        self.max_iterations = max_iterations or config.LNS_ITERATIONS
        self.destroy_ratio = destroy_ratio or config.LNS_DESTROY_RATIO

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def solve(
        self,
        orders,
        available_drones: List[Drone],
        current_time: float,
    ) -> Solution:
        """Run LNS and return the best solution found."""
        current_solution = self._create_initial_solution(
            orders, available_drones, current_time
        )
        best_solution = copy.deepcopy(current_solution)
        best_cost = self.route_evaluator.estimate_route_costs(best_solution.routes)

        for _ in range(self.max_iterations):
            destroyed = self._destroy(current_solution)
            repaired = self._repair(destroyed, current_time)

            cost = self.route_evaluator.estimate_route_costs(repaired.routes)
            if self._is_solution_valid(repaired):
                if cost < best_cost:
                    best_solution = copy.deepcopy(repaired)
                    best_cost = cost

                if cost <= self.route_evaluator.estimate_route_costs(
                    current_solution.routes
                ):
                    current_solution = repaired
                elif random.random() < 0.1:  # Accept worse solution with small probability
                    current_solution = repaired

        return best_solution

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_initial_solution(
        self,
        orders,
        available_drones: List[Drone],
        current_time: float,
    ) -> Solution:
        """Greedy insertion to build an initial solution."""
        routes: List[Route] = []
        for drone in available_drones:
            route = Route(
                drone_id=drone.id,
                drone=drone,
                visits=[],
                start_position=drone.position.copy(),
                depot_position=drone.depot.get_center().copy(),
                start_time=current_time,
            )
            routes.append(route)

        unassigned = []
        for order in orders:
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

            insert_result = self._find_best_insertion(
                routes, store_visit, customer_visit, current_time
            )
            if insert_result is None:
                unassigned.append(order)
                continue

            route, store_idx, customer_idx = insert_result
            route.visits.insert(store_idx, store_visit)
            route.visits.insert(customer_idx, customer_visit)

        return Solution(routes=routes, unassigned_orders=unassigned)

    def _destroy(self, solution: Solution) -> Solution:
        """Randomly remove a portion of assigned orders."""
        destroyed = copy.deepcopy(solution)
        removed_orders = []

        for route in destroyed.routes:
            orders_in_route = route.get_orders()
            if not orders_in_route:
                continue

            remove_count = max(1, int(len(orders_in_route) * self.destroy_ratio))
            remove_count = min(remove_count, len(orders_in_route))
            orders_to_remove = random.sample(orders_in_route, remove_count)

            route.visits = [
                visit for visit in route.visits if visit.order not in orders_to_remove
            ]

            removed_orders.extend(orders_to_remove)

        for order in removed_orders:
            if order not in destroyed.unassigned_orders:
                destroyed.unassigned_orders.append(order)

        return destroyed

    def _repair(self, solution: Solution, current_time: float) -> Solution:
        """Reinsert unassigned orders using cheapest insertion."""
        repaired = copy.deepcopy(solution)

        for order in repaired.unassigned_orders[:]:
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

            insert_result = self._find_best_insertion(
                repaired.routes, store_visit, customer_visit, current_time
            )
            if insert_result is None:
                continue

            route, store_idx, customer_idx = insert_result
            route.visits.insert(store_idx, store_visit)
            route.visits.insert(customer_idx, customer_visit)
            repaired.unassigned_orders.remove(order)

        return repaired

    def _find_best_insertion(
        self,
        routes: List[Route],
        store_visit: Visit,
        customer_visit: Visit,
        current_time: float,
    ) -> Optional[Tuple[Route, int, int]]:
        """Find the lowest cost insertion positions for an order."""
        best_option = None
        best_cost = float("inf")

        for route in routes:
            if len(route.get_orders()) >= config.DRONE_CAPACITY:
                continue

            base_cost = self.route_evaluator.estimate_route_cost(route)

            for store_idx in range(len(route.visits) + 1):
                for customer_idx in range(store_idx + 1, len(route.visits) + 2):
                    test_route = copy.deepcopy(route)
                    test_route.visits.insert(store_idx, store_visit)
                    test_route.visits.insert(customer_idx, customer_visit)
                    is_valid, reason = test_route.is_valid(current_time)
                    if not is_valid:
                        continue

                    new_cost = self.route_evaluator.estimate_route_cost(test_route)
                    cost_increase = new_cost - base_cost
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_option = (route, store_idx, customer_idx)

        return best_option

    def _is_solution_valid(self, solution: Solution) -> bool:
        """Check basic validity of all routes."""
        for route in solution.routes:
            reference_time = route.start_time if route.start_time is not None else 0.0
            is_valid, _ = route.is_valid(reference_time)
            if not is_valid:
                return False
        return True


