"""
Route evaluation helpers shared by multi-delivery algorithms.
"""

from __future__ import annotations

from typing import List

import config
from ..routing import RoutingAlgorithm
from ...models.entities import Position
from .solution_representation import Route


class RouteEvaluator:
    """Provides fast estimates vs. exact route generation."""

    def __init__(self, routing_algorithm: RoutingAlgorithm):
        self.routing_algorithm = routing_algorithm

    def estimate_route_cost(self, route: Route) -> float:
        """Return straight-line distance as a fast proxy cost."""
        return route.get_total_distance_estimate()

    def estimate_route_costs(self, routes: List[Route]) -> float:
        """Return aggregated cost for a set of routes."""
        return sum(self.estimate_route_cost(route) for route in routes)

    def estimate_battery_usage(self, route: Route) -> float:
        """Return conservative battery usage estimate."""
        return route.get_battery_required_estimate()

    def check_battery_feasibility(self, route: Route) -> bool:
        """Return True if the drone has enough battery for the route."""
        required = self.estimate_battery_usage(route)
        available = route.drone.battery_level * config.DRONE_BATTERY_LIFE
        return required <= available

    def calculate_exact_route(self, route: Route) -> List[Position]:
        """Generate the full waypoint list using the configured routing algorithm."""
        if not route.start_position or not route.depot_position:
            return []

        waypoints = [visit.position for visit in route.visits]
        return self.routing_algorithm.calculate_route(
            route.start_position,
            waypoints,
            route.depot_position,
        )

