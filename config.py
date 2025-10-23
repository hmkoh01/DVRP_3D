# Configuration file for DVRP simulation

# Map configuration
MAP_SEED = 123
ORDER_SEED = 456
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
TOTAL_BUILDINGS = 50
BUILDING_MIN_SIZE = 20
BUILDING_MAX_SIZE = 80

# Depot configuration
TOTAL_DEPOTS = 3
DRONES_PER_DEPOT = 5
DEPOT_SIZE = 20

# Simulation configuration
SIMULATION_SPEED = 1.0  # Real-time multiplier
ORDER_GENERATION_RATE = 0.001  # Orders per second
MAX_ORDER_DELAY = 300  # Maximum seconds to wait for order

# Drone configuration
DRONE_SPEED = 50  # Units per second
DRONE_CAPACITY = 1  # Number of orders per drone
DRONE_BATTERY_LIFE = 3600  # Seconds

# Clustering configuration
CLUSTERING_ALGORITHM = "kmeans"  # "kmeans" or "dbscan"
MIN_CLUSTER_SIZE = 5

# Colors for visualization
COLORS = {
    'building': (100, 100, 100),
    'store': (0, 255, 0),
    'customer': (255, 0, 0),
    'depot': (0, 0, 255),
    'drone': (255, 255, 0),
    'route': (255, 165, 0),
    'background': (240, 240, 240)
}
