# Configuration file for DVRP simulation (3D)

# Map configuration
MAP_SEED = 123
ORDER_SEED = 456
NODE_OFFSET = 5.0

# 3D Map dimensions
MAP_WIDTH = 1000  # X-axis (horizontal)
MAP_HEIGHT = 1000  # Z-axis (horizontal, called HEIGHT for backward compatibility)
MAP_DEPTH = 1000  # Z-axis (depth, same as MAP_HEIGHT)
MAX_MAP_HEIGHT = 150  # Y-axis (maximum altitude for drones)

# Building configuration
TOTAL_BUILDINGS = 30
BUILDING_MIN_SIZE = 20  # Minimum width/depth
BUILDING_MAX_SIZE = 80  # Maximum width/depth
BUILDING_MIN_HEIGHT = 50  # Minimum building height (Y-axis)
BUILDING_MAX_HEIGHT = 200  # Maximum building height (Y-axis)
FLOOR_HEIGHT = 10.0  # Height of each floor in meters
BUILDING_SAFETY_MARGIN = 15.0  # Safety distance between buildings (in meters)

# Building type ratios (should add up to <= 1.0)
STORE_RATIO = 0.3  # 30% of buildings are stores
CUSTOMER_RATIO = 0.5  # 50% of buildings are customers
# Remaining buildings (20%) will be empty buildings

# Depot configuration
TOTAL_DEPOTS = 3
DRONES_PER_DEPOT = 5
DEPOT_SIZE = 20
DEPOT_SAFETY_MARGIN = 30.0  # Safety distance from buildings (in meters)

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

# Routing visualization
VISUALIZE_FIRST_ROUTE = False  # Visualize the first delivery route
VISUALIZE_ALL_ROUTES = False  # Visualize all delivery routes (warning: can be slow)

# Colors for visualization
COLORS = {
    'building': (255, 255, 255),  # Empty buildings: white
    'store': (0, 255, 0),
    'customer': (255, 0, 0),
    'depot': (0, 0, 255),
    'drone': (255, 255, 0),
    'route': (255, 165, 0),
    'background': (240, 240, 240)
}