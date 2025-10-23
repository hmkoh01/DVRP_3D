"""
Clustering algorithms for grouping stores and customers
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import List, Tuple, Dict
from ..models.entities import Building, Position, EntityType, Map
import config


class ClusteringAlgorithm:
    """Base class for clustering algorithms"""
    
    def cluster_entities(self, buildings: List[Building]) -> Dict[int, List[Building]]:
        """Cluster buildings and return cluster assignments"""
        raise NotImplementedError


class KMeansClustering(ClusteringAlgorithm):
    """K-means clustering implementation"""
    
    def __init__(self, n_clusters: int = config.TOTAL_DEPOTS):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def cluster_entities(self, buildings: List[Building]) -> Dict[int, List[Building]]:
        """Cluster buildings using K-means"""
        if len(buildings) < self.n_clusters:
            # Not enough buildings to cluster
            clusters = {}
            for i, building in enumerate(buildings):
                clusters[i] = [building]
            return clusters
        
        # Extract positions
        positions = []
        for building in buildings:
            center = building.get_center()
            positions.append([center.x, center.y])
        
        positions = np.array(positions)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(positions)
        
        # Group buildings by cluster
        clusters = {}
        for i, building in enumerate(buildings):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(building)
        
        return clusters
    
    def get_cluster_centers(self) -> List[Tuple[float, float]]:
        """Get the centers of clusters"""
        if hasattr(self.kmeans, 'cluster_centers_'):
            return [(center[0], center[1]) for center in self.kmeans.cluster_centers_]
        return []


class DBSCANClustering(ClusteringAlgorithm):
    """DBSCAN clustering implementation"""
    
    def __init__(self, eps: float = 100.0, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    def cluster_entities(self, buildings: List[Building]) -> Dict[int, List[Building]]:
        """Cluster buildings using DBSCAN"""
        if len(buildings) < self.min_samples:
            # Not enough buildings for DBSCAN
            clusters = {}
            for i, building in enumerate(buildings):
                clusters[i] = [building]
            return clusters
        
        # Extract positions
        positions = []
        for building in buildings:
            center = building.get_center()
            positions.append([center.x, center.y])
        
        positions = np.array(positions)
        
        # Perform clustering
        cluster_labels = self.dbscan.fit_predict(positions)
        
        # Group buildings by cluster
        clusters = {}
        for i, building in enumerate(buildings):
            cluster_id = cluster_labels[i]
            if cluster_id == -1:  # Noise points
                # Assign noise points to nearest cluster or create individual clusters
                cluster_id = len(clusters)
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(building)
        
        return clusters
    
    def get_cluster_centers(self, clusters: Dict[int, List[Building]]) -> List[Tuple[float, float]]:
        """Calculate cluster centers from clustered buildings"""
        centers = []
        for cluster_buildings in clusters.values():
            if cluster_buildings:
                # Calculate centroid
                total_x = sum(building.get_center().x for building in cluster_buildings)
                total_y = sum(building.get_center().y for building in cluster_buildings)
                center_x = total_x / len(cluster_buildings)
                center_y = total_y / len(cluster_buildings)
                centers.append((center_x, center_y))
        return centers


class MixedClustering:
    """Clustering that considers both stores and customers"""
    
    def __init__(self, clustering_algorithm: str = config.CLUSTERING_ALGORITHM):
        if clustering_algorithm == "kmeans":
            self.algorithm = KMeansClustering()
        elif clustering_algorithm == "dbscan":
            self.algorithm = DBSCANClustering()
        else:
            self.algorithm = KMeansClustering()  # Default
    
    def cluster_stores_and_customers(self, map_obj: Map) -> Dict[int, Dict[str, List[Building]]]:
        """Cluster both stores and customers together"""
        # Get all buildings with entities
        all_buildings = [building for building in map_obj.buildings if building.entity_type]
        
        # Perform clustering
        clusters = self.algorithm.cluster_entities(all_buildings)
        
        # Separate stores and customers within each cluster
        separated_clusters = {}
        for cluster_id, buildings in clusters.items():
            stores = [b for b in buildings if b.entity_type == EntityType.STORE]
            customers = [b for b in buildings if b.entity_type == EntityType.CUSTOMER]
            
            separated_clusters[cluster_id] = {
                'stores': stores,
                'customers': customers
            }
        
        return separated_clusters
    
    def calculate_cluster_metrics(self, clusters: Dict[int, Dict[str, List[Building]]]) -> Dict[int, Dict]:
        """Calculate metrics for each cluster"""
        metrics = {}
        
        for cluster_id, cluster_data in clusters.items():
            stores = cluster_data['stores']
            customers = cluster_data['customers']
            
            # Calculate cluster center
            all_buildings = stores + customers
            if all_buildings:
                center_x = sum(b.get_center().x for b in all_buildings) / len(all_buildings)
                center_y = sum(b.get_center().y for b in all_buildings) / len(all_buildings)
                center = Position(center_x, center_y)
            else:
                center = None
            
            # Calculate demand (number of customers)
            demand = len(customers)
            
            # Calculate supply (number of stores)
            supply = len(stores)
            
            # Calculate average distance from center
            avg_distance = 0
            if center and all_buildings:
                total_distance = sum(b.get_center().distance_to(center) for b in all_buildings)
                avg_distance = total_distance / len(all_buildings)
            
            # Calculate store-customer ratio
            store_customer_ratio = supply / max(demand, 1)
            
            metrics[cluster_id] = {
                'center': center,
                'demand': demand,
                'supply': supply,
                'avg_distance': avg_distance,
                'store_customer_ratio': store_customer_ratio,
                'total_buildings': len(all_buildings)
            }
        
        return metrics
    
    def get_optimal_depot_positions(self, clusters: Dict[int, Dict[str, List[Building]]], 
                                  map_obj: Map) -> List[Position]:
        """Get optimal depot positions based on cluster centers and metrics"""
        metrics = self.calculate_cluster_metrics(clusters)
        depot_positions = []
        
        # Sort clusters by demand (number of customers)
        sorted_clusters = sorted(metrics.items(), 
                               key=lambda x: x[1]['demand'], reverse=True)
        
        for cluster_id, cluster_metrics in sorted_clusters[:config.TOTAL_DEPOTS]:
            center = cluster_metrics['center']
            if center:
                # Try to place depot at cluster center
                depot_pos = self._find_valid_depot_position(center, map_obj)
                if depot_pos:
                    depot_positions.append(depot_pos)
        
        return depot_positions
    
    def _find_valid_depot_position(self, preferred_pos: Position, map_obj: Map) -> Position:
        """Find a valid position for depot near preferred position"""
        import math
        
        depot_size = config.DEPOT_SIZE
        
        # Try the preferred position first
        if map_obj.is_position_valid(preferred_pos, depot_size):
            return preferred_pos
        
        # Try positions in expanding circles around preferred position
        max_radius = min(map_obj.width, map_obj.height) / 4
        radius_step = depot_size
        
        for radius in range(int(radius_step), int(max_radius), int(radius_step)):
            # Try positions at different angles
            for angle in range(0, 360, 30):  # Every 30 degrees
                angle_rad = math.radians(angle)
                offset_x = radius * math.cos(angle_rad)
                offset_y = radius * math.sin(angle_rad)
                
                test_pos = Position(
                    preferred_pos.x + offset_x,
                    preferred_pos.y + offset_y
                )
                
                if map_obj.is_position_valid(test_pos, depot_size):
                    return test_pos
        
        return None


class ClusterAnalyzer:
    """Analyzes cluster quality and provides insights"""
    
    @staticmethod
    def calculate_silhouette_score(clusters: Dict[int, List[Building]]) -> float:
        """Calculate silhouette score for cluster quality (simplified version)"""
        # This is a simplified implementation
        # In a real scenario, you would use sklearn.metrics.silhouette_score
        
        total_score = 0
        total_points = 0
        
        for cluster_id, buildings in clusters.items():
            if len(buildings) <= 1:
                continue
                
            cluster_center = Position(
                sum(b.get_center().x for b in buildings) / len(buildings),
                sum(b.get_center().y for b in buildings) / len(buildings)
            )
            
            for building in buildings:
                # Distance to own cluster center
                a = building.get_center().distance_to(cluster_center)
                
                # Distance to nearest other cluster center
                min_b = float('inf')
                for other_cluster_id, other_buildings in clusters.items():
                    if other_cluster_id != cluster_id and other_buildings:
                        other_center = Position(
                            sum(b.get_center().x for b in other_buildings) / len(other_buildings),
                            sum(b.get_center().y for b in other_buildings) / len(other_buildings)
                        )
                        b_dist = building.get_center().distance_to(other_center)
                        min_b = min(min_b, b_dist)
                
                if min_b != float('inf'):
                    silhouette = (min_b - a) / max(a, min_b)
                    total_score += silhouette
                    total_points += 1
        
        return total_score / max(total_points, 1)
    
    @staticmethod
    def print_cluster_summary(clusters: Dict[int, Dict[str, List[Building]]]):
        """Print a summary of cluster analysis"""
        print("\n=== Cluster Analysis Summary ===")
        
        for cluster_id, cluster_data in clusters.items():
            stores = cluster_data['stores']
            customers = cluster_data['customers']
            
            print(f"Cluster {cluster_id}:")
            print(f"  - Stores: {len(stores)}")
            print(f"  - Customers: {len(customers)}")
            print(f"  - Total buildings: {len(stores) + len(customers)}")
            
            if stores and customers:
                # Calculate average distance between stores and customers in cluster
                total_distance = 0
                count = 0
                for store in stores:
                    for customer in customers:
                        total_distance += store.get_center().distance_to(customer.get_center())
                        count += 1
                
                if count > 0:
                    avg_distance = total_distance / count
                    print(f"  - Average store-customer distance: {avg_distance:.2f}")
        
        print("=" * 35)
