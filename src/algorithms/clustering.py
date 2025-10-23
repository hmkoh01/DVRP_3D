"""
Clustering algorithms for grouping stores and customers (3D environment)
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import List, Tuple, Dict, Union
from ..models.entities import Building, Position, EntityType, Map, Store, Customer
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
    """Clustering that considers both stores and customers in 3D environment
    
    Uses 2D projection (x, z) with floor-based density weighting.
    Buildings with more floors will have more weight in clustering.
    """
    
    def __init__(self, n_clusters: int = config.TOTAL_DEPOTS):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers_ = None
    
    def cluster_stores_and_customers(self, map_obj: Map) -> Dict[int, Dict[str, List[Union[Store, Customer]]]]:
        """Cluster both stores and customers together based on 2D projection with density
        
        Creates a flat list of 2D coordinates (x, z) from all Store and Customer entities.
        Does NOT remove duplicates - if a building has 10 floors, (x, z) appears 10 times.
        This naturally weights K-means toward high-density (multi-floor) areas.
        
        Args:
            map_obj: Map containing Store and Customer entities
            
        Returns:
            Dictionary mapping cluster_id to {'stores': [...], 'customers': [...]}
        """
        # 1. Create flat 2D projection list with ALL entities (including floor duplicates)
        all_2d_coords = []
        all_stores = []
        all_customers = []
        
        # Add all stores (one entry per floor)
        for store in map_obj.stores:
            pos = store.get_center()
            all_2d_coords.append([pos.x, pos.z])  # Use (x, z) for horizontal plane
            all_stores.append(store)
        
        # Add all customers (one entry per floor)
        for customer in map_obj.customers:
            pos = customer.get_center()
            all_2d_coords.append([pos.x, pos.z])  # Use (x, z) for horizontal plane
            all_customers.append(customer)
        
        all_entities = all_stores + all_customers
        
        print(f"  Clustering {len(all_stores)} stores + {len(all_customers)} customers")
        print(f"  Total 2D points (with floor duplicates): {len(all_2d_coords)}")
        
        if len(all_2d_coords) < self.n_clusters:
            print(f"  Warning: Not enough entities for {self.n_clusters} clusters")
            # Create simple clusters
            clusters = {}
            for i, entity in enumerate(all_entities):
                cluster_id = i % self.n_clusters
                if cluster_id not in clusters:
                    clusters[cluster_id] = {'stores': [], 'customers': []}
                
                if isinstance(entity, Store):
                    clusters[cluster_id]['stores'].append(entity)
                else:
                    clusters[cluster_id]['customers'].append(entity)
            
            return clusters
        
        # 2. Run K-means on the complete 2D coordinate list (with all duplicates)
        coords_array = np.array(all_2d_coords)
        cluster_labels = self.kmeans.fit_predict(coords_array)
        self.cluster_centers_ = self.kmeans.cluster_centers_  # Save for get_optimal_depot_positions
        
        print(f"  K-means completed. Cluster centers:")
        for i, center in enumerate(self.cluster_centers_):
            print(f"    Cluster {i}: ({center[0]:.1f}, {center[1]:.1f})")
        
        # 3. Group entities by cluster
        clusters = {}
        for i, entity in enumerate(all_entities):
            cluster_id = int(cluster_labels[i])
            
            if cluster_id not in clusters:
                clusters[cluster_id] = {'stores': [], 'customers': []}
            
            if isinstance(entity, Store):
                clusters[cluster_id]['stores'].append(entity)
            else:
                clusters[cluster_id]['customers'].append(entity)
        
        return clusters
    
    def calculate_cluster_metrics(self, clusters: Dict[int, Dict[str, List[Union[Store, Customer]]]]) -> Dict[int, Dict]:
        """Calculate metrics for each cluster (3D entities)"""
        metrics = {}
        
        for cluster_id, cluster_data in clusters.items():
            stores = cluster_data['stores']
            customers = cluster_data['customers']
            
            # Calculate cluster center in 3D
            all_entities = stores + customers
            if all_entities:
                center_x = sum(e.get_center().x for e in all_entities) / len(all_entities)
                center_y = sum(e.get_center().y for e in all_entities) / len(all_entities)
                center_z = sum(e.get_center().z for e in all_entities) / len(all_entities)
                center = Position(center_x, center_y, center_z)
            else:
                center = None
            
            # Calculate demand (number of customer entities = total floors with customers)
            demand = len(customers)
            
            # Calculate supply (number of store entities = total floors with stores)
            supply = len(stores)
            
            # Calculate average 3D distance from center
            avg_distance = 0
            if center and all_entities:
                total_distance = sum(e.get_center().distance_to(center) for e in all_entities)
                avg_distance = total_distance / len(all_entities)
            
            # Calculate store-customer ratio
            store_customer_ratio = supply / max(demand, 1)
            
            metrics[cluster_id] = {
                'center': center,
                'demand': demand,
                'supply': supply,
                'avg_distance': avg_distance,
                'store_customer_ratio': store_customer_ratio,
                'total_entities': len(all_entities)
            }
        
        return metrics
    
    def get_optimal_depot_positions(self, clusters: Dict[int, Dict[str, List[Union[Store, Customer]]]], 
                                  map_obj: Map) -> List[Position]:
        """Get optimal depot positions from K-means cluster centers
        
        Returns the 2D cluster centers computed by K-means as ground-level positions.
        These positions are naturally weighted by floor density.
        
        Args:
            clusters: Cluster dictionary (not directly used, but kept for compatibility)
            map_obj: Map object (not directly used, but kept for compatibility)
            
        Returns:
            List of 2D Position objects (x, 0, z) representing depot locations
        """
        depot_positions = []
        
        if self.cluster_centers_ is None:
            print("  Warning: No cluster centers available")
            return depot_positions
        
        # Convert K-means cluster centers to Position objects (ground level)
        print(f"  Converting {len(self.cluster_centers_)} cluster centers to depot positions...")
        for i, center in enumerate(self.cluster_centers_):
            # center is [x, z] from K-means
            # Create ground-level position (y=0)
            depot_pos = Position(
                x=float(center[0]),
                y=0,  # Ground level
                z=float(center[1])
            )
            depot_positions.append(depot_pos)
            print(f"    - Depot position {i}: ({depot_pos.x:.1f}, 0, {depot_pos.z:.1f})")
        
        return depot_positions


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
    def print_cluster_summary(clusters: Dict[int, Dict[str, List[Union[Store, Customer]]]]):
        """Print a summary of cluster analysis (3D entities)"""
        print("\n=== Cluster Analysis Summary ===")
        
        total_stores = 0
        total_customers = 0
        
        for cluster_id, cluster_data in clusters.items():
            stores = cluster_data['stores']
            customers = cluster_data['customers']
            
            total_stores += len(stores)
            total_customers += len(customers)
            
            print(f"Cluster {cluster_id}:")
            print(f"  - Store entities: {len(stores)} (floors)")
            print(f"  - Customer entities: {len(customers)} (floors)")
            print(f"  - Total entities: {len(stores) + len(customers)}")
            
            # Calculate unique buildings per cluster
            if stores:
                unique_store_buildings = len(set(s.building_id for s in stores))
                print(f"  - Unique store buildings: {unique_store_buildings}")
            
            if customers:
                unique_customer_buildings = len(set(c.building_id for c in customers))
                print(f"  - Unique customer buildings: {unique_customer_buildings}")
            
            if stores and customers:
                # Calculate average 3D distance between stores and customers in cluster
                total_distance = 0
                count = 0
                # Sample to avoid O(n^2) for large clusters
                sample_size = min(10, len(stores), len(customers))
                import random
                sample_stores = random.sample(stores, sample_size) if len(stores) > sample_size else stores
                sample_customers = random.sample(customers, sample_size) if len(customers) > sample_size else customers
                
                for store in sample_stores:
                    for customer in sample_customers:
                        total_distance += store.get_center().distance_to(customer.get_center())
                        count += 1
                
                if count > 0:
                    avg_distance = total_distance / count
                    print(f"  - Avg store-customer 3D distance (sampled): {avg_distance:.2f}m")
        
        print(f"\nTotal: {total_stores} store entities + {total_customers} customer entities")
        print("=" * 50)
