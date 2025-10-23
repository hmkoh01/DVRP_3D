#!/usr/bin/env python3
"""
Main application for Drone Vehicle Routing Problem (DVRP) 2D Simulation
도심환경에서의 드론 음식 배달 경로 최적화 및 시뮬레이션

Author: DVRP Team
Date: 2025
"""
import sys
import time
import threading
from typing import Optional

# Add src to path
sys.path.append('src')

from src.models.entities import Map
from src.algorithms.map_generator import MapGenerator, DepotPlacer
from src.algorithms.clustering import MixedClustering, ClusterAnalyzer
from src.algorithms.order_manager import OrderManager
from src.simulation.simulation_engine import SimulationEngine
try:
    from src.visualization.visualizer import Visualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: pygame not available, visualization disabled")
import config

class DVRPApplication:
    """Main application class for DVRP simulation"""
    
    def __init__(self, map_seed: Optional[int] = None, order_seed: Optional[int] = None):
        self.map_seed = map_seed
        self.order_seed = order_seed
        self.map: Optional[Map] = None
        self.simulation_engine: Optional[SimulationEngine] = None
        self.visualizer: Optional[Visualizer] = None
        self.order_manager: Optional[OrderManager] = None
        
        print("=" * 60)
        print("도심환경에서의 드론 음식 배달 경로 최적화 및 시뮬레이션")
        print("Drone Vehicle Routing Problem (DVRP) 2D Simulation")
        print("=" * 60)
    
    # --- ✨ 초기화 순서를 바로잡은 함수 ---
    def initialize(self):
        """Initialize the application in the correct order."""
        # 1. 지도와 건물 생성
        print("\n1. 지도 생성 중...")
        map_generator = MapGenerator(seed=self.map_seed)
        self.map = map_generator.generate_map()
        
        # 2. Depot 생성 및 배치 (주문 관리자 생성 전으로 이동)
        print("\n2. 클러스터링 및 Depot 배치 중...")
        self._setup_depots()
        
        # 3. 주문 관리 시스템 초기화
        print("\n3. 주문 관리 시스템 초기화 중...")
        self.order_manager = OrderManager(self.map, seed=self.order_seed)
        
        # 4. 시뮬레이션 엔진 초기화
        print("\n4. 시뮬레이션 엔진 초기화 중...")
        self.simulation_engine = SimulationEngine(self.map, self.order_manager)
        
        # 5. 시각화 시스템 초기화
        print("\n5. 시각화 시스템 초기화 중...")
        if VISUALIZATION_AVAILABLE:
            self.visualizer = Visualizer()
            self.visualizer.set_map(self.map)
            self.visualizer.set_simulation_engine(self.simulation_engine)
        else:
            print("시각화 시스템 사용 불가 (pygame 미설치)")
            return False
            
        print("\n초기화 완료!")
        return True

    # --- ✨ Depot 생성 로직을 별도 함수로 다시 추가 ---
    def _setup_depots(self):
        """Setup depots using clustering"""
        clustering = MixedClustering()
        clusters = clustering.cluster_stores_and_customers(self.map)
        ClusterAnalyzer.print_cluster_summary(clusters)
        depot_positions = clustering.get_optimal_depot_positions(clusters, self.map)
        
        depot_placer = DepotPlacer(self.map)
        depots = depot_placer.create_depots_with_drones(depot_positions)
        
        print(f"  - Depot 수: {len(depots)}")
        print(f"  - 총 드론 수: {sum(len(depot.drones) for depot in depots)}")

    def run(self):
        """Run simulation with visualization"""
        if not VISUALIZATION_AVAILABLE:
            print("\n시각화가 사용 불가능하여 시뮬레이션을 실행할 수 없습니다.")
            return

        print("\n시각화 모드로 시뮬레이션 실행 중...")
        self.simulation_engine.start_simulation()

        try:
            self.visualizer.start_visualization()
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        finally:
            print("\n시각화 종료. 시뮬레이션을 중지합니다...")
            self.simulation_engine.stop_simulation()
            self._print_final_statistics()

    def _print_final_statistics(self):
        """Print final simulation statistics"""
        stats = self.simulation_engine.get_simulation_state()['stats']
        
        print("\n" + "=" * 60)
        print("최종 시뮬레이션 결과")
        print("=" * 60)
        print(f"총 시뮬레이션 시간: {stats['simulation_duration']:.1f}초")
        print(f"총 처리된 주문: {stats['total_orders_processed']}")
        print(f"완료된 배달: {stats['total_deliveries_completed']}")
        print(f"평균 배달 시간: {stats['average_delivery_time']:.1f}초")
        print(f"총 드론 이동 거리: {stats['total_drone_distance']:.1f}m")
        
        if stats['total_orders_processed'] > 0:
            success_rate = (stats['total_deliveries_completed'] / stats['total_orders_processed']) * 100
            print(f"배달 성공률: {success_rate:.1f}%")
        
        print("=" * 60)

def main():
    """Main function"""
    MAP_SEED: Optional[int] = None
    ORDER_SEED: Optional[int] = None
    
    app = DVRPApplication(map_seed=MAP_SEED, order_seed=ORDER_SEED)
    
    try:
        if app.initialize():
            app.run()
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())