"""
Visualization system for the drone delivery simulation
"""

import pygame
import math
import threading
from typing import List, Dict, Tuple, Optional
from ..models.entities import Building, Depot, Drone, Order, EntityType, DroneStatus, OrderStatus
from ..simulation.simulation_engine import SimulationEngine
import config


class Visualizer:
    """Main visualization class for the simulation"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Drone Delivery Simulation - DVRP 2D")
        
        self.clock = pygame.time.Clock()
        self.running = False
        
        # Scaling factors for map to screen
        self.map_width = config.MAP_WIDTH
        self.map_height = config.MAP_HEIGHT
        self.scale_x = (width - 200) / self.map_width
        self.scale_y = (height - 100) / self.map_height
        
        # UI elements
        self.ui_panel_width = 200
        self.status_bar_height = 100
        
        # Colors
        self.colors = {
            'background': config.COLORS['background'],
            'building': config.COLORS['building'],
            'store': config.COLORS['store'],
            'customer': config.COLORS['customer'],
            'depot': config.COLORS['depot'],
            'drone': config.COLORS['drone'],
            'route': config.COLORS['route'],
            'text': (0, 0, 0),
            'ui_bg': (240, 240, 240),
            'ui_border': (200, 200, 200),
            'food': (139, 69, 19) 
        }
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        self.animation_time = 0
        self.drone_trails = {}

    def map_to_screen(self, pos) -> Tuple[int, int]:
        screen_x = int(pos.x * self.scale_x) + 10
        screen_y = int(pos.y * self.scale_y) + 10
        return screen_x, screen_y
    
    def change_simulation_speed(self, delta: float):
        """Changes the simulation speed by a given delta."""
        if hasattr(self, 'simulation_engine'):
            current_speed = self.simulation_engine.speed_multiplier
            new_speed = round(current_speed + delta, 1)
            # 속도 범위를 0.1배 ~ 10배로 제한
            new_speed = max(0.1, min(new_speed, 10.0)) 
            self.simulation_engine.set_speed(new_speed)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.toggle_simulation()
                # '+' 키를 누르면 속도 증가 (숫자패드 + 포함)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    self.change_simulation_speed(0.5)
                # '-' 키를 누르면 속도 감소 (숫자패드 - 포함)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.change_simulation_speed(-0.5)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)
    
    def draw_ui(self):
        """Draw the UI panel"""
        ui_x = self.width - self.ui_panel_width
        pygame.draw.rect(self.screen, self.colors['ui_bg'], (ui_x, 0, self.ui_panel_width, self.height - self.status_bar_height))
        pygame.draw.rect(self.screen, self.colors['ui_border'], (ui_x, 0, self.ui_panel_width, self.height - self.status_bar_height), 2)
        y_offset = 20
        title_text = self.font_large.render("Simulation", True, self.colors['text'])
        self.screen.blit(title_text, (ui_x + 10, y_offset)); y_offset += 40
        
        # UI에 배속 조절 키 안내 추가
        controls_text = [
            "Controls:", 
            "SPACE - Pause/Resume", 
            "'+/-' - Change Speed", # 안내 문구 추가
            "ESC - Exit", 
            "", 
            "Status:"
        ]
        for text in controls_text:
            self.screen.blit(self.font_small.render(text, True, self.colors['text']), (ui_x + 10, y_offset)); y_offset += 20
        
        if hasattr(self, 'simulation_engine'):
            state = self.simulation_engine.get_simulation_state()
            # 'Speed' 항목이 현재 배속을 정확히 표시해 줌
            status_texts = [
                f"Running: {'Yes' if state['is_running'] else 'No'}", 
                f"Time: {state['simulation_time']:.1f}s", 
                f"Speed: {state['speed_multiplier']:.1f}x", 
                f"Orders: {state['active_orders']}", 
                f"Completed: {state['completed_orders']}",
                f"Total Drone Distance: {state['stats']['total_drone_distance']:.1f}m"
            ]
            for text in status_texts:
                self.screen.blit(self.font_small.render(text, True, self.colors['text']), (ui_x + 10, y_offset)); y_offset += 20

    # ----------------------------------------------------
    # (이 아래의 다른 함수들은 수정할 필요가 없으므로 생략합니다)
    # (기존 코드 그대로 두시면 됩니다)
    # ----------------------------------------------------

    def screen_to_map(self, screen_pos: Tuple[int, int]):
        from ..models.entities import Position
        map_x = (screen_pos[0] - 10) / self.scale_x
        map_y = (screen_pos[1] - 10) / self.map_height
        return Position(map_x, map_y)

    def start_visualization(self):
        self.running = True
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        pygame.quit()

    def stop_visualization(self):
        self.running = False
        
    def handle_mouse_click(self, pos: Tuple[int, int]):
        if pos[0] < self.width - self.ui_panel_width and pos[1] < self.height - self.status_bar_height:
            map_pos = self.screen_to_map(pos)
            print(f"Clicked at map position: {map_pos.x:.2f}, {map_pos.y:.2f}")

    def toggle_simulation(self):
        if hasattr(self, 'simulation_engine'):
            if self.simulation_engine.paused:
                self.simulation_engine.resume_simulation()
            else:
                self.simulation_engine.pause_simulation()

    def update(self):
        self.animation_time += 1
        if hasattr(self, 'simulation_engine'):
            drone_positions = self.simulation_engine.get_drone_positions()
            for drone_id, drone_info in drone_positions.items():
                if drone_id not in self.drone_trails:
                    self.drone_trails[drone_id] = []
                screen_pos = self.map_to_screen(drone_info['position'])
                self.drone_trails[drone_id].append(screen_pos)
                if len(self.drone_trails[drone_id]) > 20:
                    self.drone_trails[drone_id].pop(0)

    def render(self):
        self.screen.fill(self.colors['background'])
        self.draw_map_area()
        if hasattr(self, 'map'):
            self.draw_buildings()
            self.draw_depots()
        if hasattr(self, 'simulation_engine'):
            self.draw_drone_trails()
            self.draw_drones()
            self.draw_routes()
            self.draw_orders()
        self.draw_legend()
        self.draw_ui()
        self.draw_status_bar()
        pygame.display.flip()

    def draw_map_area(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (10, 10, self.width - self.ui_panel_width - 20, self.height - self.status_bar_height - 20), 2)

    def draw_buildings(self):
        for building in self.map.buildings:
            screen_x, screen_y = self.map_to_screen(building.position)
            screen_width = int(building.width * self.scale_x)
            screen_height = int(building.height * self.scale_y)
            if building.entity_type == EntityType.STORE: color = self.colors['store']
            elif building.entity_type == EntityType.CUSTOMER: color = self.colors['customer']
            else: color = self.colors['building']
            pygame.draw.rect(self.screen, color, (screen_x, screen_y, screen_width, screen_height))
            pygame.draw.rect(self.screen, (0, 0, 0), (screen_x, screen_y, screen_width, screen_height), 1)
            text = self.font_small.render(str(building.id), True, self.colors['text'])
            self.screen.blit(text, (screen_x + 2, screen_y + 2))

    def draw_depots(self):
        for depot in self.map.depots:
            screen_x, screen_y = self.map_to_screen(depot.position)
            depot_size = int(config.DEPOT_SIZE * self.scale_x)
            pygame.draw.circle(self.screen, self.colors['depot'], (screen_x + depot_size//2, screen_y + depot_size//2), depot_size//2)
            pygame.draw.circle(self.screen, (0, 0, 0), (screen_x + depot_size//2, screen_y + depot_size//2), depot_size//2, 2)
            text = self.font_medium.render(f"D{depot.id}", True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen_x + depot_size//2, screen_y + depot_size//2))
            self.screen.blit(text, text_rect)
            available_drones = len([d for d in depot.drones if d.status == DroneStatus.IDLE])
            drone_count_text = self.font_small.render(f"{available_drones}/{len(depot.drones)}", True, self.colors['text'])
            self.screen.blit(drone_count_text, (screen_x, screen_y + depot_size))

    def draw_drones(self):
        """Draw all active drones and food if they are carrying it"""
        # 직접 self.map에 접근하여 최신 드론 상태를 사용합니다.
        for depot in self.map.depots:
            for drone in depot.drones:
                
                if drone.status == DroneStatus.IDLE:
                    continue
                    
                screen_x, screen_y = self.map_to_screen(drone.position)
                
                # 드론 그리기 (노란색 원)
                pygame.draw.circle(self.screen, (255, 255, 0), (screen_x, screen_y), 5)
                pygame.draw.circle(self.screen, (0, 0, 0), (screen_x, screen_y), 5, 1)
                
                # 드론의 남은 경로가 2개일 때(가게->고객 구간)만 음식을 그립니다.
                if drone.route and len(drone.route) == 2:
                    food_size = 6
                    food_offset = 5 # 드론 옆에 표시될 간격
                    pygame.draw.rect(self.screen, self.colors['food'], 
                                     (screen_x + food_offset, screen_y - food_size // 2, food_size, food_size))
                    pygame.draw.rect(self.screen, (0,0,0), 
                                     (screen_x + food_offset, screen_y - food_size // 2, food_size, food_size), 1)

                # 드론 ID 텍스트
                text = self.font_small.render(str(drone.id).split('_')[-1], True, self.colors['text'])
                self.screen.blit(text, (screen_x - 10, screen_y - 20))

    def draw_drone_trails(self):
        for drone_id, trail in self.drone_trails.items():
            if len(trail) > 1:
                pygame.draw.lines(self.screen, self.colors['route'], False, trail, 2)

    def draw_routes(self):
        """Draw the entire planned route for each active drone."""
        for depot in self.map.depots:
            for drone in depot.drones:
                # 드론에게 할당된 전체 경로(route)가 있는지 확인합니다.
                if drone.route and len(drone.route) > 0:
                    # 경로 시각화를 위해 현재 드론 위치와 전체 경로 지점들을 하나의 리스트로 합칩니다.
                    # 예: [현재위치, 상점, 고객, 복귀지점]
                    all_points_map = [drone.position] + drone.route
                    
                    # 맵 좌표 리스트를 화면 좌표 리스트로 변환합니다.
                    screen_points = [self.map_to_screen(pos) for pos in all_points_map]
                    
                    # 변환된 좌표들을 사용하여 선으로 연결합니다.
                    # 점이 2개 이상 있어야 선을 그릴 수 있습니다.
                    if len(screen_points) > 1:
                        # pygame.draw.lines 함수는 여러 점을 한 번에 이어서 그려줍니다.
                        pygame.draw.lines(self.screen, self.colors['route'], False, screen_points, 2)


    def draw_orders(self):
        if not hasattr(self, 'simulation_engine'): return
        order_manager = self.simulation_engine.order_manager
        for order in order_manager.orders:
            if order.status == OrderStatus.PENDING:
                store_screen = self.map_to_screen(order.store_position)
                customer_screen = self.map_to_screen(order.customer_position)
                self.draw_dashed_line(store_screen, customer_screen, (255, 0, 0), 2)
                mid_x = (store_screen[0] + customer_screen[0]) // 2
                mid_y = (store_screen[1] + customer_screen[1]) // 2
                text = self.font_small.render(f"O{order.id}", True, (255, 0, 0))
                self.screen.blit(text, (mid_x, mid_y))

    def draw_dashed_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], width: int, dash_length: int = 10):
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        dashes = int(distance / dash_length)
        for i in range(dashes):
            if i % 2 == 0:
                start_ratio = i / dashes
                end_ratio = (i + 1) / dashes
                dash_start = (int(start[0] + (end[0] - start[0]) * start_ratio), int(start[1] + (end[1] - start[1]) * start_ratio))
                dash_end = (int(start[0] + (end[0] - start[0]) * end_ratio), int(start[1] + (end[1] - start[1]) * end_ratio))
                pygame.draw.line(self.screen, color, dash_start, dash_end, width)

    def draw_status_bar(self):
        """Draw the status bar at the bottom"""
        status_y = self.height - self.status_bar_height
        pygame.draw.rect(self.screen, self.colors['ui_bg'], (0, status_y, self.width, self.status_bar_height))
        pygame.draw.rect(self.screen, self.colors['ui_border'], (0, status_y, self.width, self.status_bar_height), 2)
        
        if hasattr(self, 'simulation_engine'):
            stats = self.simulation_engine.get_simulation_state()['stats']
            
            status_texts = [
                f"Total Orders: {stats['total_orders_processed']}",
                f"Completed: {stats['total_deliveries_completed']}",
                f"Avg Delivery Time: {stats['average_delivery_time']:.1f}s",
                f"Total Drone Distance: {stats['total_drone_distance']:.1f}m"  # 'Drone Utilization' 대신 추가
            ]
            
            x_offset = 20
            for text in status_texts:
                text_surface = self.font_medium.render(text, True, self.colors['text'])
                self.screen.blit(text_surface, (x_offset, status_y + 20))
                x_offset += 250

    def draw_legend(self):
        legend_width = 120
        legend_height = 140
        legend_x = self.width - self.ui_panel_width - legend_width - 10
        legend_y = 10
        s = pygame.Surface((legend_width, legend_height))
        s.set_alpha(200)
        s.fill((255,255,255))
        self.screen.blit(s, (legend_x, legend_y))
        pygame.draw.rect(self.screen, (0, 0, 0), (legend_x, legend_y, legend_width, legend_height), 2)
        title_text = self.font_small.render("Legend", True, self.colors['text'])
        self.screen.blit(title_text, (legend_x + 5, legend_y + 5))
        y_offset = legend_y + 25
        items = {"Store": self.colors['store'], "Customer": self.colors['customer'], "Depot": self.colors['depot'], "Drone": (255, 255, 0), "Route": self.colors['route'], "Food": self.colors['food']}
        for name, color in items.items():
            if name in ["Store", "Customer", "Food"]:
                pygame.draw.rect(self.screen, color, (legend_x + 5, y_offset, 12, 8))
                pygame.draw.rect(self.screen, (0,0,0), (legend_x + 5, y_offset, 12, 8), 1)
            elif name == "Route":
                pygame.draw.line(self.screen, color, (legend_x + 5, y_offset + 4), (legend_x + 17, y_offset + 4), 2)
            else:
                 pygame.draw.circle(self.screen, color, (legend_x + 11, y_offset + 4), 5)
                 pygame.draw.circle(self.screen, (0,0,0), (legend_x + 11, y_offset + 4), 5, 1)
            text = self.font_small.render(name, True, self.colors['text'])
            self.screen.blit(text, (legend_x + 25, y_offset - 1))
            y_offset += 18

    def set_simulation_engine(self, simulation_engine: SimulationEngine):
        self.simulation_engine = simulation_engine
        self.simulation_engine.register_event_handler('delivery_completed', lambda e: print(f"Delivery completed: {e['data']}"))
        self.simulation_engine.register_event_handler('order_assigned', lambda e: print(f"Order assigned: {e['data']}"))

    def set_map(self, map_obj):
        self.map = map_obj