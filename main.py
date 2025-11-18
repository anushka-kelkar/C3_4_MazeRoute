import pygame
import sys
from maze2 import generate_maze, load_terrain_images, assign_costs_to_grid, draw_obstacle_indicator, calculate_path_cost
from pathfinding import a_star
from dijkstra import dijkstra

pygame.init()

# Constants for menu
MENU_WIDTH, MENU_HEIGHT = 600, 500
GAME_ROWS, GAME_COLS = 10, 10
CELL_SIZE = 50
UI_HEIGHT = 120
GAME_WIDTH = GAME_COLS * CELL_SIZE
GAME_HEIGHT = GAME_ROWS * CELL_SIZE + UI_HEIGHT
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
orange= (255, 165, 0, 255)
BLUE = (0, 0, 255)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (100, 100, 100)
LIGHT_GREEN = (144, 238, 144)
LIGHT_RED = (255, 182, 193)
LIGHT_BLUE = (173, 216, 230)

def draw_button_menu(surface, rect, text, color, hover=False):
    """Draw a styled button for menu."""
    button_color = tuple(min(c + 30, 255) for c in color) if hover else color
    pygame.draw.rect(surface, button_color, rect, border_radius=10)
    pygame.draw.rect(surface, BLACK, rect, 3, border_radius=10)
    
    font = pygame.font.Font(None, 32)
    text_surface = font.render(text, True, WHITE if not hover else BLACK)
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)

def show_menu():
    """Display main menu and return user choice."""
    screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
    pygame.display.set_caption("Terrain Navigation System")
    clock = pygame.time.Clock()
    
    # Button definitions
    button_width = 400
    button_height = 60
    button_spacing = 80
    start_y = 100
    
    predict_button = pygame.Rect((MENU_WIDTH - button_width) // 2, start_y, button_width, button_height)
    astar_button = pygame.Rect((MENU_WIDTH - button_width) // 2, start_y + button_spacing, button_width, button_height)
    dijkstra_button = pygame.Rect((MENU_WIDTH - button_width) // 2, start_y + button_spacing * 2, button_width, button_height)
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # Draw background
        screen.fill(LIGHT_GRAY)
        
        # Title
        font_title = pygame.font.Font(None, 48)
        title = font_title.render("Terrain Navigation System", True, DARK_GRAY)
        title_rect = title.get_rect(center=(MENU_WIDTH // 2, 40))
        screen.blit(title, title_rect)
        
        # Draw buttons
        predict_hover = predict_button.collidepoint(mouse_pos)
        astar_hover = astar_button.collidepoint(mouse_pos)
        dijkstra_hover = dijkstra_button.collidepoint(mouse_pos)
        
        draw_button_menu(screen, predict_button, "Terrain Prediction", orange, predict_hover)
        draw_button_menu(screen, astar_button, "Pathfinding (A*)", BLUE, astar_hover)
        draw_button_menu(screen, dijkstra_button, "Pathfinding (Dijkstra)", GREEN, dijkstra_hover)
        
        # Instructions
        font_small = pygame.font.Font(None, 18)
        instruction = font_small.render("Click a button to continue or press ESC to exit", True, DARK_GRAY)
        instruction_rect = instruction.get_rect(center=(MENU_WIDTH // 2, MENU_HEIGHT - 30))
        screen.blit(instruction, instruction_rect)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if predict_button.collidepoint(event.pos):
                    return "predict"
                elif astar_button.collidepoint(event.pos):
                    return "astar"
                elif dijkstra_button.collidepoint(event.pos):
                    return "dijkstra"
        
        pygame.display.flip()
        clock.tick(60)

def show_prediction_window():
    """Display terrain prediction input window."""
    screen = pygame.display.set_mode((700, 500))
    pygame.display.set_caption("Terrain Prediction")
    clock = pygame.time.Clock()
    
    # Import here to avoid circular imports
    try:
        from model import predict_terrain_type, predict_difficulty, feature_cols
    except ImportError:
        # Show error if model.py is not available
        font = pygame.font.Font(None, 36)
        running = True
        while running:
            screen.fill(WHITE)
            error_text = font.render("Error: model.py not found!", True, RED)
            error_rect = error_text.get_rect(center=(350, 200))
            screen.blit(error_text, error_rect)
            
            instruction = pygame.font.Font(None, 24).render("Press ESC or BACKSPACE to return to menu", True, BLACK)
            instruction_rect = instruction.get_rect()
            instruction_rect.x = 50    
            instruction_rect.y = 250
            screen.blit(instruction, instruction_rect)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_BACKSPACE]:
                        return
            
            pygame.display.flip()
            clock.tick(60)
        return
    
    # Input fields
    input_boxes = []
    feature_values = []
    active_box = 0
    
    for i, feature in enumerate(feature_cols):
        input_boxes.append(pygame.Rect(300, 80 + i * 40, 200, 30))
        feature_values.append("")
    
    prediction_result = None
    
    running = True
    while running:
        screen.fill(WHITE)
        
        # Title
        font_title = pygame.font.Font(None, 36)
        title = font_title.render("Enter Terrain Features", True, DARK_GRAY)
        screen.blit(title, (225, 20))
        
        # Draw input fields
        font_label = pygame.font.Font(None, 24)
        font_input = pygame.font.Font(None, 22)
        
        for i, (box, feature) in enumerate(zip(input_boxes, feature_cols)):
            # Label
            label = font_label.render(f"{feature.replace('_', ' ').title()}:", True, BLACK)
            screen.blit(label, (100, box.y + 5))
            
            # Input box
            color = BLUE if i == active_box else DARK_GRAY
            pygame.draw.rect(screen, color, box, 2)
            
            # Input text
            text_surface = font_input.render(feature_values[i], True, BLACK)
            screen.blit(text_surface, (box.x + 5, box.y + 5))
        
        # Predict button
        predict_btn = pygame.Rect(250, 80 + len(feature_cols) * 40 + 20, 200, 40)
        pygame.draw.rect(screen, orange, predict_btn, border_radius=5)
        pygame.draw.rect(screen, BLACK, predict_btn, 2, border_radius=5)
        btn_text = font_label.render("Predict", True, WHITE)
        btn_rect = btn_text.get_rect(center=predict_btn.center)
        screen.blit(btn_text, btn_rect)
        
        # Show prediction result
        if prediction_result:
            result_y = 80 + len(feature_cols) * 40 + 80
            terrain_text = font_label.render(f"Terrain: {prediction_result[0]}", True, BLUE)
            screen.blit(terrain_text, (300, result_y))
            
            difficulty_text = font_label.render(f"Difficulty: {prediction_result[1]:.2f}", True, RED)
            screen.blit(difficulty_text, (300, result_y + 30))
        
        # Instructions
        font_small = pygame.font.Font(None, 18)
        instruction = font_small.render("Press TAB to navigate, ENTER on Predict button, ESC/BACKSPACE to return", True, DARK_GRAY)
        screen.blit(instruction, (125, 470))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_BACKSPACE]:
                    return
                
                if event.key == pygame.K_TAB:
                    active_box = (active_box + 1) % len(input_boxes)
                
                elif event.key == pygame.K_RETURN:
                    # Try to predict
                    try:
                        features = [float(val) if val else 0.0 for val in feature_values]
                        terrain = predict_terrain_type(features)
                        difficulty = predict_difficulty(features)
                        prediction_result = (terrain, difficulty)
                    except ValueError:
                        prediction_result = ("Invalid input", 0.0)
                
                elif event.key == pygame.K_BACKSPACE:
                    feature_values[active_box] = feature_values[active_box][:-1]
                
                else:
                    # Add character to active input
                    if event.unicode.isprintable():
                        feature_values[active_box] += event.unicode
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicking on input boxes
                for i, box in enumerate(input_boxes):
                    if box.collidepoint(event.pos):
                        active_box = i
                
                # Check if clicking predict button
                if predict_btn.collidepoint(event.pos):
                    try:
                        features = [float(val) if val else 0.0 for val in feature_values]
                        terrain = predict_terrain_type(features)
                        difficulty = predict_difficulty(features)
                        prediction_result = (terrain, difficulty)
                    except ValueError:
                        prediction_result = ("Invalid input", 0.0)
        
        pygame.display.flip()
        clock.tick(60)

def run_pathfinding_game(algorithm="astar"):
    """Run the pathfinding game with selected algorithm."""
    # Constants
    ROWS, COLS = GAME_ROWS, GAME_COLS
    WIDTH, HEIGHT = GAME_WIDTH, GAME_HEIGHT
    
    # Create game window
    algo_name = "A*" if algorithm == "astar" else "Dijkstra's"
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Terrain Pathfinding ({algo_name})")
    clock = pygame.time.Clock()
    
    # Load terrain images
    loaded_images = load_terrain_images(CELL_SIZE, pygame)
    
    # Load character image
    try:
        character_img = pygame.image.load("character.png")
        character_img = pygame.transform.scale(character_img, (CELL_SIZE - 10, CELL_SIZE - 10))
    except:
        # Create placeholder character
        character_img = pygame.Surface((CELL_SIZE - 10, CELL_SIZE - 10))
        character_img.fill(RED)
        font = pygame.font.Font(None, 30)
        text = font.render("C", True, WHITE)
        text_rect = text.get_rect(center=((CELL_SIZE - 10)//2, (CELL_SIZE - 10)//2))
        character_img.blit(text, text_rect)
    
    def draw_maze(maze, obstacle_grid, path=None, character_pos=None, start=None, goal=None):
        """Draw the terrain grid with obstacle indicators, optional path and character."""
        # Draw terrain with obstacles
        for r in range(ROWS):
            for c in range(COLS):
                terrain = maze[r][c]
                obstacle_prob = obstacle_grid[r][c]
                
                # Draw terrain image
                img = loaded_images[terrain]
                screen.blit(img, (c * CELL_SIZE, r * CELL_SIZE))
                
                # Draw obstacle indicator
                indicator_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                draw_obstacle_indicator(indicator_surface, CELL_SIZE, obstacle_prob, pygame)
                screen.blit(indicator_surface, (c * CELL_SIZE, r * CELL_SIZE))
                
                # Draw border
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Draw start position (green circle)
        if start:
            center = (start[1] * CELL_SIZE + CELL_SIZE // 2, start[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, GREEN, center, 8)
        
        # Draw goal position (red circle)
        if goal:
            center = (goal[1] * CELL_SIZE + CELL_SIZE // 2, goal[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, RED, center, 8)
        
        # Draw path
        if path:
            for i in range(len(path) - 1):
                start_pos = (path[i][1] * CELL_SIZE + CELL_SIZE // 2, 
                            path[i][0] * CELL_SIZE + CELL_SIZE // 2)
                end_pos = (path[i + 1][1] * CELL_SIZE + CELL_SIZE // 2, 
                          path[i + 1][0] * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.line(screen, YELLOW, start_pos, end_pos, 3)
        
        # Draw character
        if character_pos:
            char_x = character_pos[1] * CELL_SIZE + 5
            char_y = character_pos[0] * CELL_SIZE + 5
            screen.blit(character_img, (char_x, char_y))
    
    def draw_ui_panel(selecting_mode, path, character_pos, start, goal, path_cost, animation_complete):
        """Draw the UI panel with instructions."""
        ui_y = ROWS * CELL_SIZE
        
        # Background
        pygame.draw.rect(screen, LIGHT_GRAY, (0, ui_y, WIDTH, UI_HEIGHT))
        pygame.draw.line(screen, DARK_GRAY, (0, ui_y), (WIDTH, ui_y), 3)
        
        # Title/Status section
        font_title = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 22)
        font_cost = pygame.font.Font(None, 24)
        
        # Show algorithm name
        algo_color = BLUE if algorithm == "astar" else GREEN
        algo_text = font_small.render(f"Algorithm: {algo_name}", True, algo_color)
        screen.blit(algo_text, (WIDTH - 150, ui_y + 5))
        
        if selecting_mode == "start":
            title = font_title.render("Select START Point", True, BLACK)
            instruction = font_small.render("Click any cell on the grid to set starting position", True, BLACK)
            # Draw indicator
            pygame.draw.circle(screen, GREEN, (30, ui_y + 30), 12)
        elif selecting_mode == "goal":
            title = font_title.render("Select END Point", True, BLACK)
            instruction = font_small.render("Click any cell on the grid to set goal position", True, BLACK)
            # Draw indicator
            pygame.draw.circle(screen, RED, (30, ui_y + 30), 12)
        elif path is None and start and goal:
            title = font_title.render("No Path Found!", True, RED)
            instruction = font_small.render("No valid path exists between these points", True, BLACK)
        elif animation_complete and path:
            title = font_title.render("Journey Complete!", True, GREEN)
            instruction = font_small.render(f"Successfully traversed {len(path)} steps", True, BLACK)
        elif character_pos is None and path:
            title = font_title.render("Path Found!", True, BLUE)
            instruction = font_small.render("Character will now traverse the optimal path", True, BLACK)
        else:
            title = font_title.render("Character Moving...", True, BLUE)
            instruction = font_small.render(f"Progress: Step {path.index(character_pos) + 1 if character_pos in path else 0} of {len(path) if path else 0}", True, BLACK)
        
        screen.blit(title, (50, ui_y + 20))
        screen.blit(instruction, (50, ui_y + 60))
        
        # Display path cost - show in center when animation is complete, otherwise top-right
        if path and len(path) > 0:
            if animation_complete:
                # Center display with larger, more prominent styling
                font_large = pygame.font.Font(None, 36)
                cost_text = font_large.render(f"Total Path Cost: {path_cost:.2f}", True, (0, 100, 0))
                cost_bg = pygame.Rect((WIDTH - 300) // 2, ui_y + 15, 300, 50)
                pygame.draw.rect(screen, (200, 255, 200), cost_bg, border_radius=8)
                pygame.draw.rect(screen, (0, 150, 0), cost_bg, 3, border_radius=8)
                text_rect = cost_text.get_rect(center=cost_bg.center)
                screen.blit(cost_text, text_rect)
            else:
                # Top-right display during movement
                cost_text = font_cost.render(f"Cost: {path_cost:.2f}", True, DARK_GRAY)
                cost_bg = pygame.Rect(WIDTH - 130, ui_y + 30, 120, 35)
                pygame.draw.rect(screen, (255, 255, 200), cost_bg, border_radius=5)
                pygame.draw.rect(screen, DARK_GRAY, cost_bg, 2, border_radius=5)
                screen.blit(cost_text, (WIDTH - 125, ui_y + 37))
        
        # Back to menu instruction
        font_tiny = pygame.font.Font(None, 18)
        back_text = font_tiny.render("Press ESC or BACKSPACE to return to menu | Press R to regenerate maze", True, DARK_GRAY)
        screen.blit(back_text, (10, ui_y + 95))
    
    def draw_legend():
        """Draw a legend showing what colors/symbols mean."""
        legend_x = WIDTH - 120
        legend_y = ROWS * CELL_SIZE + 10
        
        font = pygame.font.Font(None, 18)
        
        # Start indicator
        pygame.draw.circle(screen, GREEN, (legend_x, legend_y + 5), 7)
        text = font.render("Start", True, BLACK)
        screen.blit(text, (legend_x + 12, legend_y))
        
        # Goal indicator
        pygame.draw.circle(screen, RED, (legend_x, legend_y + 28), 7)
        text = font.render("Goal", True, BLACK)
        screen.blit(text, (legend_x + 12, legend_y + 23))
        
        # Path indicator
        pygame.draw.line(screen, YELLOW, (legend_x - 6, legend_y + 51), (legend_x + 6, legend_y + 51), 3)
        text = font.render("Path", True, BLACK)
        screen.blit(text, (legend_x + 12, legend_y + 46))
    
    def get_cell_from_mouse(pos):
        """Convert mouse position to grid cell coordinates."""
        x, y = pos
        if y < ROWS * CELL_SIZE:  # Only if clicking on grid
            col = x // CELL_SIZE
            row = y // CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                return (row, col)
        return None
    
    # Initialize game state
    maze, obstacle_grid, difficulty_grid = generate_maze(ROWS, COLS)
    cost_grid = assign_costs_to_grid(maze, obstacle_grid, difficulty_grid)
    start = None
    goal = None
    path = None
    path_cost = 0.0
    character_pos = None
    path_index = 0
    animation_started = False
    animation_complete = False
    selecting_mode = "start"
    
    # Game loop
    running = True
    while running:
        screen.fill(WHITE)
        
        # Draw everything
        draw_maze(maze, obstacle_grid, path, character_pos, start, goal)
        draw_ui_panel(selecting_mode, path, character_pos, start, goal, path_cost, animation_complete)
        draw_legend()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicking on grid
                cell = get_cell_from_mouse(event.pos)
                if cell:
                    if selecting_mode == "start":
                        start = cell
                        selecting_mode = "goal"
                        path = None
                        path_cost = 0.0
                        character_pos = None
                        animation_started = False
                        animation_complete = False
                    elif selecting_mode == "goal":
                        goal = cell
                        selecting_mode = None
                        # Calculate path with selected algorithm
                        if start and goal:
                            if algorithm == "astar":
                                path = a_star(maze, cost_grid, start, goal)
                            else:  # dijkstra
                                path = dijkstra(maze, cost_grid, start, goal)
                            
                            character_pos = None
                            path_index = 0
                            animation_started = False
                            animation_complete = False
                            # Calculate path cost
                            if path:
                                path_cost = calculate_path_cost(path, cost_grid)
                                animation_started = True
                                character_pos = path[0]
                            else:
                                path_cost = 0.0
            
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_BACKSPACE]:
                    return  # Return to menu
                
                if event.key == pygame.K_r:
                    # Regenerate maze
                    maze, obstacle_grid, difficulty_grid = generate_maze(ROWS, COLS)
                    cost_grid = assign_costs_to_grid(maze, obstacle_grid, difficulty_grid)
                    start = None
                    goal = None
                    path = None
                    path_cost = 0.0
                    character_pos = None
                    path_index = 0
                    animation_started = False
                    animation_complete = False
                    selecting_mode = "start"
        
        # Animate character along path
        if animation_started and path and path_index < len(path):
            character_pos = path[path_index]
            path_index += 1
            if path_index >= len(path):
                animation_started = False
                animation_complete = True  # Mark animation as complete
        
        pygame.display.flip()
        clock.tick(FPS)

def main():
    """Main function to run the application."""
    while True:
        choice = show_menu()
        
        if choice == "predict":
            show_prediction_window()
        elif choice == "astar":
            run_pathfinding_game("astar")
        elif choice == "dijkstra":
            run_pathfinding_game("dijkstra")

if __name__ == "__main__":
    main()