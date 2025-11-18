import numpy as np
import pandas as pd
import random

# Load the terrain dataset
terrain_df = pd.read_csv('terrain_dataset_final.csv')

# Terrain definitions with colors (difficulty comes from dataset)
terrain_types = {
    "grass": {"image": "grass.png", "color": (34, 139, 34)},
    "mud": {"image": "mud.png", "color": (139, 69, 19)},
    "water": {"image": "water2.png", "color": (0, 119, 190)},
    "sand": {"image": "sand2.png", "color": (237, 201, 175)},
    "rock": {"image": "rock.png", "color": (105, 105, 105)}
}

# Obstacle image files based on probability ranges
def get_obstacle_image_file(obstacle_prob):
    """Return image filename based on obstacle probability."""
    if obstacle_prob < 0.33:
        return "obstacle_low.png"  # Low obstacles
    elif obstacle_prob < 0.67:
        return "obstacle_medium.png"  # Medium obstacles
    else:
        return "obstacle_high.png"  # High obstacles

# Cache for loaded obstacle images
_obstacle_image_cache = {}

def load_obstacle_images(cell_size, pygame_module):
    """Load and cache obstacle indicator images."""
    global _obstacle_image_cache
    
    if _obstacle_image_cache:  # Return cached images if already loaded
        return _obstacle_image_cache
    
    obstacle_levels = [
        ("obstacle_low.png", (0, 200, 0), "✓"),
        ("obstacle_medium.png", (255, 165, 0), "▲"),
        ("obstacle_high.png", (255, 0, 0), "✖")
    ]
    
    # Obstacle icon size (small icon in corner)
    icon_size = cell_size // 4
    
    for filename, color, fallback_symbol in obstacle_levels:
        try:
            # Try to load the image
            img = pygame_module.image.load(filename)
            img = pygame_module.transform.scale(img, (icon_size, icon_size))
            _obstacle_image_cache[filename] = img
        except:
            # Create fallback image with symbol
            img = pygame_module.Surface((icon_size, icon_size), pygame_module.SRCALPHA)
            
            # Draw semi-transparent background circle
            pygame_module.draw.circle(img, (*color[:3], 180), 
                                     (icon_size // 2, icon_size // 2), 
                                     icon_size // 2)
            
            # Draw symbol
            font = pygame_module.font.Font(None, icon_size)
            text = font.render(fallback_symbol, True, (255, 255, 255))
            text_rect = text.get_rect(center=(icon_size // 2, icon_size // 2))
            img.blit(text, text_rect)
            
            _obstacle_image_cache[filename] = img
    
    return _obstacle_image_cache

terrains = list(terrain_types.keys())

def generate_maze(rows, cols):
    """Generate random terrain grid from dataset rows."""
    maze = []
    obstacle_grid = []
    difficulty_grid = []
    
    for _ in range(rows):
        row = []
        obstacle_row = []
        difficulty_row = []
        for _ in range(cols):
            # Randomly select a row from the dataset
            random_row = terrain_df.sample(n=1).iloc[0]
            terrain = random_row['terrain']
            obstacle_prob = random_row['obstacle_probability']
            
            # Get difficulty from dataset (assuming column name is 'difficulty' or similar)
            # Check for various possible column names
            if 'difficulty' in random_row:
                difficulty = random_row['difficulty']
            elif 'encoded_terrain' in random_row:
                # If using encoded_terrain as difficulty measure
                difficulty = random_row['encoded_terrain'] / 4.0  # Normalize to 0-1 range
            else:
                # Calculate difficulty from features if available
                if all(col in random_row for col in ['friction', 'slope']):
                    # Combine friction (lower is harder) and slope (higher is harder)
                    difficulty = (1 - random_row['friction']) * 0.5 + random_row['slope'] * 0.5
                else:
                    # Fallback to default terrain difficulty
                    difficulty = 0.5
            
            row.append(terrain)
            obstacle_row.append(obstacle_prob)
            difficulty_row.append(difficulty)
        
        maze.append(row)
        obstacle_grid.append(obstacle_row)
        difficulty_grid.append(difficulty_row)
    
    return np.array(maze), np.array(obstacle_grid), np.array(difficulty_grid)

def load_terrain_images(cell_size, pygame_module):
    """Load and scale terrain images."""
    loaded_images = {}
    font = pygame_module.font.Font(None, 24)
    
    for terrain, data in terrain_types.items():
        try:
            img = pygame_module.image.load(data["image"])
            img = pygame_module.transform.scale(img, (cell_size, cell_size))
            loaded_images[terrain] = img
        except:
            # Create placeholder
            img = pygame_module.Surface((cell_size, cell_size))
            img.fill(data["color"])
            text = font.render(terrain[:4].upper(), True, (255, 255, 255))
            text_rect = text.get_rect(center=(cell_size//2, cell_size//2))
            img.blit(text, text_rect)
            loaded_images[terrain] = img
    
    return loaded_images

def draw_obstacle_indicator(surface, cell_size, obstacle_prob, pygame_module):
    """Draw obstacle image at bottom right of cell."""
    # Lazy load obstacle images if not already loaded
    if not _obstacle_image_cache:
        load_obstacle_images(cell_size, pygame_module)
    
    image_file = get_obstacle_image_file(obstacle_prob)
    obstacle_img = _obstacle_image_cache[image_file]
    
    # Position at bottom-right corner
    icon_size = cell_size // 4
    x_pos = cell_size - icon_size - 2
    y_pos = cell_size - icon_size - 2
    
    surface.blit(obstacle_img, (x_pos, y_pos))

def assign_costs_to_grid(maze, obstacle_grid, difficulty_grid):
    """Assign movement costs based on terrain difficulty from dataset and obstacles."""
    cost_grid = np.zeros(maze.shape, dtype=float)
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            base_difficulty = difficulty_grid[i, j]
            obstacle_factor = obstacle_grid[i, j]
            
            # Combine base difficulty with obstacle probability
            # Higher obstacle probability increases the cost
            combined_cost = base_difficulty * (1 + obstacle_factor)
            cost_grid[i, j] = combined_cost
    
    return cost_grid

def calculate_path_cost(path, cost_grid):
    """Calculate the total cost of traversing a path."""
    if not path or len(path) == 0:
        return 0.0
    
    total_cost = 0.0
    for position in path:
        total_cost += cost_grid[position]
    
    return total_cost

def draw_terrain_grid(screen, grid, obstacle_grid, terrain_images, cell_size, pygame_module):
    """Draw the terrain grid with obstacle indicators."""
    rows, cols = grid.shape
    BLACK = (0, 0, 0)
    
    for i in range(rows):
        for j in range(cols):
            terrain = grid[i, j]
            obstacle_prob = obstacle_grid[i, j]
            
            # Draw terrain image
            x = j * cell_size
            y = i * cell_size
            screen.blit(terrain_images[terrain], (x, y))
            
            # Create a temporary surface for the obstacle indicator
            indicator_surface = pygame_module.Surface((cell_size, cell_size), pygame_module.SRCALPHA)
            draw_obstacle_indicator(indicator_surface, cell_size, obstacle_prob, pygame_module)
            screen.blit(indicator_surface, (x, y))
            
            # Draw border
            rect = pygame_module.Rect(x, y, cell_size, cell_size)
            pygame_module.draw.rect(screen, BLACK, rect, 1)