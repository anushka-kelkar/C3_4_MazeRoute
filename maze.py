import numpy as np
import random

# Terrain definitions with costs
terrain_types = {
    "grass": {"image": "grass.png", "difficulty": 0.5425, "color": (34, 139, 34)},
    "mud": {"image": "mud.png", "difficulty": 0.4804, "color": (139, 69, 19)},
    "water": {"image": "water2.png", "difficulty": 0.5683, "color": (0, 119, 190)},
    "sand": {"image": "sand2.png", "difficulty": 0.4868, "color": (237, 201, 175)},
    "rock": {"image": "rock.png", "difficulty": 0.4942, "color": (105, 105, 105)}
}

terrains = list(terrain_types.keys())

def generate_maze(rows, cols):
    """Generate random terrain grid."""
    return np.array([[random.choice(terrains) for _ in range(cols)] for _ in range(rows)])

def load_terrain_images(cell_size, pygame_module):
    """Load and scale terrain images."""
    loaded_images = {}
    for terrain, data in terrain_types.items():
        try:
            img = pygame_module.image.load(data["image"])
            img = pygame_module.transform.scale(img, (cell_size, cell_size))
            loaded_images[terrain] = img
        except:
            # Create placeholder
            img = pygame_module.Surface((cell_size, cell_size))
            img.fill(data["color"])
            font = pygame_module.font.Font(None, 18)
            text = font.render(terrain[:4].upper(), True, (255, 255, 255))
            text_rect = text.get_rect(center=(cell_size//2, cell_size//2))
            img.blit(text, text_rect)
            loaded_images[terrain] = img
    return loaded_images

def assign_costs_to_grid(grid):
    """Assign movement costs based on terrain type."""
    cost_grid = np.zeros(grid.shape, dtype=float)
    for terrain in terrains:
        difficulty = terrain_types[terrain]["difficulty"]
        cost_grid[grid == terrain] = difficulty
    return cost_grid