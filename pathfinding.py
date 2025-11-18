import heapq
import numpy as np

def heuristic(a, b):
    """Manhattan distance heuristic for grid navigation."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, cost_grid, start, goal):
    """A* pathfinding algorithm."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4 directions
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                tentative_g = g_score[current] + cost_grid[neighbor]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def reconstruct_path(came_from, current):
    """Reconstruct path from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# If you want to test this file independently, use this:
if __name__ == "__main__":
    # Test code only runs when this file is executed directly
    print("This is the A* pathfinding module.")
    print("Import this module in your main file to use the a_star function.")