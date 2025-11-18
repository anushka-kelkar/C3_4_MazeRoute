import heapq
import numpy as np

def dijkstra(grid, cost_grid, start, goal):
    """
    Dijkstra's pathfinding algorithm.
    
    Key difference from A*: Does NOT use heuristic function.
    Explores nodes purely based on actual cost from start.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}  # Cost from start to each node

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        # Goal reached
        if current == goal:
            return reconstruct_path(came_from, current)

        # Skip if we've found a better path already
        if current_cost > g_score.get(current, float('inf')):
            continue

        # Explore neighbors (4 directions: up, down, left, right)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check if neighbor is within bounds
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Calculate cost to reach neighbor
                tentative_g = g_score[current] + cost_grid[neighbor]

                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # Push to heap with only g_score (no heuristic)
                    heapq.heappush(open_set, (tentative_g, neighbor))

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
    print("This is Dijkstra's pathfinding module.")
    print("Import this module in your main file to use the dijkstra function.")
    print("\nKey differences from A*:")
    print("- No heuristic function")
    print("- Explores ALL reachable nodes based on actual cost")
    print("- Guarantees shortest path but may explore more nodes")
    print("- Slower than A* but more thorough")