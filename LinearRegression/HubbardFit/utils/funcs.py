import numpy as np

def get_neighbors(grid_size, edges):
    # For each edge, find the index of the neighbors
    grid_width, grid_height = grid_size

    # Helper function to get the row and column from vertex index
    def get_row_col(vertex, width):
        return vertex // width, vertex % width

    def get_idx(role, col, grid_width):
        return role * grid_width + col

    # Initialize the count array
    neighbor_counts = []
    other_neighbor = []

    for edge in edges:
        count = 0
        neighbor = []

        # Get row and column for both vertices
        row, col = get_row_col(edge, grid_width)

        # Determine the edge orientation and count neighbors
        if np.diff(row):  # Horizontal (x) edge
            # Check left and right neighbors
            row.sort()
            if row[0] > 0:
                neighbor.append(
                    [
                        get_idx(row[0], col[0], grid_width),
                        get_idx(row[0] - 1, col[0], grid_width),
                    ]
                )
                count += 1
            if row[1] < grid_height - 1:
                neighbor.append(
                    [
                        get_idx(row[1], col[1], grid_width),
                        get_idx(row[1] + 1, col[1], grid_width),
                    ]
                )
                count += 1
        elif np.diff(col):  # Vertical (y) edge
            # Check top and bottom neighbors
            col.sort()
            if col[0] > 0:
                neighbor.append(
                    [
                        get_idx(row[0], col[0], grid_width),
                        get_idx(row[0], col[0] - 1, grid_width),
                    ]
                )
                count += 1
            if col[1] < grid_width - 1:
                neighbor.append(
                    [
                        get_idx(row[1], col[1], grid_width),
                        get_idx(row[1], col[1] + 1, grid_width),
                    ]
                )
                count += 1
        neighbor_counts.append(count)
        other_neighbor.append(neighbor)
    return np.array(neighbor_counts), other_neighbor
