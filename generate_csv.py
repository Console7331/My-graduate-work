import csv
import random
import numpy as np

# Set the seed so the results are reproducible
random.seed(1)
np.random.seed(1)

# Number of nodes
num_nodes = 100

# Generate node positions
node_positions = [
    (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)
]

# Open the CSV file
with open("graph.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(
        ["start_node_x", "start_node_y", "end_node_x", "end_node_y", "weight"]
    )

    # For each node
    for i in range(num_nodes):
        start_node_x, start_node_y = node_positions[i]

        # Determine number of edges for this node based on a normal distribution
        num_edges = min(max(int(np.random.normal(2.5, 0.5)), 1), 4)

        # Choose other nodes to connect to
        other_nodes = random.sample([j for j in range(num_nodes) if j != i], num_edges)

        # For each edge
        for j in other_nodes:
            end_node_x, end_node_y = node_positions[j]

            # Compute weight as Euclidean distance (you can modify this to suit your needs)
            weight = (
                (end_node_x - start_node_x) ** 2 + (end_node_y - start_node_y) ** 2
            ) ** 0.5

            # Write edge to file
            writer.writerow(
                [start_node_x, start_node_y, end_node_x, end_node_y, weight]
            )
