# A\* Bidirectional Parallel Algorithm

This repository contains a Python implementation of the A\* search algorithm, enhanced with bidirectional search and parallel processing. The A\* algorithm is a powerful and flexible pathfinding algorithm used in various applications such as game development, robotics, transport networks, and many more.

Also this is my graduate work for master degree at MIREA - Russian Technological University.

## Data Imput

Please use your data as a data.csv file with this filling:

```csv
start_node_x,start_node_y,end_node_x,end_node_y,weight
1,1,1,2,1.0
1,2,2,2,1.5
...
```

## Installing

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/Console7331/My-graduate-work.git
    cd My-graduate-work
    ```

2. Install all the necessary components into a Virtual Enviroment:

    ```bash
    python -m pip install --user virtualenv
    python -m venv venv
    source venv/bin/activate
    python -m pip install -r requirements.txt
    ```

3. Start the A\* algorithm with ```python main.py``` command.

4. You can also create a random CSV file with ```python generate_csv.py``` command. Nodes in this file has X and Y values between 0 and 100.

## Contents

The main components of the implementation include:

- `Node`: This class represents a node or a point in the graph with coordinates x and y. Each node keeps a dictionary of its neighboring nodes along with the weight of the edge connecting them.

- `PriorityQueue`: A helper class that uses a heap to maintain elements in priority order. It is used to keep track of nodes to be explored next based on their heuristic value.

- `euclidean_distance(node1, node2)`: A helper function that calculates the Euclidean distance between two nodes. This is used as a heuristic in the A* algorithm.

- `load_graph_from_csv(file_path)`: A function is used to load a graph from a CSV file. This function reads a CSV file line by line, creating a new Node for each unique x, y pair, and adding neighbors to each node based on the file's data.

- `a_star(graph, start, goal)`: The main A* algorithm function. It takes in a graph, a start node, and a goal node, and returns the shortest path from start to goal.

- `reconstruct_path(came_from, goal)`: A helper function to backtrack from the goal node to the start node by following the path stored in the came_from dictionary.

- `a_star_parallel(start, goal, graph, results, i)`: A wrapper function for the A* algorithm to be used in parallel processing.

- `a_star_bidirectional(start, goal, graph)`: The main function that applies bidirectional search and parallel processing to the A\* algorithm. It creates two processes that run the A\* algorithm from both the start node and the goal node in parallel.

Please note that this implementation assumes an undirected graph. For use with a directed graph, modifications will be necessary.

## Usage

To use this code, you will need to create a graph and define start and goal nodes. The graph should be a dictionary where the keys are Node objects and the values are dictionaries representing the neighboring nodes and the weights of the edges connecting them.
