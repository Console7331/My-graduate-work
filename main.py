import heapq
import math
import multiprocessing as mp

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def load_graph_from_csv(file_path):
    df = pd.read_csv(file_path)
    graph = {}

    for _, row in df.iterrows():
        start = Node(row["start_node_x"], row["start_node_y"])
        end = Node(row["end_node_x"], row["end_node_y"])

        if start not in graph:
            graph[start] = start
        if end not in graph:
            graph[end] = end

        graph[start].add_neighbor(graph[end], row["weight"])
        graph[end].add_neighbor(graph[start], row["weight"])

    return graph


def find_closest_node(graph, x, y):
    closest_node = None
    closest_distance = float("inf")

    for node in graph.values():
        dist = distance.euclidean([node.x, node.y], [x, y])
        if dist < closest_distance:
            closest_distance = dist
            closest_node = node

    return closest_node


def a_star(graph, start, goal):
    open_set = PriorityQueue()
    open_set.put(start, 0)
    closed_set = set()

    came_from = {}
    g_score = {node: float("inf") for node in graph}
    g_score[start] = 0
    f_score = {node: float("inf") for node in graph}
    f_score[start] = euclidean_distance(start, goal)

    while not open_set.is_empty():
        current = open_set.get()

        if current == goal:
            return reconstruct_path(came_from, goal)

        closed_set.add(current)

        for neighbor, weight in current.neighbors.items():
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(
                    neighbor, goal
                )
                open_set.put(neighbor, f_score[neighbor])

    return None


def reconstruct_path(came_from, goal):
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def a_star_parallel(start_node, end_node, results, index):
    # Load the graph in this process
    graph = load_graph_from_csv("graph.csv")

    # Check if start and goal nodes are in the graph
    if start_node not in graph or end_node not in graph:
        return

    # Get the start and goal nodes from the graph
    start_node = graph[start_node]
    goal_node = graph[end_node]

    # Run A* algorithm
    path = a_star(graph, start_node, goal_node)
    results[index] = path


def a_star_bidirectional(start_node, end_node, file_path):
    graph = load_graph_from_csv(file_path)

    # create a manager object to handle the shared memory
    manager = mp.Manager()

    # create a list in server process memory
    results = manager.list([[], []])

    # creating new processes
    p1 = mp.Process(target=a_star_parallel, args=(start_node, end_node, results, 0))
    p2 = mp.Process(target=a_star_parallel, args=(end_node, start_node, results, 1))

    # starting processes
    p1.start()
    p2.start()

    # wait until processes finish
    p1.join()
    p2.join()

    # Ensure both processes have found a path
    if len(results[0]) > 0 and len(results[1]) > 0:
        return results[0] + results[1][::-1]
    else:
        print("Path not found")
        return []


def plot_graph(graph, path=None):
    # Create a new figure
    plt.figure()

    # Draw edges
    for node in graph.values():
        for neighbor in node.neighbors:
            plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color="black")

    # Draw nodes
    for node in graph.values():
        plt.scatter(node.x, node.y, color="blue")

    # Draw path, if one is provided
    if path is not None:
        for i in range(len(path) - 1):
            plt.plot(
                [path[i].x, path[i + 1].x], [path[i].y, path[i + 1].y], color="red"
            )

    # Show the plot
    plt.show()


def main():
    # Get start and end nodes from user input
    start_node_x = float(input("Enter start node x coordinate: "))
    start_node_y = float(input("Enter start node y coordinate: "))
    end_node_x = float(input("Enter end node x coordinate: "))
    end_node_y = float(input("Enter end node y coordinate: "))

    # Load the graph
    graph = load_graph_from_csv("graph.csv")

    # Find the closest nodes to the input coordinates
    start_node = find_closest_node(graph, start_node_x, start_node_y)
    end_node = find_closest_node(graph, end_node_x, end_node_y)

    # Run A* algorithm
    path = a_star_bidirectional(start_node, end_node, "graph.csv")

    # Plot the graph and the path
    plot_graph(graph, path)


if __name__ == "__main__":
    main()
