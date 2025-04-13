import numpy as np
import matplotlib.pyplot as plt
import time

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

def greedy_search(start, goal, h_values):
    start_time = time.time()
    open_list = [Node(start, None, 0, h_values[start])]
    closed_list = set()
    nodes_visited = 0

    while open_list:
        # Get the node with the lowest heuristic value
        current_node = min(open_list, key=lambda n: n.heuristic)
        open_list.remove(current_node)

        nodes_visited += 1
        if current_node.state == goal:
            execution_time = time.time() - start_time
            print(f"Greedy Search: Goal {goal} found! Nodes visited: {nodes_visited}, Execution time: {execution_time:.4f} seconds")
            return current_node

        closed_list.add(current_node.state)
        neighbors = get_neighbors(current_node.state)

        for state in neighbors:
            if state not in closed_list:
                neighbor_node = Node(state, current_node, current_node.cost + 1, h_values[state])
                if neighbor_node not in open_list:
                    open_list.append(neighbor_node)

    return None

def a_star_search(start, goal, h_values):
    start_time = time.time()
    open_list = [Node(start, None, 0, h_values[start])]
    closed_list = set()
    nodes_visited = 0

    while open_list:
        current_node = min(open_list, key=lambda n: n.total_cost)
        open_list.remove(current_node)

        nodes_visited += 1
        if current_node.state == goal:
            execution_time = time.time() - start_time
            print(f"A* Search: Goal {goal} found! Nodes visited: {nodes_visited}, Execution time: {execution_time:.4f} seconds")
            return current_node

        closed_list.add(current_node.state)
        neighbors = get_neighbors(current_node.state)

        for state in neighbors:
            if state not in closed_list:
                cost = current_node.cost + 1
                neighbor_node = Node(state, current_node, cost, h_values[state])
                if neighbor_node not in open_list:
                    open_list.append(neighbor_node)

    return None

def ida_star(start, goal, h_values):
    def search(node, g, threshold):
        f = g + node.heuristic
        if f > threshold:
            return f
        if node.state == goal:
            return None
        min_threshold = float('inf')
        neighbors = get_neighbors(node.state)
        for state in neighbors:
            neighbor_node = Node(state, node, g + 1, h_values[state])
            t = search(neighbor_node, g + 1, threshold)
            if t is None:
                return None
            if t < min_threshold:
                min_threshold = t
        return min_threshold

    start_time = time.time()
    threshold = h_values[start]
    nodes_visited = 0
    while True:
        nodes_visited += 1
        root = Node(start, None, 0, h_values[start])
        t = search(root, 0, threshold)
        if t is None:
            execution_time = time.time() - start_time
            print(f"IDA* Search: Goal {goal} found! Nodes visited: {nodes_visited}, Execution time: {execution_time:.4f} seconds")
            return root
        if t == float('inf'):
            return None
        threshold = t

def a_star_memory_limited(start, goal, h_values, memory_limit):
    start_time = time.time()
    open_list = [Node(start, None, 0, h_values[start])]
    closed_list = set()
    nodes_visited = 0

    while open_list:
        current_node = min(open_list, key=lambda n: n.total_cost)
        open_list.remove(current_node)

        nodes_visited += 1
        if current_node.state == goal:
            execution_time = time.time() - start_time
            print(f"A* Memory Limited: Goal {goal} found! Nodes visited: {nodes_visited}, Execution time: {execution_time:.4f} seconds")
            return current_node

        closed_list.add(current_node.state)
        neighbors = get_neighbors(current_node.state)

        for state in neighbors:
            if state not in closed_list:
                cost = current_node.cost + 1
                neighbor_node = Node(state, current_node, cost, h_values[state])
                if neighbor_node not in open_list and len(open_list) < memory_limit:
                    open_list.append(neighbor_node)

    return None

def get_neighbors(state):
    # This function should return the neighboring states for a given state
    # For this example, we'll assume a simple graph structure
    graph = {
        'S': ['A', 'B'],
        'A': ['C', 'B'],
        'B': ['G', 'D'],
        'C': ['G'],
        'D': ['G'],
        'G': [],
    }
    return graph.get(state, [])

def extract_path(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path[::-1]  # Return reversed path

def draw_tree(path):
    if not path:
        print("No path found to draw.")
        return
    
    plt.figure(figsize=(10, 5))
    x = np.arange(len(path))
    y = np.zeros(len(path))

    for i, node in enumerate(path):
        y[i] = i

    plt.plot(x, y, marker='o')
    plt.xticks(x, [node.state for node in path])
    plt.title("Path to Goal")
    plt.xlabel("Node")
    plt.ylabel("Step")
    plt.grid(True)
    plt.show()

# Heuristic values for the states
h_values = {
    'S': 4,
    'A': 2,
    'B': 5,
    'C': 2,
    'D': 3,
    'G': 0,
}

# Perform searches
greedy_result = greedy_search('S', 'G', h_values)
if greedy_result:
    path = extract_path(greedy_result)
    draw_tree(path)

a_star_result = a_star_search('S', 'G', h_values)
if a_star_result:
    path = extract_path(a_star_result)
    draw_tree(path)

ida_star_result = ida_star('S', 'G', h_values)
if ida_star_result:
    path = extract_path(ida_star_result)
    draw_tree(path)

a_star_memory_result = a_star_memory_limited('S', 'G', h_values, memory_limit=3)
if a_star_memory_result:
    path = extract_path(a_star_memory_result)
    draw_tree(path)
