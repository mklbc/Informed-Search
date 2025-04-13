import heapq
import time

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

def greedy_search(graph, start, goal, heuristics):
    frontier = []
    heapq.heappush(frontier, Node(start, heuristic=heuristics[start]))
    explored = set()
    node_count = 0
    start_time = time.time()

    while frontier:
        node_count += 1
        current_node = heapq.heappop(frontier)
        if current_node.state == goal:
            return reconstruct_path(current_node), node_count, time.time() - start_time

        explored.add(current_node.state)

        for neighbor, cost in graph[current_node.state]:
            if neighbor not in explored:
                heapq.heappush(frontier, Node(neighbor, current_node, cost, heuristics[neighbor]))

    return None, node_count, time.time() - start_time

def a_star(graph, start, goal, heuristics):
    frontier = []
    heapq.heappush(frontier, Node(start, heuristic=heuristics[start]))
    explored = set()
    node_count = 0
    start_time = time.time()

    while frontier:
        node_count += 1
        current_node = heapq.heappop(frontier)
        if current_node.state == goal:
            return reconstruct_path(current_node), node_count, time.time() - start_time

        explored.add(current_node.state)

        for neighbor, cost in graph[current_node.state]:
            if neighbor not in explored:
                heapq.heappush(frontier, Node(neighbor, current_node, current_node.cost + cost, heuristics[neighbor]))

    return None, node_count, time.time() - start_time

def iterative_a_star(graph, start, goal, heuristics):
    def search(graph, node, f_limit):
        if node.state == goal:
            return reconstruct_path(node), node.cost
        min_f_limit = float('inf')
        for neighbor, cost in graph[node.state]:
            f_value = node.cost + cost + heuristics[neighbor]
            if f_value <= f_limit:
                result, total_cost = search(graph, Node(neighbor, node, node.cost + cost, heuristics[neighbor]), f_limit)
                if result is not None:
                    return result, total_cost
            else:
                min_f_limit = min(min_f_limit, f_value)
        return None, min_f_limit

    f_limit = heuristics[start]
    start_node = Node(start, heuristic=heuristics[start])
    node_count = 0
    start_time = time.time()

    while True:
        result, f_limit = search(graph, start_node, f_limit)
        node_count += 1
        if result is not None:
            return result, node_count, time.time() - start_time
        if f_limit == float('inf'):
            return None, node_count, time.time() - start_time

# Example graph and heuristic data
graph = {
    'A': [('B', 1), ('C', 2)],
    'B': [('D', 4), ('E', 2)],
    'C': [('F', 6), ('G', 3)],
    'D': [],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

heuristics = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 3,
    'E': 4,
    'F': 3,
    'G': 0
}

# Running the algorithms
greedy_path, greedy_nodes, greedy_time = greedy_search(graph, 'A', 'G', heuristics)
a_star_path, a_star_nodes, a_star_time = a_star(graph, 'A', 'G', heuristics)
iterative_a_star_path, iterative_a_star_nodes, iterative_a_star_time = iterative_a_star(graph, 'A', 'G', heuristics)

# Print the results
print(f"Greedy Search Path: {greedy_path}, Nodes visited: {greedy_nodes}, Time: {greedy_time:.4f} seconds")
print(f"A* Path: {a_star_path}, Nodes visited: {a_star_nodes}, Time: {a_star_time:.4f} seconds")
print(f"Iterative A* Path: {iterative_a_star_path}, Nodes visited: {iterative_a_star_nodes}, Time: {iterative_a_star_time:.4f} seconds")
