from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from networkx.drawing.nx_agraph import graphviz_layout

"""
# **Water Jug Problem**

The Water Jug Problem is a classic example in AI and search algorithms. Given two jugs with known capacities (e.g., 4 liters and 3 liters), the task is to measure out an exact quantity of water (e.g., 2 liters), using only the allowed operations: filling a jug, emptying a jug, or pouring water from one jug into the other. This problem models real-world resource constraints and is used to demonstrate search strategies like Breadth-First Search (BFS), Depth-First Search (DFS), and A*.

In this demo, we solve a bounded version of the problem where:
- Jug A has a capacity of 4 liters.
- Jug B has a capacity of 3 liters.
- The goal is to measure exactly 2 liters in Jug A.

The state space is defined by the pair $(a, b)$ where:
- $a$ is the current amount of water in Jug A.
- $b$ is the current amount in Jug
"""
def _init():
    global print_solution
    global JUG_A_CAPACITY
    global JUG_B_CAPACITY
    global MAX_DEPTH
    global GOAL_STATE

    # Constants
    JUG_A_CAPACITY = 4
    JUG_B_CAPACITY = 3
    MAX_DEPTH = 7
    GOAL_STATE = (2, 0)
    def print_solution(path):
        """
        Prints the solution path in a human-readable format.
        """
        if path is None:
            print("No solution found within depth limit.")
        else:
            print("Solution found in", len(path) - 1, "moves:\n")
            for i, state in enumerate(path):
                print(f"Step {i}: Jug A = {state[0]}, Jug B = {state[1]}")
            print()
"""
# **Generating Successor States**

This function defines the state transition model for the water jug problem. Given a current state (a, b) representing the amounts of water in Jug A and Jug B, it returns all possible valid next states reachable via one of the following legal operations:

*   Filling a jug completely (to its maximum capacity).
*   Emptying a jug entirely.
*   Pouring water from one jug to the other until either the source is empty or the destination is full.

These transitions define the edges in the state space graph and are used by the search algorithm (e.g., BFS) to explore all reachable configurations from a given node. The function returns a list of unique successor states, ensuring that each represents a valid one-step move from the current state.
"""
def _get_neighbors():
    global get_neighbors
    def get_neighbors(state):
        """
        Returns all possible next states from the given state.
        """
        a, b = state
        successors = set()

        # Fill either jug
        successors.add((JUG_A_CAPACITY, b))  # Fill A
        successors.add((a, JUG_B_CAPACITY))  # Fill B

        # Empty either jug
        successors.add((0, b))               # Empty A
        successors.add((a, 0))               # Empty B

        # Pour A -> B
        pour = min(a, JUG_B_CAPACITY - b)
        successors.add((a - pour, b + pour))

        # Pour B -> A
        pour = min(b, JUG_A_CAPACITY - a)
        successors.add((a + pour, b - pour))

        return list(successors)

"""
## **State Space Graph Construction**

We build the entire state space graph by exploring all reachable states from the start using BFS. Each state $(a, b)$ is represented as a node, and each valid operation is an edge.

This allows for structural analysis and visualization of the complete search space, which is useful for comparing BFS with other algorithms later (e.g., DFS, UCS, A*).
"""
def _ssgc():
    global build_state_space
    global draw_state_space
    def build_state_space(start):
        G = nx.DiGraph()
        queue = deque()
        queue.append((start, 0))
        visited = set()

        while queue:
            state, depth = queue.popleft()
            if state in visited or depth > MAX_DEPTH:
                continue
            visited.add(state)

            for neighbor in get_neighbors(state):
                G.add_edge(state, neighbor)
                queue.append((neighbor, depth + 1))

        return G

    def draw_state_space(G):
        # pos = nx.spring_layout(G, seed=42)  # Stable layout
        pos = nx.kamada_kawai_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="monospace")

        plt.title("Water Jug Problem: State Space")
        plt.axis("off")
        plt.show()

"""
# **Breadth-First Search (BFS) for Finding a Solution Path**

This function implements Breadth-First Search (BFS) to solve the water jug problem by exploring the state space level by level. Starting from the initial state, it systematically explores all reachable states while tracking the path taken to reach each one.

A queue (FIFO) is used to ensure nodes are explored in breadth-first order.

Each queue entry contains the current state and the path taken to reach it.

A visited set prevents re-exploring the same state, improving efficiency and avoiding cycles.

A depth limit (`MAX_DEPTH`) ensures the search remains bounded and avoids infinite loops in large or improperly constrained state spaces.

When the goal state is found, the function returns the entire sequence of states from start to goal, representing a valid solution path.
"""
def _bfs():
    global bfs
    def bfs(start_state):
        """
        Breadth-First Search to find a path from start_state to GOAL_STATE.
        """
        queue = deque()
        queue.append((start_state, [start_state]))  # (current state, path taken)
        visited = set()

        while queue:
            current_state, path = queue.popleft()

            if current_state in visited or len(path) > MAX_DEPTH:
                continue

            visited.add(current_state)

            if current_state == GOAL_STATE:
                return path

            for neighbor in get_neighbors(current_state):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None

"""
Run everything
"""
def run_everything():
    initial_state = (0, 0)
    state_graph = build_state_space(initial_state)
    draw_state_space(state_graph)
    initial_state = (0, 0)
    print("Solving the Water Jug problem using BFS...\n")
    solution_path = bfs(initial_state)
    print_solution(solution_path)

def _run():
    _init()
    _get_neighbors()
    _ssgc()
    _bfs()
    run_everything()

def main():
    _run()
