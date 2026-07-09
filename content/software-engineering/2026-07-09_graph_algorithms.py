"""
Graph Algorithms
BFS, DFS, Dijkstra, Topological Sort.
"""
from collections import defaultdict, deque
import heapq

class Graph:
    def __init__(self, directed=True):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def bfs(self, start):
        """Breadth-First Search. Returns visit order and distances."""
        visited = {start: 0}
        queue = deque([start])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)

        return order, visited

    def dfs(self, start):
        """Depth-First Search (iterative). Returns visit order."""
        visited = set()
        stack = [start]
        order = []

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for neighbor, _ in reversed(self.adj[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

        return order

    def dijkstra(self, start):
        """Shortest paths from start (non-negative weights)."""
        dist = {start: 0}
        prev = {start: None}
        heap = [(0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue

            for v, weight in self.adj[u]:
                new_dist = d + weight
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        return dist, prev

    def topological_sort(self):
        """Kahn's algorithm for topological ordering (DAG only)."""
        in_degree = defaultdict(int)
        for u in self.adj:
            for v, _ in self.adj[u]:
                in_degree[v] += 1

        queue = deque([u for u in self.adj if in_degree[u] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.adj):
            raise ValueError("Graph has a cycle — no topological order exists")
        return order

    def shortest_path(self, start, end):
        """Reconstruct shortest path using Dijkstra."""
        dist, prev = self.dijkstra(start)
        if end not in dist:
            return None, float("inf")

        path = []
        node = end
        while node is not None:
            path.append(node)
            node = prev[node]
        return list(reversed(path)), dist[end]


if __name__ == "__main__":
    g = Graph(directed=False)
    g.add_edge("A", "B", 4)
    g.add_edge("A", "C", 2)
    g.add_edge("B", "D", 3)
    g.add_edge("C", "D", 1)
    g.add_edge("C", "E", 5)
    g.add_edge("D", "E", 2)

    print("BFS:", g.bfs("A")[0])
    print("DFS:", g.dfs("A"))

    path, cost = g.shortest_path("A", "E")
    print(f"Shortest A→E: {path} (cost: {cost})")

    # Topological sort (DAG)
    dag = Graph(directed=True)
    dag.add_edge("compile", "test")
    dag.add_edge("compile", "lint")
    dag.add_edge("test", "deploy")
    dag.add_edge("lint", "deploy")
    print(f"Build order: {dag.topological_sort()}")
