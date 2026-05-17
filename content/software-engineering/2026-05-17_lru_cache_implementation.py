"""
LRU Cache Implementation
Using a doubly-linked list + hashmap for O(1) operations.
"""

class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    Least Recently Used Cache.
    get() and put() both run in O(1) time.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Sentinel nodes (avoid null checks)
        self.head = Node()  # Most recently used
        self.tail = Node()  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Remove node from doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node):
        """Add node right after head (most recent position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # Move to front (most recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: int, value: int):
        if key in self.cache:
            # Update existing
            self._remove(self.cache[key])
            del self.cache[key]

        # Add new node
        node = Node(key, value)
        self._add_to_front(node)
        self.cache[key] = node

        # Evict if over capacity
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        items = []
        node = self.head.next
        while node != self.tail:
            items.append(f"{node.key}:{node.value}")
            node = node.next
        return f"LRUCache([{', '.join(items)}])"


if __name__ == "__main__":
    cache = LRUCache(3)

    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    print(cache)  # [3:c, 2:b, 1:a]

    cache.get(1)   # Access 1, moves to front
    print(cache)  # [1:a, 3:c, 2:b]

    cache.put(4, "d")  # Evicts key 2 (least recent)
    print(cache)  # [4:d, 1:a, 3:c]

    print(f"Get 2: {cache.get(2)}")  # -1 (evicted)
    print(f"Get 3: {cache.get(3)}")  # c
