"""
Hash Table Implementation
Open addressing with linear probing and dynamic resizing.
"""

class HashTable:
    DELETED = object()  # Sentinel for deleted slots

    def __init__(self, initial_capacity=8, load_factor_threshold=0.7):
        self.capacity = initial_capacity
        self.load_factor_threshold = load_factor_threshold
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, key):
        """Linear probing to find slot for key."""
        index = self._hash(key)
        first_deleted = None

        for _ in range(self.capacity):
            if self.keys[index] is None:
                return first_deleted if first_deleted is not None else index
            if self.keys[index] is self.DELETED:
                if first_deleted is None:
                    first_deleted = index
            elif self.keys[index] == key:
                return index
            index = (index + 1) % self.capacity

        return first_deleted  # Table is full of deleted/occupied

    def _resize(self):
        """Double capacity and rehash all entries."""
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0

        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)

    def put(self, key, value):
        if self.size / self.capacity >= self.load_factor_threshold:
            self._resize()

        index = self._probe(key)
        is_new = self.keys[index] is None or self.keys[index] is self.DELETED
        self.keys[index] = key
        self.values[index] = value
        if is_new:
            self.size += 1

    def get(self, key, default=None):
        index = self._hash(key)
        for _ in range(self.capacity):
            if self.keys[index] is None:
                return default
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.capacity
        return default

    def delete(self, key):
        index = self._hash(key)
        for _ in range(self.capacity):
            if self.keys[index] is None:
                raise KeyError(key)
            if self.keys[index] == key:
                self.keys[index] = self.DELETED
                self.values[index] = None
                self.size -= 1
                return
            index = (index + 1) % self.capacity
        raise KeyError(key)

    def __contains__(self, key):
        return self.get(key, self.DELETED) is not self.DELETED

    def __len__(self):
        return self.size

    def __repr__(self):
        items = []
        for k, v in zip(self.keys, self.values):
            if k is not None and k is not self.DELETED:
                items.append(f"{k!r}: {v!r}")
        return "{" + ", ".join(items) + "}"


if __name__ == "__main__":
    ht = HashTable()

    # Insert
    for i in range(20):
        ht.put(f"key_{i}", i * 10)

    print(f"Size: {len(ht)}, Capacity: {ht.capacity}")
    print(f"Get key_5: {ht.get('key_5')}")
    print(f"Contains key_15: {'key_15' in ht}")

    # Delete
    ht.delete("key_5")
    print(f"After delete, get key_5: {ht.get('key_5', 'NOT FOUND')}")
    print(f"Size after delete: {len(ht)}")
