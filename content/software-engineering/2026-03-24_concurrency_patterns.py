"""
Concurrency Patterns in Python
Thread pool, async/await, producer-consumer.
"""
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from dataclasses import dataclass
from typing import Callable


# === 1. Thread Pool Pattern ===
def thread_pool_example():
    """Process multiple tasks concurrently with a thread pool."""

    def fetch_url(url):
        time.sleep(0.5)  # Simulate I/O
        return f"Response from {url}"

    urls = [f"https://api.example.com/page/{i}" for i in range(10)]

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_url, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error fetching {url}: {e}")

    return results


# === 2. Async/Await Pattern ===
async def async_example():
    """Async I/O for high-concurrency scenarios."""

    async def fetch_data(session_id: int):
        await asyncio.sleep(0.3)  # Simulate async I/O
        return {"session": session_id, "data": f"result_{session_id}"}

    # Run many concurrent requests
    tasks = [fetch_data(i) for i in range(20)]
    results = await asyncio.gather(*tasks)
    return results


# === 3. Producer-Consumer Pattern ===
@dataclass
class Task:
    task_id: int
    payload: str


def producer_consumer_example():
    """Classic producer-consumer with a bounded queue."""
    queue = Queue(maxsize=10)
    results = []
    lock = threading.Lock()

    def producer(n_tasks):
        for i in range(n_tasks):
            task = Task(task_id=i, payload=f"data_{i}")
            queue.put(task)  # Blocks if queue is full
        # Poison pills to signal consumers to stop
        for _ in range(3):
            queue.put(None)

    def consumer(consumer_id):
        while True:
            task = queue.get()  # Blocks if queue is empty
            if task is None:
                break
            # Process task
            result = f"Consumer-{consumer_id} processed task-{task.task_id}"
            with lock:
                results.append(result)
            queue.task_done()

    # Start producer and consumers
    producer_thread = threading.Thread(target=producer, args=(20,))
    consumer_threads = [
        threading.Thread(target=consumer, args=(i,)) for i in range(3)
    ]

    producer_thread.start()
    for t in consumer_threads:
        t.start()

    producer_thread.join()
    for t in consumer_threads:
        t.join()

    return results


# === 4. Rate Limiter (Token Bucket) ===
class TokenBucketRateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # Tokens per second
        self.capacity = capacity  # Max tokens
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait_and_acquire(self):
        while not self.acquire():
            time.sleep(0.01)


if __name__ == "__main__":
    # Thread pool
    results = thread_pool_example()
    print(f"Thread pool: {len(results)} results")

    # Async
    results = asyncio.run(async_example())
    print(f"Async: {len(results)} results")

    # Producer-Consumer
    results = producer_consumer_example()
    print(f"Producer-Consumer: {len(results)} results")

    # Rate limiter
    limiter = TokenBucketRateLimiter(rate=5, capacity=5)
    allowed = sum(1 for _ in range(10) if limiter.acquire())
    print(f"Rate limiter: {allowed}/10 requests allowed")
