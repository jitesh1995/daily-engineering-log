"""
Kafka Consumer Pattern
Reliable message consumption with error handling and offset management.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "my-consumer-group"
    topic: str = "events"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000


class ReliableConsumer:
    """
    Kafka consumer with:
    - Manual offset commits (at-least-once)
    - Dead letter queue for failed messages
    - Exponential backoff on failures
    - Graceful shutdown
    """

    def __init__(self, config: ConsumerConfig, process_fn: Callable):
        self.config = config
        self.process_fn = process_fn
        self.running = False
        self.stats = {"processed": 0, "failed": 0, "dlq": 0}

    def _deserialize(self, raw_message):
        """Deserialize message value from bytes."""
        try:
            return json.loads(raw_message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Deserialization failed: {e}")
            return None

    def _send_to_dlq(self, message, error):
        """Send failed message to dead letter queue."""
        dlq_record = {
            "original_topic": self.config.topic,
            "original_message": message,
            "error": str(error),
            "timestamp": time.time(),
            "consumer_group": self.config.group_id,
        }
        logger.warning(f"Sending to DLQ: {dlq_record}")
        self.stats["dlq"] += 1
        # In production: produce to {topic}.dlq topic

    def _process_batch(self, messages):
        """Process a batch of messages with error handling."""
        for msg in messages:
            data = self._deserialize(msg.value)
            if data is None:
                self._send_to_dlq(msg.value, "Deserialization failed")
                continue

            retries = 0
            max_retries = 3

            while retries <= max_retries:
                try:
                    self.process_fn(data)
                    self.stats["processed"] += 1
                    break
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries exceeded for message: {e}"
                        )
                        self._send_to_dlq(data, e)
                        self.stats["failed"] += 1
                    else:
                        wait = min(2 ** retries, 30)
                        logger.warning(
                            f"Retry {retries}/{max_retries} in {wait}s: {e}"
                        )
                        time.sleep(wait)

    def run(self):
        """Main consumer loop (pseudo-code for Kafka client)."""
        logger.info(f"Starting consumer for topic: {self.config.topic}")
        self.running = True

        # In production, use confluent_kafka or aiokafka
        # consumer = KafkaConsumer(...)

        try:
            while self.running:
                # messages = consumer.poll(timeout_ms=1000, max_records=config.max_poll_records)
                # self._process_batch(messages)
                # consumer.commit()  # Manual commit after processing
                logger.info(f"Stats: {self.stats}")
                time.sleep(1)  # Placeholder for poll
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            # consumer.close()
            logger.info(f"Final stats: {self.stats}")

    def stop(self):
        self.running = False
