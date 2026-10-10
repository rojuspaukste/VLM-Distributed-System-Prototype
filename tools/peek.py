"""Print the newest messages of a Kafka topic, with every image_b64 replaced by its length.

    python tools/peek.py --topic frames [-n 5] [--bootstrap 127.0.0.1:9092]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

from confluent_kafka import Consumer, KafkaException, TopicPartition


def newest_messages(bootstrap: str, topic: str, n: int, timeout_s: float = 10.0) -> list:
    """Returns the newest `n` messages of `topic` across all partitions, oldest first."""
    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": "peek", "enable.auto.commit": False})
    try:
        metadata = consumer.list_topics(topic, timeout=timeout_s).topics[topic]
        if metadata.error is not None:
            raise KafkaException(metadata.error)

        # The newest n messages overall are among the last n of each partition.
        assignment, end_offsets = [], {}
        for partition in metadata.partitions:
            low, high = consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=timeout_s)
            if high > low:
                assignment.append(TopicPartition(topic, partition, max(low, high - n)))
                end_offsets[partition] = high
        consumer.assign(assignment)

        messages = []
        deadline = time.monotonic() + timeout_s
        while end_offsets and time.monotonic() < deadline:
            msg = consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            messages.append(msg)
            if msg.offset() + 1 >= end_offsets.get(msg.partition(), 0):
                end_offsets.pop(msg.partition(), None)  # reached the end of this partition
        messages.sort(key=lambda m: m.timestamp()[1])
        return messages[-n:]
    finally:
        consumer.close()


def without_images(value):
    """Returns a copy of a JSON value with every image_b64 field (also nested ones, as in query
    bundles) replaced by its length."""
    if isinstance(value, dict):
        return {key: f"<{len(item)} chars>" if key == "image_b64" and isinstance(item, str) else without_images(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [without_images(item) for item in value]
    return value


def describe(value: bytes) -> str:
    """Formats a message value as JSON without the base64 images."""
    try:
        doc = json.loads(value)
    except (TypeError, ValueError):
        return repr(value[:200])
    return json.dumps(without_images(doc), ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the newest messages of a Kafka topic.")
    parser.add_argument("--topic", required=True, help="topic name, e.g. frames")
    parser.add_argument("-n", type=int, default=5, help="number of messages (default 5)")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
                        help="Kafka bootstrap servers (default: $KAFKA_BOOTSTRAP or 127.0.0.1:9092)")
    args = parser.parse_args()

    try:
        messages = newest_messages(args.bootstrap, args.topic, args.n)
    except KafkaException as exc:
        print(f"Cannot read topic {args.topic} via {args.bootstrap}: {exc}")
        return 1
    if not messages:
        print(f"Topic {args.topic} is empty.")
    for msg in messages:
        ts = datetime.fromtimestamp(msg.timestamp()[1] / 1000).strftime("%H:%M:%S.%f")[:-3]
        key = msg.key().decode() if msg.key() else None
        size_kb = len(msg.value() or b"") / 1024
        print(f"[{ts}] partition={msg.partition()} offset={msg.offset()} key={key} size={size_kb:.1f} KB")
        print(f"  {describe(msg.value())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
