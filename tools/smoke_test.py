"""Kafka round-trip test: create a temporary topic, produce one message and consume it back.

Prints OK or the error, and exits with status 0 or 1. Run it from WSL2 and from Windows:
    python tools/smoke_test.py [--bootstrap 127.0.0.1:9092]
"""
import argparse
import os
import sys
import time
import uuid

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic


def round_trip(bootstrap: str, timeout_s: float) -> float:
    """Sends one message through a temporary topic and returns the round-trip time in ms."""
    topic = f"smoke-test-{uuid.uuid4().hex[:8]}"
    payload = uuid.uuid4().hex.encode()
    admin = AdminClient({"bootstrap.servers": bootstrap})
    admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic].result(timeout_s)
    try:
        delivery_errors = []

        def on_delivery(err, _msg):
            if err is not None:
                delivery_errors.append(err)

        producer = Producer({"bootstrap.servers": bootstrap})
        start = time.monotonic()
        producer.produce(topic, value=payload, on_delivery=on_delivery)
        if producer.flush(timeout_s) > 0 or delivery_errors:
            raise KafkaException(delivery_errors[0] if delivery_errors else "message was not delivered")

        consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": topic, "enable.auto.commit": False})
        consumer.assign([TopicPartition(topic, 0, 0)])  # read the partition from the start
        try:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                msg = consumer.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())
                if msg.value() == payload:
                    return (time.monotonic() - start) * 1000
            raise TimeoutError(f"message not consumed within {timeout_s:.0f} s")
        finally:
            consumer.close()
    finally:
        admin.delete_topics([topic])[topic].result(timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kafka produce/consume round-trip test.")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
                        help="Kafka bootstrap servers (default: $KAFKA_BOOTSTRAP or 127.0.0.1:9092)")
    parser.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for each step")
    args = parser.parse_args()

    try:
        ms = round_trip(args.bootstrap, args.timeout)
    except Exception as exc:  # report any failure in one line
        detail = str(exc) or "no answer from the broker (is Kafka running and reachable?)"
        print(f"FAILED: Kafka round trip via {args.bootstrap}: {type(exc).__name__}: {detail}")
        return 1
    print(f"OK: Kafka round trip via {args.bootstrap} took {ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
