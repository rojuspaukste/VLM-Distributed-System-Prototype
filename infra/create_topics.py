"""Create the pipeline's Kafka topics. Idempotent: topics that already exist are skipped.
The `keyframes` topic of the first design (replaced by `queries`) is deleted if it still exists.

Run once after starting Kafka (uses the tools environment):
    python infra/create_topics.py [--bootstrap 127.0.0.1:9092]
"""
import argparse
import logging
import os
import sys
import time

from confluent_kafka.admin import AdminClient, NewTopic

log = logging.getLogger("create_topics")

# Frames and query bundles contain images, so they are kept for one hour only. Kafka deletes whole
# log segments, so a new segment is started every 10 minutes; otherwise old images would stay until
# a 1 GB segment fills up.
SHORT_RETENTION = {"retention.ms": str(60 * 60 * 1000), "segment.ms": str(10 * 60 * 1000)}

# topic name -> (partitions, topic config)
TOPICS = {
    "frames": (4, SHORT_RETENTION),
    "queries": (4, SHORT_RETENTION),
    "answers": (1, {}),
}
# Topics of earlier designs that are deleted when found.
OBSOLETE_TOPICS = ["keyframes"]


def wait_for_topics(admin: AdminClient, names, timeout_s: float = 10.0) -> dict:
    """Returns topic metadata once all `names` are visible (new topics can take a moment)."""
    deadline = time.monotonic() + timeout_s
    while True:
        topics = admin.list_topics(timeout=timeout_s).topics
        if all(name in topics for name in names) or time.monotonic() > deadline:
            return topics
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the frames, queries and answers topics.")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
                        help="Kafka bootstrap servers (default: $KAFKA_BOOTSTRAP or 127.0.0.1:9092)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    admin = AdminClient({"bootstrap.servers": args.bootstrap})
    existing = admin.list_topics(timeout=10).topics

    to_create = []
    for name, (partitions, config) in TOPICS.items():
        if name in existing:
            log.info("Topic %s already exists, skipping", name)
        else:
            to_create.append(NewTopic(name, num_partitions=partitions, replication_factor=1, config=config))
    if to_create:
        for name, future in admin.create_topics(to_create).items():
            future.result()  # raises if the broker rejected the topic
            log.info("Created topic %s", name)

    obsolete = [name for name in OBSOLETE_TOPICS if name in existing]
    if obsolete:
        for name, future in admin.delete_topics(obsolete).items():
            future.result()
            log.info("Deleted obsolete topic %s", name)

    # Print the final state so the partition counts can be checked at a glance.
    topics = wait_for_topics(admin, TOPICS)
    all_ok = True
    for name, (partitions, _) in TOPICS.items():
        actual = len(topics[name].partitions) if name in topics else 0
        ok = actual == partitions
        all_ok = all_ok and ok
        log.info("%-9s partitions=%d %s", name, actual, "OK" if ok else f"(expected {partitions})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
