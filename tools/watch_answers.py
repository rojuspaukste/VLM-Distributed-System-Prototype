"""Live view of the VLM answers: prints every answer as it arrives on the `answers` topic.

    python tools/watch_answers.py [--device-id robot-1] [--from-beginning] [--bootstrap 127.0.0.1:9092]
"""
import argparse
import json
import os
import sys
from datetime import datetime

from confluent_kafka import OFFSET_BEGINNING, OFFSET_END, Consumer, KafkaException, TopicPartition

TOPIC = "answers"


def show(answer: dict) -> None:
    latency = answer["latency_ms"]
    answered = datetime.fromtimestamp(answer["timestamps"]["inference_end_ms"] / 1000).strftime("%H:%M:%S")
    print(f"[{answered}] {answer['device_id']} question {answer['query_id']} (run {answer['run_id']})\n"
          f"  Q: {answer['question']}\n"
          f"  A: {answer['answer']}\n"
          f"  {latency['end_to_end'] / 1000:.1f} s end to end (queue {latency['queue'] / 1000:.1f} s, "
          f"inference {latency['inference'] / 1000:.1f} s); "
          f"{answer['context_frames_used']} earlier frames + current, frames {answer['frame_ids_used']}",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the VLM answers live.")
    parser.add_argument("--device-id", help="only show answers for this device")
    parser.add_argument("--from-beginning", action="store_true", help="also show the answers already in Kafka")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
                        help="Kafka bootstrap servers (default: $KAFKA_BOOTSTRAP or 127.0.0.1:9092)")
    args = parser.parse_args()

    # Partitions are assigned directly and no offsets are committed, so the viewer leaves no group state.
    consumer = Consumer({"bootstrap.servers": args.bootstrap, "group.id": "watch-answers",
                         "enable.auto.commit": False})
    try:
        metadata = consumer.list_topics(TOPIC, timeout=10).topics[TOPIC]
        if metadata.error is not None:
            raise KafkaException(metadata.error)
        start = OFFSET_BEGINNING if args.from_beginning else OFFSET_END
        consumer.assign([TopicPartition(TOPIC, p, start) for p in metadata.partitions])
        print(f"Watching {TOPIC} on {args.bootstrap}" + (f" for {args.device_id}" if args.device_id else "")
              + ". Ctrl+C stops.", flush=True)
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Kafka: {msg.error()}", file=sys.stderr)
                continue
            answer = json.loads(msg.value())
            if args.device_id is None or answer.get("device_id") == args.device_id:
                show(answer)
    except KafkaException as exc:
        print(f"Cannot read {TOPIC} via {args.bootstrap}: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        consumer.close()


if __name__ == "__main__":
    sys.exit(main())
