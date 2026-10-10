"""VLM service: answers every question from the Kafka topic `queries` with a vision-language model.

For each query bundle (the current frame plus context frames from the processor's sliding window):
  1. take the newest CONTEXT_FRAMES context frames plus the current frame (0 = current frame only);
  2. run the model on the GPU (greedy decoding);
  3. publish the answer to the topic `answers` and insert the same document (no images) into MongoDB.

Questions are answered one at a time, in order, and none is skipped: the consumer offset is committed
only after an answer has been published and stored, so a restarted service continues with the first
unanswered question.

Run inside WSL2; all settings come from environment variables (see .env.example):
    python vlm_service/service.py
"""
import base64
import io
import json
import logging
import math
import os
import signal
import threading
import time

import torch
from confluent_kafka import Consumer, Producer
from PIL import Image
from pymongo import MongoClient

from backends import QwenVLBackend, VLMBackend


def pixel_count(value: str) -> int:
    """Parses a pixel count such as "200704" or "448*448"."""
    return math.prod(int(part) for part in value.split("*"))


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_PIXELS = pixel_count(os.environ.get("MAX_PIXELS", "448*448"))
# 8 earlier frames show about 3 s of a gesture (motion turns almost every frame into a keyframe).
CONTEXT_FRAMES = int(os.environ.get("CONTEXT_FRAMES", "8"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
GROUP_ID = os.environ.get("GROUP_ID", "vlm-service")

QUERIES_TOPIC = "queries"
ANSWERS_TOPIC = "answers"
MONGO_DB = "edge_vlm"

log = logging.getLogger("vlm_service")


def now_ms() -> int:
    return int(time.time() * 1000)


def mib(num_bytes: int) -> int:
    return round(num_bytes / 2**20)


def decode(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def answer_query(backend: VLMBackend, query: dict) -> dict:
    """Answers one query bundle and returns the answer document (without images)."""
    context = query["context_frames"][-CONTEXT_FRAMES:] if CONTEXT_FRAMES > 0 else []
    frames = context + [query["current_frame"]]  # oldest first; the last one is the current frame

    torch.cuda.reset_peak_memory_stats()
    inference_start_ms = now_ms()
    answer = backend.answer([decode(frame["image_b64"]) for frame in frames], query["question"])
    inference_end_ms = now_ms()

    asked_ms = query["asked_ts_ms"]
    return {
        "run_id": RUN_ID,
        "device_id": query["device_id"],
        "query_id": query["query_id"],
        "question": query["question"],
        "answer": answer,
        "model": backend.name,
        "context_frames_used": len(context),
        "frame_ids_used": [frame["frame_id"] for frame in frames],
        "max_pixels": MAX_PIXELS,
        "timestamps": {
            "asked_ms": asked_ms,
            "bundled_ms": query["bundled_ts_ms"],
            "inference_start_ms": inference_start_ms,
            "inference_end_ms": inference_end_ms,
        },
        "latency_ms": {
            "end_to_end": inference_end_ms - asked_ms,
            "queue": inference_start_ms - asked_ms,
            "inference": inference_end_ms - inference_start_ms,
        },
        "peak_vram_mb": mib(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo.admin.command("ping")  # fail fast if MongoDB is not reachable
    answers_collection = mongo[MONGO_DB].answers

    log.info("Loading %s (bfloat16, max_pixels=%d) on %s", MODEL_ID, MAX_PIXELS, torch.cuda.get_device_name(0)
             if torch.cuda.is_available() else "no GPU")
    backend = QwenVLBackend(MODEL_ID, MAX_PIXELS, MAX_NEW_TOKENS)
    log.info("Model loaded: peak VRAM %d MiB allocated, %d MiB reserved by PyTorch",
             mib(torch.cuda.max_memory_allocated()), mib(torch.cuda.max_memory_reserved()))
    # The first generation on a fresh GPU context is about a second slower (CUDA kernels are loaded on
    # first use). One warm-up answer keeps that delay out of the first real answer and the statistics.
    backend.answer([Image.new("RGB", (448, 336))], "What is in the image?")

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        # A new consumer group starts with the oldest query still in Kafka, so no question is skipped.
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,  # committed by hand after each answer
    })
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP, "client.id": "vlm-service"})
    consumer.subscribe([QUERIES_TOPIC])
    log.info("run_id=%s, group %s, CONTEXT_FRAMES=%d, MAX_NEW_TOKENS=%d; waiting for questions on %s. "
             "Ctrl+C stops the service.", RUN_ID, GROUP_ID, CONTEXT_FRAMES, MAX_NEW_TOKENS, QUERIES_TOPIC)

    # Ctrl+C (or SIGTERM) finishes the current answer, then stops.
    stop_requested = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop_requested.set())
    try:
        while not stop_requested.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka: %s", msg.error())
                continue
            doc = answer_query(backend, json.loads(msg.value()))
            producer.produce(ANSWERS_TOPIC, key=doc["device_id"], value=json.dumps(doc))
            producer.flush(10)
            answers_collection.insert_one(dict(doc))  # a copy: insert_one adds an _id field
            consumer.commit(message=msg, asynchronous=False)
            log.info("Answer %s/%d (%d+1 frames, inference %.1f s, end-to-end %.1f s, peak VRAM %d MiB) \"%s\" -> %s",
                     doc["device_id"], doc["query_id"], doc["context_frames_used"],
                     doc["latency_ms"]["inference"] / 1000, doc["latency_ms"]["end_to_end"] / 1000,
                     doc["peak_vram_mb"], doc["question"], doc["answer"])
    finally:
        log.info("Stopping")
        consumer.close()
        producer.flush(10)


if __name__ == "__main__":
    main()
