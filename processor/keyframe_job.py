"""Spark Structured Streaming job: keyframes, visual memory and question bundles.

For every micro-batch read from the Kafka topic `frames`:
  1. Executors decode each JPEG and compute a 64-bit difference hash (dHash) in a pandas UDF.
  2. The driver collects only the small metadata columns and walks each device's frames in frame_id
     order. A frame is a keyframe if it is the device's first frame, a scene change (Hamming distance
     to the last keyframe >= CHANGE_THRESHOLD) or a heartbeat (no keyframe for more than HEARTBEAT_S
     seconds). The first frame that carries a new, non-empty question triggers a query.
  3. Executors downscale just those frames (a second pandas UDF) and the driver collects them.
  4. The driver keeps a sliding window of recent keyframes per device: its visual memory, held in
     memory only. Each new question becomes ONE message on the Kafka topic `queries`: the current
     frame plus the window, so the VLM can answer questions about actions and changes. The bundles
     are built from driver memory, so the driver publishes them with a Kafka producer.
  5. The keyframes' metadata (no images) and one `batch_stats` document go to MongoDB.

Run inside WSL2; all settings come from environment variables (see .env.example):
    python processor/keyframe_job.py
"""
import base64
import io
import json
import logging
import os
import signal
import sys
import threading
import operator
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import pandas as pd
import pyspark
from confluent_kafka import Producer
from PIL import Image
from pymongo import MongoClient
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
# 6 of 64 bits: still webcam scenes vary by 0-3 bits, a hand gesture by about 6-10.
CHANGE_THRESHOLD = int(os.environ.get("CHANGE_THRESHOLD", "6"))
HEARTBEAT_S = float(os.environ.get("HEARTBEAT_S", "5"))
TRIGGER_MS = int(os.environ.get("TRIGGER_MS", "500"))
MAX_OFFSETS_PER_TRIGGER = int(os.environ.get("MAX_OFFSETS_PER_TRIGGER", "500"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.expanduser("~/edge-vlm-data/checkpoint"))
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
WINDOW_MAX_FRAMES = int(os.environ.get("WINDOW_MAX_FRAMES", "8"))
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", "30"))
WINDOW_IMAGE_MAX_SIDE = int(os.environ.get("WINDOW_IMAGE_MAX_SIDE", "448"))

FRAMES_TOPIC = "frames"
QUERIES_TOPIC = "queries"
MONGO_DB = "edge_vlm"
# The connector must match the installed PySpark version exactly; PySpark 4.x uses Scala 2.13.
KAFKA_CONNECTOR = f"org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}"
# Kafka's default message size limit is about 1 MB; query bundles stay below this.
QUERY_MAX_BYTES = 900_000
DOWNSCALE_JPEG_QUALITY = 85

FRAME_SCHEMA = StructType([
    StructField("device_id", StringType()),
    StructField("frame_id", LongType()),
    StructField("capture_ts_ms", LongType()),
    StructField("instruction_id", LongType()),
    StructField("instruction", StringType()),
    StructField("width", IntegerType()),
    StructField("height", IntegerType()),
    StructField("image_b64", StringType()),
])
REASONS = ("first", "change", "heartbeat")

log = logging.getLogger("keyframe_job")


def now_ms() -> int:
    return int(time.time() * 1000)


# --- Runs on the executors -------------------------------------------------------------------

def dhash(image_b64: str) -> int:
    """64-bit difference hash: shrink to 9x8 grey pixels and compare each pixel with its right
    neighbour (8 rows x 8 comparisons). Returned as a signed integer to fit Spark's LongType."""
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int.from_bytes(np.packbits(bits).tobytes(), "big", signed=True)


@pandas_udf(LongType())
def dhash_udf(images: pd.Series) -> pd.Series:
    return images.map(dhash)


def downscale(image_b64: str) -> str:
    """Shrinks a JPEG so that its longest side is at most WINDOW_IMAGE_MAX_SIDE pixels."""
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    if max(image.size) <= WINDOW_IMAGE_MAX_SIDE:
        return image_b64  # small enough already: keep the original JPEG
    image = image.convert("RGB")
    image.thumbnail((WINDOW_IMAGE_MAX_SIDE, WINDOW_IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=DOWNSCALE_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pandas_udf(StringType())
def downscale_udf(images: pd.Series) -> pd.Series:
    return images.map(downscale)


# --- Runs on the driver ----------------------------------------------------------------------

def hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit hashes."""
    return ((a ^ b) & 0xFFFF_FFFF_FFFF_FFFF).bit_count()


@dataclass
class WindowFrame:
    """A frame in a device's visual memory. The downscaled image is kept in driver memory only."""
    frame_id: int
    capture_ts_ms: int
    reason: str      # keyframe reason, or "query" for a question's frame
    image_b64: str


@dataclass
class DeviceState:
    """What the driver remembers about a device between micro-batches."""
    last_frame_id: int
    last_hash: int                  # dHash of the last keyframe
    last_keyframe_ts_ms: int        # capture time of the last keyframe
    last_instruction_id: int = 0    # 0 = no question asked yet
    window: deque = field(default_factory=deque)  # WindowFrames, oldest first


@dataclass
class Decision:
    """A frame the driver needs: a keyframe, a question's frame, or both."""
    device_id: str
    frame_id: int
    capture_ts_ms: int
    instruction_id: int
    reason: str | None              # keyframe reason; None if the frame is only a question's frame
    hash_distance: int | None       # None for a device's first frame
    question: str | None            # set if this frame triggers a query
    state: DeviceState              # the device state at decision time (holds the window)


def decide(frames: list, states: dict) -> list:
    """Applies the keyframe rules and the query trigger to one batch of frame metadata.

    Frames are walked in frame_id order per device; `states` (device_id -> DeviceState) is updated.
    The heartbeat uses capture timestamps, so the result does not depend on processing speed.
    """
    decisions = []
    for frame in sorted(frames, key=lambda f: (f.device_id, f.frame_id)):
        state = states.get(frame.device_id)
        if state is not None and frame.frame_id == state.last_frame_id:
            continue  # the same frame delivered twice
        if state is not None and frame.frame_id < state.last_frame_id:
            # A lower frame_id means the device was restarted: start over, with an empty window.
            log.info("Device %s restarted (frame %d after %d); resetting its state",
                     frame.device_id, frame.frame_id, state.last_frame_id)
            state = None

        distance = None
        if state is None:
            reason = "first"
            state = DeviceState(frame.frame_id, frame.dhash, frame.capture_ts_ms)
            states[frame.device_id] = state
        else:
            distance = hamming(frame.dhash, state.last_hash)
            if distance >= CHANGE_THRESHOLD:
                reason = "change"
            elif frame.capture_ts_ms - state.last_keyframe_ts_ms > HEARTBEAT_S * 1000:
                reason = "heartbeat"
            else:
                reason = None
            if reason:
                state.last_hash, state.last_keyframe_ts_ms = frame.dhash, frame.capture_ts_ms
        state.last_frame_id = frame.frame_id

        question = None
        if frame.instruction_id != state.last_instruction_id:
            state.last_instruction_id = frame.instruction_id
            question = (frame.instruction or "").strip() or None  # an empty question asks nothing

        if reason or question:
            decisions.append(Decision(frame.device_id, frame.frame_id, frame.capture_ts_ms,
                                      frame.instruction_id, reason, distance, question, state))
    return decisions


def build_query(decision: Decision, image_b64: str, context: list) -> str:
    """Returns the `queries` message for a question: the current frame plus the context frames
    (oldest to newest). If the message would be too large, the oldest context frames are dropped."""
    message = {
        "device_id": decision.device_id,
        "query_id": decision.instruction_id,
        "question": decision.question,
        "asked_ts_ms": decision.capture_ts_ms,
        "bundled_ts_ms": now_ms(),
        "current_frame": {"frame_id": decision.frame_id, "capture_ts_ms": decision.capture_ts_ms,
                          "image_b64": image_b64},
        "context_frames": [{"frame_id": f.frame_id, "capture_ts_ms": f.capture_ts_ms, "reason": f.reason,
                            "image_b64": f.image_b64} for f in context],
    }
    value = json.dumps(message)
    while len(value) > QUERY_MAX_BYTES and message["context_frames"]:
        message["context_frames"].pop(0)
        value = json.dumps(message)
    if len(message["context_frames"]) < len(context):
        log.warning("Query %s/%d: dropped %d oldest context frames to stay under %d bytes",
                    decision.device_id, decision.instruction_id,
                    len(context) - len(message["context_frames"]), QUERY_MAX_BYTES)
    log.info("Query %s/%d \"%s\": current frame %d + %d context frames, %.0f KB",
             decision.device_id, decision.instruction_id, decision.question, decision.frame_id,
             len(message["context_frames"]), len(value) / 1024)
    return value


def update_windows(decisions: list, images: dict) -> list:
    """Replays the decisions in order: bundles each question with its device's window, then adds the
    frame to the window. Returns the query messages as (device_id, json) pairs."""
    queries = []
    for decision in decisions:
        image_b64 = images[(decision.device_id, decision.frame_id)]
        window = decision.state.window
        # Visual memory: only frames from the last WINDOW_SECONDS before this frame.
        while window and window[0].capture_ts_ms < decision.capture_ts_ms - WINDOW_SECONDS * 1000:
            window.popleft()
        if decision.question:
            queries.append((decision.device_id, build_query(decision, image_b64, list(window))))
        window.append(WindowFrame(decision.frame_id, decision.capture_ts_ms, decision.reason or "query",
                                  image_b64))
        while len(window) > WINDOW_MAX_FRAMES:
            window.popleft()
    return queries


class KeyframeBatchProcessor:
    """foreachBatch handler; keeps the per-device state and windows on the driver."""

    def __init__(self, spark: SparkSession, mongo_db, producer: Producer):
        self.spark = spark
        self.mongo_db = mongo_db
        self.producer = producer
        self.states = {}
        self.config = {
            "spark_master": SPARK_MASTER,
            "cores": spark.sparkContext.defaultParallelism,
            "change_threshold": CHANGE_THRESHOLD,
            "heartbeat_s": HEARTBEAT_S,
            "trigger_ms": TRIGGER_MS,
            "max_offsets_per_trigger": MAX_OFFSETS_PER_TRIGGER,
            "window_max_frames": WINDOW_MAX_FRAMES,
            "window_seconds": WINDOW_SECONDS,
            "window_image_max_side": WINDOW_IMAGE_MAX_SIDE,
        }

    def process_batch(self, batch: DataFrame, batch_id: int) -> None:
        batch_start_ms = now_ms()
        # The batch is used twice (metadata collect, image collect). Caching it avoids reading Kafka
        # and hashing twice; MEMORY_ONLY guarantees that images are never spilled to disk.
        batch.persist(StorageLevel.MEMORY_ONLY)
        try:
            frames = batch.select("device_id", "frame_id", "instruction_id", "instruction",
                                  "capture_ts_ms", "dhash").collect()
            if not frames:
                return
            selected_ts_ms = now_ms()  # all frames of the batch are decoded and hashed at this point
            decisions = decide(frames, self.states)
            queries = []
            if decisions:
                images = self.collect_downscaled(batch, decisions)
                queries = update_windows(decisions, images)
            if queries:
                self.write_queries_to_kafka(queries)
            self.write_to_mongo(batch_id, batch_start_ms, frames, decisions, len(queries), selected_ts_ms)
        finally:
            batch.unpersist()

    @staticmethod
    def collect_downscaled(batch: DataFrame, decisions: list) -> dict:
        """Downscales only the decided frames on the executors and collects them:
        returns {(device_id, frame_id): image_b64}."""
        keys = sorted({(d.device_id, d.frame_id) for d in decisions})
        wanted = reduce(operator.or_, [(F.col("device_id") == device_id) & (F.col("frame_id") == frame_id)
                                       for device_id, frame_id in keys])
        rows = (batch.where(wanted)
                .select("device_id", "frame_id", downscale_udf("image_b64").alias("image_b64"))
                .collect())
        return {(row.device_id, row.frame_id): row.image_b64 for row in rows}

    def write_queries_to_kafka(self, queries: list) -> None:
        # The bundles exist only in driver memory, so the driver publishes them itself. (A Spark write
        # would first copy them into a temporary file and costs about 0.7 s per batch.) One idempotent
        # producer keeps each device's queries in order.
        failed = []
        for device_id, value in queries:
            self.producer.produce(QUERIES_TOPIC, key=device_id, value=value,
                                  on_delivery=lambda err, _msg: failed.append(err) if err else None)
        self.producer.flush(10)
        if failed:
            log.error("%d of %d queries were not delivered: %s", len(failed), len(queries), failed[0])

    def write_to_mongo(self, batch_id: int, batch_start_ms: int, frames: list, decisions: list,
                       queries_out: int, selected_ts_ms: int) -> None:
        keyframes = [d for d in decisions if d.reason]
        if keyframes:
            self.mongo_db.keyframes.insert_many([
                {
                    "run_id": RUN_ID,
                    "device_id": d.device_id,
                    "frame_id": d.frame_id,
                    "instruction_id": d.instruction_id,
                    "reason": d.reason,
                    "hash_distance": d.hash_distance,
                    "capture_ts_ms": d.capture_ts_ms,
                    "selected_ts_ms": selected_ts_ms,
                }
                for d in keyframes
            ])

        reasons = Counter(d.reason for d in keyframes)
        delays = [selected_ts_ms - f.capture_ts_ms for f in frames]
        batch_end_ms = now_ms()
        stats = {
            "run_id": RUN_ID,
            "batch_id": batch_id,
            "batch_start_ms": batch_start_ms,
            "batch_end_ms": batch_end_ms,
            "duration_ms": batch_end_ms - batch_start_ms,
            "frames_in": len(frames),
            "keyframes_out": len(keyframes),
            "queries_out": queries_out,
            "drop_ratio": round(1 - len(keyframes) / len(frames), 4),
            "reasons": {reason: reasons.get(reason, 0) for reason in REASONS},
            "devices": dict(Counter(f.device_id for f in frames)),
            "ingest_delay_ms": {"mean": round(sum(delays) / len(delays)), "max": max(delays)},
            "config": self.config,
        }
        self.mongo_db.batch_stats.insert_one(stats)
        log.info("batch %d: %d frames in, %d keyframes out (drop %.1f%%), %d queries out, %s, %d ms",
                 batch_id, len(frames), len(keyframes), 100 * stats["drop_ratio"], queries_out,
                 " ".join(f"{r}={n}" for r, n in stats["reasons"].items()), stats["duration_ms"])


def create_spark() -> SparkSession:
    # Python UDFs run in worker processes; make them use this interpreter (the virtual environment).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = (SparkSession.builder
             .appName("edge-vlm-keyframes")
             .master(SPARK_MASTER)
             .config("spark.jars.packages", KAFKA_CONNECTOR)
             .config("spark.sql.shuffle.partitions", "4")  # tiny local job; avoid 200 empty tasks
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("py4j").setLevel(logging.WARNING)  # hide py4j's callback-server chatter

    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo.admin.command("ping")  # fail fast if MongoDB is not reachable
    spark = create_spark()
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP, "client.id": "keyframe-job",
                         "enable.idempotence": True})
    processor = KeyframeBatchProcessor(spark, mongo[MONGO_DB], producer)
    log.info("run_id=%s, Kafka %s, MongoDB %s, checkpoint %s, config %s",
             RUN_ID, KAFKA_BOOTSTRAP, MONGO_URI, CHECKPOINT_DIR, processor.config)

    frames = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
              .option("subscribe", FRAMES_TOPIC)
              .option("startingOffsets", "latest")
              .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
              # Split Kafka partitions into more Spark tasks, so every local core gets work.
              .option("minPartitions", spark.sparkContext.defaultParallelism)
              # Frames expire after one hour; if the job was stopped longer, skip what is gone.
              .option("failOnDataLoss", "false")
              .load()
              .select(F.from_json(F.col("value").cast("string"), FRAME_SCHEMA).alias("frame"))
              .select("frame.*")
              .where(F.col("device_id").isNotNull() & F.col("image_b64").isNotNull())
              .withColumn("dhash", dhash_udf("image_b64")))

    query = (frames.writeStream
             .foreachBatch(processor.process_batch)
             .trigger(processingTime=f"{TRIGGER_MS} milliseconds")
             .option("checkpointLocation", CHECKPOINT_DIR)
             .start())
    log.info("Streaming query started (Spark %s); waiting for frames on %s. Ctrl+C stops the job.",
             spark.version, FRAMES_TOPIC)

    # Ctrl+C (or SIGTERM) only sets a flag and the query is stopped between polls. PySpark's own
    # SIGINT handler would interrupt the blocking Py4J call and end the job with a Py4JError.
    stop_requested = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop_requested.set())
    try:
        while query.isActive and not stop_requested.is_set():
            query.awaitTermination(1)
        log.info("Stopping the streaming query")
        query.stop()
    finally:
        producer.flush(10)
        spark.stop()


if __name__ == "__main__":
    main()
