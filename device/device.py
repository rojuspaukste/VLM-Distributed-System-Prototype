"""Simulated camera device: streams frames plus the current question to Kafka and prints the answers.

Frames come from a webcam or a video file. Each frame is resized, JPEG-encoded and published to the
`frames` topic at a fixed rate, with the device id as message key. Every frame carries the current
question (instruction_id + text). The device starts without a question (instruction_id 0); every line
typed on stdin is a new question, and `quit` stops the device. A --questions-file asks questions
automatically. A background consumer prints this device's answers from the `answers` topic.

The terminal shows only the questions and answers; everything else is logged to a file
(default logs/<device-id>.log in the repository).

    python device/device.py --preview
    python device/device.py --source data/libero-1.mp4 --no-input --questions-file eval/libero_questions.tsv
    python device/device.py --source data/libero-2.mp4 --loop --no-input --device-id robot-2
"""
import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
from confluent_kafka import OFFSET_END, Consumer, KafkaException, Producer, TopicPartition

FRAMES_TOPIC = "frames"
ANSWERS_TOPIC = "answers"
STATS_INTERVAL_S = 5.0
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

log = logging.getLogger("device")              # everything: goes to the log file
console = logging.getLogger("device.console")  # questions and answers: the terminal (and the log file)


def setup_logging(log_file: Path) -> None:
    """Sends all log records to `log_file` and only the `console` logger's to the terminal, so status
    lines never break into a question being typed."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    console.addHandler(terminal)  # console records also reach the log file through the root logger
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV would warn on stderr


class Questions:
    """The current question, shared by the input thread and the capture loop.

    instruction_id 0 with an empty text means that no question has been asked yet.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._id = 0
        self._text = ""

    def current(self) -> tuple[int, str]:
        with self._lock:
            return self._id, self._text

    def ask(self, text: str, typed: bool = False) -> None:
        """Makes `text` the current question. The id grows even if the text repeats. A typed question
        is already on the screen, so only questions from a file or --instruction are printed."""
        with self._lock:
            self._id += 1
            self._text = text
            question_id = self._id
        (log if typed else console).info("Q%d: %s", question_id, text)


def read_stdin(questions: Questions, stop: threading.Event) -> None:
    """Reads stdin: each non-empty line is a new question, `quit` stops the device."""
    for line in sys.stdin:
        text = line.strip()
        if text.lower() == "quit":
            stop.set()
            return
        if text:
            questions.ask(text, typed=True)


def load_questions_file(path: str, fps: float) -> list[tuple[int, str]]:
    """Reads "<seconds><TAB><question>" lines and returns (frame_index, question) pairs in time order.

    The time is converted into a number of frames sent (seconds x fps), so a video file gets each
    question at exactly the same frame on every run. Blank lines and lines starting with # are skipped.
    """
    schedule = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            seconds, tab, question = line.rstrip("\r\n").partition("\t")
            try:
                frame_index = round(float(seconds) * fps)
            except ValueError:
                frame_index = None
            if not tab or not question.strip() or frame_index is None or frame_index < 0:
                raise SystemExit(f"{path}, line {number}: expected '<seconds><TAB><question>', got {line!r}")
            schedule.append((frame_index, question.strip()))
    schedule.sort(key=lambda item: item[0])  # stable: equal times keep the file order
    return schedule


def print_answers(device_id: str, bootstrap: str, stop: threading.Event) -> None:
    """Logs this device's answers, reading the `answers` topic from its current end.

    Partitions are assigned directly and no offsets are committed, so no consumer group state is kept.
    """
    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": f"device-{device_id}",
                         "enable.auto.commit": False, "logger": log})
    try:
        metadata = consumer.list_topics(ANSWERS_TOPIC, timeout=10).topics[ANSWERS_TOPIC]
        if metadata.error is not None:
            raise KafkaException(metadata.error)
        consumer.assign([TopicPartition(ANSWERS_TOPIC, p, OFFSET_END) for p in metadata.partitions])
        while not stop.is_set():
            msg = consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                log.warning("Answers topic: %s", msg.error())
                continue
            answer = json.loads(msg.value())
            if answer.get("device_id") != device_id:
                continue
            console.info("A%d: %s  [%.1f s, %d earlier frames from the last %.0f s + current]", answer["query_id"],
                         answer["answer"], answer["latency_ms"]["end_to_end"] / 1000, answer["context_frames_used"],
                         answer.get("context_span_s", 0))
    except KafkaException as exc:
        log.warning("Cannot read answers from %s: %s", ANSWERS_TOPIC, exc)
    finally:
        consumer.close()


class Stats:
    """Counts sent frames and logs a summary line every STATS_INTERVAL_S seconds."""

    def __init__(self):
        self.total = 0
        self._start_window(time.monotonic())

    def _start_window(self, now: float) -> None:
        self._window_start = now
        self._frames = 0
        self._jpeg_bytes = 0
        self._failed = 0
        self._last_error = None

    def frame_sent(self, jpeg_size: int) -> None:
        self.total += 1
        self._frames += 1
        self._jpeg_bytes += jpeg_size

    def delivery_failed(self, error) -> None:
        self._failed += 1
        self._last_error = error

    def log_if_due(self) -> None:
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed < STATS_INTERVAL_S:
            return
        avg_kb = self._jpeg_bytes / self._frames / 1024 if self._frames else 0.0
        log.info("sent %d frames in %.1f s (%.1f fps), avg JPEG %.1f KB, total %d",
                 self._frames, elapsed, self._frames / elapsed, avg_kb, self.total)
        if self._failed:
            log.warning("%d frames not delivered, last error: %s", self._failed, self._last_error)
        self._start_window(now)


def open_source(source: str) -> cv2.VideoCapture:
    """Opens a webcam (a number such as "0") or a video file (a path)."""
    if source.isdigit():
        # DirectShow opens webcams much faster than the default backend on Windows.
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(int(source), backend)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video source {source!r}")
    return capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream webcam or video frames plus questions to Kafka.")
    parser.add_argument("--source", default="0", help="webcam index (default 0) or path to a video file")
    parser.add_argument("--loop", action="store_true", help="restart the video file when it ends")
    parser.add_argument("--device-id", default="robot-1", help="device id, also the Kafka message key")
    parser.add_argument("--fps", type=float, default=10, help="frames per second to send (default 10)")
    parser.add_argument("--width", type=int, default=640, help="frame width after resizing (default 640)")
    parser.add_argument("--height", type=int, default=480, help="frame height after resizing (default 480)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default 80)")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
                        help="Kafka bootstrap servers (default: $KAFKA_BOOTSTRAP or 127.0.0.1:9092)")
    parser.add_argument("--instruction", help="first question, asked at startup (default: none)")
    parser.add_argument("--questions-file",
                        help="file of '<seconds><TAB><question>' lines, asked automatically after the start")
    parser.add_argument("--preview", action="store_true", help="show the frames in a window")
    parser.add_argument("--no-input", action="store_true", help="don't read questions from stdin (scripted runs)")
    parser.add_argument("--log-file", type=Path,
                        help="log file for everything except questions and answers (default: logs/<device-id>.log)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_file = args.log_file or LOG_DIR / f"{args.device_id}.log"
    setup_logging(log_file)

    schedule = load_questions_file(args.questions_file, args.fps) if args.questions_file else []
    capture = open_source(args.source)
    is_file = not args.source.isdigit()
    producer = Producer({"bootstrap.servers": args.bootstrap, "client.id": args.device_id, "logger": log})
    stats = Stats()

    def on_delivery(err, _msg):
        if err is not None:
            stats.delivery_failed(err)

    stop = threading.Event()
    answers_thread = threading.Thread(target=print_answers, args=(args.device_id, args.bootstrap, stop),
                                      daemon=True)
    answers_thread.start()
    log.info("Streaming %s as %s to %s at %g fps.", args.source, args.device_id, args.bootstrap, args.fps)
    questions = Questions()
    if args.instruction:
        questions.ask(args.instruction)
    if schedule:
        log.info("%d questions scheduled from %s", len(schedule), args.questions_file)
    if not args.no_input:
        threading.Thread(target=read_stdin, args=(questions, stop), daemon=True).start()
    console.info("%s is streaming%s. Log: %s", args.device_id,
                 "" if args.no_input else "; type a question and press Enter ('quit' stops)", log_file)

    frame_id = 0
    frames_this_pass = 0  # frames read since the video (re)started
    interval = 1.0 / args.fps
    next_frame_at = time.monotonic()
    try:
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                if is_file and args.loop and frames_this_pass > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart the video
                    frames_this_pass = 0
                    continue
                log.info("End of video" if is_file else "The camera returned no frame")
                break
            frames_this_pass += 1
            capture_ts_ms = int(time.time() * 1000)

            while schedule and frame_id >= schedule[0][0]:
                questions.ask(schedule.pop(0)[1])
                if not schedule:
                    log.info("All scheduled questions have been asked")

            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            instruction_id, instruction_text = questions.current()
            message = {
                "device_id": args.device_id,
                "frame_id": frame_id,
                "capture_ts_ms": capture_ts_ms,
                "instruction_id": instruction_id,
                "instruction": instruction_text,
                "width": args.width,
                "height": args.height,
                "image_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
            }
            producer.produce(FRAMES_TOPIC, key=args.device_id, value=json.dumps(message), on_delivery=on_delivery)
            producer.poll(0)  # serve delivery callbacks
            stats.frame_sent(len(jpeg))
            frame_id += 1

            if args.preview:
                cv2.imshow(f"device {args.device_id}", frame)
                cv2.waitKey(1)
            stats.log_if_due()

            # Pace to the target FPS. When behind schedule, continue from now instead of bursting.
            next_frame_at += interval
            delay = next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_at = time.monotonic()
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        stop.set()
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()
        undelivered = producer.flush(10)
        answers_thread.join(timeout=5)
        log.info("Stopped after %d frames (%d undelivered)", frame_id, undelivered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
