# VLM-Distributed-System-Prototype

**Edge VLM pipeline: ask a camera questions.** Coursework prototype for the *Cloud Computing & Big Data* course
(VILNIUS TECH). A webcam points at a desk or a room, and you type free-form questions about what it sees ("What
objects are on the desk?", "What did the person just do?"). Camera devices stream frames and questions to an edge
server. There, a Spark Structured Streaming job reduces the stream to **keyframes** and keeps a **visual memory** of
the last 30 seconds per device. Each question goes, together with frames from that memory, to an open-source
vision-language model (**Qwen2.5-VL-3B-Instruct**) on the GPU. **Nothing reaches the model unless you ask.**

**Status (October 2026): Phase 1 is complete, and the edge server also runs in Docker containers.** The whole system
runs natively on one Windows PC: Kafka, MongoDB, Spark and the VLM service in WSL2 Ubuntu, the camera device on
Windows. It has been tested with the LIBERO robot videos and with a USB webcam (Logitech C930e); the measured results
are in section 6. With Docker Compose the same edge server starts with one command (section 4). Deploying it
somewhere other than the development PC is the next step.

```
 Windows                              │ WSL2 Ubuntu
                                      │
 device/device.py ── frames ──────────┼──▶ Kafka topic frames (4 partitions)
   webcam or video file               │      │
   questions: typed or from a file    │      ▼
                                      │   processor/keyframe_job.py  (Spark Structured Streaming)
                                      │      executors: decode + dHash every frame, downscale selected frames
                                      │      driver:    keyframes, visual memory per device ──▶ MongoDB: batch_stats, keyframes
                                      │      │ one bundle per question: current frame + 8 memory frames
                                      │      ▼
                                      │   Kafka topic queries (4 partitions)
                                      │      │
                                      │      ▼
                                      │   vlm_service/service.py  (RTX 3060, Qwen2.5-VL-3B-Instruct)
                                      │      │                                         ──▶ MongoDB: answers
 device.py prints its own answers ◀───┼── Kafka topic answers (1 partition)
 tools/watch_answers.py shows all ◀───┤
                                      │   Windows reaches Kafka and MongoDB at 127.0.0.1:9092 and 127.0.0.1:27017
```

**How it works**

1. **Keyframes.** Spark executors decode every frame and compute a 64-bit difference hash (dHash) in parallel. The
   driver then walks through each device's frames in `frame_id` order. A frame becomes a keyframe if it is:

   | Reason | Condition |
   |---|---|
   | `first` | the first frame seen from that device |
   | `change` | Hamming distance to the last keyframe's hash ≥ `CHANGE_THRESHOLD` (default 6 of 64 bits) |
   | `heartbeat` | more than `HEARTBEAT_S` seconds (default 5) since the device's last keyframe |

   A still scene gives one keyframe every 5 s; movement gives a keyframe on almost every frame. Keyframe metadata and
   per-batch statistics go to MongoDB (never images).
2. **Visual memory.** For each device, the driver keeps up to `WINDOW_STORE_MAX` keyframes (default 32) from the
   last `WINDOW_SECONDS` (default 30), downscaled on the executors to at most `WINDOW_IMAGE_MAX_SIDE` pixels on the
   longest side (default 448). When the memory is full, it drops the frame that is closest in time to its
   predecessor, so even a long movement leaves frames spread over the whole 30 s. The images live only in the
   driver's memory.
3. **Questions.** Every frame carries the device's current question. When the question changes, the processor
   publishes **one** bundle to the `queries` topic: the downscaled current frame plus `WINDOW_MAX_FRAMES` (default 8)
   memory frames spread evenly in time, always including the newest one. Without questions nothing is published and
   the GPU stays idle.
4. **Answers.** The VLM service answers every bundle in order, with the newest `CONTEXT_FRAMES` (default 8) frames of
   the bundle plus the current frame. It publishes the answer to the `answers` topic, where the asking device prints
   it, and stores it in MongoDB with timestamps for every stage and the time span the images covered.

---

## Prerequisites

- **Windows 10/11 with WSL2 and Ubuntu** (`wsl --install -d Ubuntu`; `wsl -l -v` must show version 2). Tested with
  Windows 11, WSL 3.0.1 and Ubuntu 26.04.1 LTS.
- **An NVIDIA GPU with at least 10 GB of VRAM** and a current Windows driver; `nvidia-smi` must work inside Ubuntu.
  Tested with an RTX 3060 (12 GB): the model needs about 7.5 GB.
- **Disk space:** about 15 GB inside WSL2 (PyTorch with CUDA ~5 GB, model weights ~7 GB). WSL2 keeps Ubuntu's
  disk on C: by default; if C: is short of space, move it first: close all WSL windows, then in PowerShell
  `wsl --shutdown` and `wsl --manage Ubuntu --move D:\WSL\Ubuntu`.
- **Python 3.11+** on Windows (tested with 3.12) and in Ubuntu (Ubuntu 26.04 ships 3.14).
- Inside Ubuntu: **Java 17**, **Kafka 4.3.1** and **MongoDB 8.0** (installed below).
- A webcam (optional: video files work as well). Tested with a Logitech C930e.
- Internet access for the installs and for the first start of the processor (Kafka connector) and of the VLM
  service (model download).

Where things run: Kafka, MongoDB, Spark and the VLM service run **inside WSL2**, because Kafka and Spark are
unreliable natively on Windows and WSL2 has GPU access. `device.py` runs **on Windows**, because WSL2 cannot access
the webcam. Windows reaches the WSL2 services at `127.0.0.1` through WSL2's localhost forwarding.

The repository lives in the WSL2 Linux filesystem (`~/edge-vlm`). Windows sees it as
`\\wsl.localhost\Ubuntu\home\<wsl-user>\edge-vlm`.

---

## 1. One-time setup in Ubuntu (WSL2)

Run these in an Ubuntu terminal.

**Java 17 and Python virtual environments**

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless python3-venv
java -version            # openjdk version "17..."
```

**Kafka 4.3.1** (binary release in `~/kafka_2.13-4.3.1`, linked as `~/kafka`, with a SHA-512 check)

```bash
cd ~ \
 && curl -fLO https://archive.apache.org/dist/kafka/4.3.1/kafka_2.13-4.3.1.tgz \
 && curl -fLO https://archive.apache.org/dist/kafka/4.3.1/kafka_2.13-4.3.1.tgz.sha512 \
 && [ "$(cut -d: -f2- kafka_2.13-4.3.1.tgz.sha512 | tr -d ' \r\n' | tr A-F a-f)" = "$(sha512sum kafka_2.13-4.3.1.tgz | cut -d' ' -f1)" ] \
 && tar -xzf kafka_2.13-4.3.1.tgz && rm kafka_2.13-4.3.1.tgz kafka_2.13-4.3.1.tgz.sha512 \
 && ln -sfn ~/kafka_2.13-4.3.1 ~/kafka && echo "Kafka OK"
```

**MongoDB Community 8.0** from MongoDB's official apt repository. MongoDB publishes no server packages for
Ubuntu 26.04 yet, so the Ubuntu 24.04 (`noble`) repository is used; all of its dependencies exist on 26.04.
On Ubuntu 24.04 itself, the same commands apply unchanged.

```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-8.0.gpg
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org
sudo systemctl enable --now mongod && systemctl is-active mongod     # active
```

`mongod` then runs as a systemd service on `127.0.0.1:27017` whenever WSL2 is running. This needs
systemd, which is the default for new Ubuntu installs: `ps -p 1 -o comm=` prints `systemd`.
Without systemd, either enable it (add `[boot]` and `systemd=true` to `/etc/wsl.conf`, then run
`wsl --shutdown` in Windows) or start MongoDB by hand in each WSL2 session:
`sudo -u mongodb mongod --config /etc/mongod.conf --fork`.

**Code and Python environments**

```bash
git clone https://github.com/rojuspaukste/VLM-Distributed-System-Prototype.git ~/edge-vlm
cd ~/edge-vlm
python3 -m venv ~/.venvs/edge-vlm-processor
~/.venvs/edge-vlm-processor/bin/pip install -r processor/requirements.txt
python3 -m venv ~/.venvs/edge-vlm-vlm
~/.venvs/edge-vlm-vlm/bin/pip install -r vlm_service/requirements.txt      # PyTorch with CUDA: ~5 GB
python3 -m venv ~/.venvs/edge-vlm-tools
~/.venvs/edge-vlm-tools/bin/pip install -r tools/requirements.txt
cp .env.example .env     # optional: every setting has a default
```

Check the GPU and download the model once (about 7 GB into `~/.cache/huggingface`; otherwise the VLM service
downloads it on its first start). Qwen2.5-VL-3B-Instruct is not gated, so no Hugging Face token is needed; its
weights are published under the Qwen Research License (non-commercial use).

```bash
~/.venvs/edge-vlm-vlm/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
~/.venvs/edge-vlm-vlm/bin/hf download Qwen/Qwen2.5-VL-3B-Instruct
```

## 2. One-time setup on Windows

In PowerShell (replace `<wsl-user>` with your Ubuntu user name):

```powershell
$repo = "\\wsl.localhost\Ubuntu\home\<wsl-user>\edge-vlm"
python -m venv "$env:USERPROFILE\.venvs\edge-vlm-device"
& "$env:USERPROFILE\.venvs\edge-vlm-device\Scripts\python.exe" -m pip install -r "$repo\device\requirements.txt"
python -m venv "$env:USERPROFILE\.venvs\edge-vlm-tools"
& "$env:USERPROFILE\.venvs\edge-vlm-tools\Scripts\python.exe" -m pip install -r "$repo\tools\requirements.txt"
```

There is **one copy of the code**: the repository in WSL2. Windows runs the scripts straight from the WSL
share (`$repo`). Only the Windows virtual environments live on the Windows drive
(`%USERPROFILE%\.venvs\edge-vlm-device` and `...\edge-vlm-tools`), because Python runs slowly from the WSL
share. Don't clone the repository a second time on Windows (e.g. into OneDrive): edits made in one copy would
not reach the other. The PowerShell commands below assume these variables. Calling the environment's
`python.exe` directly avoids PowerShell's script execution policy, which often blocks `Activate.ps1`.

```powershell
$repo = "\\wsl.localhost\Ubuntu\home\<wsl-user>\edge-vlm"
$devicePy = "$env:USERPROFILE\.venvs\edge-vlm-device\Scripts\python.exe"
$toolsPy = "$env:USERPROFILE\.venvs\edge-vlm-tools\Scripts\python.exe"
```

---

## 3. Running the system

Start the components in this order:

| Terminal | Where | Runs |
|---|---|---|
| 1 | Ubuntu | Kafka broker (keep it open) |
| (none) | Ubuntu | MongoDB, as a systemd service |
| 2 | Ubuntu | the processor (Spark job) |
| 3 | Ubuntu | the VLM service (GPU) |
| 4 | Windows | one or more devices; you type questions here and read the answers |
| 5 | Windows or Ubuntu | optional tools: `watch_answers.py`, `peek.py`, reports |

**Terminal 1: Kafka** (Ubuntu)

```bash
bash ~/edge-vlm/infra/kafka_start.sh
```

On the first run the script formats the storage directory (`~/edge-vlm-data/kafka`). Keep this terminal
open: it runs the broker, and it keeps WSL2 running (WSL2 may shut the Ubuntu VM down when no terminal is
open). Ctrl+C stops the broker. If a broker is already running, the script refuses to start a second one.

**First time only: create the topics and test the connection** (any Ubuntu terminal)

```bash
cd ~/edge-vlm && source ~/.venvs/edge-vlm-tools/bin/activate
python infra/create_topics.py     # frames (4 partitions), queries (4), answers (1); existing topics are skipped
python tools/smoke_test.py        # OK: Kafka round trip via 127.0.0.1:9092 took 58 ms
```

`create_topics.py` also deletes the `keyframes` topic of an earlier version if it still exists. Then run the same
test from Windows: `& $toolsPy "$repo\tools\smoke_test.py"`.

**Terminal 2: processor** (Ubuntu)

```bash
cd ~/edge-vlm && source ~/.venvs/edge-vlm-processor/bin/activate
set -a; source .env; set +a      # optional: load the settings from .env
python processor/keyframe_job.py
```

The first start downloads the Kafka connector from Maven Central. Wait for `Streaming query started`. The job
logs one line per micro-batch and writes a `batch_stats` document for each, and one line per question bundle with
the time span its memory frames cover:

```
batch 315: 5 frames in, 1 keyframes out (drop 80.0%), 0 queries out, first=0 change=1 heartbeat=0, 210 ms
Query robot-1/5 "What did the robot just do?": current frame 500 + 8 of 31 memory frames over the last 30.0 s, 206 KB
```

**Terminal 3: VLM service** (Ubuntu)

```bash
cd ~/edge-vlm && source ~/.venvs/edge-vlm-vlm/bin/activate
set -a; source .env; set +a      # optional
python vlm_service/service.py
```

Loading takes about 10 s (longer the first time, while the model downloads). Wait for `waiting for questions`.
The service logs the peak VRAM after loading and one line per answer:

```
Model loaded: peak VRAM 7162 MiB allocated, 7236 MiB reserved by PyTorch
Answer robot-1/5 (8+1 frames over 30.0 s, inference 1.8 s, end-to-end 2.4 s, peak VRAM 7511 MiB) "What did the robot just do?" -> The robot placed the coffee cup on the table.
```

**Terminal 4: device** (Windows)

With a webcam (`--preview` shows the frames in a window; use `--source 1` if index 0 is the wrong camera):

```powershell
& $devicePy "$repo\device\device.py" --preview
```

The device starts **without a question**. Type a question and press Enter; every line is a new question, even if
it repeats an earlier one. The answer appears in the same terminal a few seconds later, with its latency and how
far back the model's images reached. Type `quit` to stop. The terminal shows only the questions and answers;
everything else (frame rate every 5 s, warnings, Kafka client messages) goes to `logs/<device-id>.log` in the
repository (`--log-file` changes it).

With a video file, first download the sample robot videos once. This fetches 4 files of the LIBERO dataset
into `data/` (about 145 MB); any other MP4 file works too. A `--questions-file` asks questions automatically
(see section 6.2):

```powershell
& $toolsPy "$repo\tools\fetch_sample_videos.py"
& $devicePy "$repo\device\device.py" --source "$repo\data\libero-1.mp4" --no-input --questions-file "$repo\eval\libero_questions.tsv"
```

Terminal output of that run (abridged):

```
robot-1 is streaming. Log: \\wsl.localhost\Ubuntu\home\<wsl-user>\edge-vlm\logs\robot-1.log
Q1: What objects are on the table?
A1: There are three mugs and two saucers on the table.  [2.7 s, 8 earlier frames from the last 10 s + current]
Q5: What did the robot just do?
A5: The robot placed the coffee cup on the table.  [2.4 s, 8 earlier frames from the last 30 s + current]
```

**Terminal 5: inspect** (Windows; the same commands work in Ubuntu with the tools environment)

```powershell
& $toolsPy "$repo\tools\watch_answers.py"                 # every answer, live (--device-id, --from-beginning)
& $toolsPy "$repo\tools\peek.py" --topic frames -n 3      # newest frames (images replaced by their length)
& $toolsPy "$repo\tools\peek.py" --topic queries -n 1     # newest question bundle and its size
```

MongoDB can be inspected from Ubuntu with `mongosh`:

```bash
mongosh edge_vlm --quiet --eval 'db.answers.find({}, {_id: 0, question: 1, answer: 1, context_span_s: 1, latency_ms: 1}).sort({_id: -1}).limit(3)'
mongosh edge_vlm --quiet --eval 'db.batch_stats.find().sort({_id: -1}).limit(1)'
```

**What you should see**

- **No question:** the processor selects keyframes (`first`, then `change` and `heartbeat`) with `0 queries out`;
  the `queries` topic stays empty and the VLM service runs no inference.
- **A question:** one `Query` line in the processor, one `Answer` line in the VLM service, and the answer in the
  device terminal 1–4 s after you press Enter (measured: 1.0–2.8 s on the LIBERO video, 2.0–3.6 s with the webcam).
- **An action, then a question about it:** the bundle contains frames of the action (`change` keyframes), so the
  model can describe what happened, not only the current frame.

**Questions about actions.** Ask in the past tense: "What did the person just do?". For "What is the person
doing?" the model describes the current frame, so an action that has just ended is missed even when it is in the
earlier frames. The defaults were tuned with the webcam pointed at a room:
- `CHANGE_THRESHOLD=6`: a still room stayed within 0–3 bits of the 64-bit dHash, while a hand wave peaked at about
  10 bits, so a threshold of 10 missed most gestures.
- `CONTEXT_FRAMES=8`: during a movement almost every frame becomes a keyframe. With fewer frames the model saw too
  little of a gesture: for the same wave it answered "waving their hand in front of the camera" with 8 frames and
  "lowered their hand after raising it earlier" with 4.

---

## 4. Run with Docker

The edge server (Kafka, MongoDB, the processor and the VLM service) also runs as Docker Compose services on the same
PC. The devices and the tools stay on Windows, unchanged: they connect to the same `127.0.0.1:9092` and
`127.0.0.1:27017` as before. Inside the containers only the addresses differ (`KAFKA_BOOTSTRAP=kafka:19092`,
`MONGO_URI=mongodb://mongo:27017`); the code is the same.

| Service | Image | Role |
|---|---|---|
| `kafka` | `apache/kafka:4.3.1` | single-node KRaft broker. Listener `INTERNAL` (`kafka:19092`) for the containers, `EXTERNAL` (port 9092, advertised as `127.0.0.1:9092`) for devices and tools |
| `topic-init` | `apache/kafka:4.3.1` | one-shot: creates `frames` (4 partitions), `queries` (4) and `answers` (1), with one-hour retention on `frames` and `queries`, then exits |
| `mongo` | `mongo:8.0.32` | MongoDB on port 27017, data in the volume `mongo-data` |
| `processor` | `edge-vlm/processor:1.0` (`processor/Dockerfile`) | Python 3.14, OpenJDK 17 and PySpark 4.1.3. The Kafka connector is downloaded at build time, so startup needs no download. Spark checkpoint in the volume `checkpoint` |
| `vlm` | `edge-vlm/vlm-service:1.0` (`vlm_service/Dockerfile`) | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime`, the same PyTorch build as natively, with the GPU. Mounts `~/.cache/huggingface` read-only, so the model is not downloaded again |

Every service has a health check, and they start in order: Kafka healthy → `topic-init` done (and MongoDB healthy) →
processor and VLM service. The processor counts as healthy once Spark's UI lists its streaming query as active; the
VLM service once it has joined its Kafka consumer group, which it does after loading the model and a warm-up answer.

**One-time setup**

- Docker Desktop with the WSL 2 engine. In *Settings → Resources → WSL integration*, switch on *Ubuntu*; `docker`
  then works in Ubuntu. If it reports `permission denied ... docker.sock`, open a new Ubuntu terminal: Docker
  Desktop has just added your user to the `docker` group.
- Disk space: about 28 GB (15 GB of images plus 12.5 GB of build cache, which `docker builder prune` frees).
  Docker Desktop keeps them on C: unless *Settings → Resources → Advanced → Disk image location* points
  elsewhere; change it before the first build if C: is short of space.
- The model in `~/.cache/huggingface` (section 1, `hf download`).
- The GPU must be visible to containers: `docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi`.

**Start.** The containers use the native services' ports, so first stop the native Kafka (Ctrl+C in its terminal),
the processor, the VLM service and MongoDB:

```bash
sudo systemctl stop mongod
cd ~/edge-vlm
docker compose up -d --build --wait     # the first build downloads about 5 GB
docker compose ps                       # every service healthy; topic-init has exited (0)
```

Then use the system as in section 3: the device, `watch_answers.py`, `peek.py` and the reports work unchanged, and
`& $toolsPy "$repo\tools\smoke_test.py"` tests the connection from Windows. Logs and MongoDB:

```bash
docker compose logs -f processor vlm
docker compose exec mongo mongosh edge_vlm --quiet --eval 'db.answers.find({}, {_id: 0, question: 1, answer: 1}).sort({_id: -1}).limit(3)'
```

**Settings and experiment runs.** The services take the environment variables of section 5 from the shell, and
`docker compose up` recreates the containers whose settings changed:

```bash
RUN_ID=P3-docker-ctx8 CONTEXT_FRAMES=8 docker compose up -d --wait
```

The Spark checkpoint survives restarts in the `checkpoint` volume. Before a new experiment run, delete it as in
section 6.1: `docker compose rm -sf processor && docker volume rm edge-vlm_checkpoint`.

**Stop**

```bash
docker compose down             # removes the containers; the volumes keep Kafka's and MongoDB's data
docker compose down -v          # also deletes the volumes
sudo systemctl start mongod     # back to the native setup (and bash infra/kafka_start.sh)
```

The containerized MongoDB starts empty: the native runs stay in the native MongoDB.

**Native vs. Docker (measured).** The same questions file (section 6.2) with `CONTEXT_FRAMES=8`, natively (run
`E2-ctx8`) and in containers (run `P3-docker-ctx8`), on the same PC:

| | native | Docker |
|---|---|---|
| end-to-end p50 / p95 | 2.68 / 2.80 s | 2.71 / 2.87 s |
| queue p50 | 0.59 s | 0.48 s |
| inference p50 / p95 | 2.09 / 2.18 s | 2.21 / 2.40 s |
| peak VRAM | 7511 MiB | 7511 MiB |
| images reach back (mean) | 24.8 s | 24.6 s |

The containers add no measurable latency: the differences of about 0.1 s are within the variation of six questions
(where a question falls within the 500 ms micro-batch interval alone moves its queue time by up to 0.5 s). The GPU is
used directly, so inference runs as fast as natively.

- **Startup** (`docker compose up -d --wait`, images built, volumes kept): 22–28 s until every service is healthy.
  MongoDB is ready after 2.5 s, Kafka after 5–9 s, the topics exist after 10–14 s, the processor's streaming query
  runs after 20–23 s, and the VLM service has loaded the model and answered its warm-up question after 22–28 s.
- **Images:** `edge-vlm/vlm-service:1.0` 10 GB (9.8 GB of it is the PyTorch CUDA base image),
  `edge-vlm/processor:1.0` 2.2 GB, `mongo:8.0.32` 1.3 GB, `apache/kafka:4.3.1` 0.7 GB.
- **Resources during the run** (`docker stats`, one device at 10 fps): the processor used 1.7 CPU cores on
  average (up to 7 during micro-batches) and up to 2.8 GB of RAM; Kafka 0.3 cores and 0.8 GB; MongoDB under 0.1
  core and 0.4 GB; the VLM service 0.2 cores and 2.0 GB of RAM, with the model on the GPU.
- Exports: `results/part3/` (`native.csv`, `docker.csv`, `startup.txt`, `images.txt`, `docker_stats.txt`).

---

## 5. Configuration

Settings are environment variables for the services (template with defaults in `.env.example`) and command-line
options for the scripts.

**Processor** (`processor/keyframe_job.py`):

| Variable | Default | Meaning |
|---|---|---|
| `KAFKA_BOOTSTRAP` | `127.0.0.1:9092` | Kafka address |
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | MongoDB address (database `edge_vlm`) |
| `SPARK_MASTER` | `local[*]` | Spark master; `local[N]` uses N cores (the scaling knob) |
| `CHANGE_THRESHOLD` | `6` | dHash Hamming distance (of 64 bits) that counts as a scene change |
| `HEARTBEAT_S` | `5` | select a keyframe at least this often per device |
| `WINDOW_SECONDS` | `30` | visual memory: keyframes of the last this many seconds |
| `WINDOW_STORE_MAX` | `32` | visual memory: at most this many keyframes per device |
| `WINDOW_MAX_FRAMES` | `8` | memory frames per question bundle, spread evenly in time, newest included |
| `WINDOW_IMAGE_MAX_SIDE` | `448` | longest side (pixels) of the images in the visual memory and the bundles |
| `TRIGGER_MS` | `500` | micro-batch trigger interval |
| `MAX_OFFSETS_PER_TRIGGER` | `500` | maximum Kafka records per micro-batch |
| `CHECKPOINT_DIR` | `~/edge-vlm-data/checkpoint` | Spark checkpoint (Linux filesystem); delete it between experiment runs |
| `RUN_ID` | start time, e.g. `20261010-153000` | tag stored in every MongoDB document |

**VLM service** (`vlm_service/service.py`):

| Variable | Default | Meaning |
|---|---|---|
| `KAFKA_BOOTSTRAP`, `MONGO_URI` | as above | addresses |
| `MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Hugging Face model, loaded in bfloat16 on the GPU |
| `MAX_PIXELS` | `448*448` | upper bound on the pixels of each image given to the model (a number or a product) |
| `CONTEXT_FRAMES` | `8` | frames of the bundle given to the model (the newest ones) plus the current frame; `0` = current frame only |
| `MAX_NEW_TOKENS` | `128` | answer length limit (greedy decoding) |
| `GROUP_ID` | `vlm-service` | Kafka consumer group; a new group answers every question still in Kafka (see 6.2) |
| `RUN_ID` | start time | tag stored with every answer |
| `HF_TOKEN` | (unset) | only for gated models |

**Device** (`device/device.py`):

| Option | Default | Meaning |
|---|---|---|
| `--source` | `0` | webcam index, or path to a video file |
| `--loop` | off | restart the video file when it ends |
| `--device-id` | `robot-1` | device id, also the Kafka message key |
| `--fps` | `10` | frames per second to send |
| `--width`, `--height` | `640`, `480` | frame size after resizing |
| `--jpeg-quality` | `80` | JPEG quality (1–100) |
| `--bootstrap` | `127.0.0.1:9092` | Kafka address (or `$KAFKA_BOOTSTRAP`) |
| `--instruction` | none | first question, asked at startup |
| `--questions-file` | none | file of `<seconds><TAB><question>` lines, asked automatically |
| `--preview` | off | show the frames in a window |
| `--no-input` | off | don't read questions from stdin (scripted and multi-device runs) |
| `--log-file` | `logs/<device-id>.log` | where everything except the questions and answers is logged |

**Tools**: the Kafka tools take `--bootstrap` (default `127.0.0.1:9092`), the reports take `--mongo-uri`. Run any
tool with `--help` for its options.

---

## 6. Experiments

Every run gets its own `RUN_ID`, and every MongoDB document carries it, so runs can be compared. Kafka and MongoDB
keep running between runs. The reports read MongoDB and work from Windows or Ubuntu; `--csv` writes the summary
for plotting (the `results/` folder is git-ignored). All results below were measured on the development PC (AMD
Ryzen 7 5700X3D with 8 cores / 16 threads, RTX 3060 12 GB) with the default settings, using the LIBERO sample videos.

### 6.1 Scaling study: Spark processing (coursework Part 2)

Varies one factor per run, for example the Spark cores (`SPARK_MASTER`) or the number of devices. Devices started
with `--no-input` ask no questions, so this measures the processing alone.

1. **Stop the processor** (Ctrl+C) and **delete the checkpoint**. The checkpoint stores Kafka offsets; if
   it is kept, the next run starts with the old run's leftover frames.
   ```bash
   rm -rf ~/edge-vlm-data/checkpoint
   ```
2. **Start the processor** with the run's settings (terminal 2) and wait for `Streaming query started`:
   ```bash
   RUN_ID=E1-local4-dev4 SPARK_MASTER=local[4] python processor/keyframe_job.py
   ```
3. **Start N devices** on Windows, each in its own window. With more than 4 devices, reuse the videos,
   for example `libero-$((($_ - 1) % 4) + 1).mp4`:
   ```powershell
   $devices = 1..4 | ForEach-Object {
       Start-Process -PassThru -FilePath $devicePy -ArgumentList "$repo\device\device.py", "--source", "$repo\data\libero-$_.mp4", "--loop", "--no-input", "--device-id", "robot-$_"
   }
   ```
4. Let it run for a few minutes (the first batch includes JVM warm-up), then **stop the devices**
   (`$devices | Stop-Process`) and the processor (Ctrl+C).
5. Repeat steps 1–4 for the other settings, e.g. `SPARK_MASTER=local[1]` with `RUN_ID=E1-local1-dev4`. Use a new
   `RUN_ID` for every run: documents of a reused `RUN_ID` would be mixed into one summary.
6. **Report** the runs side by side:
   ```powershell
   & $toolsPy "$repo\tools\processing_report.py" --run-id E1-local1-dev4 E1-local4-dev4 --csv results\scaling.csv
   ```

The report shows, per run: throughput (`frames_in_per_s`, `keyframes_out_per_s`), `queries_out`, `drop_ratio` and
the counts per reason, batch duration (mean, p95), ingest delay (processing time minus capture time; mean, p50,
p95 from the per-batch means weighted by frame count, plus the maximum), `keyframe_delay`
(`selected_ts_ms − capture_ts_ms` of each keyframe) and the config used.

Results: 4 LIBERO devices at 10 fps each, 90 s per run (runs `E1-local1-dev4` and `E1-local4-dev4`):

| metric | `local[1]` | `local[4]` |
|---|---|---|
| frames in / s | 39.2 | 39.9 |
| keyframes (drop ratio) | 420 (0.883) | 419 (0.883) |
| batch duration mean / p95 (ms) | 521 / 593 | 260 / 300 |
| ingest delay p50 / p95 (ms) | 716 / 916 | 495 / 610 |
| keyframe delay p50 / p95 (ms) | 748 / 1076 | 509 / 718 |

Four cores halve the batch duration and cut the delays by about a third. With one core, a batch (~520 ms) takes
longer than the 500 ms trigger interval, so Spark logs `Current batch is falling behind` and the delays grow. The
selection itself does not depend on the cores: both runs keep the same 11.7% of the frames.

Notes for interpreting the results:
- **Parallelism.** Spark normally runs one task per Kafka partition (`frames` has 4). Because devices are
  assigned to partitions by hashing their id, several devices can share a partition, so the job asks Spark to
  split the partitions into as many tasks as there are cores (`minPartitions`). This keeps every core busy.
- **Images.** Only metadata of all frames is collected to the driver; the images of the keyframes and question
  frames are downscaled on the executors and collected separately (a few per batch).
- **Robot videos move a lot.** The robot arm is almost always moving, so 11.7% of the frames are keyframes. In a
  10-minute webcam session in a room (mostly still, with gestures and objects shown to the camera) it was 4.1%.
- **Device side.** Watch the devices' frame rates too (every 5 s in `logs/robot-N.log`): with many devices, JPEG
  encoding on Windows can become the bottleneck instead of Spark. In the runs above, all four devices held 10.0 fps.

### 6.2 Evaluation: answers and latency with and without visual memory

Compares `CONTEXT_FRAMES` values (0 = current frame only) on the same video and the same questions. A questions file
holds `<seconds><TAB><question>` lines (`#` starts a comment). The time is counted in frames sent
(seconds × fps), so every run asks each question at exactly the same video frame. `eval/libero_questions.tsv` asks
six questions, 10 s apart, about `libero-1.mp4`.

1. **Start the processor** (terminal 2) with any `RUN_ID`, e.g. `RUN_ID=E2-processor`. It can keep running for all
   evaluation runs: a restarted device starts with frame 0, which the processor treats as a new device session
   (fresh visual memory).
2. **Start the VLM service** for the first setting (terminal 3) and wait for `waiting for questions`:
   ```bash
   RUN_ID=E2-ctx0 CONTEXT_FRAMES=0 python vlm_service/service.py
   ```
3. **Run the device** with the questions file (terminal 4) and wait until it has printed the answer to the last
   question (`A6`, at about 60 s), then stop it with Ctrl+C:
   ```powershell
   & $devicePy "$repo\device\device.py" --source "$repo\data\libero-1.mp4" --no-input --questions-file "$repo\eval\libero_questions.tsv"
   ```
4. **Stop the VLM service** (Ctrl+C) and repeat steps 2–3 with the other settings: `RUN_ID=E2-ctx4
   CONTEXT_FRAMES=4` and `RUN_ID=E2-ctx8 CONTEXT_FRAMES=8`.
5. **Report** latency, time span and peak VRAM per run and per number of context frames used, and compare the
   answers:
   ```powershell
   & $toolsPy "$repo\tools\latency_report.py" --run-id E2-ctx0 E2-ctx4 E2-ctx8 --csv results\latency.csv
   ```
   ```bash
   mongosh edge_vlm --quiet --eval 'db.answers.find({run_id: /^E2-ctx/}, {_id: 0, run_id: 1, query_id: 1, question: 1, answer: 1}).sort({query_id: 1, run_id: 1})'
   ```

`latency_report.py` shows the count and the mean, p50 and p95 of `end_to_end` (answer time minus the capture time
of the frame that asked), `queue` (inference start minus capture time: Kafka, Spark micro-batch and bundling) and
`inference` (image decoding, preprocessing and generation); `context_span_s` (how far back the model's images
reached: mean and minimum); and the peak VRAM.

Results (runs `E2-ctx0`, `E2-ctx4`, `E2-ctx8`; 6 questions each):

| `CONTEXT_FRAMES` | end-to-end p50 | queue p50 | inference p50 | images reach back (mean) | peak VRAM |
|---|---|---|---|---|---|
| 0 | 1.2 s | 0.4 s | 0.8 s | 0 s | 7207 MiB |
| 4 | 1.9 s | 0.4 s | 1.5 s | 11.5 s | 7358 MiB |
| 8 | 2.7 s | 0.6 s | 2.1 s | 24.8 s | 7511 MiB |

Each bundle carries 8 memory frames spread over the last 30 s, so `CONTEXT_FRAMES=4` uses the newer half (about 13 s
back) and `CONTEXT_FRAMES=8` all of it. The first questions reach back less (10 s and 20 s) because the video had
only been running that long. Each extra image adds about 0.15 s of inference and 40 MiB of VRAM.

| Question | `CONTEXT_FRAMES=0` | `CONTEXT_FRAMES=4` | `CONTEXT_FRAMES=8` |
|---|---|---|---|
| What is the robot arm doing? | The robot arm is pouring coffee into a red mug. | The robot arm is moving towards the red mug on the table. | The robot arm is moving towards the red mug on the table. |
| What did the robot just do? | The robot placed the coffee maker on the counter. | The robot placed the red mug on the table next to the gray mug. | The robot placed the coffee cup on the table. |
| Describe the scene. | The robot arm is holding a coffee cup on a table. | The robotic arm is positioned above a coffee cup on a table, preparing to interact with it. | The robotic arm is holding a mug with a handle, moving it towards a microwave oven on a kitchen counter. |

Without earlier frames the model guesses the action from a single image (there is no pouring and no coffee maker
being moved); with them it describes the movement.

**Same bundles, different settings.** The `queries` topic keeps every bundle for one hour. A VLM service with a
new consumer group answers all of them again, so settings can also be compared on identical inputs (only one
model fits on a 12 GB GPU, so stop the main service first). For such replays only the `inference` latency is
meaningful:

```bash
RUN_ID=E2-replay-ctx0 CONTEXT_FRAMES=0 GROUP_ID=replay-ctx0 python vlm_service/service.py
```

---

## 7. Troubleshooting

**Windows cannot reach Kafka or MongoDB (timeouts or `Connect ... failed`)**

- **Use `127.0.0.1`, not `localhost`.** WSL2 forwards a port to Windows only for the address family of the
  Linux socket, so IPv4 services are reachable at `127.0.0.1` only. Windows, however, resolves `localhost`
  to `::1` first. For the same reason `infra/kafka_start.sh` starts the broker on IPv4
  (`-Djava.net.preferIPv4Stack=true`) and advertises `127.0.0.1:9092`.
- **Check the forwarding.** `Get-NetTCPConnection -LocalPort 9092,27017 -State Listen` should list
  `wslrelay` on `127.0.0.1` for both ports. If it doesn't, check that Kafka runs (terminal 1) and that
  `systemctl is-active mongod` prints `active`.
- **Keep WSL2 running.** WSL2 may stop the Ubuntu VM when no terminal is open, which also stops MongoDB.
  Keeping the Kafka terminal open prevents this.
- **Test both sides.** `tools/smoke_test.py` from Ubuntu and from Windows tells you which side fails.

**Kafka**

- `Kafka is already running (port 9092 is in use)`: a broker runs in another terminal or in the background. Use
  that one, or stop it first (Ctrl+C in its terminal). Never run two brokers on the same data directory.
- **Kafka storage:** if Kafka refuses to start after a broken first run, delete `~/edge-vlm-data/kafka`
  (this removes all topics), start Kafka again and rerun `infra/create_topics.py`.

**Webcam**

- `Cannot open video source '0'` means no camera at that index. Try `--source 1`.
- Check that Windows sees the camera: `Get-PnpDevice -Class Camera,Image -PresentOnly` must list it. A camera
  that is unplugged or in use by another program (Teams, the Camera app) cannot be opened.
- Check Windows Settings → Privacy & security → Camera → *Let desktop apps access your camera*.

**VLM service**

- `CUDA is not available`: `nvidia-smi` must work inside Ubuntu, and the CUDA build of PyTorch must be installed
  (`vlm_service/requirements.txt` uses PyTorch's CUDA 13.0 packages, which need a recent NVIDIA driver).
- `torch.OutOfMemoryError`: only one model fits on a 12 GB GPU, so don't run two VLM services at once. Lower
  `MAX_PIXELS` or `CONTEXT_FRAMES` if a single service runs out of memory.
- No answers: check that the processor logs `Query` lines and that `peek.py --topic queries` shows bundles. Every
  question is answered in order, so a service that was stopped answers the questions it missed when it starts again.

**Docker**

- `The command 'docker' could not be found in this WSL 2 distro`: Docker Desktop is not running, or its WSL
  integration for Ubuntu is off.
- `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`: the terminal was
  opened before your user joined the `docker` group. Open a new one.
- `ports are not available` or `address already in use` for 9092 or 27017: the native Kafka or MongoDB still runs.
- A service stays `starting` or becomes `unhealthy`: `docker compose logs <service>`. The VLM service needs about
  7.5 GB of free VRAM, and other programs (browsers, game overlays) take some too.

**Kafka connector version**

- The job adds `org.apache.spark:spark-sql-kafka-0-10_2.13:<installed PySpark version>` through
  `spark.jars.packages`, so the connector always matches PySpark exactly. PySpark 4.x is built for
  Scala 2.13 (3.5.x used 2.12).
- If you change the PySpark version, keep the `_2.13` suffix in `processor/keyframe_job.py` in line with
  PySpark's Scala version.
- `Failed to find data source: kafka` means the connector was not loaded. The first start needs internet
  access, and the jars are cached in `~/.ivy2.5.2`.
- `NoSuchMethodError` or `ClassNotFoundException` errors usually mean a version mismatch.

**Other issues**

- **The processor starts with a burst of old frames:** the checkpoint was not deleted (see section 6.1).
  Frames older than Kafka's one-hour retention are skipped (`failOnDataLoss=false`).
- **A question is answered twice after restarting the processor:** the processor keeps no state across restarts,
  so it asks each running device's current question once more.
- **`The value of the provided token to the AccumulatorServer is not correct`** at processor startup: a stray local
  connection to a PySpark-internal port. It is harmless and was seen once in 17 test runs.
- **Java:** `java -version` must show 17; Spark 4 does not run on older Java.
- **MongoDB:** check `systemctl status mongod` and `/var/log/mongodb/mongod.log`.
- **Video files:** the LIBERO videos are AV1-encoded, which the FFmpeg bundled with `opencv-python`
  decodes. Any other MP4 works too.

---

## 8. Data and privacy

- **Images exist only in Kafka and in memory.** `frames` and `queries` keep them for one hour (`retention.ms`, with
  10-minute segments so that old data really is deleted on time). Spark caches each micro-batch in memory only
  (`MEMORY_ONLY`) and avoids shuffles, so frames are never spilled to disk. The visual memory lives in the
  processor's memory, and the processor publishes the bundles itself, straight from memory.
- **MongoDB stores no images:** `batch_stats` and `keyframes` hold metadata, `answers` holds the question, the
  answer, frame ids, timestamps and latencies.
- The downloaded sample videos in `data/`, the device logs in `logs/` and the report exports in `results/` are
  git-ignored.
- **Docker:** the images contain code and libraries only, no data and no model weights. Kafka's data, MongoDB's
  data and the Spark checkpoint live in named volumes; `docker compose down -v` deletes them.

## 9. Repository layout

```
.env.example                  settings template with all defaults (copy to .env)
docker-compose.yml            the edge server in containers: Kafka, topic-init, MongoDB, processor, VLM service
infra/kafka_start.sh          (WSL2) formats storage on first run, starts the single-node KRaft broker
infra/create_topics.py        creates frames / queries / answers (idempotent)
device/device.py              (Windows) webcam or video producer; questions in, answers out
processor/keyframe_job.py     (WSL2) Spark job: keyframes, visual memory, question bundles
processor/Dockerfile          processor image: Python 3.14, Java 17, PySpark and the Kafka connector
vlm_service/service.py        (WSL2, GPU) answers every question, stores the answers
vlm_service/backends.py       VLM backend interface and the Qwen2.5-VL implementation
vlm_service/Dockerfile        VLM service image: PyTorch 2.14.1 with CUDA 13.0
tools/smoke_test.py           Kafka round-trip test
tools/peek.py                 newest messages of a topic, without image data
tools/watch_answers.py        live view of the answers
tools/fetch_sample_videos.py  downloads LIBERO sample videos into data/
tools/processing_report.py    scaling study: processing summary per RUN_ID
tools/run_scaling_study.py    scaling study: runs the processor with N devices per setting
tools/clock_check.py          (Windows) compares the WSL2 and Windows clocks that the latencies rely on
tools/latency_report.py       evaluation: answer latency per RUN_ID and number of context frames
eval/libero_questions.tsv     scheduled questions for the LIBERO sample video
```

Each of `device/`, `processor/`, `vlm_service/` and `tools/` has its own `requirements.txt`;
`infra/create_topics.py` uses the tools environment.
