"""Part 2 scaling study: runs the processor with N camera devices for each setting.

Runs in WSL2 with Kafka and MongoDB running; the VLM service should be stopped. For each run it clears the
Spark checkpoint, starts the processor with the run's SPARK_MASTER and RUN_ID, waits for "Streaming query
started", starts N devices on Windows with the LIBERO videos (--loop --no-input; videos are reused when
N > 4), lets them stream, stops the devices and the processor, and records each device's actual frame rate
from its log. Summarise the runs afterwards with processing_report.py.

    python tools/run_scaling_study.py --seconds 60 --run E1-local1 "local[1]" 4 --run E1-local4 "local[4]" 4

Logs go to logs/study/<run_id>/; the frame rates to --fps-csv (default results/study/devices_fps.csv).
"""
import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pymongo import MongoClient

REPO = Path(__file__).resolve().parent.parent
PROCESSOR_PYTHON = Path.home() / ".venvs/edge-vlm-processor/bin/python"
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", Path.home() / "edge-vlm-data/checkpoint"))
DEVICE_PYTHON = r"$env:USERPROFILE\.venvs\edge-vlm-device\Scripts\python.exe"  # expanded by PowerShell
VIDEOS = 4  # data/libero-1.mp4 ... libero-4.mp4 (tools/fetch_sample_videos.py)
FPS_LINE = re.compile(r"sent (\d+) frames in ([\d.]+) s \(([\d.]+) fps\).*total (\d+)")


def windows_path(path: Path) -> str:
    """The Windows (\\\\wsl.localhost\\...) path of a WSL2 path."""
    return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def powershell(script: str, script_file: Path) -> str:
    """Runs a PowerShell script on Windows and returns its output. The script goes through a file, because
    quotes inside a -Command argument get mangled on the way from WSL2 to Windows."""
    script_file.write_text(script)
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                           windows_path(script_file)], capture_output=True, text=True, cwd="/mnt/c").stdout


def start_devices(count: int, log_dir: Path) -> None:
    """Starts `count` hidden device processes on Windows, each with its own log file in `log_dir`."""
    repo = windows_path(REPO)
    lines = [f'$py = "{DEVICE_PYTHON}"']
    for i in range(1, count + 1):
        video = rf"{repo}\data\libero-{(i - 1) % VIDEOS + 1}.mp4"
        log_file = rf"{windows_path(log_dir)}\robot-{i}.log"
        lines.append(f'Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList '
                     f'"{repo}\\device\\device.py","--source","{video}","--loop","--no-input",'
                     f'"--device-id","robot-{i}","--log-file","{log_file}"')
    powershell("\n".join(lines), log_dir / "start_devices.ps1")


def stop_devices(log_dir: Path) -> None:
    """Stops the device processes of this run (found by their log directory in the command line)."""
    powershell(f"Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
               f"Where-Object {{ $_.CommandLine -like \"*{windows_path(log_dir)}\\*\" }} | "
               f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}",
               log_dir / "stop_devices.ps1")


def start_processor(run_id: str, spark_master: str, log_file: Path) -> subprocess.Popen:
    """Starts the processor and waits until its streaming query runs."""
    env = dict(os.environ, RUN_ID=run_id, SPARK_MASTER=spark_master)
    process = subprocess.Popen([str(PROCESSOR_PYTHON), str(REPO / "processor/keyframe_job.py")], env=env,
                               stdout=log_file.open("w"), stderr=subprocess.STDOUT, cwd=REPO)
    deadline = time.monotonic() + 180
    while "Streaming query started" not in log_file.read_text():
        if process.poll() is not None or time.monotonic() > deadline:
            raise SystemExit(f"The processor did not start; see {log_file}")
        time.sleep(1)
    return process


def stop_processor(process: subprocess.Popen) -> None:
    process.send_signal(signal.SIGINT)  # the job stops its streaming query cleanly on Ctrl+C
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()


def device_fps(run_id: str, spark_master: str, devices: int, log_dir: Path) -> list[dict]:
    """Reads each device's "sent N frames in X s (Y fps)" lines: one row per device."""
    rows = []
    for i in range(1, devices + 1):
        log_file = log_dir / f"robot-{i}.log"
        matches = FPS_LINE.findall(log_file.read_text(encoding="utf-8")) if log_file.exists() else []
        fps = [float(m[2]) for m in matches]
        rows.append({"run_id": run_id, "spark_master": spark_master, "devices": devices, "device_id": f"robot-{i}",
                     "video": f"libero-{(i - 1) % VIDEOS + 1}.mp4",
                     "mean_fps": round(sum(fps) / len(fps), 2) if fps else None,
                     "min_fps": min(fps) if fps else None,
                     "frames_sent": int(matches[-1][3]) if matches else 0})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Part 2 scaling study.")
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("RUN_ID", "SPARK_MASTER", "DEVICES"),
                        help='one run, e.g. --run E1-local4 "local[4]" 4 (repeat for more runs)')
    parser.add_argument("--seconds", type=float, default=60, help="how long the devices stream per run (default 60)")
    parser.add_argument("--fps-csv", type=Path, default=REPO / "results/study/devices_fps.csv",
                        help="where to write the devices' frame rates")
    args = parser.parse_args()

    db = MongoClient(os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"), serverSelectionTimeoutMS=5000)["edge_vlm"]
    for run_id, _, _ in args.run:
        if db.batch_stats.count_documents({"run_id": run_id}):
            raise SystemExit(f"Run {run_id} already exists in MongoDB; choose another RUN_ID.")

    fps_rows = []
    for run_id, spark_master, devices in args.run:
        devices = int(devices)
        log_dir = REPO / "logs/study" / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)  # otherwise the run starts with old offsets
        print(f"{run_id}: SPARK_MASTER={spark_master}, {devices} devices, {args.seconds:g} s", flush=True)
        processor = start_processor(run_id, spark_master, log_dir / "processor.log")
        try:
            start_devices(devices, log_dir)
            time.sleep(args.seconds)
        finally:
            stop_devices(log_dir)
            stop_processor(processor)
        rows = device_fps(run_id, spark_master, devices, log_dir)
        fps_rows += rows
        print("  device fps: " + " ".join(f"{r['mean_fps']}" for r in rows), flush=True)

    args.fps_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.fps_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fps_rows[0]))
        writer.writeheader()
        writer.writerows(fps_rows)
    print(f"Wrote {len(fps_rows)} device rows to {args.fps_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
