"""Compare the WSL2 clock with the Windows clock (run on Windows).

Latencies compare capture timestamps from Windows with processing timestamps from WSL2, so they are only
as accurate as the two clocks agree. This calls `wsl date +%s%N` a few times and compares the result with
the Windows time at the midpoint of each call; the call with the shortest round trip is used.

    python tools/clock_check.py
"""
import subprocess
import sys
import time

TRIES = 5
LIMIT_MS = 50


def measure() -> tuple[float, float]:
    """Returns (offset, uncertainty) in ms: WSL2 time minus Windows time, and half the round trip."""
    start = time.time()
    output = subprocess.run(["wsl", "date", "+%s%N"], capture_output=True, text=True, check=True).stdout
    end = time.time()
    return (int(output.strip()) / 1e9 - (start + end) / 2) * 1000, (end - start) / 2 * 1000


def main() -> int:
    offset_ms, uncertainty_ms = min((measure() for _ in range(TRIES)), key=lambda m: m[1])
    print(f"WSL2 clock minus Windows clock: {offset_ms:+.1f} ms (+/- {uncertainty_ms:.1f} ms, best of {TRIES} calls)")
    if abs(offset_ms) > LIMIT_MS:
        print(f"WARNING: the clocks differ by more than {LIMIT_MS} ms, so the latencies are off by that much. "
              "Fix: run 'sudo hwclock -s' in an Ubuntu terminal, then check again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
