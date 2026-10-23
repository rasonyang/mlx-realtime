"""Sample per-process memory of the stack: phys_footprint (and its peak) and RSS.

    python3 -I spikes/realstack/memsample.py OUT.jsonl INTERVAL_S [DURATION_S]

Finds the processes by command line each time (so restarts are followed): the gateway
(mlx_realtime ... serve), the llm (mlxrt_llm.serve), asr (:8102) and tts (:8103) servers.
Also records vm_stat's free/active/wired pages, swap use and the memory_pressure level.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time

OUT = sys.argv[1]
INTERVAL = float(sys.argv[2])
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else float("inf")


def processes() -> dict[str, tuple[int, int]]:
    """name -> (pid, rss KB)."""
    found: dict[str, tuple[int, int]] = {}
    out = subprocess.run(["ps", "-axo", "pid=,rss=,command="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, command = int(parts[0]), int(parts[1]), parts[2]
        if command.split()[0].endswith("uv"):  # uv run's wrapper process, not the server
            continue
        if "mlx_realtime" in command and command.endswith("serve") and "python" in command:
            found["gateway"] = (pid, rss)
        elif "mlxrt_llm.serve" in command:
            found["llm"] = (pid, rss)
        elif "mlx_audio.server" in command and "--port 8102" in command:
            found["asr"] = (pid, rss)
        elif "mlx_audio.server" in command and "--port 8103" in command:
            found["tts"] = (pid, rss)
    return found


def footprints(pids: list[int]) -> dict[int, tuple[int, int]]:
    args = ["footprint", "-f", "bytes", "--noCategories"]
    for pid in pids:
        args += ["-p", str(pid)]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    result: dict[int, tuple[int, int]] = {}
    current = None
    phys = peak = 0
    for line in out.splitlines():
        if match := re.search(r"\[(\d+)\]: .*Footprint: (\d+) B", line):
            current = int(match.group(1))
        elif (match := re.search(r"phys_footprint: (\d+) B", line)) and current is not None:
            phys = int(match.group(1))
        elif (match := re.search(r"phys_footprint_peak: (\d+) B", line)) and current is not None:
            peak = int(match.group(1))
            result[current] = (phys, peak)
    return result


def system() -> dict[str, object]:
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(Pages [\w ]+|Swapouts):\s+(\d+)", vm, re.M)}
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    pressure = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout.strip()
    return {"free_gb": round(pages.get("Pages free", 0) * 16384 / 2**30, 2),
            "wired_gb": round(pages.get("Pages wired down", 0) * 16384 / 2**30, 2),
            "swapouts": pages.get("Swapouts"), "swap": swap, "pressure": pressure[-60:]}


start = time.monotonic()
while True:
    procs = processes()
    fps = footprints([pid for pid, _ in procs.values()]) if procs else {}
    record: dict[str, object] = {"t": time.strftime("%H:%M:%S"), "elapsed_s": round(time.monotonic() - start)}
    for name, (pid, rss) in sorted(procs.items()):
        phys, peak = fps.get(pid, (0, 0))
        record[name] = {"pid": pid, "footprint_gb": round(phys / 2**30, 3),
                        "peak_gb": round(peak / 2**30, 3), "rss_gb": round(rss / 2**20, 3)}
    record["system"] = system()
    with open(OUT, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if time.monotonic() - start + INTERVAL > DURATION:
        break
    time.sleep(INTERVAL)
