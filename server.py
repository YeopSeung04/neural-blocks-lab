#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    import psutil
except ImportError:
    psutil = None

if psutil:
    psutil.cpu_percent(interval=None)


def run_command(command, timeout=2.0):
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def total_memory_bytes():
    if psutil:
        return int(psutil.virtual_memory().total)
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return None


def mac_cpu_memory():
    output = run_command(["top", "-l", "1", "-n", "0", "-stats", "cpu,mem"], 3.0)
    cpu_match = re.search(
        r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys,\s*([\d.]+)% idle",
        output,
    )
    memory_match = re.search(r"PhysMem:\s*([^ ]+) used .*?,\s*([^ ]+) unused", output)
    total = total_memory_bytes()
    cpu_usage = None
    if cpu_match:
        cpu_usage = min(100.0, float(cpu_match.group(1)) + float(cpu_match.group(2)))

    def parse_size(value):
        match = re.match(r"([\d.]+)([KMGTP])", value, re.IGNORECASE)
        if not match:
            return None
        powers = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}
        return int(float(match.group(1)) * 1024 ** powers[match.group(2).upper()])

    used = parse_size(memory_match.group(1)) if memory_match else None
    if used is None and total is not None and memory_match:
        free = parse_size(memory_match.group(2))
        if free is not None:
            used = max(0, total - free)
    return cpu_usage, used, total, "macOS top"


def linux_cpu_memory():
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    cores = os.cpu_count() or 1
    cpu_usage = min(100.0, load / cores * 100)
    memory_info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            memory_info[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    total = memory_info.get("MemTotal") or total_memory_bytes()
    available = memory_info.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    return cpu_usage, used, total, "load average"


def windows_cpu_memory():
    script = (
        "$cpu=(Get-CimInstance Win32_Processor | "
        "Measure-Object -Property LoadPercentage -Average).Average;"
        "$os=Get-CimInstance Win32_OperatingSystem;"
        "[pscustomobject]@{cpu=$cpu;total=[double]$os.TotalVisibleMemorySize*1024;"
        "free=[double]$os.FreePhysicalMemory*1024}|ConvertTo-Json -Compress"
    )
    output = run_command(["powershell", "-NoProfile", "-Command", script], 3.0)
    try:
        data = json.loads(output)
        total = int(data["total"])
        used = total - int(data["free"])
        return float(data["cpu"]), used, total, "Windows CIM"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None, total_memory_bytes(), "unavailable"


def system_cpu_memory():
    if psutil:
        memory = psutil.virtual_memory()
        return (
            float(psutil.cpu_percent(interval=None)),
            int(memory.used),
            int(memory.total),
            "psutil",
        )
    system = platform.system()
    if system == "Darwin":
        return mac_cpu_memory()
    if system == "Windows":
        return windows_cpu_memory()
    return linux_cpu_memory()


def gpu_static_info():
    if platform.system() == "Darwin":
        output = run_command(["system_profiler", "SPDisplaysDataType", "-json"], 5.0)
        try:
            displays = json.loads(output).get("SPDisplaysDataType", [])
            if displays:
                return displays[0].get("sppci_model") or displays[0].get("_name")
        except json.JSONDecodeError:
            pass
    return None


GPU_NAME = gpu_static_info()


def nvidia_metrics():
    if not shutil.which("nvidia-smi"):
        return None
    output = run_command([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], 3.0)
    if not output.strip():
        return None
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) != 4:
        return None
    try:
        used = float(parts[2]) * 1024 ** 2
        total = float(parts[3]) * 1024 ** 2
        return {
            "name": parts[0],
            "usagePercent": float(parts[1]),
            "memoryUsedBytes": int(used),
            "memoryAllocatedBytes": int(used),
            "memoryTotalBytes": int(total),
            "memoryType": "dedicated VRAM",
            "source": "nvidia-smi",
        }
    except ValueError:
        return None


def mac_gpu_metrics():
    accelerator_class = "AGXAccelerator" if platform.machine() == "arm64" else "IOAccelerator"
    output = run_command(
        ["ioreg", "-r", "-d", "1", "-c", accelerator_class],
        3.0,
    )
    usage = re.search(r'"Device Utilization %"=(\d+)', output)
    allocated = re.search(r'"Alloc system memory"=(\d+)', output)
    in_use = re.search(r'"In use system memory"=(\d+)', output)
    name = re.search(r'"model"\s*=\s*"([^"]+)"', output)
    if not any((usage, allocated, in_use, name, GPU_NAME)):
        return None
    return {
        "name": name.group(1) if name else GPU_NAME or "Apple GPU",
        "usagePercent": float(usage.group(1)) if usage else None,
        "memoryUsedBytes": int(in_use.group(1)) if in_use else None,
        "memoryAllocatedBytes": int(allocated.group(1)) if allocated else None,
        "memoryTotalBytes": total_memory_bytes(),
        "memoryType": "unified memory",
        "source": "macOS IOAccelerator",
    }


def gpu_metrics():
    nvidia = nvidia_metrics()
    if nvidia:
        return nvidia
    if platform.system() == "Darwin":
        return mac_gpu_metrics()
    return {
        "name": GPU_NAME or "GPU",
        "usagePercent": None,
        "memoryUsedBytes": None,
        "memoryAllocatedBytes": None,
        "memoryTotalBytes": None,
        "memoryType": "unavailable",
        "source": "unsupported OS bridge",
    }


def collect_metrics():
    cpu_usage, memory_used, memory_total, source = system_cpu_memory()
    memory_percent = None
    if memory_used is not None and memory_total:
        memory_percent = memory_used / memory_total * 100
    return {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "cpu": {
            "usagePercent": cpu_usage,
            "logicalCores": os.cpu_count(),
            "source": source,
        },
        "memory": {
            "usedBytes": memory_used,
            "totalBytes": memory_total,
            "usagePercent": memory_percent,
            "source": source,
        },
        "gpu": gpu_metrics(),
    }


class NeuralBlocksHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/api/system-metrics":
            payload = json.dumps(collect_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, format_string, *args):
        if self.path != "/api/system-metrics":
            super().log_message(format_string, *args)


def main():
    parser = argparse.ArgumentParser(description="Neural Blocks Lab local server")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), NeuralBlocksHandler)
    print(f"Neural Blocks Lab: http://{args.bind}:{args.port}")
    print("System metrics endpoint: /api/system-metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
