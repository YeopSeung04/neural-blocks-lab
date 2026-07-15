#!/usr/bin/env python3
import argparse
import hmac
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from backend import ApiError, NeuralBlocksBackend

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.environ.get("NBL_DATABASE_PATH", ROOT / ".data" / "neural_blocks.db")
)
AUTH_WINDOW_SECONDS = 15 * 60
AUTH_MAX_ATTEMPTS = 12
AUTH_ATTEMPTS = {}
AUTH_ATTEMPTS_LOCK = threading.Lock()

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


BACKEND = NeuralBlocksBackend(DATABASE_PATH)


def allow_auth_attempt(key):
    now = time.time()
    with AUTH_ATTEMPTS_LOCK:
        recent = [
            attempt for attempt in AUTH_ATTEMPTS.get(key, [])
            if now - attempt < AUTH_WINDOW_SECONDS
        ]
        if len(recent) >= AUTH_MAX_ATTEMPTS:
            AUTH_ATTEMPTS[key] = recent
            return False
        recent.append(now)
        AUTH_ATTEMPTS[key] = recent
        return True


class NeuralBlocksHandler(SimpleHTTPRequestHandler):
    backend = BACKEND
    max_json_bytes = 2 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "worker-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'self'",
        )
        super().end_headers()

    def api_path(self):
        return urlparse(self.path).path

    def protected_static_path(self, path):
        decoded = unquote(path).replace("\\", "/")
        parts = [part for part in decoded.split("/") if part not in ("", ".")]
        if any(part in {".data", ".git", "__pycache__"} for part in parts):
            return True
        filename = parts[-1].lower() if parts else ""
        return filename.endswith((".db", ".db-shm", ".db-wal", ".py", ".pyc"))

    def send_json(self, status, data, headers=None):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers or []:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self):
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ApiError(415, "Content-Type must be application/json", "unsupported_media_type")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(400, "Invalid Content-Length", "invalid_request")
        if length <= 0:
            return {}
        if length > self.max_json_bytes:
            raise ApiError(413, "JSON request body is too large", "payload_too_large")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "Request body must be valid JSON", "invalid_json")
        if not isinstance(payload, dict):
            raise ApiError(400, "JSON request body must be an object", "invalid_json")
        return payload

    def session_token(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        value = cookie.get("nbl_session")
        return value.value if value else None

    def require_auth(self):
        return self.backend.authenticate(self.session_token())

    def require_csrf(self, auth):
        provided = self.headers.get("X-CSRF-Token", "")
        expected = auth.get("csrfToken", "")
        if not provided or not hmac.compare_digest(provided, expected):
            raise ApiError(403, "CSRF token is invalid", "csrf_failed")

    def session_cookie(self, token, clear=False):
        parts = [
            f"nbl_session={'' if clear else token}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            f"Max-Age={0 if clear else 7 * 24 * 60 * 60}",
        ]
        if os.environ.get("NBL_SECURE_COOKIES") == "1":
            parts.append("Secure")
        return "; ".join(parts)

    def auth_rate_key(self, payload):
        email = str(payload.get("email") or "").strip().lower()
        return f"{self.client_address[0]}:{email}"

    def handle_api_error(self, error):
        self.send_json(
            error.status,
            {"error": {"code": error.code, "message": error.message}},
        )

    def do_GET(self):
        path = self.api_path()
        if path == "/api/system-metrics":
            self.send_json(200, collect_metrics())
            return
        if path == "/api/health":
            self.send_json(200, {"status": "ok", "database": "sqlite"})
            return
        if path.startswith("/api/"):
            try:
                auth = self.require_auth()
                if path == "/api/auth/me":
                    self.send_json(200, auth)
                    return
                if path == "/api/courses":
                    self.send_json(200, {"courses": self.backend.list_courses(auth)})
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/assignments", path)
                if match:
                    self.send_json(
                        200,
                        {"assignments": self.backend.list_assignments(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/projects", path)
                if match:
                    self.send_json(
                        200,
                        {"projects": self.backend.list_projects(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/submissions", path)
                if match:
                    self.send_json(
                        200,
                        {"submissions": self.backend.list_submissions(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/projects/([^/]+)", path)
                if match:
                    self.send_json(200, {"project": self.backend.get_project(auth, match.group(1))})
                    return
                raise ApiError(404, "API endpoint was not found", "not_found")
            except ApiError as error:
                self.handle_api_error(error)
            except Exception as error:
                self.log_error("Unhandled API GET error: %s", error)
                self.send_json(500, {"error": {"code": "server_error", "message": "Internal server error"}})
            return
        if self.protected_static_path(path):
            self.send_error(404, "Not Found")
            return
        super().do_GET()

    def do_HEAD(self):
        path = self.api_path()
        if path.startswith("/api/") or self.protected_static_path(path):
            self.send_error(404, "Not Found")
            return
        super().do_HEAD()

    def do_POST(self):
        path = self.api_path()
        if not path.startswith("/api/"):
            self.send_error(405, "Method Not Allowed")
            return
        try:
            payload = self.read_json()
            if path in ("/api/auth/register", "/api/auth/login"):
                key = self.auth_rate_key(payload)
                if not allow_auth_attempt(key):
                    raise ApiError(429, "Too many authentication attempts", "rate_limited")
                auth = self.backend.register(payload) if path.endswith("register") else self.backend.login(payload)
                token = auth.pop("sessionToken")
                self.send_json(
                    201 if path.endswith("register") else 200,
                    auth,
                    [("Set-Cookie", self.session_cookie(token))],
                )
                return

            auth = self.require_auth()
            self.require_csrf(auth)
            if path == "/api/auth/logout":
                self.backend.logout(self.session_token())
                self.send_json(
                    200,
                    {"status": "logged_out"},
                    [("Set-Cookie", self.session_cookie("", clear=True))],
                )
                return
            if path == "/api/courses":
                self.send_json(201, {"course": self.backend.create_course(auth, payload)})
                return
            if path == "/api/courses/join":
                self.send_json(200, {"course": self.backend.join_course(auth, payload)})
                return
            match = re.fullmatch(r"/api/courses/([^/]+)/assignments", path)
            if match:
                self.send_json(
                    201,
                    {"assignment": self.backend.create_assignment(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(r"/api/courses/([^/]+)/projects", path)
            if match:
                self.send_json(
                    201,
                    {"project": self.backend.save_project(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(r"/api/assignments/([^/]+)/submissions", path)
            if match:
                self.send_json(
                    201,
                    {"submission": self.backend.submit_assignment(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(r"/api/submissions/([^/]+)/grade", path)
            if match:
                self.send_json(
                    200,
                    {"submission": self.backend.grade_submission(auth, match.group(1), payload)},
                )
                return
            raise ApiError(404, "API endpoint was not found", "not_found")
        except ApiError as error:
            self.handle_api_error(error)
        except Exception as error:
            self.log_error("Unhandled API POST error: %s", error)
            self.send_json(500, {"error": {"code": "server_error", "message": "Internal server error"}})

    def log_message(self, format_string, *args):
        if self.api_path() not in ("/api/system-metrics", "/api/health"):
            super().log_message(format_string, *args)


def main():
    parser = argparse.ArgumentParser(description="Neural Blocks Lab local server")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), NeuralBlocksHandler)
    print(f"Neural Blocks Lab: http://{args.bind}:{args.port}")
    print(f"Classroom database: {DATABASE_PATH}")
    print("Authentication endpoints: /api/auth/register, /api/auth/login, /api/auth/me")
    print("System metrics endpoint: /api/system-metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
