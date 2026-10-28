"""Find, start and stop the local llama-server."""

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request


class LaunchError(RuntimeError):
    """The model server could not be started."""


_DEVICE_LINE = re.compile(r"^\s*(Vulkan|CUDA|ROCm|HIP|Metal|SYCL|MUSA)\d*:", re.MULTILINE)


def detect_device(server_bin, run=subprocess.run):
    """"gpu" when llama.cpp lists an accelerator, else "cpu"."""
    try:
        result = run([server_bin, "--list-devices"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    return "gpu" if _DEVICE_LINE.search(result.stdout or "") else "cpu"


def weights_cached(hf, env=os.environ):
    """Whether the -hf repo:quant is already in the Hugging Face cache (then the server can start offline)."""
    repo, _, quant = hf.partition(":")
    org, _, name = repo.partition("/")
    hub = env.get("HF_HUB_CACHE") or (Path(env["HF_HOME"]) / "hub" if env.get("HF_HOME")
                                      else Path.home() / ".cache" / "huggingface" / "hub")
    pattern = str(Path(hub) / f"models--{org}--{name}" / "snapshots" / "*" / f"*{quant}*.gguf")
    return bool(glob.glob(pattern))


def server_args(cfg, device, port, cached):
    args = ["-hf", cfg.hf, "--jinja", "--reasoning-budget", "0", "--alias", cfg.alias,
            "--host", "127.0.0.1", "--port", str(port)]
    if cached:
        args.append("--offline")
    if device == "gpu":
        args += ["-ngl", "99", "-fa", "on", "-c", "32768", "-np", str(cfg.slots), "--image-max-tokens", "1120"]
    else:
        args += ["-ngl", "0", "-c", "8192", "-np", "2"]
    return args


def free_port(start=8060, end=8079):
    for port in range(start, end + 1):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise LaunchError(f"No free port between {start} and {end}")


@dataclass
class ServerHandle:
    url: str
    process: subprocess.Popen | None = None

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        # The server runs in its own session; signal the whole group so helper processes go too.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        except ProcessLookupError:
            pass


def _healthy(url):
    try:
        with urllib.request.urlopen(url.removesuffix("/v1") + "/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def start_server(cfg, log_path, timeout=600.0):
    if cfg.url:
        return ServerHandle(cfg.url)
    device = detect_device(cfg.server_bin) if cfg.device == "auto" else cfg.device
    port = cfg.port or free_port()
    argv = [cfg.server_bin, *server_args(cfg, device, port, weights_cached(cfg.hf))]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "wb") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        raise LaunchError(f"Cannot start {cfg.server_bin}: {exc}") from exc
    handle = ServerHandle(f"http://127.0.0.1:{port}/v1", process)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-30:]
            raise LaunchError("Model server stopped during startup:\n" + "\n".join(tail))
        if _healthy(handle.url):
            return handle
        time.sleep(0.25)
    handle.stop()
    raise LaunchError(f"Model server not ready after {timeout:.0f}s; see {log_path}")
