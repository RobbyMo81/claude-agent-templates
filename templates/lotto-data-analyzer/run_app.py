#!/usr/bin/env python3
"""
run_app.py
-------------
Robust launcher for the LottoDataAnalyzer application.

Features:
- Preflight checks for required files and Python dependencies
- Optional automatic installation of missing dependencies
- Starts Streamlit using the current Python interpreter
- Captures stdout/stderr, writes rotating logs, and emits a JSON run report
- Graceful shutdown on SIGINT/SIGTERM

Usage (example):
    python run_app.py --port 8501

Run with --install-missing to attempt to pip-install missing libraries automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional
from core.device import detect_gpus


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "logs"
DEFAULT_LOG.mkdir(exist_ok=True)
LOG_FILE = DEFAULT_LOG / "run_app.log"
REPORT_FILE = ROOT / "data" / "run_app_report.json"
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


REQUIRED_FILES = [
    ROOT / "app.py",
    ROOT / "data" / "powerball_complete_dataset.csv",
]

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
]


def setup_logger(log_path: Path = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger("run_app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3)
        fmt = "%(asctime)s %(levelname)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console)
    return logger


def preflight_check() -> Dict:
    """Check for required files and importable packages."""
    missing_files = [str(p) for p in REQUIRED_FILES if not p.exists()]
    missing_packages = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception:
            missing_packages.append(pkg)

    return {"missing_files": missing_files, "missing_packages": missing_packages}


def find_project_venv_python() -> Optional[Path]:
    """Return the path to the project's virtualenv python executable if it exists.

    Supports typical layouts on Windows and Unix.
    """
    win_path = ROOT / "venv" / "Scripts" / "python.exe"
    posix_path = ROOT / "venv" / "bin" / "python"
    if win_path.exists():
        return win_path
    if posix_path.exists():
        return posix_path
    return None


def check_imports_with_python(python_exe: str, packages: List[str]) -> List[str]:
    """Check which packages are missing for the given python executable.

    Returns a list of package names that failed to import.
    """
    missing = []
    for pkg in packages:
        cmd = [python_exe, "-c", f"import {pkg}"]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing.append(pkg)
        except FileNotFoundError:
            # python_exe doesn't exist
            missing = packages.copy()
            break
    return missing


def pip_install(packages: List[str], logger: logging.Logger, python_exe: Optional[str] = None) -> bool:
    """Attempt to pip install the given packages using the specified Python interpreter.

    If python_exe is None, uses sys.executable.
    """
    if not packages:
        return True
    python_exe = python_exe or sys.executable
    cmd = [python_exe, "-m", "pip", "install"] + packages
    logger.info("Installing missing packages using %s: %s", python_exe, packages)
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Automatic install failed: %s", e)
        return False


def stream_reader(pipe, logger, buffer: deque, label: str):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            line = line.rstrip("\n")
            buffer.append(line)
            # keep last 500 lines
            if len(buffer) > 500:
                buffer.popleft()
            logger.info(f"[%s] %s", label, line)
    except Exception as e:
        logger.exception("Error reading %s: %s", label, e)


def write_report(report_path: Path, report: Dict, logger: logging.Logger):
    try:
        tmp = report_path.with_suffix(".tmp.json")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        tmp.replace(report_path)
        logger.info("Run report written to %s", report_path)
    except Exception as e:
        logger.exception("Failed to write run report: %s", e)


def start_streamlit(app_py: Path, host: str, port: int, extra_args: List[str], logger: logging.Logger, python_exe: Optional[str] = None):
    """Start streamlit using the provided python_exe (or sys.executable by default)."""
    python_exe = python_exe or sys.executable
    cmd = [python_exe, "-m", "streamlit", "run", str(app_py), "--server.port", str(port), "--server.address", host]

    # Ensure the event-based file watcher is disabled to avoid RuntimeError in some Watchdog versions
    # Respect explicit user-supplied --server.fileWatcherType in extra_args
    has_watcher_arg = False
    if extra_args:
        for a in extra_args:
            if isinstance(a, str) and "fileWatcherType" in a:
                has_watcher_arg = True
                break
        cmd += extra_args

    if not has_watcher_arg:
        cmd += ["--server.fileWatcherType", "none"]

    logger.info("Starting Streamlit: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout_buf = deque()
    stderr_buf = deque()

    t_out = threading.Thread(target=stream_reader, args=(proc.stdout, logger, stdout_buf, "STDOUT"), daemon=True)
    t_err = threading.Thread(target=stream_reader, args=(proc.stderr, logger, stderr_buf, "STDERR"), daemon=True)
    t_out.start()
    t_err.start()

    return proc, stdout_buf, stderr_buf


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run LottoDataAnalyzer (Streamlit) with robust reporting")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind Streamlit")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit server")
    parser.add_argument("--install-missing", action="store_true", help="Attempt to pip install missing dependencies")
    parser.add_argument("--report-file", default=str(REPORT_FILE), help="Path to write JSON run report")
    parser.add_argument("--log-file", default=str(LOG_FILE), help="Log file path")
    parser.add_argument("--streamlit-arg", action="append", help="Extra argument to pass to streamlit (can use multiple)")
    args = parser.parse_args(argv)

    logger = setup_logger(Path(args.log_file))

    start_time = datetime.utcnow()
    report = {
        "start_time": start_time.isoformat() + "Z",
        "host": args.host,
        "port": args.port,
        "missing_files": [],
        "missing_packages": [],
    "gpu_info": {},
        "exit_code": None,
        "end_time": None,
        "stdout_tail": [],
        "stderr_tail": [],
    }

    # Detect project venv and prefer it for launching and package operations
    project_venv = find_project_venv_python()
    venv_path_str = str(project_venv) if project_venv else None
    using_python = sys.executable
    if project_venv:
        if Path(sys.executable).resolve() != project_venv.resolve():
            logger.warning("Project virtualenv detected at %s but current interpreter is %s. Using project venv for launching.", project_venv, sys.executable)
        using_python = str(project_venv)

    report["venv_detected"] = bool(project_venv)
    report["venv_path"] = venv_path_str
    report["used_python"] = using_python

    # Detect GPUs and include in report
    try:
        report["gpu_info"] = detect_gpus()
    except Exception:
        report["gpu_info"] = {}

    # Run preflight checks against the chosen interpreter
    checks = preflight_check()
    # Override package check by probing the chosen python executable
    missing_for_venv = check_imports_with_python(using_python, REQUIRED_PACKAGES)
    checks["missing_packages"] = missing_for_venv
    report["missing_files"] = checks["missing_files"]
    report["missing_packages"] = checks["missing_packages"]

    if checks["missing_files"]:
        logger.warning("Missing required files: %s", checks["missing_files"])
        logger.warning("Please ensure the dataset and app.py exist before starting the server.")

    if checks["missing_packages"]:
        logger.warning("Missing Python packages: %s", checks["missing_packages"])
        if args.install_missing:
            ok = pip_install(checks["missing_packages"], logger, python_exe=using_python)
            if not ok:
                logger.error("Failed to install missing dependencies. Exiting.")
                report["exit_code"] = 2
                report["end_time"] = datetime.utcnow().isoformat() + "Z"
                write_report(Path(args.report_file), report, logger)
                return 2
        else:
            logger.info("Run with --install-missing to attempt automatic installation using %s: -m pip install -r requirements.txt", using_python)

    app_py = ROOT / "app.py"
    if not app_py.exists():
        logger.error("app.py not found in project root (%s). Aborting.", app_py)
        report["exit_code"] = 3
        report["end_time"] = datetime.utcnow().isoformat() + "Z"
        write_report(Path(args.report_file), report, logger)
        return 3

    extra = args.streamlit_arg if args.streamlit_arg else []

    # Ensure fileWatcherType is set to 'none' to avoid watchdog RuntimeError in some environments
    # Allow user override via --streamlit-arg if they explicitly specify fileWatcherType.
    if not any('--server.fileWatcherType' in a for a in extra):
        extra += ["--server.fileWatcherType", "none"]

    proc, stdout_buf, stderr_buf = start_streamlit(app_py, args.host, args.port, extra, logger, python_exe=using_python)

    def handle_signal(sig, frame):
        logger.info("Received signal %s, terminating Streamlit process %s", sig, proc.pid)
        try:
            proc.terminate()
        except Exception:
            proc.kill()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        exit_code = proc.wait()
        logger.info("Streamlit exited with code %s", exit_code)
        report["exit_code"] = exit_code
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating child process")
        try:
            proc.terminate()
        except Exception:
            proc.kill()
        report["exit_code"] = -1
    finally:
        # Give reader threads a moment to flush
        time.sleep(0.5)
        report["end_time"] = datetime.utcnow().isoformat() + "Z"
        report["stdout_tail"] = list(list(stdout_buf)[-200:])
        report["stderr_tail"] = list(list(stderr_buf)[-200:])
        write_report(Path(args.report_file), report, logger)

    return report.get("exit_code", 0)


if __name__ == "__main__":
    rc = main()
    if isinstance(rc, int):
        sys.exit(rc)