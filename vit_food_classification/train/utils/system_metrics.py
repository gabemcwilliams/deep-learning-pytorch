"""
system_metrics.py

Continuously logs system resource usage to MLflow during model training or inference.

Features:
- Logs CPU utilization (%)
- Logs RAM usage (in GB)
- Logs GPU utilization and memory usage (in MB), if available via `pynvml`

Usage:
    Run in a separate thread alongside your training loop:

    import threading
    import system_metrics

    stop_event = threading.Event()
    metrics_thread = threading.Thread(target=system_metrics.log_system_metrics, args=(stop_event,))
    metrics_thread.start()

    # ... training code ...

    stop_event.set()
    metrics_thread.join()
"""

import psutil
import mlflow
import time

# Optional GPU monitoring via NVIDIA Management Library
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except (ImportError, Exception):  # Handle both missing library and GPU errors
    print("[WARN] pynvml not available or GPU inaccessible — GPU metrics will be skipped.")
    gpu_available = False


def log_system_metrics(stop_event, interval=10):
    """
    Continuously logs system metrics to MLflow until the stop_event is set.

    Args:
        stop_event (threading.Event): Event to signal stopping the logging loop.
        interval (int): Time (in seconds) between metric log intervals.
    """
    while not stop_event.is_set():
        # CPU and RAM usage
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / 1e9  # Convert bytes to GB

        mlflow.log_metric("cpu_percent", cpu)
        mlflow.log_metric("ram_used_gb", ram)

        if gpu_available:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6  # Convert bytes to MB
                mlflow.log_metric("gpu_util_percent", gpu_util)
                mlflow.log_metric("gpu_mem_used_mb", gpu_mem)
            except pynvml.NVMLError:
                print("[WARN] Failed to read GPU metrics this interval.")

        stop_event.wait(interval)
