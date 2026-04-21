import gc
import os
import logging

logger = logging.getLogger(__name__)


def force_gc():
    """Force garbage collection and attempt to release memory to OS."""
    gc.collect()
    # On Linux, try to release freed memory back to OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def log_memory_usage(label: str = ""):
    """Log current process memory usage."""
    try:
        pid = os.getpid()
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    logger.info(f"[Memory {label}] RSS: {rss_kb // 1024} MB")
                    return rss_kb
    except (FileNotFoundError, PermissionError):
        pass
    return 0
