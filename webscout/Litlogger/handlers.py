import os
import socket
import http.client
from datetime import datetime
from pathlib import Path
from typing import Optional

from .levels import LogLevel

RESET = "\033[0m"
LEVEL_COLORS = {
    LogLevel.TRACE: "\033[90m",
    LogLevel.DEBUG: "\033[36m",
    LogLevel.INFO: "\033[32m",
    LogLevel.WARNING: "\033[33m",
    LogLevel.ERROR: "\033[31m",
    LogLevel.CRITICAL: "\033[41m\033[97m",
}

class Handler:
    def __init__(self, level: LogLevel = LogLevel.DEBUG):
        self.level = level

    def emit(self, message: str, level: LogLevel):
        raise NotImplementedError

class ConsoleHandler(Handler):
    def __init__(self, stream=None, level: LogLevel = LogLevel.DEBUG):
        super().__init__(level)
        self.stream = stream or os.sys.stdout

    def emit(self, message: str, level: LogLevel):
        color = LEVEL_COLORS.get(level, "")
        self.stream.write(f"{color}{message}{RESET}\n")
        self.stream.flush()

class FileHandler(Handler):
    def __init__(self, path: str, level: LogLevel = LogLevel.DEBUG, max_bytes: int = 0, backups: int = 0):
        super().__init__(level)
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self._open()

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def _rotate(self):
        if self.backups <= 0:
            self._file.close()
            self.path.unlink(missing_ok=True)
            self._open()
            return
        self._file.close()
        for i in range(self.backups, 0, -1):
            src = self.path.with_suffix(f".{i}") if i == 1 else self.path.with_suffix(f".{i-1}")
            dst = self.path.with_suffix(f".{i}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        self._open()

    def emit(self, message: str, level: LogLevel):
        if self.level and level < self.level:
            return
        self._file.write(message + "\n")
        self._file.flush()
        if self.max_bytes and self._file.tell() >= self.max_bytes:
            self._rotate()

class NetworkHandler(Handler):
    def __init__(self, host: str, port: int, use_https: bool = False, level: LogLevel = LogLevel.DEBUG):
        super().__init__(level)
        self.host = host
        self.port = port
        self.use_https = use_https

    def emit(self, message: str, level: LogLevel):
        if level < self.level:
            return
        if self.use_https:
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=5)
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            conn.request("POST", "/", body=message.encode(), headers={"Content-Type": "text/plain"})
            conn.getresponse().read()
        finally:
            conn.close()

class TCPHandler(Handler):
    def __init__(self, host: str, port: int, level: LogLevel = LogLevel.DEBUG):
        super().__init__(level)
        self.host = host
        self.port = port

    def emit(self, message: str, level: LogLevel):
        if level < self.level:
            return
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.sendall(message.encode() + b"\n")

class JSONFileHandler(FileHandler):
    def __init__(self, path: str, level: LogLevel = LogLevel.DEBUG, max_bytes: int = 0, backups: int = 0):
        super().__init__(path, level, max_bytes, backups)

    def emit(self, message: str, level: LogLevel):
        # Expect message to be a JSON string or dict
        if level < self.level:
            return
        import json
        if isinstance(message, dict):
            log_entry = json.dumps(message)
        else:
            log_entry = message
        self._file.write(log_entry + "\n")
        self._file.flush()
        if self.max_bytes and self._file.tell() >= self.max_bytes:
            self._rotate()
