import asyncio
import sys
import traceback
from datetime import datetime
from typing import List, Optional

from .levels import LogLevel
from .formats import LogFormat
from .handlers import Handler, ConsoleHandler

class Logger:
    def __init__(
        self,
        name: str = "LitLogger",
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[Handler]] = None,
        fmt: str = LogFormat.DEFAULT,  # <--- use LogFormat.DEFAULT
        async_mode: bool = False,
        include_context: bool = False,  # New flag to include thread/process info
    ):
        import threading
        import multiprocessing

        self.name = name
        self.level = level
        self.format = fmt
        self.async_mode = async_mode
        self.include_context = include_context
        self.handlers = handlers or [ConsoleHandler()]
        self._thread = threading
        self._multiprocessing = multiprocessing

    def _format(self, level: LogLevel, message: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.include_context:
            thread_name = self._thread.current_thread().name
            process_id = self._multiprocessing.current_process().pid
            # Check if format is JSON format
            if self.format.strip().startswith('{') and self.format.strip().endswith('}'):
                # Format as JSON string with extra fields
                return self.format.format(
                    time=now,
                    level=level.name,
                    name=self.name,
                    message=message,
                    thread=thread_name,
                    process=process_id
                )
            else:
                # For non-JSON formats, add thread and process info if placeholders exist
                try:
                    return self.format.format(
                        time=now,
                        level=level.name,
                        name=self.name,
                        message=message,
                        thread=thread_name,
                        process=process_id
                    )
                except KeyError:
                    # If thread/process placeholders not in format, append them manually
                    base = self.format.format(time=now, level=level.name, name=self.name, message=message)
                    return f"{base} | Thread: {thread_name} | Process: {process_id}"
        else:
            return self.format.format(time=now, level=level.name, name=self.name, message=message)

    def set_format(self, fmt: str, include_context: bool = False):
        """Dynamically change the log format and context inclusion."""
        self.format = fmt
        self.include_context = include_context

    def _should_log(self, level: LogLevel) -> bool:
        return level >= self.level

    async def _log_async(self, level: LogLevel, message: str):
        if not self._should_log(level):
            return
        record = self._format(level, message)
        tasks = []
        for h in self.handlers:
            if level >= h.level:
                if asyncio.iscoroutinefunction(h.emit):
                    tasks.append(h.emit(record, level))
                else:
                    tasks.append(asyncio.to_thread(h.emit, record, level))
        if tasks:
            await asyncio.gather(*tasks)

    def _log(self, level: LogLevel, message: str):
        if not self._should_log(level):
            return
        record = self._format(level, message)
        for h in self.handlers:
            if level >= h.level:
                h.emit(record, level)

    def log(self, level: LogLevel, message: str):
        if self.async_mode:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.create_task(self._log_async(level, message))
            return loop.run_until_complete(self._log_async(level, message))
        self._log(level, message)

    def trace(self, message: str):
        self.log(LogLevel.TRACE, message)

    def debug(self, message: str):
        self.log(LogLevel.DEBUG, message)

    def info(self, message: str):
        self.log(LogLevel.INFO, message)

    def warning(self, message: str):
        self.log(LogLevel.WARNING, message)

    def error(self, message: str):
        self.log(LogLevel.ERROR, message)

    def critical(self, message: str):
        self.log(LogLevel.CRITICAL, message)

    def exception(self, message: str):
        exc = sys.exc_info()
        formatted = f"{message}\n" + "".join(traceback.format_exception(*exc))
        self.error(formatted)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.exception(str(exc))
        return False
