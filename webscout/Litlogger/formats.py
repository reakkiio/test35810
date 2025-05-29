DEFAULT_FORMAT = "{time} | {level} | {name} | {message}"

SIMPLE_FORMAT = "{level}: {message}"

DETAILED_FORMAT = "{time} | {level} | {name} | {message} | Thread: {thread} | Process: {process}"

JSON_FORMAT = '{{"time": "{time}", "level": "{level}", "name": "{name}", "message": "{message}"}}'

class LogFormat:
    DEFAULT = DEFAULT_FORMAT
    SIMPLE = SIMPLE_FORMAT
    DETAILED = DETAILED_FORMAT
    JSON = JSON_FORMAT
