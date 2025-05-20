# LitLogger

A minimal yet flexible logging library built from scratch without external dependencies. It provides colored console output, file logging with rotation, simple network logging, and optional asynchronous support.

```python
from webscout.litlogger import Logger, LogLevel, FileHandler

logger = Logger(name="demo", level=LogLevel.DEBUG, handlers=[FileHandler("app.log")])
logger.info("hello world")
```
