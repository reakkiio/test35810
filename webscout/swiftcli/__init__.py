"""
SwiftCLI - Build Beautiful Command-Line Applications at Light Speed

A modern, feature-rich CLI framework for building awesome command-line applications.
Built with love for the Python community!

Basic Usage:
    >>> from swiftcli import CLI
    >>> app = CLI(name="my-app", help="My awesome CLI app")
    >>> @app.command()
    ... def hello(name: str):
    ...     '''Say hello to someone'''
    ...     print(f"Hello {name}!")
    >>> app.run()

For more examples and documentation, visit:
https://github.com/OEvortex/Webscout/tree/main/webscout/swiftcli
"""

from .core.cli import CLI
from .core.group import Group
from .core.context import Context
from .plugins.base import Plugin
from .exceptions import (
    SwiftCLIException,
    UsageError,
    BadParameter,
    ConfigError,
    PluginError
)

# Command decorators
from .decorators.command import (
    command,
    group,
    argument,
    flag,
    pass_context
)

# Option decorators
from .decorators.options import (
    option,
    envvar,
    config_file,
    version_option,
    help_option
)

# Output decorators
from .decorators.output import (
    table_output,
    progress,
    panel_output,
    format_output,
    pager_output
)


__all__ = [
    # Core classes
    'CLI',
    'Group',
    'Context',
    'Plugin',
    
    # Exceptions
    'SwiftCLIException',
    'UsageError',
    'BadParameter',
    'ConfigError',
    'PluginError',
    
    # Command decorators
    'command',
    'group',
    'argument',
    'flag',
    'pass_context',
    
    # Option decorators
    'option',
    'envvar',
    'config_file',
    'version_option',
    'help_option',
    
    # Output decorators
    'table_output',
    'progress',
    'panel_output',
    'format_output',
    'pager_output',

]
