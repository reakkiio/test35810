"""Main CLI application class."""

import sys
from typing import Any, Dict, List, Optional, Union

from rich.console import Console

from ..exceptions import UsageError
from ..plugins.manager import PluginManager
from ..utils.formatting import format_error, format_success
from .context import Context

console = Console()

class CLI:
    """
    Main CLI application class.
    
    The CLI class is the core of SwiftCLI. It handles command registration,
    argument parsing, and command execution. It also manages plugins and
    provides the main entry point for CLI applications.
    
    Attributes:
        name: Application name
        help: Application description
        version: Application version
        debug: Debug mode flag
        commands: Registered commands
        groups: Command groups
        plugin_manager: Plugin manager instance
        
    Example:
        >>> app = CLI(name="myapp", version="1.0.0")
        >>> @app.command()
        ... def greet(name: str):
        ...     '''Greet someone'''
        ...     print(f"Hello {name}!")
        >>> app.run()
    """
    
    def __init__(
        self,
        name: str,
        help: Optional[str] = None,
        version: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize CLI application.
        
        Args:
            name: Application name
            help: Application description
            version: Application version
            debug: Enable debug mode
        """
        self.name = name
        self.help = help
        self.version = version
        self.debug = debug
        
        self.commands: Dict[str, Dict[str, Any]] = {}
        self.groups: Dict[str, 'Group'] = {}  # type: ignore
        self.plugin_manager = PluginManager()
        
        # Initialize plugin manager with this CLI instance
        self.plugin_manager.init_plugins(self)
    
    def command(
        self,
        name: Optional[str] = None,
        help: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        hidden: bool = False
    ):
        """
        Decorator to register a command.
        
        Args:
            name: Command name (defaults to function name)
            help: Command help text
            aliases: Alternative command names
            hidden: Hide from help output
            
        Example:
            @app.command()
            def hello(name: str):
                '''Say hello'''
                print(f"Hello {name}!")
        """
        def decorator(f):
            cmd_name = name or f.__name__
            self.commands[cmd_name] = {
                'func': f,
                'help': help or f.__doc__,
                'aliases': aliases or [],
                'hidden': hidden
            }
            
            # Register aliases
            for alias in (aliases or []):
                self.commands[alias] = self.commands[cmd_name]
            
            return f
        return decorator
    
    def group(
        self,
        name: Optional[str] = None,
        help: Optional[str] = None,
        **kwargs
    ):
        """
        Create a command group.
        
        Args:
            name: Group name
            help: Group help text
            **kwargs: Additional group options
            
        Example:
            @app.group()
            def db():
                '''Database commands'''
                pass
                
            @db.command()
            def migrate():
                '''Run migrations'''
                pass
        """
        from .group import Group  # Import here to avoid circular dependency
        
        def decorator(f):
            group_name = name or f.__name__
            group = Group(
                name=group_name,
                help=help or f.__doc__,
                parent=self,
                **kwargs
            )
            self.groups[group_name] = group
            return group
        return decorator
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI application.
        
        Args:
            args: Command line arguments (defaults to sys.argv[1:])
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            args = args or sys.argv[1:]
            
            # Show help if no arguments
            if not args or args[0] in ['-h', '--help']:
                self._print_help()
                return 0
            
            # Show version if requested
            if args[0] in ['-v', '--version'] and self.version:
                console.print(self.version)
                return 0
            
            command_name = args[0]
            command_args = args[1:]
            
            # Check if it's a group command
            if command_name in self.groups:
                return self.groups[command_name].run(command_args)
            
            # Check if it's a regular command
            if command_name not in self.commands:
                format_error(f"Unknown command: {command_name}")
                self._print_help()
                return 1
            
            # Create command context
            ctx = Context(self, command=command_name, debug=self.debug)
            
            # Run command through plugin system
            if not self.plugin_manager.before_command(command_name, command_args):
                return 1
            
            try:
                command = self.commands[command_name]
                result = command['func'](**self._parse_args(command, command_args))
                self.plugin_manager.after_command(command_name, command_args, result)
                return 0
            except Exception as e:
                self.plugin_manager.on_error(command_name, e)
                if self.debug:
                    raise
                format_error(str(e))
                return 1
            
        except KeyboardInterrupt:
            console.print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            if self.debug:
                raise
            format_error(str(e))
            return 1
    
    def _parse_args(self, command: Dict[str, Any], args: List[str]) -> Dict[str, Any]:
        """Parse command arguments."""
        from ..utils.parsing import (
            parse_args, validate_required, convert_type,
            validate_choice, get_env_var
        )
        
        params = {}
        func = command['func']
        
        # Parse command-line arguments
        parsed_args = parse_args(args)
        
        # Handle options
        if hasattr(func, '_options'):
            for opt in func._options:
                # Use the longest parameter name (usually the --long-form) for the parameter name
                param_names = [p.lstrip('-').replace('-', '_') for p in opt['param_decls']]
                name = max(param_names, key=len)  # Use the longest name

                # Check all possible parameter names in parsed args
                value = None
                found = False
                for param_name in param_names:
                    if param_name in parsed_args:
                        value = parsed_args[param_name]
                        found = True
                        break

                if found:
                    if 'type' in opt:
                        value = convert_type(value, opt['type'], name)
                    if 'choices' in opt and opt['choices']:
                        validate_choice(
                            value,
                            opt['choices'],
                            name,
                            opt.get('case_sensitive', True)
                        )
                    params[name] = value
                elif opt.get('required', False):
                    raise UsageError(f"Missing required option: {name}")
                elif 'default' in opt:
                    params[name] = opt['default']
        
        # Handle arguments
        if hasattr(func, '_arguments'):
            for i, arg in enumerate(func._arguments):
                name = arg['name']
                if f'arg{i}' in parsed_args:
                    value = parsed_args[f'arg{i}']
                    if 'type' in arg:
                        value = convert_type(value, arg['type'], name)
                    params[name] = value
                elif arg.get('required', True):
                    raise UsageError(f"Missing required argument: {name}")
                elif 'default' in arg:
                    params[name] = arg['default']
        
        # Handle environment variables
        if hasattr(func, '_envvars'):
            for env in func._envvars:
                name = env['name'].lower()
                value = get_env_var(
                    env['name'],
                    env.get('type', str),
                    env.get('required', False),
                    env.get('default')
                )
                if value is not None:
                    params[name] = value
        
        return params
    
    def _print_help(self) -> None:
        """Print application help message."""
        console.print(f"\n[bold]{self.name}[/]")
        if self.help:
            console.print(f"\n{self.help}")
        
        # Show commands
        console.print("\n[bold]Commands:[/]")
        for name, cmd in self.commands.items():
            if not cmd.get('hidden', False):
                console.print(f"  {name:20} {cmd['help'] or ''}")
        
        # Show command groups
        for name, group in self.groups.items():
            console.print(f"\n[bold]{name} commands:[/]")
            for cmd_name, cmd in group.commands.items():
                if not cmd.get('hidden', False):
                    console.print(f"  {cmd_name:20} {cmd['help'] or ''}")
        
        console.print("\nUse -h or --help with any command for more info")
        if self.version:
            console.print("Use -v or --version to show version")
    
    def __repr__(self) -> str:
        return f"<CLI name={self.name}>"
