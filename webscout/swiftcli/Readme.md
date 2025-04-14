<div align="center">
  <h1>⚡ SwiftCLI</h1>
  <p><strong>Build Beautiful Command-Line Applications at Light Speed</strong></p>
  
  <!-- Badges -->
  <p>
    <a href="#-installation"><img src="https://img.shields.io/badge/Easy-Installation-success?style=flat-square" alt="Easy Installation"></a>
    <a href="#-features"><img src="https://img.shields.io/badge/Rich-Features-blue?style=flat-square" alt="Rich Features"></a>
    <a href="#-examples"><img src="https://img.shields.io/badge/Practical-Examples-orange?style=flat-square" alt="Practical Examples"></a>
  </p>
</div>

> [!NOTE]
> SwiftCLI is a powerful framework for building beautiful, feature-rich command-line applications with minimal code. It combines simplicity with advanced features like command groups, progress bars, and rich output formatting.

## 📋 Table of Contents

- [🚀 Installation](#-installation)
- [🔥 Quick Start](#-quick-start)
- [✨ Features](#-features)
  - [🎯 Command Groups](#-command-groups)
  - [🎨 Beautiful Tables](#-beautiful-tables)
  - [⚡ Progress Bars](#-progress-bars)
  - [🔧 Command Options](#-command-options)
  - [🌍 Environment Variables](#-environment-variables)
  - [⚙️ Config Files](#️-config-files)
- [🌟 Advanced Features](#-advanced-features)
  - [🔗 Command Chaining](#-command-chaining)
  - [🎮 Context Passing](#-context-passing)
  - [🔌 Plugin System](#-plugin-system)
- [📚 Examples](#-examples)
  - [🚀 Deployment CLI](#-deployment-cli)
  - [📊 Data Processing](#-data-processing)
  - [🔍 Search Tool](#-search-tool)
- [💡 Best Practices](#-best-practices)
- [📘 API Reference](#-api-reference)

## 🚀 Installation

SwiftCLI is included in the Webscout package. Install or update Webscout to get access:

```bash
pip install -U webscout
```

## 🔥 Quick Start

Create a simple CLI application in just a few lines of code:

```python
from webscout import CLI

# Create your application
app = CLI(name="my-app", help="My awesome CLI application")

# Add a command
@app.command()
def greet(name: str):
    """Say hello to someone"""
    print(f"Hello {name}!")

# Run the application
app.run()
```

Usage:
```bash
$ python app.py greet Alice
Hello Alice!

$ python app.py --help
Usage: app.py [OPTIONS] COMMAND [ARGS]...

  My awesome CLI application

Options:
  --help  Show this message and exit.

Commands:
  greet  Say hello to someone
```

## ✨ Features

### 🎯 Command Groups

Organize related commands into logical groups:

```python
@app.group()
def db():
    """Database management commands"""
    pass

@db.command()
def migrate():
    """Run database migrations"""
    print("Running migrations... 🚀")

@db.command()
def backup():
    """Backup the database"""
    print("Creating backup... 💾")
```

Usage:
```bash
$ python app.py db migrate
Running migrations... 🚀

$ python app.py db backup
Creating backup... 💾
```

### 🎨 Beautiful Tables

Display data in well-formatted tables:

```python
@app.command()
@table_output(headers=["ID", "Name", "Status"])
def users():
    """List all users"""
    return [
        [1, "John", "Active"],
        [2, "Jane", "Inactive"],
        [3, "Bob", "Active"]
    ]
```

Output:
```
┌────┬──────┬──────────┐
│ ID │ Name │ Status   │
├────┼──────┼──────────┤
│ 1  │ John │ Active   │
│ 2  │ Jane │ Inactive │
│ 3  │ Bob  │ Active   │
└────┴──────┴──────────┘
```

### ⚡ Progress Bars

Show progress for long-running operations:

```python
@app.command()
@progress("Processing files")
def process(files: list):
    """Process multiple files"""
    for file in files:
        # Your processing logic here
        yield f"Processing {file}..."
```

Output:
```
Processing files [######################] 100% | ETA: 00:00:00 | 3/3
✓ Processing file1.txt...
✓ Processing file2.txt...
✓ Processing file3.txt...
```

### 🔧 Command Options

Add flexible options to your commands:

```python
@app.command()
@option("--count", "-c", type=int, default=1, help="Number of items to export")
@option("--format", "-f", type=str, choices=["json", "yaml", "csv"], help="Output format")
@option("--verbose", is_flag=True, help="Enable verbose output")
def export(count: int, format: str, verbose: bool):
    """Export data with options"""
    if verbose:
        print(f"Verbose mode enabled")
    print(f"Exporting {count} items as {format}")
```

Usage:
```bash
$ python app.py export --count 5 --format json --verbose
Verbose mode enabled
Exporting 5 items as json
```

### 🌍 Environment Variables

Load configuration from environment variables:

```python
@app.command()
@envvar("API_KEY", required=True)
@envvar("API_URL", default="https://api.example.com")
def api_call(api_key: str, api_url: str):
    """Make API call using environment variables"""
    print(f"Connecting to: {api_url}")
    print(f"Using API key: {api_key[:4]}...")
```

Usage:
```bash
$ export API_KEY=my-secret-key
$ python app.py api_call
Connecting to: https://api.example.com
Using API key: my-s...
```

### ⚙️ Config Files

Load settings from configuration files:

```python
@app.command()
@config_file("~/.myapp/config.json")
def setup(config: dict):
    """Setup using configuration file"""
    print(f"Database: {config.get('database', 'default')}")
    print(f"Debug mode: {config.get('debug', False)}")
```

Example config.json:
```json
{
  "database": "postgres",
  "debug": true,
  "api_keys": {
    "service1": "key1",
    "service2": "key2"
  }
}
```

## 🌟 Advanced Features

### 🔗 Command Chaining

Run multiple commands in sequence:

```python
@app.group(chain=True)
def process():
    """Process data pipeline"""
    pass

@process.command()
def validate():
    """Validate input data"""
    print("Validating... ✅")
    return {"valid": True}

@process.command()
def transform():
    """Transform data"""
    print("Transforming... 🔄")
    return {"transformed": True}

@process.command()
def load():
    """Load processed data"""
    print("Loading... 📥")
    return {"loaded": True}
```

Usage:
```bash
$ python app.py process validate transform load
Validating... ✅
Transforming... 🔄
Loading... 📥
```

### 🎮 Context Passing

Access application context in your commands:

```python
@app.command()
@pass_context
def status(ctx):
    """Get application status with context"""
    print(f"Application: {ctx.cli.name}")
    print(f"Version: {ctx.cli.version}")
    print(f"Debug mode: {ctx.cli.debug}")
    print(f"Available commands: {len(ctx.cli.commands)}")
```

### 🔌 Plugin System

Extend functionality with plugins:

```python
class LoggingPlugin(Plugin):
    def before_command(self, command, args):
        print(f"[LOG] Running command: {command}")
        print(f"[LOG] Arguments: {args}")

    def after_command(self, command, args, result):
        print(f"[LOG] Command {command} completed")
        if result:
            print(f"[LOG] Result: {result}")

# Register your plugin
app.plugin_manager.register(LoggingPlugin())
```

## 📚 Examples

### 🚀 Deployment CLI

Create a deployment tool with progress tracking:

```python
@app.group()
def deploy():
    """Deployment commands"""
    pass

@deploy.command()
@option("--env", type=str, default="dev", help="Deployment environment")
@option("--force", is_flag=True, help="Force deployment")
@progress("Deploying application")
def start(env: str, force: bool):
    """Start deployment process"""
    if force:
        yield "Force mode enabled, skipping checks..."
    
    yield "Building application..."
    time.sleep(1)
    
    yield "Running tests..."
    time.sleep(1)
    
    yield f"Deploying to {env} environment..."
    time.sleep(1)
    
    yield "Updating documentation..."
    time.sleep(0.5)
    
    yield "Deployment complete!"
```

### 📊 Data Processing

Build a data processing tool with table output:

```python
@app.command()
@option("--input", "-i", type=str, required=True, help="Input file path")
@option("--output", "-o", type=str, required=True, help="Output file path")
@option("--format", "-f", type=str, default="csv", help="Output format")
@table_output(headers=["File", "Records", "Status", "Time"])
def process(input: str, output: str, format: str):
    """Process data files"""
    results = []
    
    # Process files (simplified example)
    files = glob.glob(input)
    for file in files:
        # Your processing logic here
        record_count = random.randint(100, 1000)
        process_time = f"{random.randint(1, 10)}s"
        results.append([file, record_count, "Success", process_time])
    
    # Write to output file
    print(f"\nProcessed {len(files)} files")
    print(f"Results written to {output} in {format} format")
    
    return results
```

### 🔍 Search Tool

Create a file search utility:

```python
@app.command()
@option("--path", "-p", type=str, default=".", help="Search path")
@option("--pattern", type=str, required=True, help="Search pattern")
@option("--recursive", "-r", is_flag=True, help="Search recursively")
@table_output(headers=["File", "Size", "Modified"])
def search(path: str, pattern: str, recursive: bool):
    """Search for files matching pattern"""
    results = []
    
    # Determine search method
    search_path = path
    if recursive:
        search_pattern = f"{search_path}/**/{pattern}"
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = f"{search_path}/{pattern}"
        files = glob.glob(search_pattern)
    
    # Process results
    for file in files:
        stats = os.stat(file)
        size = f"{stats.st_size / 1024:.1f} KB"
        modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")
        results.append([file, size, modified])
    
    print(f"\nFound {len(results)} files matching '{pattern}'")
    return results
```

## 💡 Best Practices

<details>
<summary><strong>1. Use Docstrings for Help Text</strong></summary>

```python
@app.command()
def analyze(file: str, verbose: bool = False):
    """
    Analyze a file and generate report.
    
    The analysis includes file statistics, content summary,
    and potential issues found in the file.
    
    Arguments:
        file: Path to the file to analyze
        verbose: Enable detailed output
    """
    # Command implementation
```
</details>

<details>
<summary><strong>2. Leverage Rich Output</strong></summary>

```python
from rich.console import Console
from rich.panel import Panel

console = Console()

@app.command()
def status():
    """Show system status"""
    console.print(Panel.fit(
        "[bold green]System Status: Online[/]",
        title="Status Check",
        border_style="green"
    ))
    
    console.print("[bold]Components:[/]")
    console.print("  [green]✓[/] Database")
    console.print("  [green]✓[/] API Server")
    console.print("  [yellow]⚠[/] Cache Server")
```
</details>

<details>
<summary><strong>3. Implement Shell Completion</strong></summary>

```python
@app.command()
@option("--service", type=str, required=True, help="Service to restart")
def restart(service: str):
    """Restart a system service"""
    print(f"Restarting {service}...")

@restart.completion()
def complete_service(ctx, incomplete):
    """Provide completion for available services"""
    services = ["nginx", "apache", "mysql", "redis", "mongodb"]
    return [s for s in services if s.startswith(incomplete)]
```
</details>

<details>
<summary><strong>4. Use Subcommands for Complex CLIs</strong></summary>

```python
# Main CLI
app = CLI(name="devops", help="DevOps toolkit")

# User management group
@app.group()
def users():
    """User management commands"""
    pass

@users.command()
def list():
    """List all users"""
    pass

@users.command()
def add(username: str):
    """Add a new user"""
    pass

# Server management group
@app.group()
def servers():
    """Server management commands"""
    pass

@servers.command()
def status():
    """Check server status"""
    pass

@servers.command()
def deploy():
    """Deploy to servers"""
    pass
```
</details>

<details>
<summary><strong>5. Handle Errors Gracefully</strong></summary>

```python
@app.command()
def risky_operation():
    """Perform a risky operation"""
    try:
        # Risky code here
        result = perform_operation()
        print(f"Operation completed successfully: {result}")
    except FileNotFoundError:
        console.print("[bold red]Error:[/] Required file not found")
        return 1
    except PermissionError:
        console.print("[bold red]Error:[/] Insufficient permissions")
        return 2
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {str(e)}")
        if app.debug:
            import traceback
            traceback.print_exc()
        return 99
```
</details>

## 📘 API Reference

<details>
<summary><strong>CLI Class</strong></summary>

```python
CLI(
    name: str,                 # Application name
    help: str = None,          # Application description
    version: str = "0.1.0",    # Application version
    debug: bool = False        # Enable debug mode
)

# Methods
CLI.command()                  # Register a command
CLI.group()                    # Create a command group
CLI.run()                      # Run the application
```
</details>

<details>
<summary><strong>Decorators</strong></summary>

```python
# Command decorators
@app.command()                 # Register a function as a command
@app.group()                   # Create a command group

# Option decorators
@option(name, type=None, required=False, default=None, help=None)
@argument(name, type=None, required=True, help=None)
@flag(name, help=None)         # Boolean flag option

# Output decorators
@table_output(headers=None)    # Format return value as table
@progress(description=None)    # Show progress bar for generator commands

# Configuration decorators
@envvar(name, required=False, default=None)  # Load from environment variable
@config_file(path)             # Load from config file
@pass_context                  # Pass CLI context to command
```
</details>

<details>
<summary><strong>Plugin System</strong></summary>

```python
class Plugin:
    def before_command(self, command, args):
        """Called before command execution"""
        pass
        
    def after_command(self, command, args, result):
        """Called after command execution"""
        pass
        
    def on_error(self, command, args, error):
        """Called when command raises an exception"""
        pass

# Register a plugin
app.plugin_manager.register(MyPlugin())
```
</details>

<div align="center">
  <p>
    <a href="https://github.com/OEvortex/Webscout"><img alt="GitHub Repository" src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href="https://t.me/PyscoutAI"><img alt="Telegram Group" src="https://img.shields.io/badge/Telegram%20Group-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  </p>
</div>
  "api_keys": {
    "service1": "key1",
    "service2": "key2"
  }
}
```

## 🌟 Advanced Features

### 🔗 Command Chaining

Run multiple commands in sequence:

```python
@app.group(chain=True)
def process():
    """Process data pipeline"""
    pass

@process.command()
def validate():
    """Validate input data"""
    print("Validating... ✅")
    return {"valid": True}

@process.command()
def transform():
    """Transform data"""
    print("Transforming... 🔄")
    return {"transformed": True}

@process.command()
def load():
    """Load processed data"""
    print("Loading... 📥")
    return {"loaded": True}
```

Usage:
```bash
$ python app.py process validate transform load
Validating... ✅
Transforming... 🔄
Loading... 📥
```

### 🎮 Context Passing

Access application context in your commands:

```python
@app.command()
@pass_context
def status(ctx):
    """Get application status with context"""
    print(f"Application: {ctx.cli.name}")
    print(f"Version: {ctx.cli.version}")
    print(f"Debug mode: {ctx.cli.debug}")
    print(f"Available commands: {len(ctx.cli.commands)}")
```

### 🔌 Plugin System

Extend functionality with plugins:

```python
class LoggingPlugin(Plugin):
    def before_command(self, command, args):
        print(f"[LOG] Running command: {command}")
        print(f"[LOG] Arguments: {args}")

    def after_command(self, command, args, result):
        print(f"[LOG] Command {command} completed")
        if result:
            print(f"[LOG] Result: {result}")

# Register your plugin
app.plugin_manager.register(LoggingPlugin())
```

## 📚 Examples

### 🚀 Deployment CLI

Create a deployment tool with progress tracking:

```python
@app.group()
def deploy():
    """Deployment commands"""
    pass

@deploy.command()
@option("--env", type=str, default="dev", help="Deployment environment")
@option("--force", is_flag=True, help="Force deployment")
@progress("Deploying application")
def start(env: str, force: bool):
    """Start deployment process"""
    if force:
        yield "Force mode enabled, skipping checks..."
    
    yield "Building application..."
    time.sleep(1)
    
    yield "Running tests..."
    time.sleep(1)
    
    yield f"Deploying to {env} environment..."
    time.sleep(1)
    
    yield "Updating documentation..."
    time.sleep(0.5)
    
    yield "Deployment complete!"
```

### 📊 Data Processing

Build a data processing tool with table output:

```python
@app.command()
@option("--input", "-i", type=str, required=True, help="Input file path")
@option("--output", "-o", type=str, required=True, help="Output file path")
@option("--format", "-f", type=str, default="csv", help="Output format")
@table_output(headers=["File", "Records", "Status", "Time"])
def process(input: str, output: str, format: str):
    """Process data files"""
    results = []
    
    # Process files (simplified example)
    files = glob.glob(input)
    for file in files:
        # Your processing logic here
        record_count = random.randint(100, 1000)
        process_time = f"{random.randint(1, 10)}s"
        results.append([file, record_count, "Success", process_time])
    
    # Write to output file
    print(f"\nProcessed {len(files)} files")
    print(f"Results written to {output} in {format} format")
    
    return results
```

### 🔍 Search Tool

Create a file search utility:

```python
@app.command()
@option("--path", "-p", type=str, default=".", help="Search path")
@option("--pattern", type=str, required=True, help="Search pattern")
@option("--recursive", "-r", is_flag=True, help="Search recursively")
@table_output(headers=["File", "Size", "Modified"])
def search(path: str, pattern: str, recursive: bool):
    """Search for files matching pattern"""
    results = []
    
    # Determine search method
    search_path = path
    if recursive:
        search_pattern = f"{search_path}/**/{pattern}"
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = f"{search_path}/{pattern}"
        files = glob.glob(search_pattern)
    
    # Process results
    for file in files:
        stats = os.stat(file)
        size = f"{stats.st_size / 1024:.1f} KB"
        modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")
        results.append([file, size, modified])
    
    print(f"\nFound {len(results)} files matching '{pattern}'")
    return results
```

## 💡 Best Practices

<details>
<summary><strong>1. Use Docstrings for Help Text</strong></summary>

```python
@app.command()
def analyze(file: str, verbose: bool = False):
    """
    Analyze a file and generate report.
    
    The analysis includes file statistics, content summary,
    and potential issues found in the file.
    
    Arguments:
        file: Path to the file to analyze
        verbose: Enable detailed output
    """
    # Command implementation
```
</details>

<details>
<summary><strong>2. Leverage Rich Output</strong></summary>

```python
from rich.console import Console
from rich.panel import Panel

console = Console()

@app.command()
def status():
    """Show system status"""
    console.print(Panel.fit(
        "[bold green]System Status: Online[/]",
        title="Status Check",
        border_style="green"
    ))
    
    console.print("[bold]Components:[/]")
    console.print("  [green]✓[/] Database")
    console.print("  [green]✓[/] API Server")
    console.print("  [yellow]⚠[/] Cache Server")
```
</details>

<details>
<summary><strong>3. Implement Shell Completion</strong></summary>

```python
@app.command()
@option("--service", type=str, required=True, help="Service to restart")
def restart(service: str):
    """Restart a system service"""
    print(f"Restarting {service}...")

@restart.completion()
def complete_service(ctx, incomplete):
    """Provide completion for available services"""
    services = ["nginx", "apache", "mysql", "redis", "mongodb"]
    return [s for s in services if s.startswith(incomplete)]
```
</details>

<details>
<summary><strong>4. Use Subcommands for Complex CLIs</strong></summary>

```python
# Main CLI
app = CLI(name="devops", help="DevOps toolkit")

# User management group
@app.group()
def users():
    """User management commands"""
    pass

@users.command()
def list():
    """List all users"""
    pass

@users.command()
def add(username: str):
    """Add a new user"""
    pass

# Server management group
@app.group()
def servers():
    """Server management commands"""
    pass

@servers.command()
def status():
    """Check server status"""
    pass

@servers.command()
def deploy():
    """Deploy to servers"""
    pass
```
</details>

<details>
<summary><strong>5. Handle Errors Gracefully</strong></summary>

```python
@app.command()
def risky_operation():
    """Perform a risky operation"""
    try:
        # Risky code here
        result = perform_operation()
        print(f"Operation completed successfully: {result}")
    except FileNotFoundError:
        console.print("[bold red]Error:[/] Required file not found")
        return 1
    except PermissionError:
        console.print("[bold red]Error:[/] Insufficient permissions")
        return 2
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {str(e)}")
        if app.debug:
            import traceback
            traceback.print_exc()
        return 99
```
</details>

## 📘 API Reference

<details>
<summary><strong>CLI Class</strong></summary>

```python
CLI(
    name: str,                 # Application name
    help: str = None,          # Application description
    version: str = "0.1.0",    # Application version
    debug: bool = False        # Enable debug mode
)

# Methods
CLI.command()                  # Register a command
CLI.group()                    # Create a command group
CLI.run()                      # Run the application
```
</details>

<details>
<summary><strong>Decorators</strong></summary>

```python
# Command decorators
@app.command()                 # Register a function as a command
@app.group()                   # Create a command group

# Option decorators
@option(name, type=None, required=False, default=None, help=None)
@argument(name, type=None, required=True, help=None)
@flag(name, help=None)         # Boolean flag option

# Output decorators
@table_output(headers=None)    # Format return value as table
@progress(description=None)    # Show progress bar for generator commands

# Configuration decorators
@envvar(name, required=False, default=None)  # Load from environment variable
@config_file(path)             # Load from config file
@pass_context                  # Pass CLI context to command
```
</details>

<details>
<summary><strong>Plugin System</strong></summary>

```python
class Plugin:
    def before_command(self, command, args):
        """Called before command execution"""
        pass
        
    def after_command(self, command, args, result):
        """Called after command execution"""
        pass
        
    def on_error(self, command, args, error):
        """Called when command raises an exception"""
        pass

# Register a plugin
app.plugin_manager.register(MyPlugin())
```
</details>

<div align="center">
  <p>
    <a href="https://github.com/OEvortex/Webscout"><img alt="GitHub Repository" src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href="https://t.me/PyscoutAI"><img alt="Telegram Group" src="https://img.shields.io/badge/Telegram%20Group-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"></a>
  </p>
</div>
