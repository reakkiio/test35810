"""
Command-line interface for webscout.Local
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from typing import Optional

from .model_manager import ModelManager
from .llm import LLMInterface
from .server import start_server

app: typer.Typer = typer.Typer(help="webscout.Local - A llama-cpp-python based LLM serving tool")
console: Console = Console()

model_manager: ModelManager = ModelManager()

# RAM requirements for different model sizes
RAM_REQUIREMENTS = {
    "1B": "2 GB",
    "3B": "4 GB",
    "7B": "8 GB",
    "13B": "16 GB",
    "33B": "32 GB",
    "70B": "64 GB",
}

@app.command("serve")
def run_model(
    model_string: str = typer.Argument(..., help="Model to run (format: 'name', 'repo_id' or 'repo_id:filename')"),
    host: Optional[str] = typer.Option(None, help="Host to bind the server to"),
    port: Optional[int] = typer.Option(None, help="Port to bind the server to"),
) -> None:
    """
    Start a model server (downloads if needed).
    """
    # First check if this is a filename that already exists
    model_path = model_manager.get_model_path(model_string)
    if model_path:
        # This is a filename that exists, find the model name
        for model_info in model_manager.list_models():
            if model_info.get("filename") == model_string or model_info.get("path") == model_path:
                model_name = model_info.get("name")
                break
        else:
            # Fallback to using the string as model name
            model_name = model_string
    else:
        # Parse the model string to see if it's a repo_id:filename format
        repo_id, _ = model_manager.parse_model_string(model_string)
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Check if model exists, if not try to download it
        if not model_manager.get_model_path(model_name):
            console.print(f"[yellow]Model {model_name} not found locally. Attempting to download...[/yellow]")
            try:
                # We don't need to use the parsed values directly as download_model handles this
                _ = model_manager.parse_model_string(model_string)  # Just to validate the format
                # Download the model
                model_name, _ = model_manager.download_model(model_string)
                console.print(f"[bold green]Model {model_name} downloaded successfully[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
                return

    # Check RAM requirements
    ram_requirement = "Unknown"
    for size, ram in RAM_REQUIREMENTS.items():
        if size in model_name:
            ram_requirement = ram
            break

    if ram_requirement != "Unknown":
        console.print(f"[yellow]This model requires approximately {ram_requirement} of RAM[/yellow]")

    # Try to load the model to verify it works
    try:
        llm = LLMInterface(model_name)
        llm.load_model(verbose=False)
        console.print(f"[bold green]Model {model_name} loaded successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
        return

    # Start the server
    console.print(f"[bold blue]Starting webscout.Local server with model {model_name}...[/bold blue]")
    start_server(host=host, port=port)

@app.command("pull")
def pull_model(
    model_string: str = typer.Argument(..., help="Model to download (format: 'repo_id' or 'repo_id:filename')"),
) -> None:
    """
    Download a model from Hugging Face without running it.
    """
    try:
        model_name, model_path = model_manager.download_model(model_string)
        console.print(f"[bold green]Model {model_name} downloaded successfully to {model_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")

@app.command("list")
def list_models() -> None:
    """
    List downloaded models.
    """
    models = model_manager.list_models()

    if not models:
        console.print("[yellow]No models found. Use 'webscout.Local pull' to download a model.[/yellow]")
        return

    table = Table(title="Downloaded Models")
    table.add_column("Name", style="cyan")
    table.add_column("Repository", style="green")
    table.add_column("Filename", style="blue")
    table.add_column("Size", style="magenta")
    table.add_column("Downloaded", style="yellow")

    for model in models:
        # Get file size in human-readable format
        file_path = model.get("path")
        file_size = "Unknown"
        if file_path:
            try:
                import os
                size_bytes = os.path.getsize(file_path)
                # Convert to human-readable format
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if size_bytes < 1024.0 or unit == 'TB':
                        file_size = f"{size_bytes:.2f} {unit}"
                        break
                    size_bytes /= 1024.0
            except Exception:
                pass

        # Format downloaded date
        downloaded_at = model.get("downloaded_at", "Unknown")
        if downloaded_at != "Unknown":
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(downloaded_at)
                downloaded_at = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        table.add_row(
            model["name"],
            model.get("repo_id", "Unknown"),
            model.get("filename", "Unknown"),
            file_size,
            downloaded_at,
        )

    console.print(table)

@app.command(name="remove", help="Remove a downloaded model")
def remove_model(
    model_string: str = typer.Argument(..., help="Name or filename of the model to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation"),
) -> None:
    """
    Remove a downloaded model.
    """
    # First check if this is a model name
    model_info = model_manager.get_model_info(model_string)

    # If not found by name, check if it's a filename
    if not model_info:
        for info in model_manager.list_models():
            if info.get("filename") == model_string:
                model_info = info
                model_string = info["name"]
                break

    if not model_info:
        console.print(f"[yellow]Model {model_string} not found.[/yellow]")
        return

    if not force:
        confirm = Prompt.ask(
            f"Are you sure you want to remove model {model_string}?",
            choices=["y", "n"],
            default="n",
        )

        if confirm.lower() != "y":
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    if model_manager.remove_model(model_string):
        console.print(f"[bold green]Model {model_string} removed successfully[/bold green]")
    else:
        console.print(f"[bold red]Error removing model {model_string}[/bold red]")

@app.command("run")
def chat(
    model_string: str = typer.Argument(..., help="Name or filename of the model to chat with"),
) -> None:
    """
    Interactive chat with a model.
    """
    # First check if this is a filename that already exists
    model_path = model_manager.get_model_path(model_string)
    if model_path:
        # This is a filename that exists, find the model name
        for model_info in model_manager.list_models():
            if model_info.get("filename") == model_string or model_info.get("path") == model_path:
                model_name = model_info.get("name")
                break
        else:
            # Fallback to using the string as model name
            model_name = model_string
    else:
        # Use the string as model name
        model_name = model_string

        # Check if model exists, if not try to download it
        if not model_manager.get_model_path(model_name):
            console.print(f"[yellow]Model {model_name} not found locally. Attempting to download...[/yellow]")
            try:
                # Parse the model string to see if it's a repo_id:filename format
                # We don't need to use the parsed values directly as download_model handles this
                _ = model_manager.parse_model_string(model_string)  # Just to validate the format
                # Download the model
                model_name, _ = model_manager.download_model(model_string)
                console.print(f"[bold green]Model {model_name} downloaded successfully[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
                return

    # Check RAM requirements
    ram_requirement = "Unknown"
    for size, ram in RAM_REQUIREMENTS.items():
        if size in model_name:
            ram_requirement = ram
            break

    if ram_requirement != "Unknown":
        console.print(f"[yellow]This model requires approximately {ram_requirement} of RAM[/yellow]")

    # Load the model
    try:
        llm = LLMInterface(model_name)
        llm.load_model(verbose=False)
    except Exception as e:
        console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
        return

    console.print(f"[bold green]Chat with {model_name}. Type '/help' for available commands or '/bye' to exit.[bold green]")

    # Chat history
    messages = []
    system_prompt = None

    # Initialize with empty system prompt
    messages.append({"role": "system", "content": ""})

    # Define help text
    help_text = """
    Available commands:
    /help or /? - Show this help message
    /bye - Exit the chat
    /set system <prompt> - Set the system prompt
    /set context <size> - Set context window size (default: 4096)
    /clear or /cls - Clear the terminal screen
    /reset - Reset all settings
    """

    while True:
        # Get user input
        user_input = input("\n> ")

        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=2)
            cmd = cmd_parts[0].lower()

            if cmd == "/bye" or user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd == "/help" or cmd == "/?":
                console.print(help_text)
                continue

            elif cmd == "/clear" or cmd == "/cls":
                # Do not clear history, just clear the terminal screen
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                console.print(f"[bold green]Chat with {model_name}. Type '/help' for available commands or '/bye' to exit.[/bold green]")
                console.print("[yellow]Screen cleared. Chat history preserved.[/yellow]")
                continue

            elif cmd == "/reset":
                messages = [{"role": "system", "content": ""}]
                system_prompt = None
                console.print("[yellow]All settings reset.[/yellow]")
                continue

            elif cmd == "/set" and len(cmd_parts) >= 2:
                if len(cmd_parts) < 3:
                    console.print("[red]Error: Missing value for setting[/red]")
                    continue

                setting = cmd_parts[1].lower()
                value = cmd_parts[2]

                if setting == "system":
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]

                    system_prompt = value
                    # Update system message
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        # Clear messages and add system prompt
                        messages = [{"role": "system", "content": system_prompt}]

                    # Print confirmation that it's been applied
                    console.print(f"[yellow]System prompt set to:[/yellow]")
                    console.print(f"[cyan]\"{system_prompt}\"[/cyan]")
                    console.print(f"[green]System prompt applied. Next responses will follow this instruction.[/green]")

                    # Force a test message to ensure the system prompt is applied
                    test_messages = messages.copy()
                    test_messages.append({"role": "user", "content": "Say 'System prompt active'."})

                    # Test if the system prompt is working
                    console.print("[dim]Testing system prompt...[/dim]")
                    response = llm.create_chat_completion(
                        messages=test_messages,
                        stream=False,
                        max_tokens=20
                    )
                    console.print("[dim]System prompt test complete.[/dim]")
                elif setting == "context":
                    try:
                        context_size = int(value)
                        # Reload the model with new context size
                        console.print(f"[yellow]Reloading model with context size: {context_size}...[/yellow]")
                        llm.load_model(n_ctx=context_size, verbose=False)
                        console.print(f"[green]Context size set to: {context_size}[/green]")
                    except ValueError:
                        console.print(f"[red]Invalid context size: {value}. Must be an integer.[/red]")
                else:
                    console.print(f"[red]Unknown setting: {setting}[/red]")
                continue
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Generate response
        console.print("\n")  # Add extra spacing between user input and response

        # Use a buffer to collect the response
        response_buffer = ""

        def print_token(token):
            nonlocal response_buffer
            response_buffer += token
            console.print(token, end="", highlight=False)

        llm.stream_chat_completion(
            messages=messages,
            callback=print_token,
        )

        # Get the full response to add to history
        response = llm.create_chat_completion(
            messages=messages,
            stream=False,
        )

        assistant_message = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_message})

        # Add extra spacing after the response
        console.print("")

@app.command("copy")
def copy_model(
    source: str = typer.Argument(..., help="Name of the source model"),
    destination: str = typer.Argument(..., help="Name for the destination model"),
) -> None:
    """
    Copy a model to a new name.
    """
    try:
        if model_manager.copy_model(source, destination):
            console.print(f"[bold green]Model {source} copied to {destination} successfully[/bold green]")
        else:
            console.print(f"[bold red]Failed to copy model {source} to {destination}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error copying model: {str(e)}[/bold red]")

@app.command("show")
def show_model(
    model_name: str = typer.Argument(..., help="Name of the model to show information for"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """
    Show detailed information about a model.
    """
    model_info = model_manager.get_model_info(model_name)

    if not model_info:
        console.print(f"[yellow]Model {model_name} not found.[/yellow]")
        return

    # Create a table for basic information
    table = Table(title=f"Model Information: {model_name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Add basic properties
    table.add_row("Name", model_info["name"])
    table.add_row("Repository", model_info.get("repo_id", "Unknown"))
    table.add_row("Filename", model_info.get("filename", "Unknown"))

    # Get file size in human-readable format
    file_path = model_info.get("path")
    if file_path:
        try:
            import os
            size_bytes = os.path.getsize(file_path)
            # Convert to human-readable format
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0 or unit == 'TB':
                    file_size = f"{size_bytes:.2f} {unit}"
                    break
                size_bytes /= 1024.0
            table.add_row("Size", file_size)
        except Exception:
            table.add_row("Size", "Unknown")

    # Format downloaded date
    downloaded_at = model_info.get("downloaded_at", "Unknown")
    if downloaded_at != "Unknown":
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(downloaded_at)
            downloaded_at = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    table.add_row("Downloaded", downloaded_at)

    # Add copied information if available
    if "copied_from" in model_info:
        table.add_row("Copied From", model_info["copied_from"])
        copied_at = model_info.get("copied_at", "Unknown")
        if copied_at != "Unknown":
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(copied_at)
                copied_at = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
        table.add_row("Copied At", copied_at)

    # Estimate RAM requirements based on model name
    ram_requirement = "Unknown"
    for size, ram in RAM_REQUIREMENTS.items():
        if size in model_name:
            ram_requirement = ram
            break
    table.add_row("Estimated RAM", ram_requirement)

    # Print the table
    console.print(table)

    # If verbose, show all properties
    if verbose:
        console.print("\n[bold]Detailed Information:[/bold]")
        for key, value in model_info.items():
            if key not in ["name", "repo_id", "filename", "path", "downloaded_at", "copied_from", "copied_at"]:
                console.print(f"[cyan]{key}:[/cyan] {value}")

@app.command("ps")
def list_running_models() -> None:
    """
    List running models.
    """
    from .server import loaded_models

    if not loaded_models:
        console.print("[yellow]No models currently running.[/yellow]")
        return

    table = Table(title="Running Models")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")

    for name in loaded_models.keys():
        table.add_row(name, "Running")

    console.print(table)

@app.command("version")
def version() -> None:
    """
    Show version information.
    """
    from webscout.Local import __version__
    console.print(f"[bold]webscout.Local[/bold] version [cyan]{__version__}[/cyan]")
    console.print("A llama-cpp-python based LLM serving tool")

if __name__ == "__main__":
    app()
