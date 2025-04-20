"""
Model management for webscout.local
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shutil

from rich.console import Console
from rich.prompt import Prompt
from huggingface_hub import hf_hub_download, HfFileSystem

from .config import config

console = Console()

class ModelManager:
    """
    Manager for downloading and managing models.
    Handles model download, listing, removal, and path resolution.
    """
    models_dir: Path

    def __init__(self) -> None:
        self.models_dir = config.models_dir

    def parse_model_string(self, model_string: str) -> Tuple[str, Optional[str]]:
        """
        Parse a model string in the format 'repo_id:filename' or just 'repo_id'.
        Args:
            model_string (str): The model string to parse.
        Returns:
            Tuple[str, Optional[str]]: (repo_id, filename)
        """
        if ":" in model_string:
            repo_id, filename = model_string.split(":", 1)
            return repo_id, filename
        else:
            return model_string, None

    def list_repo_gguf_files(self, repo_id: str) -> List[str]:
        """
        List all GGUF files in a repository.
        Args:
            repo_id (str): The Hugging Face repository ID.
        Returns:
            List[str]: List of filenames.
        """
        fs = HfFileSystem()
        try:
            files = fs.ls(repo_id, detail=False)
            gguf_files = [os.path.basename(f) for f in files if f.endswith(".gguf")]
            return gguf_files
        except Exception as e:
            console.print(f"[bold red]Error listing files in repository {repo_id}: {str(e)}[/bold red]")
            return []

    def select_file_interactive(self, repo_id: str) -> Optional[str]:
        """
        Interactively select a file from a repository.
        Args:
            repo_id (str): The Hugging Face repository ID.
        Returns:
            Optional[str]: Selected filename or None if cancelled.
        """
        gguf_files = self.list_repo_gguf_files(repo_id)
        if not gguf_files:
            console.print(f"[bold red]No GGUF files found in repository {repo_id}[/bold red]")
            return None
        console.print(f"[bold blue]Available GGUF files in {repo_id}:[/bold blue]")
        for i, filename in enumerate(gguf_files):
            console.print(f"  [{i+1}] {filename}")
        choice = Prompt.ask(
            "Select a file to download (number or filename)",
            default="1"
        )
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(gguf_files):
                return gguf_files[idx]
        except ValueError:
            if choice in gguf_files:
                return choice
        console.print(f"[bold red]Invalid selection: {choice}[/bold red]")
        return None

    def download_model(self, model_string: str, filename: Optional[str] = None) -> Tuple[str, Path]:
        """
        Download a model from Hugging Face Hub.
        Args:
            model_string (str): The model string in format 'repo_id' or 'repo_id:filename'.
            filename (Optional[str]): Specific filename to download, overrides filename in model_string.
        Returns:
            Tuple[str, Path]: (model_name, model_path)
        """
        repo_id, file_from_string = self.parse_model_string(model_string)
        filename = filename or file_from_string
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        model_dir = config.get_model_path(model_name)
        model_dir.mkdir(exist_ok=True, parents=True)
        model_info: Dict[str, Any] = {
            "repo_id": repo_id,
            "name": model_name,
            "downloaded_at": datetime.datetime.now().isoformat(),
        }
        with open(model_dir / "info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        if not filename:
            console.print(f"[yellow]No filename provided, searching for GGUF files in {repo_id}...[/yellow]")
            filename = self.select_file_interactive(repo_id)
            if not filename:
                raise ValueError(f"No GGUF file selected from repository {repo_id}")
            console.print(f"[green]Selected GGUF file: {filename}[/green]")
        console.print(f"[bold blue]Downloading {filename} from {repo_id}...[/bold blue]")
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
            )
        except Exception as e:
            console.print(f"[bold red]Error downloading file: {str(e)}[/bold red]")
            raise
        console.print(f"[bold green]Model downloaded to {model_path}[/bold green]")
        model_info["filename"] = filename
        model_info["path"] = str(model_path)
        with open(model_dir / "info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        return model_name, Path(model_path)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a downloaded model.
        Args:
            model_name (str): Name of the model.
        Returns:
            Optional[Dict[str, Any]]: Model info dict or None if not found.
        """
        model_dir = config.get_model_path(model_name)
        info_file = model_dir / "info.json"
        if not info_file.exists():
            return None
        with open(info_file, "r") as f:
            return json.load(f)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded models with their information.
        Returns:
            List[Dict[str, Any]]: List of model info dicts.
        """
        models: List[Dict[str, Any]] = []
        seen_paths: set = set()
        if not config.models_dir.exists():
            return []
        model_dirs = [d for d in config.models_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            if ":" in model_dir.name:
                continue
            info_file = model_dir / "info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        info = json.load(f)
                    if "path" in info and info["path"] in seen_paths:
                        continue
                    if "path" in info:
                        seen_paths.add(info["path"])
                    models.append(info)
                except Exception:
                    pass
        return models

    def remove_model(self, model_name: str) -> bool:
        """
        Remove a downloaded model.
        Args:
            model_name (str): Name of the model to remove.
        Returns:
            bool: True if removed, False if not found.
        """
        model_dir = config.get_model_path(model_name)
        if not model_dir.exists():
            return False
        shutil.rmtree(model_dir)
        return True

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the path to a model file.
        Args:
            model_name (str): Name or filename of the model.
        Returns:
            Optional[str]: Path to the model file or None if not found.
        """
        info = self.get_model_info(model_name)
        if not info or "path" not in info:
            for model_info in self.list_models():
                if model_info.get("filename") == model_name:
                    return model_info.get("path")
            return None
        return info["path"]

    def copy_model(self, source_model: str, destination_model: str) -> bool:
        """
        Copy a model to a new name.
        Args:
            source_model (str): Name of the source model.
            destination_model (str): Name for the destination model.
        Returns:
            bool: True if copied successfully, False otherwise.
        """
        # Get source model info
        source_info = self.get_model_info(source_model)
        if not source_info or "path" not in source_info:
            console.print(f"[bold red]Source model {source_model} not found[/bold red]")
            return False

        # Create destination directory
        dest_dir = config.get_model_path(destination_model)
        dest_dir.mkdir(exist_ok=True, parents=True)

        # Copy the model file
        source_path = Path(source_info["path"])
        dest_path = dest_dir / source_path.name

        try:
            console.print(f"[bold blue]Copying model from {source_path} to {dest_path}...[/bold blue]")
            shutil.copy2(source_path, dest_path)

            # Create info file for the destination model
            dest_info = source_info.copy()
            dest_info["name"] = destination_model
            dest_info["path"] = str(dest_path)
            dest_info["copied_from"] = source_model
            dest_info["copied_at"] = datetime.datetime.now().isoformat()

            with open(dest_dir / "info.json", "w") as f:
                json.dump(dest_info, f, indent=2)

            console.print(f"[bold green]Model copied successfully to {dest_path}[/bold green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error copying model: {str(e)}[/bold red]")
            # Clean up if there was an error
            if dest_path.exists():
                dest_path.unlink()
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            return False
