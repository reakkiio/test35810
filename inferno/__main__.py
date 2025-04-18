"""
Entry point for running inferno as a module.
This allows running the CLI using 'python -m inferno'.
"""

from webscout.Local.cli import app

if __name__ == "__main__":
    app()
