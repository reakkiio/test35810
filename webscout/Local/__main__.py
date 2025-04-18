"""
Entry point for running webscout.Local as a module.
This allows running the CLI using 'python -m webscout.Local'.
"""

from webscout.Local.cli import app

if __name__ == "__main__":
    app()
