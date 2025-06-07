import importlib.metadata
from typing import Optional, Dict, Any, Literal
import requests
from packaging import version

# Constants
PYPI_URL = "https://pypi.org/pypi/webscout/json"
YOUTUBE_URL = "https://youtube.com/@OEvortex"

# Create a session for HTTP requests
session = requests.Session()

# Version comparison result type
VersionCompareResult = Literal[-1, 0, 1]

def get_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package by name."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

def get_installed_version() -> Optional[str]:
    """Get the currently installed version of webscout.

    Returns:
        Optional[str]: The installed version string or None if not found

    Examples:
        >>> version = get_installed_version()
        >>> print(version)
        '1.2.3'
    """
    return get_package_version('webscout')

def get_pypi_version() -> Optional[str]:
    """Get the latest version available on PyPI.

    Returns:
        Optional[str]: The latest version string or None if retrieval failed

    Examples:
        >>> latest = get_pypi_version()
        >>> print(latest)
        '2.0.0'
    """
    try:
        response = session.get(PYPI_URL, timeout=10)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data.get('info', {}).get('version')
    except (requests.RequestException, KeyError, ValueError):
        return None

def version_compare(v1: str, v2: str) -> VersionCompareResult:
    """Compare two version strings.

    Returns:
        -1 if v1 < v2, 1 if v1 > v2, 0 if equal or invalid

    Examples:
        >>> version_compare('1.0.0', '2.0.0')
        -1
    """
    try:
        version1 = version.parse(v1)
        version2 = version.parse(v2)
        if version1 < version2:
            return -1
        if version1 > version2:
            return 1
        return 0
    except Exception:
        return 0

def get_update_message(installed: str, latest: str) -> str:
    """Generate appropriate update message based on version comparison."""
    comparison_result = version_compare(installed, latest)
    youtube_msg = f'\033[1;32mSubscribe to my YouTube Channel: {YOUTUBE_URL}\033[0m'
    if comparison_result < 0:
        # Bold red for update available
        return (
            f"\033[1;31mNew Webscout version available: {latest} "
            f"- Update with: pip install --upgrade webscout\033[0m\n{youtube_msg}"
        )
    elif comparison_result > 0:
        # Bold yellow for dev version
        return (
            f"\033[1;33mYou're running a development version ({installed}) "
            f"ahead of latest release ({latest})\033[0m\n{youtube_msg}"
        )
    # Bold green for up-to-date

def check_for_updates() -> Optional[str]:
    """Check if a newer version of Webscout is available.

    Returns:
        Optional[str]: Update message if newer version exists, None otherwise
    """
    installed_version = get_installed_version()
    if not installed_version:
        return "\033[1;31mWebscout is not installed.\033[0m"

    latest_version = get_pypi_version()
    if not latest_version:
        return "\033[1;31mCould not retrieve latest version from PyPI.\033[0m"

    return get_update_message(installed_version, latest_version)

if __name__ == "__main__":
    try:
        update_message = check_for_updates()
        if update_message:
            print(update_message)
    except KeyboardInterrupt:
        print("\nUpdate check canceled by user.")
    except Exception as e:
        print(f"\033[1;31mUpdate check failed: {str(e)}\033[0m")