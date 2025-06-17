#!/usr/bin/env python3
"""
Setup script for Parallax Maker development and packaging.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True


def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")
    dirs_to_remove = ["dist", "build", "*.egg-info"]
    for pattern in dirs_to_remove:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                path.unlink()
                print(f"Removed file: {path}")


def build_package():
    """Build the package."""
    print("Building package...")
    return run_command("python -m build", "Building wheel and source distribution")


def check_package():
    """Check the built package."""
    print("Checking package...")
    return run_command("python -m twine check dist/*", "Checking package integrity")


def install_dev_tools():
    """Install development tools."""
    print("Installing development tools...")
    tools = ["build", "twine", "pytest", "black", "flake8"]
    for tool in tools:
        if not run_command(f"pip install {tool}", f"Installing {tool}"):
            return False
    return True


def run_tests():
    """Run tests."""
    print("Running tests...")
    return run_command("python -m pytest", "Running test suite")


def format_code():
    """Format code with black."""
    print("Formatting code...")
    return run_command("python -m black .", "Formatting code with black")


def main():
    """Main setup function."""
    if len(sys.argv) < 2:
        print("Usage: python setup_dev.py <command>")
        print("Commands:")
        print("  install-tools  - Install development tools")
        print("  clean         - Clean build artifacts")
        print("  build         - Build the package")
        print("  check         - Check the package")
        print("  test          - Run tests")
        print("  format        - Format code")
        print("  full-build    - Clean, build, and check")
        print("  dev-setup     - Full development setup")
        sys.exit(1)

    command = sys.argv[1]

    if command == "install-tools":
        install_dev_tools()
    elif command == "clean":
        clean_build()
    elif command == "build":
        build_package()
    elif command == "check":
        check_package()
    elif command == "test":
        run_tests()
    elif command == "format":
        format_code()
    elif command == "full-build":
        clean_build()
        if build_package():
            check_package()
    elif command == "dev-setup":
        install_dev_tools()
        format_code()
        run_tests()
        clean_build()
        if build_package():
            check_package()
            print("\n✅ Development setup complete!")
            print("📦 Package built successfully!")
            print("\nNext steps:")
            print("1. Test install: pip install dist/*.whl")
            print(
                "2. Upload to TestPyPI: python -m twine upload --repository testpypi dist/*"
            )
            print("3. Upload to PyPI: python -m twine upload dist/*")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
