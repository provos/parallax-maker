#!/usr/bin/env python3
"""
Test script to verify the parallax-maker package installation.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, description, capture_output=True, timeout=120):
    """Run a command and return success status."""
    print(f"Testing: {description}")
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            print("✅ Success")
            if capture_output and result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print("❌ Failed")
            if capture_output and result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("⏱️ Timeout")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    """Test the package installation."""
    print("🧪 Testing Parallax Maker Package Installation")
    print("=" * 50)

    # Find the wheel file
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist/ directory found. Run 'python -m build' first.")
        sys.exit(1)

    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel files found in dist/")
        sys.exit(1)

    wheel_file = wheel_files[0]
    print(f"📦 Found wheel: {wheel_file}")

    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_dir = Path(temp_dir) / "test_env"

        print(f"\n🔧 Creating test environment in {venv_dir}")

        # Create virtual environment
        if not run_command(
            f"python -m venv {venv_dir}", "Creating virtual environment"
        ):
            return False

        # Determine activation script
        if sys.platform == "win32":
            activate_script = venv_dir / "Scripts" / "activate"
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            activate_script = venv_dir / "bin" / "activate"
            python_exe = venv_dir / "bin" / "python"

        # Upgrade pip first
        upgrade_cmd = f"{python_exe} -m pip install --upgrade pip"
        if not run_command(upgrade_cmd, "Upgrading pip"):
            return False

        # Install the wheel with dependencies
        install_cmd = f"{python_exe} -m pip install {wheel_file.absolute()}"
        if not run_command(
            install_cmd, "Installing parallax-maker wheel", capture_output=False
        ):
            return False

        # Test importing the package
        import_cmd = f'{python_exe} -c "import parallax_maker; print(f\\"Parallax Maker v{{parallax_maker.__version__}} imported successfully\\")"'
        if not run_command(import_cmd, "Testing package import"):
            return False

        # Test CLI commands
        help_cmd = f"{python_exe} -c \"import sys; sys.argv=['parallax-maker', '--help']; from parallax_maker.webui import main; main()\""
        if not run_command(help_cmd, "Testing parallax-maker --help"):
            return False

        gltf_help_cmd = f"{python_exe} -c \"import sys; sys.argv=['parallax-gltf-cli', '--help']; from parallax_maker.gltf_cli import main; main()\""
        if not run_command(gltf_help_cmd, "Testing parallax-gltf-cli --help"):
            return False

        print("\n🎉 All tests passed!")
        print("📦 Package is ready for PyPI upload!")
        print("\nNext steps:")
        print(
            "1. Upload to TestPyPI: python -m twine upload --repository testpypi dist/*"
        )
        print(
            "2. Test from TestPyPI: pip install --index-url https://test.pypi.org/simple/ parallax-maker"
        )
        print("3. Upload to PyPI: python -m twine upload dist/*")

        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
