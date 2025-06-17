#!/usr/bin/env python3
"""
Script to fix absolute imports to relative imports in the parallax_maker package.
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """Fix absolute imports to relative imports in a single file."""
    print(f"Fixing imports in {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    original_content = content

    # List of modules that should be relative imports
    local_modules = [
        "utils",
        "segmentation",
        "upscaler",
        "stabilityai",
        "camera",
        "slice",
        "controller",
        "constants",
        "components",
        "clientside",
        "depth",
        "instance",
        "inpainting",
        "gltf",
        "automatic1111",
        "comfyui",
    ]

    # Fix "import module" patterns
    for module in local_modules:
        # Replace "import module" with "from . import module"
        pattern = f"^import {module}$"
        replacement = f"from . import {module}"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Replace "import module as alias" with "from . import module as alias"
        pattern = f"^import {module} as (.+)$"
        replacement = f"from . import {module} as \\1"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Fix "from module import ..." patterns
    for module in local_modules:
        # Replace "from module import ..." with "from .module import ..."
        pattern = f"^from {module} import (.+)$"
        replacement = f"from .{module} import \\1"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Special case for constants imports
    content = re.sub(
        r"^import constants as C$",
        "from . import constants as C",
        content,
        flags=re.MULTILINE,
    )

    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"  ✅ Updated {file_path}")
        return True
    else:
        print(f"  ➖ No changes needed in {file_path}")
        return False


def main():
    """Fix all imports in the parallax_maker package."""
    package_dir = Path("parallax_maker")

    if not package_dir.exists():
        print("❌ parallax_maker directory not found!")
        return

    python_files = list(package_dir.glob("*.py"))
    print(f"Found {len(python_files)} Python files")

    updated_files = 0
    for py_file in python_files:
        # Skip __init__.py and test files for now
        if py_file.name.startswith("__") or py_file.name.startswith("test_"):
            continue

        if fix_imports_in_file(py_file):
            updated_files += 1

    print(f"\n✅ Updated {updated_files} files")
    print("🏁 Import fixing complete!")


if __name__ == "__main__":
    main()
