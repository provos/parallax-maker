# Building and Publishing Parallax Maker to PyPI

This document provides instructions for building wheel packages and publishing to PyPI.

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Configure PyPI credentials (one-time setup):
```bash
pip install keyring
# For PyPI
python -m keyring set https://upload.pypi.org/legacy/ your-username
# For TestPyPI
python -m keyring set https://test.pypi.org/legacy/ your-username
```

## Building the Package

1. Clean any previous builds:
```bash
rm -rf dist/ build/ *.egg-info
```

2. Build Tailwind CSS assets (if changed):
```bash
npm run build
```

3. Build the wheel and source distribution:
```bash
python -m build
```

This will create:
- `dist/parallax_maker-*.whl` (wheel package)
- `dist/parallax-maker-*.tar.gz` (source distribution)

## Testing the Package Locally

1. Create a test environment:
```bash
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
```

2. Install the wheel:
```bash
pip install dist/parallax_maker-*.whl
```

3. Test the installation:
```bash
parallax-maker --help
parallax-gltf-cli --help
```

## Publishing to PyPI

### Test on TestPyPI first (recommended)

1. Upload to TestPyPI:
```bash
python -m twine upload --repository testpypi dist/*
```

2. Test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ parallax-maker
```

### Upload to PyPI

1. Upload to PyPI:
```bash
python -m twine upload dist/*
```

2. Verify the upload at https://pypi.org/project/parallax-maker/

## Version Management

Update the version in `pyproject.toml` before building:

```toml
[project]
version = "1.0.2"  # Update this
```

## Automated Publishing with GitHub Actions

Consider setting up GitHub Actions for automated publishing. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Notes

- The package name is `parallax-maker` (with hyphen) but the Python module is `parallax_maker` (with underscore)
- Entry points are configured for both the web UI (`parallax-maker`) and CLI tool (`parallax-gltf-cli`)
- All assets (CSS, JS) are included via `MANIFEST.in`
