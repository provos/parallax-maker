# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parallax Maker is a Python-based computer vision and AI tool that converts 2D images into 2.5D animations using depth estimation, segmentation, and inpainting. It features a web UI built with Dash and exports to glTF for 3D applications.

## Development Commands

### Setup and Dependencies
```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .[dev]

# Install TailwindCSS dependencies (only if modifying styles)
npm install -D tailwindcss
```

### Running the Application
```bash
# Start the web UI
parallax-maker

# Or run the module directly
python -m parallax_maker.webui

# Prefetch models to avoid download delays
parallax-maker --prefetch-models=default
```

### Development Workflow
```bash
# Run tests with coverage
pytest

# Lint code
flake8 . --max-line-length=127

# Format code
black .

# Type checking
mypy parallax_maker

# Build TailwindCSS (when modifying styles)
npm run build

# Watch TailwindCSS changes during development
npm run watch
```

### Running Individual Tests
```bash
# Run specific test file
pytest parallax_maker/test_filename.py

# Run specific test
pytest parallax_maker/test_filename.py::test_function_name
```

## Architecture Overview

### Core Components

1. **Web UI (webui.py)**: Main Dash application that orchestrates the workflow
   - Handles image loading, depth generation, segmentation, inpainting, and export
   - State management via AppState class

2. **Depth Estimation (depth.py)**: Supports multiple models
   - MiDaS, DINOv2, ZoeDepth for depth map generation
   - Configurable model selection and parameters

3. **Segmentation (segmentation.py)**: Interactive image segmentation
   - Segment Anything Model (SAM) with point-based selection
   - Manual card creation and depth manipulation

4. **Inpainting (inpainting.py)**: Multiple backends for mask inpainting
   - Local: Stable Diffusion XL, SD3 Medium
   - Remote: Automatic1111 (automatic1111.py), ComfyUI (comfyui.py)
   - API: StabilityAI

5. **3D Export (gltf.py)**: glTF 2.0 scene generation
   - Creates depth-based card arrangements
   - Supports depth displacement for realistic geometry
   - Command-line tool: parallax-gltf-cli

### Key Design Patterns

- **AppState**: Central state management for the entire workflow
- **Component Registry**: UI components registered in components.py
- **Model Management**: Lazy loading of AI models to optimize memory
- **Callback Architecture**: Dash callbacks handle all UI interactions

### Data Flow

1. Image loaded → Depth model generates depth map
2. Depth map → Segmentation creates cards/slices
3. Cards → Inpainting fills masked regions
4. Processed cards → glTF export or GIF animation

### File Organization

- `parallax_maker/`: Main package directory
  - `assets/`: CSS and JavaScript files
  - `components.py`: All Dash UI components
  - `controller.py`: Main workflow orchestration
  - `constants.py`: Configuration and defaults
  - Model-specific files: depth.py, segmentation.py, inpainting.py
  - Integration files: automatic1111.py, comfyui.py
  - Export: gltf.py, gltf_cli.py

### State Persistence

- Application state saved to `appstate-*` directories
- Contains processed images, depth maps, masks, and metadata
- JSON serialization for state data

## Important Considerations

- First-time model downloads can take several minutes
- GPU recommended for reasonable performance
- Memory usage scales with image size and model complexity
- TailwindCSS compilation required only when modifying styles
- SD3 Medium requires latest diffusers from GitHub