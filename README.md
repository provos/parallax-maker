![Hacking On Parallax Maker](https://raw.githubusercontent.com/provos/parallax-maker/main/example/hacking.gif)

# Installation and First Usage

## Prerequisites
- Python 3.10, 3.11, or 3.12
- pip (for package management)

## Prerequisites (additional, for building the web UI)
- Node.js 20+ and npm (only needed to build the bundled Svelte frontend; a
  pip install of a released package/wheel already includes a prebuilt copy)

## Installation Methods

Create a new environment with Python 3.10+, install the project in
development mode, and build the Svelte frontend it serves:

```bash
# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install the project and dependencies
pip install -e .

# Build the Svelte frontend into parallax_maker/static/app/ (skip this if
# you installed from a released wheel, which already ships a built copy)
npm --prefix frontend ci
npm run build:frontend
```

## Running the Application
After installation, you can start the application using the entry point:

```bash
# Using the installed entry point
parallax-maker

# Or run the module directly
python -m parallax_maker.server
```

You can then reach the web UI via [http://127.0.0.1:8050/](http://127.0.0.1:8050/). Be prepared that the first time, any new functionality is used, the corresponding models need to be downloaded first. This can take a few minutes based on your connection speed. If you want to prefetch the default models, you can start the application with:

```bash
parallax-maker --prefetch-models=default
```

`parallax-maker` runs as a single process: it serves the JSON API (under
`/api/v1`) and the built Svelte app (everything else) from one Flask server,
using an in-memory per-project registry and job queue that is not shared
across worker processes.

## Development workflow

For frontend development with hot reload, run the Python backend and the
Vite dev server side by side. Vite proxies `/api` and `/__e2e__` requests to
the backend (see `frontend/vite.config.ts`), so the browser only ever talks
to one origin:

```bash
# Terminal 1: the backend (serves /api/v1; the /static/app UI it also
# serves is irrelevant here since Vite serves its own dev copy instead)
parallax-maker --port 8050

# Terminal 2: the frontend dev server with hot module reload
npm --prefix frontend run dev
```

Then open the URL Vite prints (typically http://localhost:5173/).

Useful commands while working on the frontend:

```bash
npm run check:frontend   # svelte-check + tsc
npm run test:frontend    # Vitest unit tests
npm run build:frontend   # production build into parallax_maker/static/app/
```

Useful commands while working on the backend:

```bash
pytest                              # Python unit/service/API tests
flake8 . --max-line-length=127      # lint
black .                             # format
```

Browser end-to-end tests (Playwright, driving the real Svelte UI against a
deterministic fake-model backend) live in `e2e/`; see `e2e/README.md` for
details:

```bash
npx playwright install chromium
npm run test:e2e
```

# Parallax-Maker

Provides a workflow for turning images into 2.5D animation like the one seen above.

## Features
 - Segmentation of images
   - Using depth models like Midas or ZeoDepth
   - Using instance segmentation via Segment Anything with multiple positive and negative point selection
   - Adding and removing of cards, direct manipulation of depth values
 - Inpainting
   - Inpainting of masks that can be padded and blurred
   - Replacing the masked regions with new images via image generation models like Stable Diffusion 1.0 XL, Stable Diffusion 3 Medium, Automatic1111 or ComyfUI endpoints as well as the StabilityAI API.
 - 3D Export
   - Generation of glTF scenes that can be imported into Blender or Unreal Engine
   - Support for depth displacement of cards to generate more realistic 3D geometry
   - In browser 3D preview of the generated glTF scene.

## Basic Examples

Using an input image, the tool runs a depth model like **Midas** or **DINOv2** to generate a depth map

![Input Image](https://raw.githubusercontent.com/provos/parallax-maker/main/example/input_plus_depth.png)

and then creates cards that can be used for 2.5 parallax animation.

![Animation](https://raw.githubusercontent.com/provos/parallax-maker/main/example/output.gif)

This animation was created using the following command:

~~~
ffmpeg -framerate 24 -i rendered_image_%03d.png -filter_complex "fps=5,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" output.gif 
~~~


# 3D Export

The tool also supports generating a glTF2.0 scene file that an be easily imported into 3D apps like Blender or Unreal Engine.

> [!TIP]
> To utilize depth of field camera effects for the Blender scene, the material needs to be changed to **ALPHA HASHED**.

> [!TIP]
> To utilize depth of field camera effects for Unreal Engine, the material needs to be changed to **Translucent Masked**.


![Blender Scene View](https://raw.githubusercontent.com/provos/parallax-maker/main/example/blender_view.png)


# Web UI

![Web UI](https://raw.githubusercontent.com/provos/parallax-maker/main/example/webui.jpg)

A Svelte 5 based Web UI (backed by a Flask/HTTP API) provides a browser assisted workflow to generate slices from images, inpaint the slices and then export them as a glTF scene to Blender or Unreal Engine. The resulting glTF scene can also be visualized within the app or manipulated via a command line tool and the state file saved by the app.

The UI is organized into workflow tabs: **Mode** (upload an image, pick a
depth model, adjust thresholds), **Segmentation** (depth/instance-point
selection, slice creation and editing, mask tools), **Inpainting** (paint a
mask, generate/fill/enhance candidates, apply or erase), **Export** (camera
and displacement settings, glTF/animation export, texture upscaling) and
**Configuration** (depth/inpainting model selection, external server/API-key
setup). A 2D/3D viewer toggle previews the input image or the exported glTF
scene in-browser.

![Web UI 3D Example](https://raw.githubusercontent.com/provos/parallax-maker/main/example/webui_3d.jpg)

# Advanced Use Cases
Parallax Maker also supports the Automatic1111 and ComfyUI API endpoints. This allows the tool to utilize GPUs remotely and potentially achieve much higher performance compared to the local GPU. It also means that it's possible to use more specialized inpainting models and workflows. Here is [an example](https://raw.githubusercontent.com/provos/parallax-maker/main/example/workflow.json) ComfyUI inpainting workflow that makes use the offset lora published by Stability AI.

![Example configuration for ComfyUI](https://raw.githubusercontent.com/provos/parallax-maker/main/example/external_config.png)

# Watch the Video
[![Watch the video](https://raw.githubusercontent.com/provos/parallax-maker/main/example/thumb.png)](https://www.youtube.com/watch?v=4JBQCz-wWYQ)

# Tutorials
## Segmentation and Inpainting Tutorial
[![Segmentation and Inpainting Tutorial](https://raw.githubusercontent.com/provos/parallax-maker/main/example/inpainting-thumb.jpg)](https://youtu.be/hb_x8z4WIeI)
## Unreal Engine Import and Rendering Tutorial
[![Unreal Import and Rendering Tutorial](https://raw.githubusercontent.com/provos/parallax-maker/main/example/unreal-thumb.jpg)](https://www.youtube.com/watch?v=fLSCCS53h_U)