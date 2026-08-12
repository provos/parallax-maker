# Parallax Maker Fork: Architecture & Codebase Analysis Report

This report evaluates the existing **Parallax Maker** codebase and designs a migration path to convert it into a specialized cinematic image-to-video / 2.5D parallax production tool for Indian history, architecture, temple, and ancient knowledge documentaries on YouTube.

---

## Part 1: Technical & Architectural Analysis

### 1. Repository Structure
The repository is structured as a Python package containing a Dash-based Web UI and several command-line utility entry points:
- `/` (Root directory): Contains configuration and installation/packaging files (`pyproject.toml`, `requirements.txt`, `Dockerfile`, `package.json`, `setup_dev.py`, etc.).
- `/example/`: Contains sample inputs, depth maps, workflow configurations (`workflow.json`), and output media gifs/images showing the 2.5D parallax results.
- `/parallax_maker/`: Core source directory containing the Python modules:
  - `webui.py`: Dash application containing layout configuration, entry points, and central web server routing.
  - `components.py`: Modular Dash/HTML/Tailwind CSS components and Dash callback logic.
  - `constants.py` (`C`): Centralized string IDs for Dash elements, stores, and inputs to avoid callback mismatch.
  - `controller.py`: Houses the central `AppState` class for project/state serialization and image caching.
  - `slice.py`: Defines the `ImageSlice` class representing an individual layered card (with source image, depth, filename, prompts, etc.).
  - `camera.py`: Encapsulates camera positioning, focal lengths, and camera projection matrices.
  - `depth.py`: Handles depth estimation via local/remote models (MiDaS, ZoeDepth, DinoV2).
  - `segmentation.py`: Implements depth-map histogram analysis and slicing operations.
  - `instance.py`: Implements point/coordinate-guided instance segmentation via Segment Anything (SAM) or Mask2Former.
  - `inpainting.py`: Local and remote inpainting pipeline (Stable Diffusion, FLUX, ComfyUI, Automatic1111).
  - `gltf.py` & `gltf_cli.py`: Creates 3D scene files (glTF) for Blender, Houdini, or Unreal Engine.
  - `upscaler.py`: Local tiled upscaling using external models.
  - `utils.py`: Helper functions for color tinting, feathering, and coordinate conversion.
  - `automatic1111.py`, `comfyui.py`, `falai.py`, `stabilityai.py`: Clients and API integrations for remote backend providers.
  - `test_*.py`: Pytest suite files validating individual modules.
  - `assets/`: Folder hosting static JS files (`utility.js`) and compiled Tailwind stylesheet (`tailwind_compiled.css`).

### 2. Application Entry Points
There are two main entry points registered in `pyproject.toml`:
- **Web UI Application:** Registered to `parallax-maker` (`python -m parallax_maker.webui`), which starts the local Flask/Dash web server on port 8050.
- **gltf Command-Line Interface:** Registered to `parallax-gltf-cli` (`python -m parallax_maker.gltf_cli`), allowing batch glTF file creations directly from saved state directories.

### 3. Current UI Architecture
The UI is built with **Plotly Dash** using **Tailwind CSS** classes. The layout inside `webui.py` is structured around a multi-column and tabbed panel system:
- **Viewer Tabs (2D vs 3D Preview):** Displays either the interactive 2D canvas with event listeners for mouse interactions or an `Iframe` containing an HTML-based 3D glTF Model Viewer (`model-viewer` script).
- **Control Tabs:** Sliced into collapsible sidebar tabs:
  1. *Mode Selector:* Choose standard interaction mode (Slicing vs SAM Instance Segmentation).
  2. *Segmentation:* Configure threshold sliders and initiate automatic depth-based slicing.
  3. *Inpainting:* Handle prompts, strengths, mask padding, blur, and execute the inpainting run.
  4. *Export:* Create glTF files or compile/render direct 2.5D push-in animation frame sequences.
  5. *Configuration:* Select local or remote models and configure external endpoints (ComfyUI / Automatic1111).
- **Communication Flow:** Managed via a clientside Javascript script (`utility.js`) and Python Dash callbacks in `components.py` communicating via transient or persistent `dcc.Store` JSON payloads.

### 4. Image Loading Pipeline
- **Input:** Users upload an image using a Dash `dcc.Upload` drag-and-drop component.
- **Conversion:** Base64-encoded strings are decoded, converted to a PIL Image, and forced into RGB mode (`imgData`).
- **Disk Cache:** The input image is written immediately to `tmp-images/<AppState-filename>/input_image.png` and served to the browser via a customized Flask route.
- **Tints Precomputation:** Grayscale and color-tinted preview copies are generated immediately to support interactive hover overlays.

### 5. Depth Estimation Pipeline
- **Models Supported:** `midas` (DPT_Large via PyTorch Hub), `zoedepth` (ZoeD_NK via PyTorch Hub), and `dinov2` (facebook/dpt-dinov2-large-nyu via Hugging Face Transformers).
- **Model Loading:** Dynamically loaded via the `load_model()` method on demand. Device selection is determined via `utils.torch_get_device()`, prioritizing CUDA (GPU), then MPS (Apple Silicon), falling back to CPU.
- **Inference:** The input image is converted to a NumPy array, downsized to a max of 1024x1024 (using bicubic interpolation for DinoV2), and run through PyTorch.
- **Output & Storage:** The raw tensor output is normalized into a 0-255 grayscale NumPy array, inverted so that farthest objects are black (0) and nearest objects are white (255), and stored both in-memory (`state.depthMapData`) and on disk (`depth_map.png`).

### 6. Segmentation Pipeline
- **Slicing Models:** Supports both depth-threshold histogram slicing and direct user click-guided instance segmentation via **Segment Anything (SAM)** (`facebook/sam-vit-huge`).
- **Mask Creation:** Slicing works by applying `cv2.inRange` filters between calculated histogram thresholds. For SAM, coordinates are mapped from UI clicks, processed via `SamProcessor` into positive/negative points, and input to SAM to generate a precise 1-channel binary mask.
- **User Interaction:** Users click points on the canvas. Standard clicks denote positive points, Ctrl+clicks denote negative points, and Shift+clicks aggregate/subtract masks.
- **Mask Persistence:** Binary masks are compiled with a feathering operation using Gaussian blur and stored as the alpha channel of an `ImageSlice` object.

### 7. Layer/Card Generation
- **Representation:** Handled by `ImageSlice` (`slice.py`).
- **3D Card Calculations:** Each slice calculates its 3D depth coordinate $z$ based on its average depth value mapped to `Camera.max_distance`.
- **Card Sizing:** Slices are transformed into 3D quadrilateral meshes by projecting 2D card width and height coordinates back through the camera's focal length and distance using the formulas:
  $$\text{width} = \frac{\text{img\_width} \cdot (z + \text{cam\_dist})}{\text{focal\_length\_px}}$$
  $$\text{height} = \frac{\text{img\_height} \cdot (z + \text{cam\_dist})}{\text{focal\_length\_px}}$$
- Slices can also represent ground planes, which are calculated with customized tilted near-and-far boundary projections.

### 8. Inpainting Pipeline
- **Implementation:** Employs localized `patch_image` preprocessing utilizing Numba JIT compiling to approximate background color fills under masked layers. This is followed by a diffusion-based inpainting step to generate clean background content.
- **Supported Backends:** Local Diffusers models (Stable Diffusion v1.5, SDXL, Stable Diffusion 3 Medium, FLUX.1 Fill), API backends (StabilityAI, Fal.ai models), or self-hosted servers (Automatic1111 / ComfyUI workflows).
- **Local vs External:** Highly modular. Local execution runs locally on PyTorch with Diffusers, while API execution requests content over REST APIs or WebSocket networks.

### 9. 2.5D/Parallax Rendering
- **Implementation:** Implemented in `segmentation.py` (`render_view` & `render_image_sequence`).
- **Warping Engine:** Utilizes OpenCV's perspective warping (`cv2.projectPoints` to project corners and `cv2.warpPerspective` to warp the flat textures).
- **Alpha Compositing:** Layer transparency is merged sequentially from back to front using alpha blending in OpenCV/NumPy.

### 10. Camera/Scene Manipulation
- **Representation:** The camera parameters (`Camera`) track position (3D array), focal length (scalar), distance to target (scalar), and maximum clip depth (scalar).
- **Focal Length:** Tracked in metric units (equivalent to mm) and mapped to screen pixels through standard sensor-width ratios ($35\text{ mm}$ default).
- **Motion:** The camera position vector is translated along the $z$-axis during direct sequences to simulate simple linear "push-in/pull-out" animations.

### 11. glTF/3D Export
- **Exporting Engine:** Built using `pygltflib` inside `gltf.py`.
- **3D Geometry Slices:** Generates a full glTF 2.0 structure. Plane meshes are subdivided (up to 500x500 subdivisions) to accommodate depth mesh displacement mapping.
- **Materials:** Creates emissive, double-sided materials with a PBR metallic-roughness setup. Alpha channel blending modes default to `"BLEND"` or `"MASK"`.
- **Packaging:** Can pack references as standalone external links or inline base64-encoded URI strings.

### 12. State/Project Persistence
- **Structure:** State is saved inside a single JSON state file (`appstate.json`) nested within a dedicated local folder (`appstate-<8-char-hash>/`).
- **Associated Media:** This folder houses all related assets (`input_image.png`, `depth_map.png`, and several numbered `image_slice_i.png` layers).
- **Encryption:** Remote API keys are encoded using a simple nonce matching the folder's name to prevent accidental plain-text exposures in project shares.

### 13. Existing Image Formats Supported
- **Input:** Standard web formats (PNG, JPEG, WebP, BMP) processed via Pillow.
- **Intermediate Layers:** Alpha-enabled 4-channel PNGs and 1-channel grayscale PNG depth representations.
- **3D Assets:** Standard `.gltf` text formats referencing inline or local binary attachments.

### 14. Existing Animation/Export Capabilities
- **Direct Video Export:** Currently, the application writes sequential `.png` frames to disk.
- **Video Assembly:** The CLI relies on a separate local `ffmpeg` shell call command outlined in the documentation to convert generated frame folders into animated `.gif` or `.mp4` videos.

### 15. Performance Bottlenecks
- **GPU/VRAM Latencies:** Running multiple Hugging Face Transformer pipelines (DinoV2 depth estimation, SAM segmentation, and FLUX/SDXL inpainting) concurrently on consumer-grade GPUs triggers VRAM out-of-memory errors or slow system swap behavior.
- **Rendering Throughput:** Rendering animations in python via single-threaded OpenCV perspective warping (`cv2.warpPerspective`) is relatively slow for high-resolution images, limiting real-time feedback.
- **Disk I/O:** Serializing large, uncompressed high-bitrate canvas state layers slows down UI callbacks.

### 16. GPU/VRAM Requirements
- **Local Depth/SAM:** Minimum 6GB VRAM.
- **Local Diffusion Inpainting (SDXL/SD3):** Minimum 10GB VRAM.
- **Local FLUX Inpainting:** Minimum 16GB VRAM (ideally 24GB).
- *Mitigation:* The tool supports remote endpoints (StabilityAI, Fal.ai, ComfyUI, Automatic1111) to offload VRAM usage from the local machine.

### 17. CPU Fallback Behavior
- **Torch Operations:** Automatically switches torch devices to `"cpu"` if CUDA/MPS are unavailable, causing depth map generation and segmentation calculations to become extremely slow (taking minutes per operation).
- **Local Inpainting:** Extremely slow on CPU fallback; local image patching (numba JIT) remains fast because it is optimized for multi-threaded CPU execution.

### 18. Large-Image Handling
- **Downscaling Safeguards:** DinoV2 downscales large images to a maximum of 1024x1024 before inference to keep VRAM usage manageable.
- **Inpainting Scaling:** Diffusers/API-based inpainting automatically downscales slices to 1024x1024 or 512x512 depending on model requirements, then rescales back using LANCZOS interpolation.

### 19. Existing Tests
- Uses `pytest` with a dedicated suite checking automatic payload calculations, upscaling components, state file serializers, depth pipelines, and Dash callbacks.
- Includes integration mock files verifying `webui.py` callbacks and endpoint connection configurations.

### 20. Existing Architectural Weaknesses
- **State Coupling:** AppState manages file paths, active layers, hardware parameters, UI state, and API credentials in a single coupled class.
- **Limited Animation Controls:** The camera system only supports simple linear push-in movements, lacks support for keyframes, and cannot handle complex panning, rotations, or multi-point paths.
- **Synchronous Rendering:** Video frame generation is executed synchronously within the main thread of Flask/Dash callbacks, which can trigger gateway timeouts on larger export workloads.

### 21. Components to Reuse Unchanged
- **`instance.py`:** SAM instance segmentation mechanics are highly refined and work reliably.
- **`depth.py`:** Excellent wrapper for loading and running depth models on multiple devices.
- **`gltf.py`:** Solid generation of 3D geometry and metadata packing.
- **`upscaler.py` / `inpainting.py`:** Provide stable connections to local and remote backends.

### 22. Components to Wrap behind Adapters
- **`camera.py`:** Needs an adapter to map custom cinematic motion paths (such as Bézier, spline-based, or keyframed curves) to the existing 3D camera properties.
- **`slice.py`:** Needs wrapping to inject customizable layers like atmospheric fog or particle overlay textures into the 3D depth layers.

### 23. Components to Replace Eventually
- **`segmentation.py` warper/renderer:** The CPU-based NumPy/OpenCV compositing system should be upgraded to a GPU-accelerated canvas engine (such as PyOpenGL or ModernGL) to enable real-time viewport feedback at 4K resolution.
- **`controller.py` State Manager:** The tightly-coupled `AppState` should be refactored into a decoupled, lightweight Scene Graph model.

### 24. Components Missing for Documentarian Workflow
- **Cinematic Timeline & Keyframing:** Lacks an editor to create camera pan, zoom, tilt, and rotation keyframes.
- **Atmospheric Layer System:** Missing options to add cinematic elements like light leaks, dust particles, and volumetric fog into the depth stack.
- **Direct Video Compiler:** No built-in automated compiler to generate high-quality `.mp4` videos directly within the UI, currently requiring external manual `ffmpeg` scripts.
- **Preset System:** Lacks quick options to export scenes in standard documentary formats (e.g., YouTube 16:9 4K at 24fps, or Shorts 9:16 vertical).

---

## Part 2: Migration Map

The following map defines how the codebase should evolve to support cinematic documentary production:

```
CURRENT COMPONENT                               KEEP / WRAP / MODIFY / REPLACE                       ROLE IN NEW APPLICATION
-----------------------------------------------------------------------------------------------------------------------------------------
parallax_maker/instance.py                      KEEP (Unchanged)                                     Provides SAM-driven object isolation.
parallax_maker/depth.py                         KEEP (Unchanged)                                     Generates accurate 16-bit depth cards.
parallax_maker/gltf.py                          KEEP (Unchanged)                                     Handles 3D geometry compilation.
parallax_maker/slice.py                         WRAP                                                 Enables complex layered card tracking.
parallax_maker/camera.py                        MODIFY                                               Adds 6-DOF coordinates and spline motions.
parallax_maker/controller.py (AppState)         MODIFY                                               Manages a decoupled, lightweight Scene Graph.
parallax_maker/segmentation.py (Renderer)       REPLACE (Later)                                      Upgrades CPU warping to GPU-accelerated rendering.
[NEW] parallax_maker/timeline.py                NEW COMPONENT                                        Manages keyframes and camera paths.
[NEW] parallax_maker/atmosphere.py              NEW COMPONENT                                        Injects fog, light leaks, and dust cards.
[NEW] parallax_maker/video_compiler.py          NEW COMPONENT                                        Compiles frames into standard MP4 outputs.
```

### Minimum Component Pipeline Diagram

```
[IMAGE INPUT]
      │
      ├───────────────────────┐
      ▼                       ▼
[DEPTH ESTIMATION]    [SAM SEGMENTATION]
      │                       │
      └───────────┬───────────┘
                  ▼
          [LAYER GENERATION] ◄─── [ATMOSPHERIC EFFECTS] (fog, dust overlays)
                  │
                  ▼
          [INPAINTING ENGINE] (remote/local background restoration)
                  │
                  ▼
            [SCENE GRAPH]
                  │
                  ▼
         [TIMELINE & CAMERA] (keyframed 6-DOF spline movement)
                  │
                  ▼
         [RENDERER / COMPOSITOR] (GPU/OpenCV perspective warper)
                  │
                  ▼
         [VIDEO EXPORT PIPELINE] (automated ffmpeg/gltf packaging)
```

---

## Part 3: Phased Implementation Plan

### Phase 0: Environment Setup & Verification
- **Objective:** Configure the cloned sandbox environment, install dependencies, and run existing unit tests to establish a baseline.
- **Why:** Ensures the codebase is stable and functions correctly before any modifications are introduced.

### Phase 1: Decoupled Scene Graph Integration
- **Objective:** Separate state management from UI logic by creating a lightweight `SceneGraph` class to track layers, atmospheric details, and camera metrics.
- **Why:** Resolves the architectural issue of state-UI coupling, simplifying the future integration of complex animation curves.

### Phase 2: Cinematic Scene Editor
- **Objective:** Expand the UI layout with dedicated controls for fine-tuning layer placement, adjusting depth offsets, and managing atmospheric elements.
- **Why:** Gives documentarians control over individual layer depth, spacing, and atmospheric effects.

### Phase 3: Timeline & Camera Path System
- **Objective:** Build an interactive keyframe timeline that supports 6-DOF camera panning, zooming, and rotations using cubic spline interpolation.
- **Why:** Replaces the current simple linear zoom with cinematic camera movements essential for professional video production.

### Phase 4: Atmospheric Layer Stack
- **Objective:** Create modular overlay cards (e.g., fog, dust particles, volumetric light leaks) that can be inserted at customizable depths.
- **Why:** Enhances visual depth and matches the atmospheric look of premium history and ancient knowledge documentaries.

### Phase 5: Production-Ready Exporter
- **Objective:** Integrate automated `ffmpeg` compiling directly into the UI, including preset options for standard YouTube formats (16:9 4K, 1080p 24fps, and 9:16 Shorts).
- **Why:** Replaces manual shell command rendering with a simple, one-click export workflow.

### Phase 6: Performance Optimization
- **Objective:** Offload rendering and processing to asynchronous background workers and optimize memory and VRAM usage.
- **Why:** Ensures smooth operation on consumer GPUs and prevents UI lockups during heavy 4K video exports.

---

## Part 4: Executive Summary

### A. What We Already Have
- High-quality depth estimation (MiDaS, ZoeDepth, DinoV2).
- SAM-based interactive object segmentation.
- Robust layer/card generation with automated backfilling (JIT Numba).
- Modular inpainting integrations (Stable Diffusion, FLUX, ComfyUI, Fal.ai).
- Standardized glTF 3D exporter for Blender/Unreal Engine.

### B. What We Should Reuse
- Core SAM-driven segmentation logic (`instance.py`).
- Pre-configured depth estimation backends (`depth.py`).
- Local/API-based diffusion inpainting systems (`inpainting.py`).
- Structural 3D model packaging and exporter (`gltf.py`).

### C. What We Should Modify
- Expand `camera.py` to support 6-DOF coordinates (translation and rotation) instead of simple linear distance tracking.
- Update `controller.py` to store keyframes, atmospheric overlays, and direct export settings.
- Upgrade `webui.py` with an interactive timeline and keyframe configuration panel.

### D. What We Should Not Touch
- Stable API endpoints for third-party tools like StabilityAI, Fal.ai, ComfyUI, and Automatic1111.
- Internal base64 coordinate mapping on the 2D interactive canvas.

### E. What Is Missing
- Timeline Editor supporting keyframes and interpolation.
- 6-DOF camera panning, rotation, and custom animation paths.
- Atmospheric overlay generation (dust, light leaks, fog) in the depth stack.
- Built-in, automated MP4 video compilation.
- Video export presets optimized for YouTube (4K/1080p, 24fps).

### F. Recommended Next Implementation Step
**Refactor state management to extract a clean `SceneGraph` core model.**
- Begin by creating a decoupled `SceneGraph` that represents layers, camera attributes, and timeline data independently of Flask/Dash.
- Write unit tests for this new model to ensure robust project saving and loading before building out the frontend timeline UI.
