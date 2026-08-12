# Parallax Maker: Deep Execution-Path & Pipeline Lifecycle Analysis Report

This report documents the detailed technical analysis of the existing Parallax Maker execution paths. It traces the lifecycle of an uploaded image, specifies exact functions and data structures, evaluates existing structures (`AppState`, `ImageSlice`, `Camera`), maps rendering projection mathematics, and designs the minimum required boundaries to convert this system into a high-performance, cinematic video production tool.

---

## Part 1: Life of an Uploaded Image (Complete Lifecycle Trace)

### Stage 1: Image Upload
- **Source File:** `parallax_maker/webui.py`
- **Function/Callback:** `update_input_image(contents, classnames)` (triggered by input `C.UPLOAD_IMAGE`)
- **Input Data Structure:** `contents` (Base64-encoded string matching format `data:<mimetype>;base64,<payload>`)
- **Output Data Structure:**
  - `filename`: String (randomly generated state directory name, e.g., `"appstate-abc123xy"`)
  - `img_uri`: String (relative URL path for the Flask route, e.g., `"/tmp-images/appstate-abc123xy/input_image.png?v=<timestamp>"`)
- **Files Written to Disk:**
  - Directory: `/tmp-images/appstate-abc123xy/` (and its nested subdirectory `/appstate-abc123xy/` inside the workspace root)
  - Raw original image: `/appstate-abc123xy/input_image.png`
- **Model Dependencies:** None
- **CPU/GPU Requirements:** Low CPU (Base64 decoding via standard Python libraries).
- **Execution Mode:** Synchronous (Dash callback execution thread).
- **Reusability:** 100% reusable without modification.
- **Wrapping Action:** Keep unchanged.

---

### Stage 2: AppState Creation
- **Source File:** `parallax_maker/controller.py`
- **Function/Method:** `AppState.from_file_or_new(None)` called inside `update_input_image(...)`
- **Input Data Structure:** `None` (for a fresh initialization) or a relative path string pointing to an existing state folder.
- **Output Data Structure:** An instantiated instance of the `AppState` class holding project configurations (e.g., active layers list, thresholds, paths, model settings).
- **Files Written to Disk:** None on creation; `state.to_file()` later writes the configuration to `/appstate-abc123xy/appstate.json`.
- **Model Dependencies:** None
- **CPU/GPU Requirements:** Very low CPU (JSON serialization/deserialization).
- **Execution Mode:** Synchronous.
- **Reusability:** Reusable as a passive data store; UI logic and rendering algorithms should be decoupled from it.
- **Wrapping Action:** Wrap state access behind clear adapter interfaces to prevent UI coupling during rendering changes.

---

### Stage 3: Depth Estimation
- **Source File:** `parallax_maker/depth.py` (orchestrated by `generate_depth_map_callback` in `webui.py`)
- **Class & Methods:** `DepthEstimationModel.depth_map(image, progress_callback)`
- **Input Data Structure:** `image` (NumPy array of shape `(H, W, 3)`, dtype `uint8`)
- **Output Data Structure:** `depth_map` (NumPy array of shape `(H, W)`, dtype `uint8`) representing normalized, inverted distance values (where 0 is black/farthest, and 255 is white/closest).
- **Files Written to Disk:** Grayscale depth map saved to `/appstate-abc123xy/depth_map.png`.
- **Model Dependencies:**
  - `midas`: Intel-ISL MiDaS `DPT_Large` via PyTorch Hub.
  - `zoedepth`: `ZoeD_NK` via PyTorch Hub (ISL Org).
  - `dinov2`: `facebook/dpt-dinov2-large-nyu` via Hugging Face.
- **CPU/GPU Requirements:** High GPU memory (VRAM) recommended. Can fallback to CPU (using PyTorch), which causes massive delays (minutes per run on CPU vs milliseconds on GPU).
- **Execution Mode:** Synchronous (runs on Flask request thread, locking the client thread until complete).
- **Reusability:** Reusable without modification.
- **Wrapping Action:** Wrap behind an adapter to support remote depth estimation API fallbacks, preventing VRAM out-of-memory errors on low-end local hardware.

---

### Stage 4: Segmentation
- **Source File:**
  - Depth Histogram: `parallax_maker/segmentation.py`
  - SAM Instance Segmentation: `parallax_maker/instance.py`
- **Class & Methods:**
  - `analyze_depth_histogram(depth_map, num_slices)`
  - `mask_from_depth(depth_map, threshold_min, threshold_max)`
  - `SegmentationModel.mask_at_point_blended(point_input)`
- **Input Data Structure:**
  - Histogram: `depth_map` (NumPy array of shape `(H, W)`, dtype `uint8`)
  - SAM: `point_input` (dictionary containing lists of `positive_points` and `negative_points` coordinates as `(x, y)` tuples).
- **Output Data Structure:** `mask` (1-channel binary NumPy array of shape `(H, W)`, dtype `uint8`, containing values 0 and 255).
- **Files Written to Disk:** Transient intermediate masks are temporarily cached inside the state folder (e.g., `/appstate-abc123xy/image_slice_i_mask.png`).
- **Model Dependencies:** Segment Anything Model `facebook/sam-vit-huge` (via Hugging Face Transformers).
- **CPU/GPU Requirements:** High GPU VRAM (SAM-huge uses large memory maps); slow and CPU-heavy on local CPU fallbacks.
- **Execution Mode:** Synchronous.
- **Reusability:** 100% reusable.
- **Wrapping Action:** Keep unchanged.

---

### Stage 5: ImageSlice Generation
- **Source File:** `parallax_maker/segmentation.py` (with the core slice data-structure class defined in `parallax_maker/slice.py`)
- **Class & Functions:**
  - `generate_image_slices(image, depth_map, thresholds, num_expand)`
  - `create_slice_from_mask(image, mask, num_expand)`
  - `ImageSlice.__init__(image, depth, filename)`
- **Input Data Structure:**
  - `image`: NumPy array `(H, W, 3)`, dtype `uint8`
  - `mask`: NumPy array `(H, W)`, dtype `uint8`
- **Output Data Structure:** An instantiated list of `ImageSlice` objects.
- **Files Written to Disk:** Individual layered slices are saved to disk as alpha-enabled PNGs: `/appstate-abc123xy/image_slice_i.png`.
- **Model Dependencies:** None.
- **CPU/GPU Requirements:** Low-to-moderate CPU (NumPy/OpenCV operations such as feathering and color compositing).
- **Execution Mode:** Synchronous.
- **Reusability:** Reusable as a data structure; needs wrapping to support position offsets and rotation parameters.
- **Wrapping Action:** Wrap inside an adapter to support custom 3D card parameters (e.g., transformations, position offsets, and atmospheric assets) without altering the core `ImageSlice` class.

---

### Stage 6: Inpainting
- **Source File:** `parallax_maker/inpainting.py`
- **Class & Functions:**
  - JIT Patching: `patch_image(image, mask_image)` (uses Numba JIT functions `find_nearest_alpha` and `patch_pixels`)
  - AI Diffusion: `InpaintingModel.inpaint(prompt, negative_prompt, init_image, mask_image, ...)`
- **Input Data Structure:**
  - `init_image`: PIL Image or NumPy array representing the layered slice.
  - `mask_image`: Grayscale PIL Image or NumPy array representing the region to be inpainted.
- **Output Data Structure:** A fully inpainted 4-channel RGBA PIL Image.
- **Files Written to Disk:** Saves a new version of the slice to `/appstate-abc123xy/image_slice_i.png` (overwriting or versioning the target file).
- **Model Dependencies:** Local Diffusers models (v1.5, SDXL, SD 3 Medium, FLUX.1 Fill) or remote third-party REST/WebSocket APIs.
- **CPU/GPU Requirements:**
  - Local JIT Patching: Low multi-threaded CPU.
  - Local AI Diffusion: Extremely high GPU VRAM (requires 10GB–16GB VRAM); extremely slow and resource-heavy on CPU.
  - Remote APIs: Low local resource usage (runs on remote servers).
- **Execution Mode:** Synchronous.
- **Reusability:** Fully reusable.
- **Wrapping Action:** Wrap local and remote inpainting backends behind a unified adapter interface, allowing users to choose the best option based on their hardware.

---

### Stage 7: Camera Configuration
- **Source File:** `parallax_maker/camera.py`
- **Class & Methods:** `Camera` (and projection methods `focal_length_px(image_width)` and `camera_matrix(image_width, image_height)`)
- **Input Data Structure:** Key metric attributes: `distance`, `max_distance`, and `focal_length`.
- **Output Data Structure:** `camera_matrix` (NumPy projection matrix of shape `(3, 3)`, dtype `float32`).
- **Files Written to Disk:** Camera parameters are serialized to JSON and saved inside `appstate.json`.
- **Model Dependencies:** None.
- **CPU/GPU Requirements:** Low CPU (simple float matrix operations).
- **Execution Mode:** Synchronous.
- **Reusability:** Highly reusable, but limited to a single linear push-in trajectory.
- **Wrapping Action:** Modify or wrap behind a cinematic motion adapter to support 6-DOF coordinate translations, rotations, and keyframed camera paths.

---

### Stage 8: 2.5D Rendering
- **Source File:** `parallax_maker/segmentation.py`
- **Function/Method:** `render_view(image_slices, camera_matrix, card_corners_3d_list, camera_position)`
- **Input Data Structure:**
  - `image_slices`: List of `ImageSlice` objects
  - `camera_matrix`: Projection matrix of shape `(3, 3)`, dtype `float32`
  - `card_corners_3d_list`: List of NumPy arrays of shape `(4, 3)`, representing the 3D coordinates of each card's corners
  - `camera_position`: NumPy array representing the 3D position of the camera: `[x, y, z]`
- **Output Data Structure:** `rendered_image` (4-channel RGBA NumPy array of shape `(H, W, 4)`, dtype `uint8`).
- **Files Written to Disk:** None.
- **Model Dependencies:** None.
- **CPU/GPU Requirements:** Moderate CPU (runs OpenCV's perspective warping and alpha compositing in a single-threaded loop).
- **Execution Mode:** Synchronous.
- **Reusability:** Fully reusable, but single-threaded CPU rendering can become a bottleneck at high resolutions.
- **Wrapping Action:** Wrap behind an adapter to support horizontal/vertical panning, rotational warping, and multi-point camera path animations.

---

### Stage 9: Frame Generation & Export (Animation Compilation)
- **Source File:** `parallax_maker/segmentation.py` (orchestrated by the `export_animation` callback in `webui.py`)
- **Function/Method:** `render_image_sequence(...)`
- **Input Data Structure:** Configuration inputs: output directory path, camera matrices, push distances, and the total frame count.
- **Output Data Structure:** None (compiles and saves sequential frame files to disk).
- **Files Written to Disk:** Sequential frame images (e.g., `/appstate-abc123xy/rendered_image_000.png` through `rendered_image_099.png`).
- **Model Dependencies:** None.
- **CPU/GPU Requirements:** Moderate CPU.
- **Execution Mode:** Synchronous (completely locks the main UI thread during the rendering loop).
- **Reusability:** Reusable, but needs to be decoupled from the main Flask request thread to prevent UI lockups.
- **Wrapping Action:** Modify to run asynchronously on a background thread, and integrate automatic FFmpeg encoding to output video files directly instead of relying on manual command-line scripts.

---

## Part 2: Detailed Component Evaluation & Design

### A. AppState Evaluation
- **Can it remain temporarily?** **YES.**
  `AppState` acts as a highly robust database for active projects, managing local file paths, API credentials, layers, and thresholds. It is not necessary to replace it immediately.
- **Clean Boundary Design:** We can introduce a clear boundaries structure around the rendering pipeline. By treating `AppState` as a read-only project database, we can extract and format its state data into structured inputs for the rendering engine without coupling the rendering logic to `AppState` directly.

### B. ImageSlice Specifications
An `ImageSlice` contains:
- `image`: RGBA image containing the slice's visual content.
- `depth`: Projected focal coordinate mapped to the Z-axis.
- `filename`: Local path of the asset.
- `is_ground_plane`: Flag denoting customized ground projections.
- `positive_prompt` / `negative_prompt`: Text strings used for inpainting backends.

#### Evaluation of Capabilities:
- **Source Image:** Yes, stored in the RGB channels of `image`.
- **Mask:** Yes, stored in the alpha channel of `image`.
- **Depth:** Yes, stored in the `depth` parameter.
- **Position:** No. The card is assumed to be centered at $(0, 0, z)$.
- **Layer Ordering:** No. Ordering is implied by its sorted index position inside the parent list.
- **Opacity:** No. Opacity is baked directly into the alpha channel of the image.
- **Transformation:** No. There is no concept of rotation, scale, or translation offsets.
- **Inpainting Result:** No. Inpainted results are written directly back into the slice image, overwriting the original content.

#### Minimum Required Architectural Changes:
To enable advanced 2.5D documentary animations, we must wrap `ImageSlice` behind an adapter that supports:
1. **Translation Offsets:** `(offset_x, offset_y, offset_z)` parameters to position layers independently.
2. **Rotation Parameters:** Euler angles or rotation matrices for angled layer adjustments.
3. **Scale Multipliers:** Independent layer scaling to fine-tune perspective adjustments.
4. **Layer Opacity:** Multiplier values to fade layers in and out (e.g., for atmospheric overlays).

---

### C. Camera Mechanics
The camera properties inside `camera.py` include:
- `camera_position`: `[x, y, z]` translation array, defaulting to `[0, 0, -distance]`.
- `camera_distance` ($d_{cam}$): Distance from camera to target plane ($z=0$).
- `max_distance` ($D_{max}$): Maximum depth threshold/clipping boundary.
- `focal_length` ($f$): Metric lens focal length (mm).
- `sensor_width` ($W_{sensor}$): Metric sensor width (mm).

#### Core Equations:
1. **Focal Length in Pixels ($f_{px}$):**
   $$f_{px} = \frac{\text{image\_width} \cdot f}{W_{sensor}}$$
2. **Camera Projection Matrix ($K$):**
   $$K = \begin{bmatrix} f_{px} & 0 & \text{image\_width}/2 \\ 0 & f_{px} & \text{image\_height}/2 \\ 0 & 0 & 1 \end{bmatrix}$$
3. **Depth Map Value to 3D Coordinate ($z_{3d}$):**
   $$z_{3d} = D_{max} \cdot \frac{255 - \text{depth\_val}}{255.0}$$
4. **Card Scale Dimensions ($W_{card}, H_{card}$) at Depth $z_{3d}$:**
   $$W_{card} = \frac{\text{image\_width} \cdot (z_{3d} + d_{cam})}{f_{px}}$$
   $$H_{card} = \frac{\text{image\_height} \cdot (z_{3d} + d_{cam})}{f_{px}}$$

---

### D. Rendering Pipeline Analysis
`render_view` implements a simple, CPU-based rendering system using OpenCV:
```python
# Iterates through layers from back to front:
for i, slice_image in enumerate(image_slices):
    # Calculates translations based on camera position:
    tvec = -camera_position.reshape(3, 1)

    # Projects the 3D corners using zero rotation:
    card_corners_2d, _ = cv2.projectPoints(card_corners_3d, np.zeros((3,1)), tvec, camera_matrix, None)

    # Warps the layer's image to the projected 2D coordinates:
    warped_slice = cv2.warpPerspective(cur_image, cv2.getPerspectiveTransform(src_corners, card_corners_2d), (W, H))

    # Blends the warped layer onto the viewport:
    blend_with_alpha(rendered_image, warped_slice)
```

#### Minimum Changes for Advanced Motion Support:
No major restructuring is needed to support advanced movements. We can achieve this with minor additions:
1. **Horizontal/Vertical Camera Movement:** Pass the X and Y coordinates of the camera position into the translation vector:
   `tvec = -camera_position` (currently, X and Y translation work automatically if non-zero coordinates are passed to `camera_position`).
2. **Push In / Pull Out:** This is already supported by adjusting the Z component of the camera position vector.
3. **Rotations (Pan, Tilt, Roll):** Pass non-zero Euler angles into the rotation vector `rvec` inside `cv2.projectPoints` (currently hardcoded to `np.zeros((3,1))`).
4. **Cinematic Multi-Point Paths:** Pass interpolated, keyframed translation and rotation coordinates to the renderer for each frame step, instead of using a simple linear increment loop.

---

### E. Performance Optimization for Low-End / Non-CUDA Machines
On machines without NVIDIA GPUs, deep neural network inference on CPU is incredibly slow. To optimize the workflow on standard hardware, we must prioritize CPU-friendly solutions:
1. **Remote API Fallbacks:** Make local depth and SAM models optional. Allow users to offload heavy calculations to fast cloud APIs (StabilityAI, Fal.ai, or external ComfyUI servers), keeping local resource usage to a minimum.
2. **Fast Local Fallback (JIT Patching):** Use the existing JIT-compiled Numba patching (`patch_image`) as a fast, local preprocessing fallback. It runs instantly on any standard CPU and provides excellent preview quality.
3. **Viewport Resolution Scaling:** Render live timeline previews at a lower resolution (e.g., 50% scale) to ensure real-time performance on standard CPUs, and only render at full resolution during the final video export.

---

### F. Inpainting Architectural Separation
To keep the tool fast and lightweight on all systems, we can divide inpainting into three distinct, optional tiers:
- **Tier 1: Fast Local Patching (No AI):** The JIT-compiled Numba patching (`patch_image`). It is extremely fast, runs natively on any CPU, and provides excellent results for quick edits.
- **Tier 2: Remote API Processing (Cloud AI):** Offload inpainting to cloud APIs (StabilityAI or Fal.ai). This provides state-of-the-art results without requiring any local GPU hardware.
- **Tier 3: Heavyweight Local Diffusion (Local AI):** Run SDXL or FLUX models locally via PyTorch/Diffusers. This requires a high-end local GPU with substantial VRAM (12GB+).
- *Implementation Strategy:* Design these tiers as interchangeable, optional modules. Users can complete the entire workflow using only Tier 1 and Tier 2, making high-end local GPU hardware completely optional.

---

### G. Video Compilation & Export
- **Current Behavior:** The application renders and saves sequential PNG frames (`rendered_image_000.png` through `rendered_image_099.png`) to the state directory. The user must manually run an external `ffmpeg` command in their terminal to compile them into a video.
- **Direct MP4 Export Solution:** We can automate this process by executing `ffmpeg` programmatically using Python's `subprocess` module.
- *Implementation Path:* Read the sequential PNG frames from the state directory, call `ffmpeg` directly from the app using optimized presets (e.g., h264 codec, yuv420p color space), and compile the video automatically.

```python
import subprocess

def compile_frames_to_mp4(state_dir, output_filepath, fps=24):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{state_dir}/rendered_image_%03d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-crf", "18",
        output_filepath
    ]
    subprocess.run(cmd, check=True)
```

---

## Part 3: Architecture & Next Steps Summary

### 1. Current Execution Architecture
The current architecture is highly functional but tightly coupled to Dash:
`dcc.Upload` (Base64 Image) → Decodes to RGB → Creates directory and saves original `input_image.png` → Calls `DepthEstimationModel` (PyTorch) → Generates and saves `depth_map.png` → Generates depth thresholds → Computes individual `ImageSlice` cards → Executes optional inpainting (JIT or Diffusion) → Saves layered PNG files → Generates sequential frame PNGs using single-threaded CPU rendering (`cv2.warpPerspective`) → Requires manual command-line compilation.

### 2. Reusable Core Components
- **`instance.py`:** Excellent SAM-driven interactive object segmentation.
- **`depth.py`:** Solid depth estimation loader and inference engine (MiDaS, ZoeDepth, DinoV2).
- **`inpainting.py`:** Lightweight Numba-JIT patching and remote API connections.
- **`gltf.py`:** Complete and robust 3D scene packaging and glTF exporter.

### 3. Key Bottlenecks
- **Heavy Local CPU/GPU Inference:** Running depth estimation, SAM, and local diffusion models on standard CPUs or low-end GPUs is extremely slow.
- **Single-Threaded Rendering:** High-resolution rendering on a single CPU thread limits performance and prevents real-time previewing.
- **Synchronous Execution:** Running long processing tasks inside the main UI thread locks the interface and can cause timeouts.

### 4. Minimum Required Architectural Boundary
To keep the system modular and stable, we should implement a clean boundary around the rendering pipeline. Instead of modifying the core `AppState` or `ImageSlice` structures, we can introduce a **Scene Motion Adapter** that intercepts data from `AppState` and translates it into structured coordinates for the camera and layers during rendering. This allows us to implement advanced animation paths and rotational camera sweeps while keeping the underlying data structures completely unchanged.

### 5. What Should Be Changed First
- **Programmatic FFmpeg Video Export:** Automate the frame compilation process. This replaces manual terminal commands with a clean, one-click MP4 export directly inside the UI.
- **6-DOF Camera Trajectories:** Update the projection calculations inside `render_view` to accept rotational matrices (`rvec`) and 3D translation vectors, enabling cinematic panning and rotation.

### 6. What Should NOT Be Changed Yet
- **`AppState` Core:** Keep the central `AppState` class intact to serve as the project's data-holder and project persistence database, preventing breaking changes or unnecessary modifications.
- **Local SAM & Depth Engines:** Keep the local segmentation and depth pipelines unchanged, as they are highly refined and work reliably.

### 7. Recommended Implementation Order
1. **Step 1: Automated Video Compilation:** Implement direct MP4 video export using Python's `subprocess` to call `FFmpeg` automatically on export.
2. **Step 2: 6-DOF Camera Warping:** Update `render_view` to support camera rotation coordinates and 3-axis translation values.
3. **Step 3: Cinematic Camera Keyframing:** Create a simple keyframe interpolation system to support complex panning, rotations, and custom paths.
4. **Step 4: Layer Offsets & Transmutations:** Wrap `ImageSlice` within an adapter to allow translation, scale, and opacity adjustments on individual layers.
5. **Step 5: Cloud API Fallbacks:** Configure cloud API integrations (such as Fal.ai) as optional fallbacks to ensure high performance on systems without high-end GPUs.
6. **Step 6: UI Controls:** Add keyframe and timeline panels to the Dash UI to give users a clean interface for editing cinematic motions.
