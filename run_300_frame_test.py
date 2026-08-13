import os
import sys
import time
import resource
import numpy as np
import cv2
from pathlib import Path

# Add repo root to path
sys.path.append(str(Path(__file__).parent))

import torch
from parallax_maker.slice import ImageSlice
from parallax_maker.camera import Camera
from parallax_maker.segmentation import (
    analyze_depth_histogram,
    generate_image_slices,
    render_image_sequence,
    reconstruct_slice_disocclusions
)

def run():
    print("Starting 300-frame Vishnu/Shesha Parallax rendering...")

    t_start = time.perf_counter()

    # Load input image and depth map
    img_path = Path("example/input.png")
    depth_path = Path("example/depth_map.png")

    if not img_path.exists() or not depth_path.exists():
        print(f"Error: input assets not found!")
        return

    # 1. Preprocessing stage
    t_pre_start = time.perf_counter()

    # Load with OpenCV and convert to RGB/Grayscale
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

    h_orig, w_orig = image.shape[:2]
    print(f"Loaded image of shape: {image.shape}")

    # Segment into 5 slices using depth histogram
    num_slices = 5
    thresholds = analyze_depth_histogram(depth_map, num_slices=num_slices)

    # Generate original slices (unpadded)
    original_slices = generate_image_slices(image, depth_map, thresholds, num_expand=0)

    # Generate reconstructed slices (padded with 10% safety margin)
    margin = 0.1
    reconstructed_slices = []
    for i, slice_image in enumerate(original_slices):
        recon_img = reconstruct_slice_disocclusions(slice_image.image, is_background=(i == 0), margin=margin)
        recon_slice = ImageSlice(image=recon_img, depth=slice_image.depth)
        reconstructed_slices.append(recon_slice)

    t_pre_end = time.perf_counter()
    pre_time = t_pre_end - t_pre_start
    print(f"Preprocessing completed in {pre_time:.4f} seconds.")

    # 2. Camera Setup
    camera = Camera(100.0, 500.0, 100.0)
    camera_matrix = camera.camera_matrix(w_orig, h_orig)

    card_corners_3d_list = []
    for i, image_slice in enumerate(reconstructed_slices):
        card = image_slice.create_card(h_orig, w_orig, camera)
        # Apply 10% margin to cards matching the reconstructed slice size
        card[:, :2] *= (1.0 + 2.0 * margin)
        card_corners_3d_list.append(card)

    # 3. Rendering stage
    output_dir = Path("/tmp/rendered_output_300")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_render_start = time.perf_counter()

    # Render the 300-frame sequence
    camera_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)
    # Let's request a large camera displacement/push distance (e.g. 150 units) to push boundaries
    push_distance = 150.0

    report = render_image_sequence(
        output_dir,
        reconstructed_slices,
        card_corners_3d_list,
        camera_matrix,
        camera_position,
        push_distance=push_distance,
        num_frames=300,
        original_size=(h_orig, w_orig),
        original_slices=original_slices,
        max_reconstruction_ratio=0.15,
        ai_threshold_ratio=0.08,
    )

    t_render_end = time.perf_counter()
    render_time = t_render_end - t_render_start

    # Peak resources tracking
    peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_ram_mb = peak_ram_kb / 1024.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Print requested stats
    print("\n" + "="*50)
    print("           FINAL VISHNU/SHESHA REPORT")
    print("="*50)
    print(f"Original Camera Displacement:   {push_distance:.2f} units")
    print(f"Actual Camera Displacement:     {push_distance:.2f} units (clamped frames: {report['percentage_clamped_frames']:.1f}%)")
    print(f"Percentage Clamped:             {report['percentage_clamped_frames']:.1f}%")
    print(f"Max Reconstructed Area:         {report['max_reconstructed_ratio'] * 100:.2f}%")
    print(f"Avg Reconstructed Area:         {report['average_reconstructed_ratio'] * 100:.2f}%")
    print(f"AI Reconstruction Usage:        {report['ai_used_count']} of 300 frames")
    print(f"Mean Temporal MAD:              {report['mean_temporal_mad']:.4f}")
    print(f"Max Temporal MAD:               {report['max_temporal_mad']:.4f}")
    print(f"Mean Mask Change:               {report['mean_mask_change']:.4f}")
    print(f"Mean Boundary Movement:         {report['mean_boundary_movement']:.4f}")
    print(f"Preprocessing Time:             {pre_time:.4f} seconds")
    print(f"Rendering Time (300 frames):    {render_time:.4f} seconds (average {(render_time/300.0)*1000.0:.1f} ms/frame)")
    print(f"Total Time:                     {pre_time + render_time:.4f} seconds")
    print(f"Peak RAM:                       {peak_ram_mb:.2f} MB")
    if torch.cuda.is_available():
        print(f"Peak VRAM:                      {peak_vram_mb:.2f} MB")
    else:
        print("Peak VRAM:                      N/A (CPU execution)")
    print("="*50)

    # Visual Frame Comparison details
    print("\nVisual Frame Comparison metrics:")
    target_frames = [0, 75, 150, 225, 299]
    for frame_idx in target_frames:
        frame_path = output_dir / f"rendered_image_{frame_idx:03d}.png"
        # We can re-render specifically to get ratio & warnings for this frame
        req_pos = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        req_pos[2] += (push_distance / 300.0) * frame_idx

        from parallax_maker.segmentation import render_view
        rendered_f = render_view(
            reconstructed_slices,
            camera_matrix,
            card_corners_3d_list,
            req_pos,
            original_size=(h_orig, w_orig),
            original_slices=original_slices,
            max_reconstruction_ratio=0.15,
            ai_threshold_ratio=0.08,
        )
        ratio = rendered_f.reconstruction_ratio
        clamped_flag = "Yes" if len(rendered_f.warnings) > 0 else "No"
        ai_flag = "Yes" if rendered_f.ai_used else "No"

        # Calculate provenance breakdown
        prov_map = rendered_f.provenance_map
        p_empty = (np.count_nonzero(prov_map == 0) / prov_map.size) * 100.0
        p_orig = (np.count_nonzero(prov_map == 1) / prov_map.size) * 100.0
        p_det = (np.count_nonzero(prov_map == 2) / prov_map.size) * 100.0
        p_ai = (np.count_nonzero(prov_map == 3) / prov_map.size) * 100.0

        print(f"  Frame {frame_idx:3d}: Reconstructed Area = {ratio*100:6.2f}%, Trajectory Clamped = {clamped_flag}, AI Mode = {ai_flag}")
        print(f"             Provenance: Orig={p_orig:5.1f}%, DetRecon={p_det:5.1f}%, AIRecon={p_ai:5.1f}%, Empty={p_empty:5.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run()
