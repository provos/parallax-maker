![Hacking On Parallax Maker](example/hacking.gif)

# Installation and First Usage

Create a new environment with python 3.10 via *conda* or *venv* and then use pip to install the dependencies.
```
$ pip install -r requirements.txt
$ python ./webui.py
```

You can then reach the web ui via [http://127.0.0.1:8050/](http://127.0.0.1:8050/). Be prepared that the first time, any new functionality is used, the corresponding models need to be downloaded first. This can take a few minutes based on your connection speed. If you want to prefetch the default models, you can start the application with
```
$ python ./webui.py --prefetch-models=default
```

> [!NOTE]
> If you want to make changes to the styles, you need to set up `node` and run `npm run build` to rebuild the tailwind css file. This requires installing `tailwindcss` via `npm install -D tailwindcss`.

> [!IMPORTANT]  
> To use Stable Diffusion 3 Medium, you will need to install the current versio of diffusers from github.

# Parallax-Maker

Provides a workflow for turning images into 2.5D animation like the one seen above.

## Features
 - Segmentation of images
   - Using depth models like Midas or ZeoDepth
   - Using instance segmentatio via Segment Anything with multiple positive and negative point selection
   - Adding and removing of cards, direct manipulation of depth values
 - Inpainting
   - Inpainting of masks that can be padded and blurred
   - Replacing the masked regions with new images via image generation models like Stable Diffusion 1.0 XL, Stable Diffusion 3 Medium, Automatic1111 or ComyfUI endpoints as well as the StabilityAI API.
 - 3D Export
   - Generation of glTF scenes that can be imported into Blender or Unreal Engine
   - Support for depth displacement of cards to generate more realistic 3D geometry
   - In browser 3D preview of the generated glTF scene.

## Basic Examples

Using an input image, the tool runs a depth model like **Midas** or **ZoeDepth** to generate a depth map

![Input Image](example/input_plus_depth.png)

and then creates cards that can be used for 2.5 parallax animation.

![Animation](example/output.gif)

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


![Blender Scene View](example/blender_view.png)


# Web UI

![Web UI](example/webui.jpg)

A Dash based Web UI provides a browser assisted workflow to generated slices from images, inpaint the slices and then export them as a glTF scene to Blender or Unreal Engine. The resulting glTF scene can also be visualized within the app or manipulated via a command line tool and the state file saved by the app.

![Web UI 3D Example](example/webui_3d.jpg)

# Advanced Use Cases
Parallax Maker also supports the Automatic1111 and ComfyUI API endpoints. This allows the tool to utilize GPUs remotely and potentially achieve much higher performance compared to the local GPU. It also means that it's possible to use more specialzied inpainting models and workflows. Here is [an example](example/workflow.json) ComfyUI inpainting workflow that makes use the offset lora published by Stability AI.

![Example configuration for ComfyUI](example/external_config.png)

# Watch the Video
[![Watch the video](example/thumb.png)](https://www.youtube.com/watch?v=4JBQCz-wWYQ)

# Tutorials
## Segmentation and Inpainting Tutorial
[![Segementation and Inpainting Tutorial](example/inpainting-thumb.jpg)](https://youtu.be/hb_x8z4WIeI)
## Unreal Engine Import and Rendering Tutorial
[![Unreal Import and Rendering Tutorial](example/unreal-thumb.jpg)](https://www.youtube.com/watch?v=fLSCCS53h_U)