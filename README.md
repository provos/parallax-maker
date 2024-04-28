# installation

Via pip use
```
$ pip install -r requirements.txt
```

# parallax-maker

Generates masked images that can be used for 2.5D animation

Using an input image, the tool runs a depth model like Midas to generate a depth map

![Input Image](example/input.png) ![Depth Map](example/depth_map.png)

and then creates cards that can be used for 2.5 parallax animation.

![Animation](example/output.gif)

This animation was created using the following command:

~~~
ffmpeg -framerate 24 -i rendered_image_%03d.png -filter_complex "fps=5,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" output.gif 
~~~


# 3D Export

The tool also supports generating a glTF2.0 scene file that an be easily imported into 3D apps like Blender.

![Blender Scene View](example/blender_view.png)


# Web UI

![Web UI](example/webui.jpg)

A simple Dash based Web UI provides a browser assisted workflow to generated slices from images, inpaint the slices and then export them as a glTF scene to Blender. The resulting glTF scene can also be visualized within the app or manipulated via a command line tool and the state file saved by the app

![Web UI 3D Example](example/webui_3d.jpg)