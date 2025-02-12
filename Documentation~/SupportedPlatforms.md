<div style="background: #ffe2d7; padding: 16px; border-radius: 4px; margin: 16px 0;">
    The Barracuda package is deprecated. Use <a href="https://docs.unity3d.com/Packages/com.unity.sentis@latest/index.html">the Sentis package</a> instead.
</div>

# Supported platforms

Barracuda supports the following platforms: 

- CPU inference: all Unity platforms are supported.

- GPU inference: all Unity platforms are supported except: 
  - `OpenGL ES` on `Android/iOS`: use Vulkan/Metal.
  - `OpenGL Core` on `Mac`: use Metal.
  - `WebGL`: use CPU inference.
