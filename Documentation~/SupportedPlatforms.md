# Supported platforms

Barracuda supports the following platforms: 

- CPU inference: all Unity platforms are supported.

- GPU inference: all Unity platforms are supported except: 
  - `OpenGL ES` on `Android/iOS`: use Vulkan/Metal.
  - `OpenGL Core` on `Mac`: use Metal.
  - `WebGL`: use CPU inference.
