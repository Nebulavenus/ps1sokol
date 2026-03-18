# PS1 Sokol Racer Architecture

This document provides a high-level overview of the game's architecture, key technical features, and module responsibilities.

## Overview
PS1 Sokol Racer is a 3D racing game built in Nim using the **Sokol** framework. It is designed with a "Retro 3D" aesthetic, emulating the technical limitations and visual style of the original PlayStation (PS1).

## Core Technical Features

### 1. Retro Rendering Pipeline (`src/shaders/default.glsl`)
The rendering engine uses several techniques to achieve its signature PS1 look:
- **Vertex Jittering**: Simulates the fixed-point coordinate limitations of the GTE (Geometry Transformation Engine) by snapping vertices to a low-resolution grid in clip-space.
- **Affine Texture Mapping**: Bypasses perspective-correct texture interpolation to create the classic "texture wobble" seen on PS1 hardware.
- **Ordered Dithering & Color Quantization**: Emulates 15-bit color depth (5 bits per channel) using a 4x4 Bayer matrix for dithering, preventing banding in gradients.
- **Fog System**: A distance-based smoothstep fog used for atmosphere and to hide draw distance limits.

### 2. Pre-baked Ambient Occlusion (`src/aobaker.nim`)
To add depth without expensive real-time lighting, the engine pre-calculates AO during asset loading:
- **Bent Normal Generation**: For each vertex, a "most open direction" (bent normal) is calculated by casting rays in a hemisphere.
- **Occlusion Strength**: The ratio of unblocked rays determines the vertex AO.
- **Static IBL**: The fragment shader uses the bent normal to sample "Sky" and "Ground" light colors, providing cheap static global illumination.

### 3. Physics & Collision (`src/physics.nim`)
- **Vehicle Dynamics**: Custom arcade physics handling acceleration, braking, drifting, and surface alignment.
- **Uniform Grid Spatial Partitioning**: Track geometry is divided into a 3D grid to allow extremely fast ray-triangle intersection tests for ground height and wall collisions.
- **Surface Info**: A downward raycast determines the road height and normal, allowing the car to tilt and follow the terrain.

### 4. Audio Engine (`src/audio.nim`, `src/qoa.nim`)
- **QOA Format**: Uses the Quite OK Audio format for high-quality, efficient storage.
- **Dynamic Engine Sound**: Real-time pitch and volume modulation based on vehicle RPM and speed.
- **Music Playlist**: Support for background music tracks with playback controls (Next/Prev/Mute).

## Module Responsibilities

### Core Loop & App
- **`src/main.nim`**: The entry point. Handles the Sokol lifecycle (`init`, `frame`, `cleanup`).
- **`src/events.nim`**: Manages all input (Keyboard/Mouse) and high-level state transitions (Menu <-> CarSelection <-> RaceSetup <-> Playing).
- **`src/types.nim`**: Defines global state, vehicle structures, and shared types.

### Graphics & Level
- **`src/renderer.nim`**: Manages Sokol pipelines, render passes (Offscreen for Post-FX), and 3D drawing procedures.
- **`src/ui.nim`**: Renders the HUD and menus using `sokol-debugtext`.
- **`src/level.nim`**: Coordinates track loading, path extraction, and bot spawning.
- **`src/mesh_loader.nim`**: Custom PLY and OBJ parsers that integrate with the AO baker.
- **`src/particles.nim`**: A simple distance-sorted particle system for smoke and exhaust effects.

### Simulation & Utilities
- **`src/ai.nim`**: Advanced path-following logic for opponent vehicles, including curvature-based speed control and varied performance traits.
- **`src/camera.nim`**: Manages smooth follow camera and hood-mounted front view.
- **`src/rtfs.nim` & `src/embedfs.nim`**: Abstraction layers for asset loading, supporting both local filesystem and embedded resources.
- **`src/math_utils.nim`**: Shared mathematical primitives and intersection helpers.

## Asset Pipeline
- **Models**: Standard PLY or OBJ files.
- **Textures**: QOI (Quite OK Image) format for fast, simple loading.
- **Audio**: QOA (Quite OK Audio).
- **Shaders**: Authored in GLSL and compiled using `sokol-shdc`.
