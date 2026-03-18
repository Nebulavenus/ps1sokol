# PS1 Sokol Racer

**PS1 Sokol Racer** is a high-speed, arcade-style 3D racing game built with Nim and the Sokol framework. It meticulously recreates the iconic "Retro 3D" aesthetic of the 32-bit era, complete with the technical quirks and visual charm of the original PlayStation.

[![Itch.io](https://img.shields.io/badge/Play%20on-Itch.io-FA5C5C?style=for-the-badge&logo=itchdotio)](https://abyss-inhabitant.itch.io/arcade-ps1-style-racing-game)

AI Overview [deepwiki](https://deepwiki.com/Nebulavenus/ps1sokol)

---

## 🏎️ Authentic PS1 Aesthetics

Experience a visual style that captures the raw charm of 1990s hardware:

*   **Signature Vertex Jitter:** Emulates the GTE's fixed-point precision limits, causing vertices to subtly snap and shift as you move.
*   **Affine Texture Mapping:** Linear texture interpolation (without perspective correction) creates the classic "texture wobble" seen on PS1.
*   **15-bit Color & Ordered Dithering:** Uses a 4x4 Bayer matrix to emulate low color depth, resulting in authentic pixelated gradients.
*   **Pre-Baked Ambient Occlusion:** Advanced **Bent Normal** calculation provides sophisticated shading (Sky/Ground light bounce) without runtime cost.
*   **Low-Poly Mastery:** Optimized models and environment assets designed to fit within tight polygon budgets while maintaining a cohesive look.

## 🛠️ Technical Architecture

The project has been recently refactored for high modularity and maintainability:

- **Modular Design:** Decomposed into specialized units (Renderer, UI, Physics, AI, Particles, Events, etc.) to keep the codebase clean and efficient.
- **Custom Asset Pipeline:** Features custom PLY/OBJ loaders with integrated AO baking and support for the **QOI** (Image) and **QOA** (Audio) formats.
- **Spatial Partitioning:** Uses a **Uniform Grid** structure for high-performance collision detection and ground-following.
- **Post-Processing:** Includes a custom CRT/Scanline post-processing pass for that final cathode-ray tube feeling.

For a deep dive into the engine, see [docs/architecture.md](docs/architecture.md).

## 🎮 Controls

### **Driving**
- **Accelerate:** `W` / `UP`
- **Brake / Reverse:** `S` / `DOWN`
- **Steer:** `A` / `D` or `LEFT` / `RIGHT`
- **Drift:** `SPACE`
- **Reset at Checkpoint:** `R`
- **Toggle Camera (Follow/Front):** `C`

### **System & Audio**
- **Pause / Back:** `ESC` / `TAB`
- **Menu Confirm:** `ENTER`
- **Next / Previous Track:** `N` / `B`
- **Toggle Music:** `M`
- **Adjust Volume:** `9` / `0`
- **Toggle Replay:** `P`

### **Debug (AO Controls)**
- **AO Strength:** `1` / `2`
- **Sky Intensity:** `3` / `4`
- **Ground Intensity:** `5` / `6`

---

## 🚀 Building from Source

To compile from source you need Nim 2.2.4 and sokol-nim@07bd978

### **Prerequisites**
- **Nim 2.0+**
- **Sokol-nim** (Ensure dependencies are installed for your OS)

### **Compilation**
To build the project for your native platform:

```bash
nimble build
```

The resulting `main.exe` (or `main` on Unix) will be created in the root directory.

---

## 📜 Recent Changes
For the latest updates, including the major modular refactor and scene transition improvements, check out the [Session Changelog (10.MD)](docs/10.MD).

---

*PS1 Sokol Racer is more than just a throwback; it's a playable slice of gaming history, crafted with passion and powered by clever techniques. Get ready to put the pedal to the pixel!*
