# 🍲 MISo – Morphoelastic Inverse problem Solver

MISo computes the stress‑free reference configuration of a hyperelastic solid whose current (observed) shape is a sphere.  
It solves the inverse morphoelastic problem:

1. Load an arbitrary solid from an STL file.
2. Compute a conformal map from its surface to the unit sphere.
3. Extend the map volumetrically using As‑Conformal‑As‑Possible (ACAP) mapping.
4. Relax shear stresses via GPU‑accelerated gradient descent while keeping the boundary fixed to the sphere.
5. Output the final pressure field inside the sphere.

## Features

- **Robust mesh preprocessing**: Automatic repair, smoothing, and tetrahedralization.
- **Fast conformal boundary map**: Boundary First Flattening (BFF) via `confmap`.
- **Accurate volumetric map**: ACAP via MIRTK (falls back to harmonic extension if MIRTK unavailable).
- **GPU‑accelerated relaxation**: Vectorized assembly using `TorchMesh` (PyTorch backend).
- **Pressure field computation**: Hydrostatic pressure from the final Cauchy stress.
- **Visualization**: Interactive 3D view with `vedo`.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MISo.git
cd MISo
