# 🍲 MISo – Morphoelastic Inverse problem Solver
MISo computes the **hydrostatic pressure field** inside a sphere that results from forcing an arbitrary solid into a spherical shape while allowing shear stresses to relax.  
It solves the inverse morphoelastic problem:

> Given a solid shape (STL), map it to a sphere using a conformal volumetric map, then relax all shear stresses under an isochoric hyperelastic material while **keeping the boundary fixed to the sphere**. The resulting pressure field is the **reaction pressure** required to maintain the spherical shape against the deviatoric (shear) stresses.

This pressure field can be interpreted as the internal stress state of a material that has been "grown" from a stress‑free sphere into the observed shape, then shear‑relaxed.


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
