# miso

**miso** — the **M**orphoelastic **I**nverse problem **SO**lver — morphs the
boundary of a triangle mesh (or a tetrahedral mesh) onto a sphere while keeping
the interior valid: a conformalized mean curvature flow (cMCF) driven
boundary-to-sphere morph with a Neo-Hookean tetrahedral interior.

Input: `.stl` (surface only) or `.tet` (volumetric). Output: the flowed mesh,
and for tet input a **volumetric map** (`volumetric_map.tet`) that takes the
original volume to the sphere with no inverted elements.

Single-file C, no dependencies beyond libm. Parallel with OpenMP on CPU, with an
optional OpenCL backend for the linear solves.

## How it works

Each flow step does three things:

1. **Boundary target** `U_b`
   - *cMCF* (while the surface is **not** star-shaped): solve
     `(M + δL) U_b = M V_old` — an implicit mean-curvature step with the
     cotangent Laplacian.
   - *Radial projection* (once the surface **is** star-shaped w.r.t. a kernel
     point `c`): push every boundary vertex along its ray from `c` onto the
     target sphere. No linear solve at all.
2. **All vertices**: inexact Newton-CG minimizing

   `E(V) = E_NH(V) + 1/2 Σ_b k_b |V_b − U_b|²`

   with Neo-Hookean energy `Ψ = μ/2 (I_C − 3) − μ ln J + κ/2 (ln J)²`, step
   clipping so no tetrahedron ever reaches `J = 0`, and Armijo backtracking.
   Surface-only input skips the interior solve and moves the boundary directly.
3. **Recentre and rescale** to constant surface area (or a volume-matched
   sphere area with `--target volume`).

**Phase 2 — shear relaxation** (tets only, runs after phase 1 even if the
boundary did not reach a sphere within `--steps`): the
interior is relaxed using only the **off-diagonal Cauchy stress** `σ_ij (i < j)`
until the relative shear-energy change drops below `--shear-tol`. If the
boundary is not yet spherical, phase 2 first approaches it with `J > 0`-safe
partial snaps; once on the sphere, constrained steps are retracted exactly onto
the sphere every line search.
The minimized functional contains **no pressure and no diagonal-deviatoric
term**: `p = tr σ/3` and the on-diagonal traceless part are never relaxed (the
only volumetric piece is a small `κ_shear (ln J)²` guard that keeps `J > 0`).
This is the actual volumetric map: boundary on the sphere, minimal interior
shear, all `J > 0`.

## Build

```sh
make          # CPU (OpenCL-free) build -> ./miso
make gpu      # adds the OpenCL backend (needs OpenCL headers + ICD loader)
```

Equivalent by hand: `cc -O3 -march=native -fopenmp miso.c -o miso -lm`

- **macOS**: `cc -O3 -DUSE_OPENCL miso.c -o miso -framework OpenCL -lm`
  (Apple clang has no `-fopenmp`; use `brew libomp` if you want OpenMP)
- **Windows (MSVC)**: `cl /O2 /openmp /DUSE_OPENCL miso.c OpenCL.lib`

## Usage

```
miso input.(stl|tet) [options]
```

```sh
./miso stl/bunny.stl                    # surface flow, video in out/flow.mp4
./miso stl/cube.stl --steps 50 --no-video
./miso stl/spot.stl --target volume -o out_spot
./miso mesh.tet --gpu on                # volumetric map
```

### Main options

| Option | Default | Meaning |
| --- | --- | --- |
| `-o DIR` | `out` | output directory |
| `--steps N` | `100` | max flow steps |
| `--tol X` | `1e-3` | stop when radius deviation < X |
| `--target area\|volume` | `area` | sphere size target |
| `--delta-scale X` | `1` | cMCF time-step scale |
| `--min-cot X` | `1e-3` | cotangent clamping |
| `--mu X --kappa X --fit X` | `2 5 50` | Neo-Hookean / boundary-fit stiffness |
| `--ref initial\|updated` | `initial` | reference configuration |
| `--no-radial` | | never switch to radial projection |
| `--radial-rate X` | `0.3` | fraction of log-radius gap closed per step |
| `--star-every N` | `5` | kernel-point search period while not star-shaped |
| `--no-shear` | | stop after the boundary reaches the sphere |
| `--shear-tol X` | `1e-4` | phase-2 stop: relative shear-energy change |
| `--shear-steps N` | `1000` | phase-2 iteration cap |
| `--gpu auto\|on\|off` | `auto` | OpenCL for the Newton-CG solve |
| `--list-devices` | | list OpenCL devices |
| `--threads N` | | OpenMP threads |
| `--allow-open` | | proceed despite open / non-manifold STL surfaces |
| `--no-video` | | disable the mp4 render |

Solver knobs: `--newton-iters 15 --newton-tol 1e-3 --max-ls 30
--cg-tol 1e-2 --cg-iters 400`. Video knobs: `--video FILE --fps 15
--video-every 1 --video-size 960x720 --xvfb --python CMD --render-script P`.
`miso --help` prints everything.

## Outputs

Written to the output directory:

| File | Contents |
| --- | --- |
| `flowed.vtk` | full mesh after the flow (open in ParaView) |
| `boundary_flowed.stl` | flowed boundary surface |
| `volumetric_map.tet` | tet input only: original volume → sphere map |
| `stress_sphere.vtk` | tet input only: pressure + deviatoric fields on the sphere |
| `flow.mp4` | progress video (unless `--no-video`) |

The log reports `dev` (boundary radius deviation), `minJ` (worst tet Jacobian —
positive means nothing inverted) and the phase-2 shear energy.

## Stress output

For tetrahedral input, `stress_sphere.vtk` (legacy VTK `POLYDATA`, points =
boundary vertices, i.e. points of the sphere) carries the stress of the final
map, computed from the full Neo-Hookean Cauchy stress

```
σ = (μ (b − I) + κ ln J I) / J,    b = F Fᵀ,    F: initial shape → sphere
```

per tetrahedron and volume-weighted-averaged onto each sphere vertex:

| Field | VTK | Definition |
| --- | --- | --- |
| pressure | `SCALARS pressure` | `p = tr σ / 3` (scalar per sphere point) |
| deviatoric | `VECTORS deviatoric` | `(σ_xx − p, σ_yy − p, σ_zz − p)` — on-diagonal, traceless |

Off-diagonal (shear) `σ_ij (i < j)` is deliberately **not** part of these two
fields — it is the quantity phase 2 relaxes toward zero.

## Input formats

- **`.stl`** — binary or ASCII, vertices are welded by position; winding is
  fixed automatically. Surface-only: no interior is modelled. The surface is
  checked for open edges, non-manifold edges and winding conflicts — miso
  refuses to run on a bad surface unless you pass `--allow-open`.
- **`.tet`** — plain text:

  ```
  nv nt
  x1 y1 z1          # nv vertex lines
  ...
  a b c d           # nt tetrahedron lines, 0-based indices
  ...
  ```

  The boundary is derived from the tets automatically (tets with a free face),
  and negative-volume tets are reoriented on load.

## Video rendering (optional)

By default `miso` pipes frames to `render_video.py`, which needs Python with:

```sh
pip install numpy pyvista imageio imageio-ffmpeg
```

Use `--xvfb` on headless machines, `--python <cmd>` to pick the interpreter,
`--render-script <path>` if the script is not next to the executable, or
`--no-video` to skip it entirely. A failed renderer never aborts the flow.

## Sample meshes

`stl/` contains `cube.stl`, `bunny.stl` and `spot.stl` for quick tests.

## License

MIT — see [LICENSE](LICENSE).
