#!/usr/bin/env python3
"""Reads frames streamed by cmcf on stdin and writes an mp4 (same look as the
original FlowVideo: light-blue smooth-shaded surface, fixed camera).
Invoked by miso; args: out.mp4 fps WxH xvfb steps"""
import struct
import sys
import numpy as np

out, fps, size, xvfb, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
w, h = (int(x) for x in size.lower().split("x"))
inp = sys.stdin.buffer


def read(n):
    b = inp.read(n)
    return b if len(b) == n else None


import pyvista as pv
import imageio
if xvfb:
    try:
        pv.start_xvfb()
    except Exception:
        pass

nb, nf = struct.unpack("<ii", read(8))
F = np.frombuffer(read(12 * nf), dtype=np.int32).reshape(-1, 3).astype(np.int64)
faces = np.hstack([np.full((nf, 1), 3), F]).ravel()

writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=None)
pl = pv.Plotter(off_screen=True, window_size=(w, h))
pl.set_background("white")
actor, cam = None, False
while True:
    head = read(12)
    if head is None:
        break
    step, dev = struct.unpack("<id", head)
    raw = read(12 * nb)
    if raw is None:
        break
    V = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    if actor is not None:
        pl.remove_actor(actor)
    actor = pl.add_mesh(pv.PolyData(V, faces), color="lightblue", smooth_shading=True,
                        specular=0.3, specular_power=20, reset_camera=False)
    if not cam:
        lo, hi = V.min(0), V.max(0)
        c, d = (lo + hi) / 2, 2.5 * np.linalg.norm(hi - lo)
        pl.camera.position = c + np.array([d, -d, 0.7 * d])
        pl.camera.focal_point = c
        pl.camera.up = (0, 0, 1)
        pl.reset_camera_clipping_range()
        cam = True
    if step < 0:        # phase 2: shear relaxation, dev carries E_shear / E_shear(start)
        label = f"shear relaxation {-step}\nE/E0={dev:.4f}"
    else:
        label = f"step {step}/{steps}\ndev={dev:.3e}" if step else "step 0"
    pl.add_text(label, position="upper_left",
                font_size=10, color="black", name="info")
    writer.append_data(pl.screenshot(return_img=True))
writer.close()
pl.close()
