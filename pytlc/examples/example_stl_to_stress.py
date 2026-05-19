#!/usr/bin/env python3
"""
Example: STL to Stress Processing Pipeline

This example demonstrates the complete pipeline from an STL surface mesh
to hyperelastic stress analysis using pytlc.

Steps:
1. Load STL surface mesh (Stanford Bunny)
2. Convert to tetrahedral volume mesh
3. Ensure injectivity using TLC lifting method
4. Apply virtual deformation (simple shear)
5. Compute Von Mises stress field (Neo-Hookean hyperelastic model)
6. Generate visual diagnostics and export results

Author: pytlc development team
Based on: "Lifting Simplices to Find Injectivity" by Xingyi Du
"""

import os
import sys
import numpy as np

# Resolve paths dynamically for module usage
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # /workspace/pytlc
grandparent_dir = os.path.dirname(parent_dir)  # /workspace
sys.path.insert(0, grandparent_dir)

import pytlc
from pytlc import (
    read_stl_with_trimesh,
    create_tetrahedral_mesh_from_surface,
    check_injectivity,
    find_injective_mapping,
    compute_shear_stress_hyperelastic,
    compute_von_mises_stress,
    write_vtk_file,
    diagnose_mesh_quality,
    plot_mesh_quality_heatmap,
    plot_stress_distribution,
    generate_diagnostic_report,
    tet_signed_volume,
)


def apply_simple_shear(vertices, shear_factor=0.3):
    """
    Apply a simple shear deformation to the mesh.
    
    Parameters
    ----------
    vertices : ndarray
        Vertex positions (n_vertices x 3)
    shear_factor : float
        Shear strain amount (gamma)
    
    Returns
    -------
    deformed_vertices : ndarray
        Deformed vertex positions
    deformation_gradient : ndarray
        Average deformation gradient tensor (3x3)
    """
    n_verts = len(vertices)
    centroid = np.mean(vertices, axis=0)
    
    # Center vertices
    centered = vertices - centroid
    
    # Apply simple shear in x-direction (F = I + gamma * e_x ⊗ e_y)
    # x' = x + gamma * y
    # y' = y
    # z' = z
    deformed = centered.copy()
    deformed[:, 0] += shear_factor * centered[:, 1]
    
    # Translate back
    deformed_vertices = deformed + centroid
    
    # Construct deformation gradient F
    F = np.eye(3)
    F[0, 1] = shear_factor
    
    return deformed_vertices, F


def main():
    print("=" * 70)
    print("STL to Stress Processing Pipeline")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load STL file
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading STL surface mesh...")
    stl_path = os.path.join(parent_dir, 'data', 'bunny.stl')
    
    if not os.path.exists(stl_path):
        print(f"ERROR: STL file not found at {stl_path}")
        print("Please ensure bunny.stl exists in pytlc/data/")
        return False
    
    mesh_data = read_stl_with_trimesh(stl_path)
    surface_vertices = mesh_data['vertices']
    surface_faces = mesh_data['faces']
    
    print(f"  Loaded surface mesh: {len(surface_vertices)} vertices, {len(surface_faces)} faces")
    
    # -------------------------------------------------------------------------
    # Step 2: Convert to tetrahedral volume mesh
    # -------------------------------------------------------------------------
    print("\n[Step 2] Converting surface to tetrahedral volume mesh...")
    vol_mesh = create_tetrahedral_mesh_from_surface(
        surface_vertices,
        surface_faces,
        method='centroid'
    )
    
    vertices = vol_mesh['vertices']
    tetrahedra = vol_mesh['tetrahedra']
    
    print(f"  Created volume mesh: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
    
    # -------------------------------------------------------------------------
    # Step 3: Check and ensure injectivity
    # -------------------------------------------------------------------------
    print("\n[Step 3] Checking mesh injectivity...")
    
    is_injective, min_vol = check_injectivity(vertices, tetrahedra)
    n_negative = sum(1 for i in range(len(tetrahedra)) 
                     if tet_signed_volume(vertices[tetrahedra[i]]) / 6.0 < 0)
    
    print(f"  Minimum element volume: {min_vol:.6e}")
    print(f"  Number of inverted elements: {n_negative}")
    
    if n_negative > 0:
        print("  Running TLC optimization to fix inverted elements...")
        opt_result = find_injective_mapping(
            vertices,
            tetrahedra,
            boundary_fixed=True,
            max_iterations=100,
            verbose=False
        )
        
        vertices = opt_result['optimized_vertices']
        
        # Re-check
        is_injective, min_vol = check_injectivity(vertices, tetrahedra)
        n_negative = sum(1 for i in range(len(tetrahedra))
                         if tet_signed_volume(vertices[tetrahedra[i]]) / 6.0 < 0)
        print(f"  After optimization: {n_negative} inverted elements")
        
        if n_negative > 0:
            print("  WARNING: Some elements remain inverted. Results may be inaccurate.")
    else:
        print("  Mesh is already injective ✓")
    
    # -------------------------------------------------------------------------
    # Step 4: Diagnose mesh quality
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running mesh quality diagnostics...")
    
    quality_diag = diagnose_mesh_quality(vertices, tetrahedra)
    avg_quality = quality_diag.details['mean_quality']
    min_quality = quality_diag.details['min_quality']
    
    print(f"  Average element quality: {avg_quality:.4f}")
    print(f"  Minimum element quality: {min_quality:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Apply virtual deformation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Applying virtual shear deformation...")
    
    shear_factor = 0.3
    deformed_vertices, F_def = apply_simple_shear(vertices, shear_factor=shear_factor)
    
    print(f"  Applied shear strain: gamma = {shear_factor}")
    print(f"  Deformation gradient F:")
    print(f"    [[{F_def[0,0]:.2f}, {F_def[0,1]:.2f}, {F_def[0,2]:.2f}],")
    print(f"     [{F_def[1,0]:.2f}, {F_def[1,1]:.2f}, {F_def[1,2]:.2f}],")
    print(f"     [{F_def[2,0]:.2f}, {F_def[2,1]:.2f}, {F_def[2,2]:.2f}]]")
    
    # -------------------------------------------------------------------------
    # Step 6: Compute hyperelastic stress
    # -------------------------------------------------------------------------
    print("\n[Step 6] Computing hyperelastic stress (Neo-Hookean)...")
    
    # Material parameters (rubber-like material)
    shear_modulus = 1.0e6  # Pa (1 MPa)
    bulk_modulus = 2.0e6   # Pa (2 MPa)
    
    # Compute deformation gradients for each tetrahedron
    from pytlc.diagnostics import compute_deformation_gradients
    
    F_elements = compute_deformation_gradients(vertices, deformed_vertices, tetrahedra)
    
    # Compute stress for each tetrahedron
    stress_tensors = compute_shear_stress_hyperelastic(
        F_elements,
        shear_modulus=shear_modulus,
        bulk_modulus=bulk_modulus
    )
    
    # Compute Von Mises stress
    von_mises = np.array([compute_von_mises_stress(s) for s in stress_tensors])
    
    print(f"  Computed stress for {len(stress_tensors)} elements")
    print(f"  Von Mises stress range: [{von_mises.min():.2f}, {von_mises.max():.2f}] Pa")
    print(f"  Mean Von Mises stress: {von_mises.mean():.2f} Pa")
    
    # Map element stresses to vertices (for visualization)
    vertex_stress = np.zeros(len(vertices))
    vertex_counts = np.zeros(len(vertices))
    
    for i, tet in enumerate(tetrahedra):
        for v_idx in tet:
            vertex_stress[v_idx] += von_mises[i]
            vertex_counts[v_idx] += 1
    
    vertex_stress /= np.maximum(vertex_counts, 1)
    
    print(f"  Vertex Von Mises stress range: [{vertex_stress.min():.2f}, {vertex_stress.max():.2f}] Pa")
    
    # -------------------------------------------------------------------------
    # Step 7: Visual diagnostics
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating visual diagnostics...")
    
    output_dir = os.path.join(grandparent_dir, 'diagnostic_reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot mesh quality heatmap
    quality_plot_path = os.path.join(output_dir, 'stl_stress_mesh_quality.png')
    plot_mesh_quality_heatmap(
        vertices,
        tetrahedra,
        title="Mesh Quality (STL to Stress Pipeline)",
        output_path=quality_plot_path,
        show=False
    )
    print(f"  Saved mesh quality plot: {quality_plot_path}")
    
    # Plot stress distribution
    stress_plot_path = os.path.join(output_dir, 'stl_stress_distribution.png')
    plot_stress_distribution(
        deformed_vertices,
        tetrahedra,
        vertex_stress,
        title="Von Mises Stress Distribution",
        cmap='hot',
        output_path=stress_plot_path,
        show=False
    )
    print(f"  Saved stress distribution plot: {stress_plot_path}")
    
    # Generate comprehensive report
    report_path = os.path.join(output_dir, 'stl_stress_full_report')
    metadata = {
        'input_file': stl_path,
        'n_vertices': len(vertices),
        'n_tetrahedra': len(tetrahedra),
        'shear_strain': shear_factor,
        'shear_modulus_Pa': shear_modulus,
        'bulk_modulus_Pa': bulk_modulus,
        'max_von_mises_Pa': float(von_mises.max()),
        'mean_von_mises_Pa': float(von_mises.mean())
    }
    
    generate_diagnostic_report(
        vertices,
        tetrahedra,
        deformed_vertices=deformed_vertices,
        stress_values=vertex_stress,
        output_prefix=report_path,
        metadata=metadata
    )
    print(f"  Saved full diagnostic report: {report_path}_*.png")
    
    # -------------------------------------------------------------------------
    # Step 8: Export results
    # -------------------------------------------------------------------------
    print("\n[Step 8] Exporting results...")
    
    # Export deformed mesh with stress to VTK
    vtk_path = os.path.join(grandparent_dir, 'data', 'bunny_stress_result.vtk')
    
    # Prepare point data
    point_data = {
        'von_mises_stress': vertex_stress,
        'displacement_x': deformed_vertices[:, 0] - vertices[:, 0],
        'displacement_y': deformed_vertices[:, 1] - vertices[:, 1],
        'displacement_z': deformed_vertices[:, 2] - vertices[:, 2]
    }
    
    write_vtk_file(
        vtk_path,
        deformed_vertices,
        tetrahedra,
        point_data=point_data,
        cell_data={'element_von_mises': von_mises}
    )
    print(f"  Saved VTK result: {vtk_path}")
    
    # Also save optimized TLC format
    tlc_path = os.path.join(grandparent_dir, 'data', 'bunny_stress_result.tlc')
    pytlc.write_result_file(tlc_path, deformed_vertices, tetrahedra)
    print(f"  Saved TLC result: {tlc_path}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"\nResults summary:")
    print(f"  • Input: {stl_path}")
    print(f"  • Vertices: {len(vertices)}")
    print(f"  • Tetrahedra: {len(tetrahedra)}")
    print(f"  • Applied shear: γ = {shear_factor}")
    print(f"  • Max Von Mises stress: {von_mises.max():.2f} Pa")
    print(f"  • Mean Von Mises stress: {von_mises.mean():.2f} Pa")
    print(f"\nOutput files:")
    print(f"  • {vtk_path}")
    print(f"  • {tlc_path}")
    print(f"  • {quality_plot_path}")
    print(f"  • {stress_plot_path}")
    print(f"  • {report_path}_*.png")
    print("\nTo visualize results, open the VTK file in ParaView or similar software.")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
