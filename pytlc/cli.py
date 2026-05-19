#!/usr/bin/env python3
"""
MISo - Morphoelastic Inverse problem Solver Command Line Interface

This CLI provides a robust and advanced pipeline for computing the morphoelastic
Jacobian determinant from STL files. The Jacobian represents the MISo solution
for a given deformed shape.

Usage:
    miso stl-to-jacobian input.stl [output.vtk] [options]
    miso relax input.stl [output.stl] [options]
    miso diagnose input.stl [options]

Examples:
    # Basic usage with default robust settings
    miso stl-to-jacobian heart.stl jacobian.vtk
    
    # With custom material parameters
    miso stl-to-jacobian artery.stl output.vtk --shear-modulus 1e5 --bulk-modulus 1e8
    
    # High precision mode
    miso stl-to-jacobian microstructure.stl output.vtk --high-precision
    
    # Relax a stressed configuration
    miso relax deformed.stl relaxed.stl --max-iterations 10000
"""

import argparse
import sys
import os
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

# Import core MISo functionality
from .core import (
    relax_boundary_preserving_sphericity,
    compute_deformation_gradient_tensor,
    compute_morphoelastic_jacobian,
    compute_deformation_map,
    compute_composed_map_jacobian,
    find_injective_mapping,
    check_injectivity,
)
from .stl_utils import (
    read_stl_file,
    create_tetrahedral_mesh_from_surface,
    extract_boundary_vertices,
)
from .io_utils import write_vtk_file


def setup_robust_defaults(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Setup the most robust and advanced method parameters as defaults.
    
    These settings are optimized for numerical stability and accuracy
    in solving the morphoelastic inverse problem.
    """
    params = {
        # Optimization parameters
        'max_iterations': args.max_iterations if hasattr(args, 'max_iterations') else 10000,
        'tolerance': args.tolerance if hasattr(args, 'tolerance') else 1e-10,
        'ftol_rel': args.ftol_rel if hasattr(args, 'ftol_rel') else 1e-10,
        'xtol_rel': args.xtol_rel if hasattr(args, 'xtol_rel') else 1e-10,
        
        # Material parameters
        'shear_modulus': args.shear_modulus if hasattr(args, 'shear_modulus') else 1.0,
        'bulk_modulus': args.bulk_modulus if hasattr(args, 'bulk_modulus') else None,
        
        # TLC energy parameters
        'form': 'harmonic',  # More robust for large deformations
        'alpha_ratio': args.alpha_ratio if hasattr(args, 'alpha_ratio') else 1e-8,
        
        # Numerical stability
        'stop_when_injective': True,
        'verbose': args.verbose if hasattr(args, 'verbose') else False,
    }
    
    # High precision mode overrides
    if hasattr(args, 'high_precision') and args.high_precision:
        params['tolerance'] = 1e-12
        params['ftol_rel'] = 1e-12
        params['xtol_rel'] = 1e-12
        params['alpha_ratio'] = 1e-10
        params['max_iterations'] = max(params['max_iterations'], 20000)
    
    # Set bulk modulus if not provided (nearly incompressible)
    if params['bulk_modulus'] is None:
        params['bulk_modulus'] = 1000.0 * params['shear_modulus']
    
    return params


def cmd_stl_to_jacobian(args: argparse.Namespace) -> int:
    """
    Compute the Jacobian determinant (MISo solution) from an STL file.
    
    This is the main MISo pipeline:
    1. Read STL surface mesh
    2. Generate tetrahedral volume mesh
    3. Identify boundary vertices
    4. Relax the configuration while preserving boundary sphericity
    5. Compute deformation gradient and Jacobian determinant
    6. Write results to VTK file
    """
    print("=" * 70)
    print("MISo - Morphoelastic Inverse problem Solver")
    print("Computing Jacobian determinant from STL file")
    print("=" * 70)
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found")
        return 1
    
    # Setup robust defaults
    params = setup_robust_defaults(args)
    
    print(f"\n[1/6] Reading STL file: {args.input}")
    try:
        vertices, faces = read_stl_file(args.input)
        print(f"  Loaded {len(vertices)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"ERROR: Failed to read STL file: {e}")
        return 1
    
    print(f"\n[2/6] Generating tetrahedral mesh...")
    try:
        tet_vertices, tet_simplices = create_tetrahedral_mesh_from_surface(vertices, faces)
        print(f"  Generated {len(tet_vertices)} vertices, {len(tet_simplices)} tetrahedra")
    except Exception as e:
        print(f"ERROR: Failed to generate tetrahedral mesh: {e}")
        return 1
    
    print(f"\n[3/6] Identifying boundary vertices...")
    boundary_indices = extract_boundary_vertices(faces)
    print(f"  Found {len(boundary_indices)} boundary vertices")
    
    # Check initial injectivity
    is_injective, min_vol = check_injectivity(tet_vertices, tet_simplices)
    print(f"  Initial mesh injectivity: {is_injective} (min volume: {min_vol:.6e})")
    
    print(f"\n[4/6] Relaxing configuration (preserving boundary sphericity)...")
    print(f"  Using parameters:")
    print(f"    - Max iterations: {params['max_iterations']}")
    print(f"    - Tolerance: {params['tolerance']}")
    print(f"    - Shear modulus: {params['shear_modulus']}")
    print(f"    - Bulk modulus: {params['bulk_modulus']}")
    print(f"    - Energy form: {params['form']}")
    
    # Compute sphere parameters for boundary constraint
    boundary_verts = tet_vertices[boundary_indices]
    sphere_center = np.mean(boundary_verts, axis=0)
    sphere_radius = np.mean(np.linalg.norm(boundary_verts - sphere_center, axis=1))
    
    try:
        relaxation_result = relax_boundary_preserving_sphericity(
            stressed_vertices=tet_vertices,
            simplices=tet_simplices,
            boundary_vertex_indices=boundary_indices,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            shear_modulus=params['shear_modulus'],
            bulk_modulus=params['bulk_modulus'],
            max_iterations=params['max_iterations'],
            tolerance=params['tolerance'],
            verbose=params['verbose']
        )
        
        relaxed_vertices = relaxation_result['relaxed_vertices']
        converged = relaxation_result['converged']
        iterations = relaxation_result['iterations']
        
        print(f"  Relaxation completed:")
        print(f"    - Converged: {converged}")
        print(f"    - Iterations: {iterations}")
        print(f"    - Final displacement: {relaxation_result['displacement_history'][-1]:.6e}")
        
        # Verify boundary sphericity
        boundary_distances = np.linalg.norm(relaxed_vertices[boundary_indices] - sphere_center, axis=1)
        radius_std = np.std(boundary_distances)
        print(f"    - Boundary sphericity (radius std): {radius_std:.6e}")
        
    except Exception as e:
        print(f"ERROR: Relaxation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n[5/6] Computing Jacobian determinant (MISo solution)...")
    try:
        # Compute the morphoelastic Jacobian
        jacobian = compute_morphoelastic_jacobian(
            source_vertices=tet_vertices,
            target_vertices=relaxed_vertices,
            simplices=tet_simplices
        )
        
        print(f"  Jacobian statistics:")
        print(f"    - Min: {np.min(jacobian):.6f}")
        print(f"    - Max: {np.max(jacobian):.6f}")
        print(f"    - Mean: {np.mean(jacobian):.6f}")
        print(f"    - Std: {np.std(jacobian):.6f}")
        
        # Check for negative Jacobians (should not happen with robust methods)
        neg_count = np.sum(jacobian <= 0)
        if neg_count > 0:
            print(f"  WARNING: {neg_count} elements have non-positive Jacobian!")
            print(f"  This may indicate mesh quality issues or insufficient relaxation.")
        
    except Exception as e:
        print(f"ERROR: Jacobian computation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n[6/6] Writing results...")
    output_file = args.output if hasattr(args, 'output') and args.output else 'jacobian_result.vtk'
    
    try:
        # Prepare data for VTK output
        cell_data = {
            'Jacobian': jacobian,
        }
        
        point_data = {}
        if hasattr(args, 'export_displacement') and args.export_displacement:
            displacement = relaxed_vertices - tet_vertices
            point_data['Displacement'] = displacement
        
        write_vtk_file(output_file, relaxed_vertices, tet_simplices, 
                      cell_data=cell_data, point_data=point_data)
        print(f"  Written to: {output_file}")
        
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("MISo computation completed successfully!")
    print("=" * 70)
    
    return 0


def cmd_relax(args: argparse.Namespace) -> int:
    """
    Relax a stressed configuration while preserving boundary constraints.
    """
    print("=" * 70)
    print("MISo - Configuration Relaxation")
    print("=" * 70)
    
    # Similar pipeline but outputs relaxed geometry
    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found")
        return 1
    
    params = setup_robust_defaults(args)
    
    print(f"\nReading STL file: {args.input}")
    vertices, faces = read_stl_file(args.input)
    print(f"  Loaded {len(vertices)} vertices, {len(faces)} faces")
    
    print(f"\nGenerating tetrahedral mesh...")
    tet_vertices, tet_simplices = create_tetrahedral_mesh_from_surface(vertices, faces)
    print(f"  Generated {len(tet_vertices)} vertices, {len(tet_simplices)} tetrahedra")
    
    boundary_indices = extract_boundary_vertices(faces)
    print(f"  Found {len(boundary_indices)} boundary vertices")
    
    # Compute sphere parameters
    boundary_verts = tet_vertices[boundary_indices]
    sphere_center = np.mean(boundary_verts, axis=0)
    sphere_radius = np.mean(np.linalg.norm(boundary_verts - sphere_center, axis=1))
    
    print(f"\nRelaxing configuration...")
    relaxation_result = relax_boundary_preserving_sphericity(
        stressed_vertices=tet_vertices,
        simplices=tet_simplices,
        boundary_vertex_indices=boundary_indices,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        **params
    )
    
    # Output relaxed configuration
    output_file = args.output if hasattr(args, 'output') and args.output else 'relaxed_result.vtk'
    
    # Extract surface from relaxed tetrahedral mesh for STL output
    # For now, output as VTK
    write_vtk_file(output_file, relaxation_result['relaxed_vertices'], tet_simplices)
    print(f"\nRelaxed configuration written to: {output_file}")
    
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """
    Run comprehensive diagnostics on an STL file.
    """
    print("=" * 70)
    print("MISo - Mesh Diagnostics")
    print("=" * 70)
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found")
        return 1
    
    from .diagnostics import run_full_diagnostic
    
    vertices, faces = read_stl_file(args.input)
    tet_vertices, tet_simplices = create_tetrahedral_mesh_from_surface(vertices, faces)
    
    print("\nRunning full diagnostic...")
    result = run_full_diagnostic(tet_vertices, tet_simplices)
    
    print("\nDiagnostic Summary:")
    print(f"  Mesh quality: {result.mesh_quality_summary}")
    print(f"  Injectivity: {result.injectivity_check}")
    print(f"  Recommendations: {result.recommendations}")
    
    return 0


def main():
    """Main entry point for MISo CLI."""
    parser = argparse.ArgumentParser(
        prog='miso',
        description='MISo - Morphoelastic Inverse problem Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # stl-to-jacobian command
    p_stl2jac = subparsers.add_parser(
        'stl-to-jacobian',
        help='Compute Jacobian determinant (MISo solution) from STL file'
    )
    p_stl2jac.add_argument('input', help='Input STL file')
    p_stl2jac.add_argument('output', nargs='?', help='Output VTK file (default: jacobian_result.vtk)')
    p_stl2jac.add_argument('--shear-modulus', type=float, default=1.0,
                          help='Shear modulus (default: 1.0)')
    p_stl2jac.add_argument('--bulk-modulus', type=float, default=None,
                          help='Bulk modulus (default: 1000 * shear_modulus)')
    p_stl2jac.add_argument('--max-iterations', type=int, default=10000,
                          help='Maximum relaxation iterations (default: 10000)')
    p_stl2jac.add_argument('--tolerance', type=float, default=1e-10,
                          help='Convergence tolerance (default: 1e-10)')
    p_stl2jac.add_argument('--alpha-ratio', type=float, default=1e-8,
                          help='TLC energy alpha ratio (default: 1e-8)')
    p_stl2jac.add_argument('--high-precision', action='store_true',
                          help='Use high precision settings (tolerance=1e-12)')
    p_stl2jac.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
    p_stl2jac.add_argument('--export-displacement', action='store_true',
                          help='Export displacement field in addition to Jacobian')
    p_stl2jac.set_defaults(func=cmd_stl_to_jacobian)
    
    # relax command
    p_relax = subparsers.add_parser(
        'relax',
        help='Relax a stressed configuration'
    )
    p_relax.add_argument('input', help='Input STL file')
    p_relax.add_argument('output', nargs='?', help='Output VTK file')
    p_relax.add_argument('--shear-modulus', type=float, default=1.0)
    p_relax.add_argument('--bulk-modulus', type=float, default=None)
    p_relax.add_argument('--max-iterations', type=int, default=10000)
    p_relax.add_argument('--tolerance', type=float, default=1e-10)
    p_relax.add_argument('--high-precision', action='store_true')
    p_relax.add_argument('--verbose', '-v', action='store_true')
    p_relax.set_defaults(func=cmd_relax)
    
    # diagnose command
    p_diag = subparsers.add_parser(
        'diagnose',
        help='Run comprehensive mesh diagnostics'
    )
    p_diag.add_argument('input', help='Input STL file')
    p_diag.set_defaults(func=cmd_diagnose)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
