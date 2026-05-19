"""
pytlc - Python implementation of Lifting Simplices to Find Injectivity

A Python wrapper for the Total Lifted Content (TLC) energy minimization method
from the paper "Lifting Simplices to Find Injectivity" by Xingyi Du et al. (SIGGRAPH 2020).

This module provides tools for computing locally injective mappings for triangular
and tetrahedral meshes by minimizing the TLC energy using gradient-based optimization.
"""

from .core import (
    find_injective_mapping,
    check_injectivity,
    tri_signed_area,
    tet_signed_volume,
    tri_area_squared_edges as tri_area,
    tet_volume_squared_edges as tet_volume,
    compute_shear_stress_from_transformation_strain,
    compute_von_mises_stress,
    compute_shear_stress_hyperelastic,
    compute_cauchy_stress_hyperelastic,
)

# Alias for convenience
compute_hyperelastic_stress = compute_shear_stress_hyperelastic

from .io_utils import (
    read_input_file,
    write_result_file,
    read_solver_options,
    write_obj_file,
    write_vtk_file,
)
from .mesh_utils import (
    extract_boundary_vertices,
    compute_squared_edge_lengths,
)
from .stl_utils import (
    read_stl_file,
    read_stl_with_trimesh,
    stl_to_tlc_input,
    create_tetrahedral_mesh_from_surface,
    check_mesh_orientation,
)
from .diagnostics import (
    run_full_diagnostic,
    diagnose_mesh_quality,
    diagnose_material_stability,
    diagnose_numerical_conditioning,
    diagnose_stress_concentration,
    DiagnosticResult,
)
from .visual_diagnostics import (
    plot_mesh_quality_heatmap,
    plot_stress_distribution,
    plot_deformation_comparison,
    plot_histogram_metrics,
    generate_diagnostic_report,
)

__version__ = '0.1.0'
__author__ = 'Based on work by Xingyi Du et al.'
__all__ = [
    'find_injective_mapping',
    'check_injectivity',
    'tri_signed_area',
    'tet_signed_volume',
    'tri_area',
    'tet_volume',
    'read_input_file',
    'write_result_file',
    'read_solver_options',
    'write_obj_file',
    'write_vtk_file',
    'extract_boundary_vertices',
    'compute_squared_edge_lengths',
    # STL utilities
    'read_stl_file',
    'read_stl_with_trimesh',
    'stl_to_tlc_input',
    'create_tetrahedral_mesh_from_surface',
    'check_mesh_orientation',
    # Stress computation utilities
    'compute_shear_stress_from_transformation_strain',
    'compute_von_mises_stress',
    'compute_shear_stress_hyperelastic',
    'compute_cauchy_stress_hyperelastic',
    # Diagnostics
    'run_full_diagnostic',
    'diagnose_mesh_quality',
    'diagnose_material_stability',
    'diagnose_numerical_conditioning',
    'diagnose_stress_concentration',
    'DiagnosticResult',
    # Visual diagnostics
    'plot_mesh_quality_heatmap',
    'plot_stress_distribution',
    'plot_deformation_comparison',
    'plot_histogram_metrics',
    'generate_diagnostic_report',
]
