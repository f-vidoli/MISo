"""
MISo - Morphoelastic Inverse problem Solver

A Python implementation of the morphoelastic inverse problem solver based on
the Total Lifted Content (TLC) energy minimization method from the paper
"Lifting Simplices to Find Injectivity" by Xingyi Du et al. (SIGGRAPH 2020).

MISo computes the Jacobian determinant J = det(F) representing the local volume
change required to transform a stressed (deformed) configuration into a relaxed
(stress-free) configuration while preserving boundary constraints.

Key Features:
- Robust boundary-preserving relaxation with sphericity constraints
- Deformation gradient tensor computation
- Jacobian determinant (the MISo solution)
- Composed map analysis for multi-stage deformations
- Complete pipeline from STL files to Jacobian fields
- Command-line interface for batch processing

Basic Usage:
    >>> from pytlc import stl_to_jacobian_pipeline
    >>> jacobian = stl_to_jacobian_pipeline('input.stl')
    
    >>> # Or use individual components:
    >>> from pytlc import (
    ...     relax_boundary_preserving_sphericity,
    ...     compute_morphoelastic_jacobian
    ... )
    >>> relaxed = relax_boundary_preserving_sphericity(...)
    >>> J = compute_morphoelastic_jacobian(stressed, relaxed, simplices)

Command Line:
    $ miso stl-to-jacobian input.stl output.vtk --high-precision
    $ miso relax deformed.stl relaxed.vtk
    $ miso diagnose mesh.stl
"""

from .core import (
    find_injective_mapping,
    check_injectivity,
    tri_signed_area,
    tet_signed_volume,
    tri_area_squared_edges,
    tet_volume_squared_edges,
    compute_shear_stress_from_transformation_strain,
    compute_von_mises_stress,
    compute_shear_stress_hyperelastic,
    compute_cauchy_stress_hyperelastic,
    # Renamed functions with MISo terminology
    relax_boundary_preserving_sphericity,
    compute_deformation_gradient_tensor,
    compute_morphoelastic_jacobian,
    compute_composed_map_jacobian,
    compute_deformation_map,
    evaluate_map_at_points,
)

# Import aliases - define locally rather than importing
tri_area = tri_area_squared_edges
tet_volume = tet_volume_squared_edges

# Aliases for backward compatibility - define locally
relax_sphere_from_shear_stress = relax_boundary_preserving_sphericity
compute_deformation_gradient_map = compute_deformation_gradient_tensor
compute_jacobian_determinant_map = compute_morphoelastic_jacobian
compose_maps_and_compute_jacobian = compute_composed_map_jacobian
compute_map_stressed_to_relaxed = compute_deformation_map

# Alias for convenience - define locally
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

__version__ = '0.2.0'
__author__ = 'MISo Development Team'
__all__ = [
    # Core TLC energy minimization
    'find_injective_mapping',
    'check_injectivity',
    'tri_signed_area',
    'tet_signed_volume',
    'tri_area',
    'tet_volume',
    # I/O utilities
    'read_input_file',
    'write_result_file',
    'read_solver_options',
    'write_obj_file',
    'write_vtk_file',
    # Mesh utilities
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
    'compute_hyperelastic_stress',
    # MISo core functions (new naming convention)
    'relax_boundary_preserving_sphericity',
    'compute_deformation_gradient_tensor',
    'compute_morphoelastic_jacobian',
    'compute_composed_map_jacobian',
    'compute_deformation_map',
    'evaluate_map_at_points',
    # Aliases for backward compatibility
    'relax_sphere_from_shear_stress',
    'compute_deformation_gradient_map',
    'compute_jacobian_determinant_map',
    'compose_maps_and_compute_jacobian',
    'compute_map_stressed_to_relaxed',
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
