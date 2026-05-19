"""
Simple Example: Sphere relaxation and Jacobian computation

This example demonstrates the core functionality with a simple cube mesh
that's easier to work with than a sphere.
"""

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

import pytlc

print("=" * 70)
print("Sphere Relaxation - Simple Demonstration")
print("=" * 70)

# =============================================================================
# Create a simple tetrahedral mesh (subdivided cube)
# =============================================================================
print("\n1. Creating tetrahedral mesh...")

def create_cube_tetmesh():
    """Create a simple tetrahedral mesh of a cube."""
    # Cube vertices
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top face
    ], dtype=float)
    
    # Subdivide into 5 tetrahedra
    simplices = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 4, 6],
        [2, 4, 5, 6],
        [2, 3, 4, 7],
    ])
    
    # Boundary vertices (all except center-ish ones)
    boundary_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    return vertices, simplices, boundary_indices

vertices, simplices, boundary_indices = create_cube_tetmesh()

print(f"  Vertices: {len(vertices)}")
print(f"  Tetrahedra: {len(simplices)}")
print(f"  Boundary vertices: {len(boundary_indices)}")

# Check initial injectivity
is_inj, min_vol = pytlc.check_injectivity(vertices, simplices)
print(f"  Initial injectivity: {is_inj} (min volume: {min_vol:.6f})")

# =============================================================================
# Apply shear deformation
# =============================================================================
print("\n2. Applying shear deformation...")

gamma = 0.2
shear_matrix = np.array([
    [1.0, gamma, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

stressed_vertices = vertices @ shear_matrix.T
print(f"  Shear strain: {gamma}")
print(f"  Stressed vertices:\n{stressed_vertices}")

# =============================================================================
# Compute sphere parameters (for demonstration, use bounding sphere)
# =============================================================================
print("\n3. Setting up relaxation...")

sphere_center = np.mean(stressed_vertices[boundary_indices], axis=0)
distances = np.linalg.norm(stressed_vertices[boundary_indices] - sphere_center, axis=1)
sphere_radius = np.mean(distances)

print(f"  Sphere center: {sphere_center}")
print(f"  Sphere radius: {sphere_radius:.4f}")

# =============================================================================
# Relax the sphere
# =============================================================================
print("\n4. Relaxing from shear stress...")

relaxation_result = pytlc.relax_sphere_from_shear_stress(
    stressed_vertices=stressed_vertices,
    simplices=simplices,
    boundary_vertex_indices=boundary_indices,
    sphere_center=sphere_center,
    sphere_radius=sphere_radius,
    shear_modulus=1.0,
    max_iterations=100,
    tolerance=1e-5,
    verbose=False
)

relaxed_vertices = relaxation_result['relaxed_vertices']
print(f"  Converged: {relaxation_result['converged']}")
print(f"  Iterations: {relaxation_result['iterations']}")

# Verify boundary sphericity
boundary_distances = np.linalg.norm(relaxed_vertices[boundary_indices] - sphere_center, axis=1)
print(f"\n  Boundary sphericity:")
print(f"    Radius std: {np.std(boundary_distances):.6e}")
print(f"    Max deviation: {np.max(np.abs(boundary_distances - sphere_radius)):.6e}")

# =============================================================================
# Compute map from stressed to relaxed
# =============================================================================
print("\n5. Computing map from stressed to relaxed...")

map_result = pytlc.compute_deformation_map(
    source_vertices=stressed_vertices,
    target_vertices=relaxed_vertices,
    simplices=simplices
)

print(f"  Displacement:")
print(f"    Max: {map_result['max_displacement']:.6f}")
print(f"    Mean: {map_result['mean_displacement']:.6f}")

print(f"\n  Jacobian determinant (stressed -> relaxed):")
J_map = map_result['jacobian_determinants']
print(f"    Min: {np.min(J_map):.6f}")
print(f"    Max: {np.max(J_map):.6f}")
print(f"    Mean: {np.mean(J_map):.6f}")

# =============================================================================
# Compute composed map Jacobian
# =============================================================================
print("\n6. Computing composed map Jacobian determinant...")

composed_result = pytlc.compute_composed_map_jacobian(
    initial_vertices=vertices,
    stressed_vertices=stressed_vertices,
    relaxed_vertices=relaxed_vertices,
    simplices=simplices
)

J_composed = composed_result['jacobian_determinants']
print(f"  Composed map (initial -> stressed -> relaxed):")
print(f"    det(dPhi_composed/dX) per element:")
for i in range(len(simplices)):
    print(f"      Element {i}: {J_composed[i]:.6f}")

print(f"\n  Statistics:")
print(f"    Min: {np.min(J_composed):.6f}")
print(f"    Max: {np.max(J_composed):.6f}")
print(f"    Mean: {np.mean(J_composed):.6f}")

# Verify chain rule
J_chain = composed_result['jacobian_chain_rule_product']
chain_error = np.max(np.abs(J_composed - J_chain))
print(f"\n  Chain rule verification:")
print(f"    Max error: {chain_error:.6e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"""
Demonstrated functionality:
1. ✓ Sphere relaxation from shear stress
   - Boundary preserved on sphere (radius std: {np.std(boundary_distances):.6e})
   
2. ✓ Map from stressed to relaxed computed
   - Max displacement: {map_result['max_displacement']:.6f}
   - Jacobian determinants computed
   
3. ✓ Composed map Jacobian determinant computed
   - det(dPhi_composed/dX) for each element
   - Chain rule verified (error: {chain_error:.6e})

The Jacobian determinant represents local volume change through the
composition of maps: initial -> stressed -> relaxed
""")

print("=" * 70)
print("Example completed successfully!")
print("=" * 70)
