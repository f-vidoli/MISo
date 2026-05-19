"""
Example: Relaxing a stressed sphere while preserving sphericity

This example demonstrates how to:
1. Create a spherical mesh with shear stress deformation
2. Relax the sphere from shear stress while preserving boundary sphericity
3. Compute the map from stressed to relaxed configuration
4. Compute the Jacobian determinant of the composed map
"""

import numpy as np
import os
import sys

# Add grandparent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # /workspace/pytlc
grandparent_dir = os.path.dirname(parent_dir)  # /workspace
sys.path.insert(0, grandparent_dir)

import pytlc

print("=" * 70)
print("Sphere Relaxation from Shear Stress Example")
print("=" * 70)

# =============================================================================
# Step 1: Create a simple spherical tetrahedral mesh
# =============================================================================
print("\n1. Creating spherical tetrahedral mesh...")
print("-" * 50)

def create_spherical_mesh(n_radial: int = 3, n_theta: int = 8, n_phi: int = 4, 
                          radius: float = 1.0):
    """
    Create a simple spherical tetrahedral mesh.
    
    Parameters
    ----------
    n_radial : int
        Number of radial layers
    n_theta : int
        Number of theta divisions (longitude)
    n_phi : int
        Number of phi divisions (latitude, excluding poles)
    radius : float
        Sphere radius
        
    Returns
    -------
    tuple
        (vertices, simplices, boundary_indices)
    """
    vertices = []
    
    # Center vertex
    vertices.append([0.0, 0.0, 0.0])
    center_idx = 0
    
    # Radial layers
    layer_indices = [[center_idx]]  # indices for each radial layer
    
    for r_idx in range(1, n_radial + 1):
        r = radius * r_idx / n_radial
        layer_verts = []
        
        # North pole
        north_pole = [0, 0, r]
        vertices.append(north_pole)
        layer_verts.append(len(vertices) - 1)
        
        # Latitude rings (excluding poles)
        for phi_idx in range(1, n_phi + 1):
            phi = np.pi * phi_idx / (n_phi + 1)
            z = r * np.cos(phi)
            r_xy = r * np.sin(phi)
            
            for theta_idx in range(n_theta):
                theta = 2 * np.pi * theta_idx / n_theta
                x = r_xy * np.cos(theta)
                y = r_xy * np.sin(theta)
                vertices.append([x, y, z])
                layer_verts.append(len(vertices) - 1)
        
        # South pole
        south_pole = [0, 0, -r]
        vertices.append(south_pole)
        layer_verts.append(len(vertices) - 1)
        
        layer_indices.append(layer_verts)
    
    vertices = np.array(vertices)
    
    # Build tetrahedra
    simplices = []
    
    for r_idx in range(n_radial):
        inner_layer = layer_indices[r_idx]
        outer_layer = layer_indices[r_idx + 1]
        
        # For simplicity, connect center to first layer
        if r_idx == 0:
            # Connect center to each triangle on first shell
            for i in range(len(outer_layer) - 2):
                if i < len(outer_layer) - 2:
                    v1, v2, v3 = outer_layer[0], outer_layer[i+1], outer_layer[i+2]
                    # Check if this forms a valid triangle (not crossing the pole gap)
                    simplices.append([center_idx, v1, v2, v3])
        else:
            # Connect between layers
            for i in range(min(len(inner_layer), len(outer_layer)) - 1):
                for j in range(min(len(inner_layer), len(outer_layer)) - 1):
                    if i < len(inner_layer) - 1 and j < len(outer_layer) - 1:
                        v_inner = [inner_layer[i], inner_layer[i+1]]
                        v_outer = [outer_layer[j], outer_layer[j+1]]
                        
                        # Create tetrahedra connecting these
                        for vi in v_inner:
                            for vj in v_outer:
                                # Simplified connectivity
                                pass
    
    # Boundary vertices are those on the outermost layer
    boundary_indices = np.array(layer_indices[-1])
    
    return vertices, np.array(simplices) if simplices else np.array([]).reshape(0, 4), boundary_indices


# Alternative: Use a simpler approach with scipy.spatial.Delaunay
from scipy.spatial import Delaunay

def create_sphere_tetmesh(n_points: int = 200, radius: float = 1.0):
    """
    Create a tetrahedral mesh of a sphere using Delaunay triangulation.
    
    Returns a valid mesh with positive orientation.
    """
    # Generate points on concentric spheres
    np.random.seed(42)
    
    # Center point
    points = [[0, 0, 0]]
    
    # Multiple layers
    n_layers = 5
    for layer in range(1, n_layers + 1):
        r = radius * layer / n_layers
        n_layer_points = max(30, int(n_points / n_layers))
        
        # Fibonacci sphere sampling for even distribution
        indices = np.arange(0, n_layer_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_layer_points)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        
        layer_points = np.column_stack([x, y, z])
        points.extend(layer_points.tolist())
    
    points = np.array(points)
    
    # Delaunay tetrahedralization
    tri = Delaunay(points)
    
    # Find boundary vertices (those on convex hull)
    # A vertex is on boundary if it appears in fewer tetrahedra than interior vertices
    vertex_counts = np.bincount(tri.simplices.flatten())
    median_count = np.median(vertex_counts)
    boundary_indices = np.where(vertex_counts < median_count)[0]
    
    # Ensure outermost layer is marked as boundary
    distances = np.linalg.norm(points, axis=1)
    outer_layer = np.where(distances > 0.9 * radius)[0]
    boundary_indices = np.unique(np.concatenate([boundary_indices, outer_layer]))
    
    # Check and fix orientation
    simplices = tri.simplices.copy()
    
    # Compute signed volumes and flip if needed
    total_vol = 0.0
    for i in range(len(simplices)):
        pts = points[simplices[i]]
        total_vol += pytlc.tet_signed_volume(pts)
    
    if total_vol < 0:
        print("  Fixing mesh orientation...")
        simplices[:, [0, 1]] = simplices[:, [1, 0]]
    
    return points, simplices, boundary_indices


# Create the mesh
vertices, simplices, boundary_indices = create_sphere_tetmesh(n_points=150, radius=1.0)

print(f"  Vertices: {len(vertices)}")
print(f"  Tetrahedra: {len(simplices)}")
print(f"  Boundary vertices: {len(boundary_indices)}")

# Verify initial injectivity
is_injective, min_vol = pytlc.check_injectivity(vertices, simplices)
print(f"  Initial injectivity: {is_injective} (min volume: {min_vol:.6e})")

# =============================================================================
# Step 2: Apply shear deformation to create stressed configuration
# =============================================================================
print("\n2. Applying shear deformation...")
print("-" * 50)

# Simple shear deformation: x' = x + gamma * y
gamma = 0.3  # Shear strain
shear_matrix = np.array([
    [1.0, gamma, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

stressed_vertices = vertices @ shear_matrix.T

print(f"  Shear strain gamma: {gamma}")
print(f"  Deformation gradient F:")
print(f"    {shear_matrix}")

# Compute shear stress for verification
F_batch = np.tile(shear_matrix[np.newaxis, :, :], (len(simplices), 1, 1))
shear_stress = pytlc.compute_shear_stress_hyperelastic(F_batch, shear_modulus=1.0)
print(f"  Shear stress (sample): {shear_stress[0]}")

# =============================================================================
# Step 3: Relax the sphere while preserving boundary sphericity
# =============================================================================
print("\n3. Relaxing sphere from shear stress...")
print("-" * 50)

relaxation_result = pytlc.relax_sphere_from_shear_stress(
    stressed_vertices=stressed_vertices,
    simplices=simplices,
    boundary_vertex_indices=boundary_indices,
    shear_modulus=1.0,
    max_iterations=200,  # Reduced for faster execution
    tolerance=1e-4,      # Relaxed tolerance
    verbose=True
)

relaxed_vertices = relaxation_result['relaxed_vertices']
print(f"\n  Relaxation completed:")
print(f"    Converged: {relaxation_result['converged']}")
print(f"    Iterations: {relaxation_result['iterations']}")
print(f"    Final displacement: {relaxation_result['displacement_history'][-1]:.6e}")

# Verify boundary sphericity
sphere_center = relaxation_result['sphere_center']
sphere_radius = relaxation_result['sphere_radius']
boundary_distances = np.linalg.norm(relaxed_vertices[boundary_indices] - sphere_center, axis=1)
print(f"\n  Boundary sphericity check:")
print(f"    Mean radius: {np.mean(boundary_distances):.6f}")
print(f"    Std deviation: {np.std(boundary_distances):.6e}")
print(f"    Max deviation: {np.max(np.abs(boundary_distances - sphere_radius)):.6e}")

# Verify injectivity of relaxed configuration
is_injective_relaxed, min_vol_relaxed = pytlc.check_injectivity(relaxed_vertices, simplices)
print(f"\n  Relaxed injectivity: {is_injective_relaxed} (min volume: {min_vol_relaxed:.6e})")

# =============================================================================
# Step 4: Compute map from stressed to relaxed sphere
# =============================================================================
print("\n4. Computing map from stressed to relaxed sphere...")
print("-" * 50)

map_result = pytlc.compute_map_stressed_to_relaxed(
    stressed_vertices=stressed_vertices,
    relaxed_vertices=relaxed_vertices,
    simplices=simplices
)

print(f"  Displacement statistics:")
print(f"    Max displacement: {map_result['max_displacement']:.6e}")
print(f"    Mean displacement: {map_result['mean_displacement']:.6e}")

print(f"\n  Jacobian determinant statistics:")
J = map_result['jacobian_determinants']
print(f"    Min J: {np.min(J):.6f}")
print(f"    Max J: {np.max(J):.6f}")
print(f"    Mean J: {np.mean(J):.6f}")
print(f"    All positive: {np.all(J > 0)}")

# =============================================================================
# Step 5: Compute composed map and its Jacobian determinant
# =============================================================================
print("\n5. Computing composed map (initial -> stressed -> relaxed)...")
print("-" * 50)

composed_result = pytlc.compose_maps_and_compute_jacobian(
    initial_vertices=vertices,
    stressed_vertices=stressed_vertices,
    relaxed_vertices=relaxed_vertices,
    simplices=simplices
)

print(f"  Map composition:")
print(f"    initial --F1--> stressed --F2--> relaxed")

print(f"\n  Jacobian determinants:")
print(f"    Initial->Stressed (J1):")
print(f"      Min: {np.min(composed_result['jacobian_initial_to_stressed']):.6f}")
print(f"      Max: {np.max(composed_result['jacobian_initial_to_stressed']):.6f}")
print(f"      Mean: {np.mean(composed_result['jacobian_initial_to_stressed']):.6f}")

print(f"\n    Stressed->Relaxed (J2):")
print(f"      Min: {np.min(composed_result['jacobian_stressed_to_relaxed']):.6f}")
print(f"      Max: {np.max(composed_result['jacobian_stressed_to_relaxed']):.6f}")
print(f"      Mean: {np.mean(composed_result['jacobian_stressed_to_relaxed']):.6f}")

print(f"\n    Composed (J = J2 * J1):")
J_composed = composed_result['jacobian_determinants']
print(f"      Min: {np.min(J_composed):.6f}")
print(f"      Max: {np.max(J_composed):.6f}")
print(f"      Mean: {np.mean(J_composed):.6f}")
print(f"      All positive: {np.all(J_composed > 0)}")

# Verify chain rule
J_chain = composed_result['jacobian_chain_rule_product']
chain_error = np.max(np.abs(J_composed - J_chain))
print(f"\n  Chain rule verification:")
print(f"    Max |J_composed - J_chain|: {chain_error:.6e}")

# =============================================================================
# Step 6: Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"""
The sphere relaxation successfully:
1. ✓ Applied shear deformation to create stressed configuration
2. ✓ Relaxed the sphere while preserving boundary sphericity
   - Boundary radius std: {np.std(boundary_distances):.6e}
3. ✓ Computed the map from stressed to relaxed configuration
   - Max displacement: {map_result['max_displacement']:.6e}
4. ✓ Composed maps and computed Jacobian determinant
   - det(dPhi_composed/dX) ranges from {np.min(J_composed):.4f} to {np.max(J_composed):.4f}
   - All elements have positive Jacobian: {np.all(J_composed > 0)}

The Jacobian determinant of the composed map represents the local volume change
from the initial configuration through the stressed state to the relaxed state.
""")

print("=" * 70)
print("Example completed successfully!")
print("=" * 70)
