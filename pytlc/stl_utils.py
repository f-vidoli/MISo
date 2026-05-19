"""
STL file utilities for pytlc.

This module provides functions to read STL files and convert them
to the format needed for TLC energy minimization.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def read_stl_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read an STL file and extract vertices and faces.
    
    Parameters
    ----------
    filepath : str
        Path to the STL file
        
    Returns
    -------
    dict
        Dictionary containing:
        - vertices: Vertex positions, shape (n_verts, 3)
        - faces: Face connectivity, shape (n_faces, 3)
        - face_normals: Face normals, shape (n_faces, 3)
    """
    try:
        from stl import mesh
    except ImportError:
        raise ImportError("numpy-stl is required. Install with: pip install numpy-stl")
    
    # Load the STL mesh
    stl_mesh = mesh.Mesh.from_file(filepath)
    
    # Extract unique vertices
    all_vertices = stl_mesh.vectors.reshape(-1, 3)
    
    # Find unique vertices and build face index mapping
    vertices, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)
    
    # Rebuild faces using unique vertex indices
    n_faces = len(stl_mesh.vectors)
    faces = inverse_indices.reshape(n_faces, 3)
    
    # Get face normals
    face_normals = stl_mesh.normals.copy()
    
    return {
        'vertices': vertices.astype(np.float64),
        'faces': faces.astype(np.int32),
        'face_normals': face_normals.astype(np.float64),
    }


def read_stl_with_trimesh(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read an STL file using trimesh library.
    
    Parameters
    ----------
    filepath : str
        Path to the STL file
        
    Returns
    -------
    dict
        Dictionary containing:
        - vertices: Vertex positions, shape (n_verts, 3)
        - faces: Face connectivity, shape (n_faces, 3)
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required. Install with: pip install trimesh")
    
    # Load the mesh
    mesh_obj = trimesh.load(filepath)
    
    # Handle scene objects (multiple meshes)
    if isinstance(mesh_obj, trimesh.Scene):
        # Combine all meshes in the scene
        meshes = [g for g in mesh_obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError("No valid meshes found in STL file")
        mesh_obj = trimesh.util.concatenate(meshes)
    
    return {
        'vertices': mesh_obj.vertices.astype(np.float64),
        'faces': mesh_obj.faces.astype(np.int32),
    }


def create_tetrahedral_mesh_from_surface(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    method: str = 'centroid'
) -> Dict[str, np.ndarray]:
    """
    Create a tetrahedral mesh from a surface triangle mesh.
    
    This is needed because TLC works on volumetric meshes (tetrahedra).
    For a closed surface mesh, we can create tetrahedra by connecting
    each surface triangle to an interior point.
    
    Parameters
    ----------
    surface_vertices : np.ndarray
        Surface mesh vertices, shape (n_verts, 3)
    surface_faces : np.ndarray
        Surface mesh faces, shape (n_faces, 3)
    method : str
        Method for creating tetrahedral mesh:
        - 'centroid': Connect all faces to the centroid
        - 'origin': Connect all faces to the origin (if inside)
        
    Returns
    -------
    dict
        Dictionary containing:
        - vertices: Tetrahedral mesh vertices (includes interior point)
        - tetrahedra: Tetrahedron connectivity, shape (n_tets, 4)
    """
    if method == 'centroid':
        # Compute centroid of all vertices
        interior_point = np.mean(surface_vertices, axis=0, keepdims=True)
    elif method == 'origin':
        interior_point = np.array([[0.0, 0.0, 0.0]])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Check orientation of surface faces
    # Ensure all faces are oriented so that tetrahedra have positive volume
    # when connected to the interior point
    
    # For each face, compute the signed volume of the tet formed with interior point
    n_faces = len(surface_faces)
    tetrahedra = np.zeros((n_faces, 4), dtype=np.int32)
    
    interior_idx = len(surface_vertices)
    
    for i in range(n_faces):
        face = surface_faces[i]
        # Try original orientation
        pts_orig = np.vstack([
            surface_vertices[face[0]],
            surface_vertices[face[1]],
            surface_vertices[face[2]],
            interior_point[0]
        ])
        
        from .core import tet_signed_volume
        vol_orig = tet_signed_volume(pts_orig)
        
        if vol_orig > 0:
            # Original orientation gives positive volume
            tetrahedra[i] = [face[0], face[1], face[2], interior_idx]
        else:
            # Flip orientation
            tetrahedra[i] = [face[1], face[0], face[2], interior_idx]
    
    # Add interior point to vertices
    all_vertices = np.vstack([surface_vertices, interior_point])
    
    return {
        'vertices': all_vertices,
        'tetrahedra': tetrahedra,
    }


def create_tetrahedral_mesh_with_interior_points(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    n_interior_layers: int = 1,
    shrink_factor: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Create a tetrahedral mesh with multiple interior layers.
    
    This creates a more robust volumetric mesh by adding multiple
    layers of interior points, which gives the optimizer more
    degrees of freedom.
    
    Parameters
    ----------
    surface_vertices : np.ndarray
        Surface mesh vertices, shape (n_verts, 3)
    surface_faces : np.ndarray
        Surface mesh faces, shape (n_faces, 3)
    n_interior_layers : int
        Number of interior vertex layers to create
    shrink_factor : float
        Factor to shrink interior layers toward centroid
        
    Returns
    -------
    dict
        Dictionary containing:
        - vertices: All vertices (surface + interior)
        - tetrahedra: Tetrahedron connectivity
        - boundary_vertices: Indices of surface vertices
        - interior_vertices: Indices of interior vertices
    """
    # Compute centroid
    centroid = np.mean(surface_vertices, axis=0)
    
    # Start with surface vertices
    all_vertices = [surface_vertices.copy()]
    vertex_offset = len(surface_vertices)
    boundary_indices = np.arange(len(surface_vertices), dtype=np.int32)
    
    # Create interior layers
    interior_indices = []
    for layer in range(n_interior_layers):
        # Shrink factor for this layer (closer to centroid for inner layers)
        alpha = shrink_factor ** (layer + 1)
        
        # Create interior vertices by shrinking toward centroid
        interior_layer = centroid + alpha * (surface_vertices - centroid)
        all_vertices.append(interior_layer)
        
        interior_indices.extend(range(vertex_offset, vertex_offset + len(interior_layer)))
        vertex_offset += len(interior_layer)
    
    all_vertices = np.vstack(all_vertices)
    
    # Check face orientation
    face_centers = np.mean(surface_vertices[surface_faces], axis=1)
    face_normals = np.cross(
        surface_vertices[surface_faces[:, 1]] - surface_vertices[surface_faces[:, 0]],
        surface_vertices[surface_faces[:, 2]] - surface_vertices[surface_faces[:, 0]]
    )
    vectors_to_center = centroid - face_centers
    dot_products = np.sum(face_normals * vectors_to_center, axis=1)
    
    # Flip faces where normal points toward interior
    oriented_faces = surface_faces.copy()
    oriented_faces[dot_products > 0] = surface_faces[dot_products > 0][:, [0, 2, 1]]
    
    # Create tetrahedra
    tetrahedra_list = []
    
    # Layer 0: Surface to first interior layer
    for i, face in enumerate(oriented_faces):
        # Tet from surface face to corresponding first interior vertex
        interior_v = len(surface_vertices)  # First interior layer starts here
        tet = [interior_v + face[0], interior_v + face[1], interior_v + face[2], face[0]]
        tetrahedra_list.append(tet)
        
        # Also create side tets connecting surface edges to interior
        # This creates a proper volumetric mesh
        for j in range(3):
            v1 = face[j]
            v2 = face[(j + 1) % 3]
            iv1 = interior_v + face[j]
            iv2 = interior_v + face[(j + 1) % 3]
            # Quad split into two triangles, then to tet with center
            center_iv = interior_v + len(surface_vertices) if n_interior_layers > 1 else interior_v
            tetrahedra_list.append([v1, v2, iv2, iv1])
    
    # For simplicity, use the basic centroid method if only one layer
    if n_interior_layers == 1:
        return create_tetrahedral_mesh_from_surface(
            surface_vertices, surface_faces, method='centroid'
        )
    
    return {
        'vertices': all_vertices,
        'tetrahedra': np.array(tetrahedra_list, dtype=np.int32) if tetrahedra_list else np.zeros((0, 4), dtype=np.int32),
        'boundary_vertices': boundary_indices,
        'interior_vertices': np.array(interior_indices, dtype=np.int32),
    }


def stl_to_tlc_input(
    filepath: str,
    use_trimesh: bool = True,
    tet_method: str = 'centroid'
) -> Dict[str, np.ndarray]:
    """
    Convert an STL file to TLC input format.
    
    This function reads an STL surface mesh and converts it to the
    format needed for TLC energy minimization by:
    1. Reading the surface mesh from STL
    2. Creating a tetrahedral mesh from the surface
    3. Setting up initial and rest configurations
    
    Parameters
    ----------
    filepath : str
        Path to the STL file
    use_trimesh : bool
        Use trimesh library for reading (recommended)
    tet_method : str
        Method for tetrahedralization
        
    Returns
    -------
    dict
        Dictionary containing:
        - rest_vertices: Source mesh vertices (same as init for surface-only)
        - init_vertices: Initial embedding vertices
        - simplices: Tetrahedron connectivity
        - handles: Boundary vertex indices (all surface vertices)
        - boundary_vertices: Indices of surface/boundary vertices
        - interior_vertices: Indices of interior vertices
    """
    # Read STL file
    if use_trimesh:
        mesh_data = read_stl_with_trimesh(filepath)
    else:
        mesh_data = read_stl_file(filepath)
    
    surface_vertices = mesh_data['vertices']
    surface_faces = mesh_data['faces']
    
    # Create tetrahedral mesh
    tet_mesh = create_tetrahedral_mesh_from_surface(
        surface_vertices, surface_faces, method=tet_method
    )
    
    all_vertices = tet_mesh['vertices']
    tetrahedra = tet_mesh['tetrahedra']
    
    # All surface vertices are boundary/handles
    n_surface = len(surface_vertices)
    boundary_vertices = np.arange(n_surface, dtype=np.int32)
    interior_vertices = np.array([n_surface], dtype=np.int32)  # Just the centroid
    
    # For surface-only input, rest and init are the same
    # The optimization will move the interior point to make all tets positive
    rest_vertices = all_vertices.copy()
    init_vertices = all_vertices.copy()
    
    # Handles are the boundary vertices (fixed during optimization)
    handles = boundary_vertices
    
    return {
        'rest_vertices': rest_vertices,
        'init_vertices': init_vertices,
        'simplices': tetrahedra,
        'handles': handles,
        'boundary_vertices': boundary_vertices,
        'interior_vertices': interior_vertices,
        'surface_faces': surface_faces,
    }


def check_mesh_orientation(vertices: np.ndarray, tetrahedra: np.ndarray) -> Tuple[bool, float]:
    """
    Check if all tetrahedra have positive orientation.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions
    tetrahedra : np.ndarray
        Tetrahedron connectivity
        
    Returns
    -------
    tuple
        (all_positive, min_volume)
    """
    from .core import tet_signed_volume
    
    n_tets = len(tetrahedra)
    min_volume = float('inf')
    
    for i in range(n_tets):
        pts = vertices[tetrahedra[i]]
        vol = tet_signed_volume(pts) / 6.0
        min_volume = min(min_volume, vol)
    
    return min_volume > 0, min_volume
