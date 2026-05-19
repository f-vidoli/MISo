"""
Mesh utility functions for TLC.

This module provides helper functions for mesh processing tasks such as
extracting boundary vertices and computing edge lengths.
"""

import numpy as np
from typing import Tuple, List, Set


def extract_boundary_vertices(simplices: np.ndarray) -> np.ndarray:
    """
    Extract boundary vertices from a mesh.
    
    For a triangle mesh, boundary edges are those that appear in exactly one triangle.
    For a tetrahedral mesh, boundary faces are those that appear in exactly one tetrahedron.
    
    Parameters
    ----------
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    np.ndarray
        Indices of boundary vertices
    """
    dim = simplices.shape[1] - 1
    
    if dim == 2:  # Triangle mesh
        return _extract_boundary_vertices_2d(simplices)
    else:  # Tetrahedral mesh
        return _extract_boundary_vertices_3d(simplices)


def _extract_boundary_vertices_2d(triangles: np.ndarray) -> np.ndarray:
    """Extract boundary vertices from a triangle mesh."""
    n_triangles = len(triangles)
    
    # Count edge occurrences
    edge_count = {}
    
    for i in range(n_triangles):
        tri = triangles[i]
        # Three edges per triangle
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    # Collect boundary vertices
    boundary_verts = set()
    for edge in boundary_edges:
        boundary_verts.add(edge[0])
        boundary_verts.add(edge[1])
    
    return np.array(sorted(boundary_verts), dtype=np.int32)


def _extract_boundary_vertices_3d(tetrahedra: np.ndarray) -> np.ndarray:
    """Extract boundary vertices from a tetrahedral mesh."""
    n_tets = len(tetrahedra)
    
    # Count face occurrences
    face_count = {}
    
    for i in range(n_tets):
        tet = tetrahedra[i]
        # Four faces per tetrahedron
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for face in faces:
            face_count[face] = face_count.get(face, 0) + 1
    
    # Boundary faces appear exactly once
    boundary_faces = [face for face, count in face_count.items() if count == 1]
    
    # Collect boundary vertices
    boundary_verts = set()
    for face in boundary_faces:
        for v in face:
            boundary_verts.add(v)
    
    return np.array(sorted(boundary_verts), dtype=np.int32)


def compute_squared_edge_lengths(vertices: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """
    Compute squared edge lengths for all simplices.
    
    This is a convenience wrapper around the core function.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    np.ndarray
        Squared edge lengths, shape (n_simplices, n_edges)
    """
    from .core import compute_squared_edge_lengths as core_func
    return core_func(vertices, simplices)


def get_simplex_edges(simplices: np.ndarray) -> np.ndarray:
    """
    Get all unique edges from a mesh.
    
    Parameters
    ----------
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    np.ndarray
        Unique edges, shape (n_edges, 2)
    """
    dim = simplices.shape[1]
    n_simplices = len(simplices)
    n_edges_per_simplex = dim * (dim + 1) // 2
    
    edges = []
    for i in range(n_simplices):
        simplex = simplices[i]
        for j in range(dim):
            for k in range(j + 1, dim + 1):
                edge = tuple(sorted([simplex[j], simplex[k]]))
                edges.append(edge)
    
    # Remove duplicates
    unique_edges = list(set(edges))
    return np.array(unique_edges, dtype=np.int32)


def get_simplex_faces(simplices: np.ndarray) -> np.ndarray:
    """
    Get all unique faces from a tetrahedral mesh.
    
    Parameters
    ----------
    simplices : np.ndarray
        Tetrahedron connectivity, shape (n_tets, 4)
        
    Returns
    -------
    np.ndarray
        Unique faces, shape (n_faces, 3)
    """
    n_tets = len(simplices)
    
    faces = []
    for i in range(n_tets):
        tet = simplices[i]
        # Four faces per tetrahedron
        face_list = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        faces.extend(face_list)
    
    # Remove duplicates
    unique_faces = list(set(faces))
    return np.array(unique_faces, dtype=np.int32)


def compute_vertex_adjacency(simplices: np.ndarray, n_verts: int) -> List[Set[int]]:
    """
    Compute vertex adjacency information.
    
    Parameters
    ----------
    simplices : np.ndarray
        Simplex connectivity
    n_verts : int
        Total number of vertices
        
    Returns
    -------
    list
        List of sets, where adjacency[i] contains indices of vertices adjacent to vertex i
    """
    adjacency = [set() for _ in range(n_verts)]
    
    for simplex in simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                adjacency[simplex[i]].add(simplex[j])
                adjacency[simplex[j]].add(simplex[i])
    
    return adjacency


def compute_vertex_valence(simplices: np.ndarray, n_verts: int) -> np.ndarray:
    """
    Compute vertex valence (number of adjacent vertices).
    
    Parameters
    ----------
    simplices : np.ndarray
        Simplex connectivity
    n_verts : int
        Total number of vertices
        
    Returns
    -------
    np.ndarray
        Valence for each vertex
    """
    adjacency = compute_vertex_adjacency(simplices, n_verts)
    return np.array([len(adj) for adj in adjacency], dtype=np.int32)
