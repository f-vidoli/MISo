"""
File I/O utilities for reading and writing TLC format files.

This module provides functions to read input files in the original TLC format
and write result files.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def read_input_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read an input file in the TLC format.
    
    The input file format is:
        [num_sourceVert] [dimension_sourceVert]
        ... (num_sourceVert * dimension_sourceVert) Matrix ...
        [num_initVert]   [dimension_initVert]
        ... (num_initVert * dimension_initVert) Matrix ...
        [num_simplex]    [simplex_size]
        ... (num_simplex * simplex_size) Matrix ...
        [num_handles]
        ... (num_handles * 1) Matrix ...
    
    Parameters
    ----------
    filepath : str
        Path to the input file
        
    Returns
    -------
    dict
        Dictionary containing:
        - rest_vertices: Source mesh vertices
        - init_vertices: Initial embedding vertices
        - simplices: Mesh connectivity
        - handles: Constrained vertex indices
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    
    # Read rest vertices
    parts = lines[idx].strip().split()
    n_rest, dim_rest = int(parts[0]), int(parts[1])
    idx += 1
    
    rest_vertices = np.zeros((n_rest, dim_rest))
    for i in range(n_rest):
        rest_vertices[i] = [float(x) for x in lines[idx].strip().split()]
        idx += 1
    
    # Read initial vertices
    parts = lines[idx].strip().split()
    n_init, dim_init = int(parts[0]), int(parts[1])
    idx += 1
    
    init_vertices = np.zeros((n_init, dim_init))
    for i in range(n_init):
        init_vertices[i] = [float(x) for x in lines[idx].strip().split()]
        idx += 1
    
    # Read simplices
    parts = lines[idx].strip().split()
    n_simplices, simplex_size = int(parts[0]), int(parts[1])
    idx += 1
    
    simplices = np.zeros((n_simplices, simplex_size), dtype=np.int32)
    for i in range(n_simplices):
        simplices[i] = [int(x) for x in lines[idx].strip().split()]
        idx += 1
    
    # Read handles
    parts = lines[idx].strip().split()
    n_handles = int(parts[0])
    idx += 1
    
    handles = np.zeros(n_handles, dtype=np.int32)
    for i in range(n_handles):
        handles[i] = int(lines[idx].strip())
        idx += 1
    
    return {
        'rest_vertices': rest_vertices,
        'init_vertices': init_vertices,
        'simplices': simplices,
        'handles': handles,
    }


def read_solver_options(filepath: str) -> Dict[str, Any]:
    """
    Read solver options file in the TLC format.
    
    Parameters
    ----------
    filepath : str
        Path to the solver options file
        
    Returns
    -------
    dict
        Dictionary containing solver options
    """
    options = {
        'form': 'Tutte',
        'alpha_ratio': 1e-6,
        'alpha': -1.0,
        'ftol_abs': 1e-8,
        'ftol_rel': 1e-8,
        'xtol_abs': 1e-8,
        'xtol_rel': 1e-8,
        'max_iterations': 10000,
        'algorithm': 'LBFGS',
        'stop_code': 'all_good',
        'record_vert': False,
        'record_energy': False,
        'record_min_area': False,
    }
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue
            
            if line == 'form':
                idx += 1
                options['form'] = lines[idx].strip()
            elif line == 'alphaRatio':
                idx += 1
                options['alpha_ratio'] = float(lines[idx].strip())
            elif line == 'alpha':
                idx += 1
                options['alpha'] = float(lines[idx].strip())
            elif line == 'ftol_abs':
                idx += 1
                options['ftol_abs'] = float(lines[idx].strip())
            elif line == 'ftol_rel':
                idx += 1
                options['ftol_rel'] = float(lines[idx].strip())
            elif line == 'xtol_abs':
                idx += 1
                options['xtol_abs'] = float(lines[idx].strip())
            elif line == 'xtol_rel':
                idx += 1
                options['xtol_rel'] = float(lines[idx].strip())
            elif line == 'maxeval':
                idx += 1
                options['max_iterations'] = int(lines[idx].strip())
            elif line == 'algorithm':
                idx += 1
                options['algorithm'] = lines[idx].strip()
            elif line == 'stopCode':
                idx += 1
                options['stop_code'] = lines[idx].strip()
            elif line == 'record':
                # Read vert flag
                idx += 1
                if lines[idx].strip().startswith('vert'):
                    parts = lines[idx].strip().split()
                    options['record_vert'] = int(parts[1]) > 0
                
                # Read energy flag
                idx += 1
                if lines[idx].strip().startswith('energy'):
                    parts = lines[idx].strip().split()
                    options['record_energy'] = int(parts[1]) > 0
                
                # Read minArea flag
                idx += 1
                if lines[idx].strip().startswith('minArea'):
                    parts = lines[idx].strip().split()
                    options['record_min_area'] = int(parts[1]) > 0
            
            idx += 1
    except FileNotFoundError:
        print(f"Warning: Solver options file not found: {filepath}, using defaults.")
    
    return options


def write_result_file(
    filepath: str,
    vertices: np.ndarray,
    rest_vertices: Optional[np.ndarray] = None,
    simplices: Optional[np.ndarray] = None,
    energy_history: Optional[List[float]] = None,
    min_measure_history: Optional[List[float]] = None,
) -> None:
    """
    Write a result file in the TLC format.
    
    Parameters
    ----------
    filepath : str
        Path to the output file
    vertices : np.ndarray
        Result vertex positions
    rest_vertices : np.ndarray, optional
        Rest mesh vertices (for reference)
    simplices : np.ndarray, optional
        Simplex connectivity (for reference)
    energy_history : list, optional
        Energy values per iteration
    min_measure_history : list, optional
        Minimum measure values per iteration
    """
    n_verts, dim = vertices.shape
    
    with open(filepath, 'w') as f:
        # Set high precision
        np.set_printoptions(precision=16)
        
        # Write result vertices
        f.write(f"resV {n_verts} {dim}\n")
        for i in range(n_verts):
            for j in range(dim):
                f.write(f"{vertices[i, j]} ")
            f.write("\n")
        
        # Write energy history if available
        if energy_history is not None:
            f.write(f"energy {len(energy_history)}\n")
            for e in energy_history:
                f.write(f"{e} ")
            f.write("\n")
        
        # Write min measure history if available
        if min_measure_history is not None:
            f.write(f"minArea {len(min_measure_history)}\n")
            for m in min_measure_history:
                f.write(f"{m} ")
            f.write("\n")


def write_obj_file(filepath: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Write mesh to OBJ format.
    
    Parameters
    ----------
    filepath : str
        Output file path
    vertices : np.ndarray
        Vertex positions
    faces : np.ndarray
        Face connectivity
    """
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]}")
            if len(v) > 2:
                f.write(f" {v[2]}")
            f.write("\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write("f")
            for idx in face:
                f.write(f" {idx + 1}")
            f.write("\n")


def write_vtk_file(filepath: str, vertices: np.ndarray, cells: np.ndarray) -> None:
    """
    Write mesh to VTK format (legacy).
    
    Parameters
    ----------
    filepath : str
        Output file path
    vertices : np.ndarray
        Vertex positions
    cells : np.ndarray
        Cell connectivity
    """
    n_verts = len(vertices)
    n_cells = len(cells)
    cell_type = cells.shape[1]
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("TLC Result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        f.write(f"POINTS {n_verts} float\n")
        for v in vertices:
            f.write(" ".join(map(str, v)) + "\n")
        
        # Cells
        n_cell_entries = n_cells * (cell_type + 1)
        f.write(f"CELLS {n_cells} {n_cell_entries}\n")
        for cell in cells:
            f.write(f"{cell_type} " + " ".join(map(str, cell)) + "\n")
        
        # Cell types
        f.write(f"CELL_TYPES {n_cells}\n")
        if cell_type == 3:
            cell_vtk_type = 5  # VTK_TRIANGLE
        elif cell_type == 4:
            cell_vtk_type = 10  # VTK_TETRA
        else:
            cell_vtk_type = 1  # VTK_VERTEX
        
        for _ in range(n_cells):
            f.write(f"{cell_vtk_type}\n")
