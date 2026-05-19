#!/usr/bin/env python3
"""
Example: 2D triangle mesh injective mapping using TLC.

This example demonstrates how to use pytlc to compute a locally injective
mapping for a 2D triangle mesh with a non-injective initial configuration.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace')

from pytlc import find_injective_mapping, check_injectivity, read_input_file


def create_example_2d():
    """Create a simple 2D example with folded triangles."""
    # Create a simple 2x2 grid of vertices
    rest_vertices = np.array([
        [0.0, 0.0],   # 0
        [1.0, 0.0],   # 1
        [0.0, 1.0],   # 2
        [1.0, 1.0],   # 3
        [0.5, 0.5],   # 4 (center)
    ], dtype=np.float64)
    
    # Create triangles (rest mesh)
    simplices = np.array([
        [0, 1, 4],  # bottom-right
        [1, 3, 4],  # top-right
        [3, 2, 4],  # top-left
        [2, 0, 4],  # bottom-left
    ], dtype=np.int32)
    
    # Create a non-injective initial embedding (folded)
    init_vertices = rest_vertices.copy()
    # Fold the center vertex to create inverted triangles
    init_vertices[4] = [0.5, 0.3]  # Move center down
    
    # Boundary vertices (handles) - all except center
    handles = np.array([0, 1, 2, 3], dtype=np.int32)
    
    return rest_vertices, init_vertices, simplices, handles


def test_with_original_example():
    """Test with the original TLC example file."""
    input_file = '/workspace/original_repo/example/input'
    
    print(f"Reading input file: {input_file}")
    data = read_input_file(input_file)
    
    rest_vertices = data['rest_vertices']
    init_vertices = data['init_vertices']
    simplices = data['simplices']
    handles = data['handles']
    
    print(f"\nMesh statistics:")
    print(f"  Vertices: {len(rest_vertices)}")
    print(f"  Dimension: {rest_vertices.shape[1]}")
    print(f"  Simplices: {len(simplices)}")
    print(f"  Handles: {len(handles)}")
    
    # Check initial injectivity
    is_inj, min_m = check_injectivity(init_vertices, simplices)
    print(f"\nInitial state:")
    print(f"  Injective: {is_inj}")
    print(f"  Min measure: {min_m:.6e}")
    
    # Run optimization
    print("\nRunning TLC optimization...")
    result = find_injective_mapping(
        rest_vertices=rest_vertices,
        init_vertices=init_vertices,
        simplices=simplices,
        handles=handles,
        form='Tutte',
        alpha_ratio=1e-6,
        max_iterations=1000,
        record_energy=True,
        record_min_measure=True,
        stop_when_injective=True,
        verbose=True
    )
    
    # Check final injectivity
    final_vertices = result['vertices']
    is_inj, min_m = check_injectivity(final_vertices, simplices)
    print(f"\nFinal state:")
    print(f"  Injective: {is_inj}")
    print(f"  Min measure: {min_m:.6e}")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final energy: {result['energy']:.6e}")
    
    if result.get('energy_history'):
        print(f"\nEnergy history (first 5): {result['energy_history'][:5]}")
        print(f"Energy history (last 5): {result['energy_history'][-5:]}")
    
    if result.get('min_measure_history'):
        print(f"\nMin measure history (first 5): {result['min_measure_history'][:5]}")
        print(f"Min measure history (last 5): {result['min_measure_history'][-5:]}")
    
    return result


def test_simple_2d():
    """Test with a simple 2D example."""
    print("=" * 60)
    print("Simple 2D Example")
    print("=" * 60)
    
    rest_vertices, init_vertices, simplices, handles = create_example_2d()
    
    print(f"\nMesh statistics:")
    print(f"  Vertices: {len(rest_vertices)}")
    print(f"  Simplices: {len(simplices)}")
    print(f"  Handles: {len(handles)}")
    
    # Check initial injectivity
    is_inj, min_m = check_injectivity(init_vertices, simplices)
    print(f"\nInitial state:")
    print(f"  Injective: {is_inj}")
    print(f"  Min measure: {min_m:.6e}")
    
    # Run optimization
    print("\nRunning TLC optimization...")
    result = find_injective_mapping(
        rest_vertices=rest_vertices,
        init_vertices=init_vertices,
        simplices=simplices,
        handles=handles,
        form='Tutte',
        alpha_ratio=1e-6,
        max_iterations=500,
        record_energy=True,
        record_min_measure=True,
        stop_when_injective=True,
        verbose=True
    )
    
    return result


if __name__ == '__main__':
    print("pytlc - Python wrapper for Lifting Simplices to Find Injectivity")
    print("=" * 60)
    
    # Test with simple 2D example
    try:
        test_simple_2d()
    except Exception as e:
        print(f"Simple 2D test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    
    # Test with original example
    try:
        test_with_original_example()
    except Exception as e:
        print(f"Original example test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
