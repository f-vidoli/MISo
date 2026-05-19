"""
Example: Using pytlc with an STL surface mesh (Stanford Bunny)

This example demonstrates how to:
1. Load a surface mesh from an STL file
2. Convert it to a tetrahedral mesh for TLC processing
3. Find an injective mapping using TLC energy minimization
4. Verify the result
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace')

import pytlc


def main():
    # Path to the STL file
    stl_path = '/workspace/data/bunny.stl'
    
    print("=" * 60)
    print("TLC Injectivity Example - Stanford Bunny")
    print("=" * 60)
    
    # Step 1: Load STL and convert to TLC input format
    print("\n[Step 1] Loading STL file and creating tetrahedral mesh...")
    tlc_input = pytlc.stl_to_tlc_input(stl_path, use_trimesh=True)
    
    rest_vertices = tlc_input['rest_vertices']
    init_vertices = tlc_input['init_vertices']
    simplices = tlc_input['simplices']  # tetrahedra
    
    # For this simple example, we'll only fix a few boundary vertices
    # to anchor the mesh, and let the rest move freely
    boundary_vertices = tlc_input['boundary_vertices']
    
    # Select a small subset of boundary vertices as handles (anchors)
    # This gives the optimizer more freedom to find an injective mapping
    n_handles = max(3, len(boundary_vertices) // 100)  # About 1% as handles
    handle_indices = boundary_vertices[::max(1, len(boundary_vertices) // n_handles)][:n_handles]
    handles = np.array(handle_indices, dtype=np.int32)
    
    print(f"  Surface vertices: {len(boundary_vertices)}")
    print(f"  Interior vertices: {len(tlc_input['interior_vertices'])}")
    print(f"  Total vertices: {len(rest_vertices)}")
    print(f"  Tetrahedra: {len(simplices)}")
    print(f"  Fixed (handle) vertices: {len(handles)}")
    print(f"  Free vertices: {len(rest_vertices) - len(handles)}")
    
    # Step 2: Check initial configuration
    print("\n[Step 2] Checking initial mesh configuration...")
    is_injective, min_vol = pytlc.check_injectivity(init_vertices, simplices)
    print(f"  Initially injective: {is_injective}")
    print(f"  Minimum tetrahedron volume: {min_vol:.6e}")
    
    # Also check orientation
    all_positive, min_vol_check = pytlc.check_mesh_orientation(init_vertices, simplices)
    print(f"  All tetrahedra positive: {all_positive}")
    
    # Step 3: Run TLC optimization to find injective mapping
    print("\n[Step 3] Running TLC optimization...")
    result = pytlc.find_injective_mapping(
        rest_vertices=rest_vertices,
        init_vertices=init_vertices,
        simplices=simplices,
        handles=handles,
        form='harmonic',  # or 'Tutte'
        alpha_ratio=1e-4,  # Increased for better convergence
        max_iterations=10000,
        stop_when_injective=True,
        verbose=True,
        record_energy=True,
        record_min_measure=True,
    )
    
    # Step 4: Analyze results
    print("\n[Step 4] Results:")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final energy: {result['energy']:.6e}")
    
    # Check final injectivity
    final_vertices = result['vertices']
    is_injective_final, min_vol_final = pytlc.check_injectivity(final_vertices, simplices)
    print(f"  Finally injective: {is_injective_final}")
    print(f"  Final minimum volume: {min_vol_final:.6e}")
    
    # Show energy history if recorded
    if 'energy_history' in result:
        energy_hist = result['energy_history']
        print(f"\n  Energy history:")
        print(f"    Initial: {energy_hist[0]:.6e}")
        print(f"    Final: {energy_hist[-1]:.6e}")
        if energy_hist[0] > 0:
            print(f"    Reduction: {(energy_hist[0] - energy_hist[-1]) / energy_hist[0] * 100:.2f}%")
    
    # Show min measure history if recorded
    if 'min_measure_history' in result:
        min_hist = result['min_measure_history']
        print(f"\n  Min measure history:")
        print(f"    Initial: {min_hist[0]:.6e}")
        print(f"    Final: {min_hist[-1]:.6e}")
    
    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    
    # Save in TLC format
    output_path = '/workspace/data/bunny_result.tlc'
    pytlc.write_result_file(
        output_path,
        vertices=final_vertices,
        energy_history=result.get('energy_history'),
        min_measure_history=result.get('min_measure_history'),
    )
    print(f"  Saved TLC result to: {output_path}")
    
    # Save as OBJ for visualization (using surface faces)
    obj_path = '/workspace/data/bunny_result.obj'
    surface_faces = tlc_input['surface_faces']
    pytlc.write_obj_file(obj_path, final_vertices[:len(surface_faces)], surface_faces)
    print(f"  Saved OBJ file to: {obj_path}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    result = main()
