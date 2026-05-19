"""
Example demonstrating diagnostic and visual debugging tools for hyperelastic simulations.

This example shows how to use the diagnostic utilities to identify bugs in mesh quality,
material stability, numerical conditioning, and stress concentrations.
"""

import numpy as np
import sys
import os

# Add grandparent directory to path for imports
# Structure: /workspace/pytlc/examples/example_diagnostics.py
#            ^-- grandparent is /workspace, which contains pytlc package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # /workspace/pytlc
grandparent_dir = os.path.dirname(parent_dir)  # /workspace
sys.path.insert(0, grandparent_dir)

import pytlc


def create_test_mesh_2d_with_issues():
    """Create a 2D triangular mesh with some intentionally bad elements."""
    # Create a simple grid - use proper orientation
    nx, ny = 10, 10
    
    # Regular grid vertices
    vertices = []
    for j in range(ny + 1):  # y first
        for i in range(nx + 1):  # x second
            x = i / nx
            y = j / ny
            vertices.append([x, y])
    
    vertices = np.array(vertices)
    
    # Create triangles with proper counter-clockwise orientation
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            
            # Two triangles per quad - both counter-clockwise
            # Triangle 1: bottom-left, bottom-right, top-left
            elements.append([n0, n1, n2])
            # Triangle 2: bottom-right, top-right, top-left  
            elements.append([n1, n3, n2])
    
    elements = np.array(elements)
    
    # Intentionally create ONE bad element for testing
    # 1. Invert one triangle by swapping vertices (element index 5)
    elements[5] = elements[5][[0, 2, 1]]  # This inverts the triangle
    
    # 2. Make one triangle degenerate (nearly zero area) at index 10
    vertices[elements[10, 0]] = vertices[elements[10, 1]] * 0.999 + vertices[elements[10, 2]] * 0.001
    
    return vertices, elements


def create_test_mesh_3d_with_issues():
    """Create a 3D tetrahedral mesh with some issues."""
    # Simple cube meshed with tetrahedra
    n = 5
    
    vertices = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                vertices.append([i/n, j/n, k/n])
    
    vertices = np.array(vertices)
    
    # Create tetrahedra using a simple pattern
    elements = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                n0 = i * (n+1)**2 + j * (n+1) + k
                n1 = n0 + 1
                n2 = n0 + (n+1)
                n3 = n0 + (n+1)**2
                n4 = n0 + (n+1)**2 + 1
                n5 = n0 + (n+1)**2 + (n+1)
                n6 = n0 + (n+1) + 1
                n7 = n0 + (n+1)**2 + (n+1) + 1
                
                # Split cube into 5-6 tetrahedra (simplified)
                elements.append([n0, n1, n2, n3])
                elements.append([n1, n2, n3, n6])
                elements.append([n1, n3, n4, n6])
                elements.append([n3, n4, n5, n6])
                elements.append([n2, n3, n5, n6])
                elements.append([n1, n4, n6, n7])
    
    elements = np.array(elements)
    
    # Remove duplicate elements and ensure valid indices
    unique_elements = []
    for elem in elements:
        if np.all(elem < len(vertices)):
            unique_elements.append(elem)
    
    elements = np.array(unique_elements)
    
    # Create an inverted element
    if len(elements) > 5:
        elements[5] = elements[5][[0, 2, 1, 3]]  # Swap to invert
    
    return vertices, elements


def test_2d_diagnostics():
    """Test diagnostics on a 2D mesh."""
    print("="*60)
    print("TESTING 2D MESH DIAGNOSTICS")
    print("="*60)
    
    vertices, elements = create_test_mesh_2d_with_issues()
    
    print(f"\nMesh: {len(vertices)} vertices, {len(elements)} elements")
    
    # Run full diagnostic
    results = pytlc.run_full_diagnostic(vertices, elements)
    
    # Print summary
    print("\n" + "-"*60)
    print("DIAGNOSTIC RESULTS:")
    print("-"*60)
    
    for check_name, result in results.items():
        if check_name.startswith('_'):
            continue
        
        status = "✓ PASS" if result.passed else "✗ FAIL"
        severity = result.severity.upper()
        
        print(f"\n[{status}] {check_name} ({severity})")
        print(f"  Message: {result.message}")
        
        if result.critical_elements is not None and len(result.critical_elements) > 0:
            print(f"  Problem elements: {result.critical_elements[:5]}{'...' if len(result.critical_elements) > 5 else ''}")
        
        # Show some details
        if 'min_quality' in result.details:
            print(f"  Min quality: {result.details['min_quality']:.4f}")
        if 'min_jacobian' in result.details:
            print(f"  Min Jacobian: {result.details['min_jacobian']:.4e}")
    
    # Summary
    summary = results['_summary']
    print("\n" + "="*60)
    print(f"OVERALL STATUS: {'PASS ✓' if summary.passed else 'FAIL ✗'}")
    print(f"Issues found: {summary.details['critical']} critical, {summary.details['errors']} errors, {summary.details['warnings']} warnings")
    print("="*60)
    
    return vertices, elements, results


def test_3d_diagnostics():
    """Test diagnostics on a 3D mesh."""
    print("\n" + "="*60)
    print("TESTING 3D MESH DIAGNOSTICS")
    print("="*60)
    
    vertices, elements = create_test_mesh_3d_with_issues()
    
    print(f"\nMesh: {len(vertices)} vertices, {len(elements)} elements")
    
    # Run full diagnostic
    results = pytlc.run_full_diagnostic(vertices, elements)
    
    # Print summary
    print("\n" + "-"*60)
    print("DIAGNOSTIC RESULTS:")
    print("-"*60)
    
    for check_name, result in results.items():
        if check_name.startswith('_'):
            continue
        
        status = "✓ PASS" if result.passed else "✗ FAIL"
        severity = result.severity.upper()
        
        print(f"\n[{status}] {check_name} ({severity})")
        print(f"  Message: {result.message}")
        
        if result.critical_elements is not None and len(result.critical_elements) > 0:
            print(f"  Problem elements: {result.critical_elements[:5]}{'...' if len(result.critical_elements) > 5 else ''}")
    
    # Summary
    summary = results['_summary']
    print("\n" + "="*60)
    print(f"OVERALL STATUS: {'PASS ✓' if summary.passed else 'FAIL ✗'}")
    print(f"Issues found: {summary.details['critical']} critical, {summary.details['errors']} errors, {summary.details['warnings']} warnings")
    print("="*60)
    
    return vertices, elements, results


def test_material_stability():
    """Test material stability diagnostics with deformation gradients."""
    print("\n" + "="*60)
    print("TESTING MATERIAL STABILITY DIAGNOSTICS")
    print("="*60)
    
    # Create a simple mesh
    vertices = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0.5, 0.5]
    ])
    elements = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ])
    
    # Create deformation gradients with various issues
    n_elements = len(elements)
    F = np.zeros((n_elements, 2, 2))
    
    # Normal deformation
    F[0] = [[1.1, 0.0], [0.0, 1.1]]
    
    # Extreme compression (J < 0.1)
    F[1] = [[0.05, 0.0], [0.0, 0.05]]
    
    # Inverted element (negative J)
    F[2] = [[-1.0, 0.0], [0.0, 1.0]]
    
    # Extreme expansion (J > 10)
    F[3] = [[5.0, 0.0], [0.0, 5.0]]
    
    print(f"Created {n_elements} deformation gradients")
    print("F[0]: Normal (J={:.3f})".format(np.linalg.det(F[0])))
    print("F[1]: Extreme compression (J={:.3f})".format(np.linalg.det(F[1])))
    print("F[2]: Inverted (J={:.3f})".format(np.linalg.det(F[2])))
    print("F[3]: Extreme expansion (J={:.3f})".format(np.linalg.det(F[3])))
    
    # Run material stability diagnostic
    result = pytlc.diagnose_material_stability(F, shear_modulus=1.0, bulk_modulus=2.0)
    
    print(f"\nMaterial Stability Result: {result.message}")
    print(f"Severity: {result.severity}")
    
    if result.critical_elements is not None:
        print(f"Problematic elements: {result.critical_elements}")
    
    return vertices, elements, F, result


def test_stress_concentration():
    """Test stress concentration diagnostics."""
    print("\n" + "="*60)
    print("TESTING STRESS CONCENTRATION DIAGNOSTICS")
    print("="*60)
    
    # Create a simple mesh
    vertices = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0.5, 0.5]
    ])
    elements = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ])
    
    # Create stress tensors with one high stress element
    n_elements = len(elements)
    stress = np.zeros((n_elements, 2, 2))
    
    # Normal stress
    stress[0] = [[1.0, 0.0], [0.0, 1.0]]
    stress[1] = [[1.2, 0.1], [0.1, 0.8]]
    stress[2] = [[0.9, -0.1], [-0.1, 1.1]]
    
    # High stress concentration
    stress[3] = [[10.0, 2.0], [2.0, 8.0]]
    
    print(f"Created {n_elements} stress tensors")
    for i in range(n_elements):
        s = stress[i]
        vm = np.sqrt(s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2 + 3*s[0,1]**2)
        print(f"Element {i}: Von Mises = {vm:.3f}")
    
    # Run stress concentration diagnostic
    result = pytlc.diagnose_stress_concentration(stress, threshold_factor=2.0)
    
    print(f"\nStress Concentration Result: {result.message}")
    print(f"Severity: {result.severity}")
    
    if result.critical_elements is not None:
        print(f"Elements with concentration: {result.critical_elements}")
    
    return vertices, elements, stress, result


def test_visual_diagnostics_2d():
    """Test visual diagnostics on 2D mesh."""
    print("\n" + "="*60)
    print("GENERATING VISUAL DIAGNOSTICS (2D)")
    print("="*60)
    
    vertices, elements, _ = test_2d_diagnostics()
    
    output_dir = "./diagnostic_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate mesh quality heatmap
        print("\nGenerating mesh quality heatmap...")
        pytlc.plot_mesh_quality_heatmap(
            vertices, elements,
            output_path=os.path.join(output_dir, "mesh_quality_2d.png"),
            show=False
        )
        print("✓ Saved mesh_quality_2d.png")
        
        # Generate histograms
        print("\nGenerating metric histograms...")
        pytlc.plot_histogram_metrics(
            vertices, elements,
            output_path=os.path.join(output_dir, "metrics_histogram_2d.png"),
            show=False
        )
        print("✓ Saved metrics_histogram_2d.png")
        
        print(f"\n✓ Visual diagnostics saved to {output_dir}/")
        
    except Exception as e:
        print(f"Warning: Could not generate visual diagnostics: {e}")
        import traceback
        traceback.print_exc()


def test_visual_diagnostics_3d():
    """Test visual diagnostics on 3D mesh."""
    print("\n" + "="*60)
    print("GENERATING VISUAL DIAGNOSTICS (3D)")
    print("="*60)
    
    vertices, elements, _ = test_3d_diagnostics()
    
    output_dir = "./diagnostic_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate mesh quality heatmap
        print("\nGenerating 3D mesh quality visualization...")
        pytlc.plot_mesh_quality_heatmap(
            vertices, elements,
            output_path=os.path.join(output_dir, "mesh_quality_3d.png"),
            show=False
        )
        print("✓ Saved mesh_quality_3d.png")
        
        # Generate histograms
        print("\nGenerating 3D metric histograms...")
        pytlc.plot_histogram_metrics(
            vertices, elements,
            output_path=os.path.join(output_dir, "metrics_histogram_3d.png"),
            show=False
        )
        print("✓ Saved metrics_histogram_3d.png")
        
        print(f"\n✓ Visual diagnostics saved to {output_dir}/")
        
    except Exception as e:
        print(f"Warning: Could not generate visual diagnostics: {e}")
        import traceback
        traceback.print_exc()


def test_full_diagnostic_report():
    """Generate a complete diagnostic report."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE DIAGNOSTIC REPORT")
    print("="*60)
    
    # Use 2D mesh for demonstration
    vertices, elements = create_test_mesh_2d_with_issues()
    
    # Create some fake stress data
    n_elements = len(elements)
    stress = np.random.randn(n_elements, 2, 2) * 0.5
    stress[0] = [[5.0, 0.0], [0.0, 5.0]]  # High stress element
    
    output_dir = "./diagnostic_reports"
    
    try:
        report_path = pytlc.generate_diagnostic_report(
            vertices, elements,
            stress_tensors=stress,
            output_dir=output_dir,
            shear_modulus=1.0,
            bulk_modulus=2.0,
            show=False
        )
        
        print(f"\n✓ Full diagnostic report generated at: {report_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate full report: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostic tests."""
    print("\n" + "#"*60)
    print("# PYTLC DIAGNOSTIC TOOLS DEMONSTRATION")
    print("#"*60 + "\n")
    
    # Test 2D diagnostics
    test_2d_diagnostics()
    
    # Test 3D diagnostics
    test_3d_diagnostics()
    
    # Test material stability
    test_material_stability()
    
    # Test stress concentration
    test_stress_concentration()
    
    # Test visual diagnostics
    test_visual_diagnostics_2d()
    test_visual_diagnostics_3d()
    
    # Generate full report
    test_full_diagnostic_report()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED")
    print("#"*60 + "\n")
    
    print("Summary of diagnostic capabilities:")
    print("  • Mesh quality checks (inverted/degenerate elements)")
    print("  • Material stability analysis")
    print("  • Numerical conditioning assessment")
    print("  • Stress concentration detection")
    print("  • Visual heatmaps and histograms")
    print("  • Comprehensive PDF reports")
    print("\nUse these tools to debug your hyperelastic simulations!")


if __name__ == "__main__":
    main()
