"""
Visual diagnostic utilities for hyperelastic simulations.

This module generates plots to visually identify mesh issues, stress concentrations,
and numerical instability in 2D and 3D simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection, PatchCollection
from typing import Optional, Tuple, List, Union
import os

# Handle matplotlib version compatibility
try:
    # Matplotlib >= 3.4.0: Axes3D is accessed differently
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    # Fallback for older versions
    try:
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        # If all else fails, we'll handle 3D plotting differently
        Axes3D = None
        Poly3DCollection = None

from .diagnostics import (
    run_full_diagnostic,
    compute_element_quality,
    compute_jacobian_determinants,
    DiagnosticResult
)


def _get_mesh_dimension(vertices: np.ndarray, elements: np.ndarray) -> int:
    """Determine mesh dimension from vertices and elements."""
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    
    dim = vertices.shape[1]
    n_nodes = elements.shape[1]
    
    # Infer from number of nodes per element
    if n_nodes == 3:
        return 2  # Triangles
    elif n_nodes == 4:
        return 3  # Tetrahedra
    else:
        return dim


def _compute_element_centroids(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Compute centroid of each element."""
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    
    elem_vertices = vertices[elements]
    centroids = np.mean(elem_vertices, axis=1)
    
    return centroids


def plot_mesh_quality_heatmap(
    vertices: np.ndarray,
    elements: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = True,
    title: str = "Mesh Quality Heatmap",
    cmap: str = 'RdYlGn_r',
    min_quality_threshold: float = 0.1
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap visualization of mesh quality.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        output_path: Path to save the figure (optional)
        show: Whether to display the plot
        title: Plot title
        cmap: Colormap name
        min_quality_threshold: Threshold for highlighting poor quality elements
        
    Returns:
        Figure and Axes objects
    """
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    dim = _get_mesh_dimension(vertices, elements)
    
    # Compute quality metrics
    qualities = compute_element_quality(vertices, elements)
    jacobians = compute_jacobian_determinants(vertices, elements)
    centroids = _compute_element_centroids(vertices, elements)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if dim == 2:
        # 2D visualization
        patches = []
        colors_list = []
        
        for i, elem in enumerate(elements):
            elem_verts = vertices[elem]
            
            # Create polygon patch
            if len(elem_verts) == 3:
                poly = plt.Polygon(elem_verts[:, :2], closed=True, linewidth=0.5)
                patches.append(poly)
                
                # Color based on quality
                q = qualities[i]
                if jacobians[i] < 0:
                    colors_list.append('red')  # Inverted
                elif q < min_quality_threshold:
                    colors_list.append('orange')  # Poor quality
                else:
                    colors_list.append(q)
        
        collection = PatchCollection(patches, cmap=cmap, alpha=0.8)
        if all(isinstance(c, (int, float)) for c in colors_list):
            collection.set_array(np.array(colors_list))
            collection.set_clim(0, 1)
        else:
            # Mixed colors (some inverted)
            color_map = {'red': 'red', 'orange': 'orange'}
            numeric_colors = [c if isinstance(c, (int, float)) else 0 for c in colors_list]
            collection.set_array(np.array(numeric_colors))
            collection.set_clim(0, 1)
        
        ax.add_collection(collection)
        
        # Set axis limits with padding
        pad = 0.1
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
        ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
        ax.set_aspect('equal')
        
    else:
        # 3D visualization (wireframe with colored edges)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw tetrahedra as wireframes
        for i, elem in enumerate(elements):
            elem_verts = vertices[elem]
            
            # Edges of tetrahedron
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            
            q = qualities[i]
            if jacobians[i] < 0:
                color = 'red'
            elif q < min_quality_threshold:
                color = 'orange'
            else:
                norm_q = (q - 0) / (1 - 0)
                color = cm.get_cmap(cmap)(norm_q)
            
            for j, k in edge_pairs:
                ax.plot3D(
                    [elem_verts[j, 0], elem_verts[k, 0]],
                    [elem_verts[j, 1], elem_verts[k, 1]],
                    [elem_verts[j, 2], elem_verts[k, 2]],
                    color=color, linewidth=0.5, alpha=0.7
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Add colorbar
    if dim == 2:
        cbar = plt.colorbar(collection, ax=ax, label='Quality')
        cbar.set_label('Element Quality (1.0 = perfect, <0 = inverted)', rotation=270, labelpad=20)
    
    ax.set_title(f"{title}\nMin: {qualities.min():.4f}, Max: {qualities.max():.4f}, Mean: {qualities.mean():.4f}")
    
    # Add statistics text box
    n_inverted = np.sum(jacobians < 0)
    n_poor = np.sum((qualities < min_quality_threshold) & (jacobians >= 0))
    stats_text = f"Inverted: {n_inverted}\nPoor (<{min_quality_threshold}): {n_poor}\nTotal: {len(elements)}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    if dim == 2:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    else:
        # For 3D, use a different approach
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved mesh quality plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_stress_distribution(
    vertices: np.ndarray,
    elements: np.ndarray,
    stress_tensors: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = True,
    title: str = "Von Mises Stress Distribution",
    cmap: str = 'jet'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap visualization of Von Mises stress distribution.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        stress_tensors: (n_elements, dim, dim) array of stress tensors
        output_path: Path to save the figure (optional)
        show: Whether to display the plot
        title: Plot title
        cmap: Colormap name
        
    Returns:
        Figure and Axes objects
    """
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    stress = np.asarray(stress_tensors)
    dim = _get_mesh_dimension(vertices, elements)
    
    # Compute Von Mises stress for each element
    von_mises = []
    for i in range(len(elements)):
        s = stress[i]
        if dim == 3:
            s_dev = s - np.trace(s) / 3 * np.eye(3)
            vm = np.sqrt(3/2 * np.sum(s_dev * s_dev))
        else:
            s_dev = s - np.trace(s) / 2 * np.eye(2)
            vm = np.sqrt(np.sum(s_dev * s_dev))
        von_mises.append(vm)
    
    von_mises = np.array(von_mises)
    centroids = _compute_element_centroids(vertices, elements)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if dim == 2:
        # 2D visualization
        patches = []
        
        for i, elem in enumerate(elements):
            elem_verts = vertices[elem]
            if len(elem_verts) == 3:
                poly = plt.Polygon(elem_verts[:, :2], closed=True, linewidth=0.5)
                patches.append(poly)
        
        collection = PatchCollection(patches, cmap=cmap, alpha=0.8)
        collection.set_array(von_mises)
        
        ax.add_collection(collection)
        
        # Set axis limits
        pad = 0.1
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
        ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(collection, ax=ax, label='Von Mises Stress')
        
    else:
        # 3D visualization
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize colors
        vmin, vmax = von_mises.min(), von_mises.max()
        norm_von_mises = (von_mises - vmin) / (vmax - vmin + 1e-10)
        
        for i, elem in enumerate(elements):
            elem_verts = vertices[elem]
            
            # Get color for this element
            color = cm.get_cmap(cmap)(norm_von_mises[i])
            
            # Draw faces of tetrahedron
            faces = [
                [elem_verts[0], elem_verts[1], elem_verts[2]],
                [elem_verts[0], elem_verts[1], elem_verts[3]],
                [elem_verts[0], elem_verts[2], elem_verts[3]],
                [elem_verts[1], elem_verts[2], elem_verts[3]]
            ]
            
            for face in faces:
                tri = Poly3DCollection([face], alpha=0.6, linewidth=0.3)
                tri.set_facecolor(color)
                ax.add_collection3d(tri)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add colorbar
        mappable = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        mappable.set_array(von_mises)
        cbar = plt.colorbar(mappable, ax=ax, label='Von Mises Stress')
    
    ax.set_title(f"{title}\nMin: {von_mises.min():.4f}, Max: {von_mises.max():.4f}, Mean: {von_mises.mean():.4f}")
    
    # Add statistics
    stats_text = f"Max: {von_mises.max():.4f}\nMean: {von_mises.mean():.4f}\nStd: {von_mises.std():.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved stress distribution plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_deformation_comparison(
    ref_vertices: np.ndarray,
    def_vertices: np.ndarray,
    elements: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = True,
    title: str = "Deformation Comparison"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create side-by-side comparison of reference and deformed configurations.
    
    Args:
        ref_vertices: Reference configuration vertices
        def_vertices: Deformed configuration vertices
        elements: Element connectivity
        output_path: Path to save the figure (optional)
        show: Whether to display the plot
        title: Plot title
        
    Returns:
        Figure and Axes objects
    """
    ref_vertices = np.asarray(ref_vertices)
    def_vertices = np.asarray(def_vertices)
    elements = np.asarray(elements)
    dim = _get_mesh_dimension(ref_vertices, elements)
    
    # Compute displacement magnitude
    displacements = def_vertices - ref_vertices
    disp_magnitude = np.linalg.norm(displacements, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    if dim == 2:
        # Reference configuration
        ax_ref = axes[0]
        patches_ref = []
        for elem in elements:
            elem_verts = ref_vertices[elem]
            if len(elem_verts) == 3:
                poly = plt.Polygon(elem_verts[:, :2], closed=True, 
                                   fill=False, edgecolor='blue', linewidth=0.5, alpha=0.5)
                patches_ref.append(poly)
        
        collection_ref = PatchCollection(patches_ref, alpha=0.5)
        ax_ref.add_collection(collection_ref)
        ax_ref.set_title("Reference Configuration")
        ax_ref.set_aspect('equal')
        
        # Deformed configuration
        ax_def = axes[1]
        patches_def = []
        colors_def = []
        for i, elem in enumerate(elements):
            elem_verts = def_vertices[elem]
            if len(elem_verts) == 3:
                poly = plt.Polygon(elem_verts[:, :2], closed=True, linewidth=0.5)
                patches_def.append(poly)
                
                # Color by average displacement at nodes
                avg_disp = np.mean([disp_magnitude[node] for node in elem])
                colors_def.append(avg_disp)
        
        collection_def = PatchCollection(patches_def, cmap='Reds', alpha=0.7)
        collection_def.set_array(np.array(colors_def))
        ax_def.add_collection(collection_def)
        ax_def.set_title("Deformed Configuration\n(colored by displacement)")
        ax_def.set_aspect('equal')
        
        cbar = plt.colorbar(collection_def, ax=ax_def, label='Displacement Magnitude')
        
        # Set same limits for both
        all_verts = np.vstack([ref_vertices, def_vertices])
        pad = 0.1
        x_min, y_min = all_verts.min(axis=0)
        x_max, y_max = all_verts.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        for ax in axes:
            ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
            ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
        
    else:
        # 3D visualization
        ax_ref = fig.add_subplot(121, projection='3d')
        ax_def = fig.add_subplot(122, projection='3d')
        
        # Reference wireframe
        for elem in elements:
            elem_verts = ref_vertices[elem]
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for j, k in edge_pairs:
                ax_ref.plot3D(
                    [elem_verts[j, 0], elem_verts[k, 0]],
                    [elem_verts[j, 1], elem_verts[k, 1]],
                    [elem_verts[j, 2], elem_verts[k, 2]],
                    color='blue', linewidth=0.3, alpha=0.3
                )
        ax_ref.set_title("Reference Configuration")
        
        # Deformed wireframe colored by displacement
        for i, elem in enumerate(elements):
            elem_verts = def_vertices[elem]
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            
            avg_disp = np.mean([disp_magnitude[node] for node in elem])
            norm_disp = (avg_disp - disp_magnitude.min()) / (disp_magnitude.max() - disp_magnitude.min() + 1e-10)
            color = cm.get_cmap('Reds')(norm_disp)
            
            for j, k in edge_pairs:
                ax_def.plot3D(
                    [elem_verts[j, 0], elem_verts[k, 0]],
                    [elem_verts[j, 1], elem_verts[k, 1]],
                    [elem_verts[j, 2], elem_verts[k, 2]],
                    color=color, linewidth=0.5, alpha=0.7
                )
        ax_def.set_title("Deformed Configuration\n(colored by displacement)")
        
        # Add colorbar
        mappable = cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(disp_magnitude.min(), disp_magnitude.max()))
        mappable.set_array(disp_magnitude)
        cbar = plt.colorbar(mappable, ax=ax_def, label='Displacement Magnitude', shrink=0.5)
        
        for ax in [ax_ref, ax_def]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved deformation comparison to {output_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_histogram_metrics(
    vertices: np.ndarray,
    elements: np.ndarray,
    stress_tensors: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    title: str = "Mesh Metrics Histograms"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create histograms of mesh quality metrics and stress values.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        stress_tensors: Optional stress tensors for stress histogram
        output_path: Path to save the figure (optional)
        show: Whether to display the plot
        title: Plot title
        
    Returns:
        Figure and list of Axes objects
    """
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    
    # Compute metrics
    qualities = compute_element_quality(vertices, elements)
    jacobians = compute_jacobian_determinants(vertices, elements)
    
    # Create figure with subplots
    n_plots = 3 if stress_tensors is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    # Quality histogram
    ax0 = axes[0] if n_plots > 1 else axes[0]
    ax0.hist(qualities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax0.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (inverted)')
    ax0.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Poor quality threshold')
    ax0.set_xlabel('Element Quality')
    ax0.set_ylabel('Count')
    ax0.set_title(f"Element Quality Distribution\nMin: {qualities.min():.4f}, Max: {qualities.max():.4f}")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    
    # Jacobian histogram
    ax1 = axes[1] if n_plots > 1 else axes[1]
    ax1.hist(jacobians, bins=50, color='green', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (inverted)')
    ax1.set_xlabel('Jacobian Determinant')
    ax1.set_ylabel('Count')
    ax1.set_title(f"Jacobian Distribution\nMin: {jacobians.min():.4e}, Max: {jacobians.max():.4e}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stress histogram (if provided)
    if stress_tensors is not None:
        stress = np.asarray(stress_tensors)
        dim = _get_mesh_dimension(vertices, elements)
        
        von_mises = []
        for i in range(len(elements)):
            s = stress[i]
            if dim == 3:
                s_dev = s - np.trace(s) / 3 * np.eye(3)
                vm = np.sqrt(3/2 * np.sum(s_dev * s_dev))
            else:
                s_dev = s - np.trace(s) / 2 * np.eye(2)
                vm = np.sqrt(np.sum(s_dev * s_dev))
            von_mises.append(vm)
        
        von_mises = np.array(von_mises)
        
        ax2 = axes[2]
        ax2.hist(von_mises, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax2.axvline(x=von_mises.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {von_mises.mean():.4f}')
        ax2.set_xlabel('Von Mises Stress')
        ax2.set_ylabel('Count')
        ax2.set_title(f"Stress Distribution\nMax: {von_mises.max():.4f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histograms to {output_path}")
    
    if show:
        plt.show()
    
    return fig, list(axes)


def generate_diagnostic_report(
    vertices: np.ndarray,
    elements: np.ndarray,
    deformation_gradients: Optional[np.ndarray] = None,
    stress_tensors: Optional[np.ndarray] = None,
    output_dir: str = "./diagnostic_report",
    shear_modulus: float = 1.0,
    bulk_modulus: float = 2.0,
    show: bool = False
) -> str:
    """
    Generate a comprehensive diagnostic report with all visualizations.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        deformation_gradients: Optional deformation gradients
        stress_tensors: Optional stress tensors
        output_dir: Directory to save report files
        shear_modulus: Shear modulus for material checks
        bulk_modulus: Bulk modulus for material checks
        show: Whether to display plots
        
    Returns:
        Path to the output directory
    """
    import os
    from datetime import datetime
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"Generating diagnostic report in: {report_dir}")
    
    # Run diagnostics
    results = run_full_diagnostic(
        vertices, elements, deformation_gradients, stress_tensors,
        shear_modulus, bulk_modulus
    )
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        if check_name.startswith('_'):
            continue
        status = "✓ PASS" if result.passed else "✗ FAIL"
        severity = result.severity.upper()
        print(f"\n[{status}] {check_name} ({severity})")
        print(f"  {result.message}")
        
        if result.critical_elements is not None and len(result.critical_elements) > 0:
            print(f"  Critical elements: {result.critical_elements[:10]}{'...' if len(result.critical_elements) > 10 else ''}")
    
    # Summary
    summary = results['_summary']
    print("\n" + "="*60)
    print(f"OVERALL: {'PASS' if summary.passed else 'FAIL'}")
    print(f"Critical: {summary.details['critical']}, Errors: {summary.details['errors']}, Warnings: {summary.details['warnings']}")
    print("="*60 + "\n")
    
    # Generate plots
    plots_generated = []
    
    # 1. Mesh quality heatmap
    quality_plot = os.path.join(report_dir, "mesh_quality.png")
    try:
        plot_mesh_quality_heatmap(vertices, elements, output_path=quality_plot, show=show)
        plots_generated.append(quality_plot)
    except Exception as e:
        print(f"Warning: Could not generate mesh quality plot: {e}")
    
    # 2. Stress distribution (if available)
    if stress_tensors is not None:
        stress_plot = os.path.join(report_dir, "stress_distribution.png")
        try:
            plot_stress_distribution(vertices, elements, stress_tensors, 
                                    output_path=stress_plot, show=show)
            plots_generated.append(stress_plot)
        except Exception as e:
            print(f"Warning: Could not generate stress plot: {e}")
    
    # 3. Histograms
    hist_plot = os.path.join(report_dir, "metrics_histograms.png")
    try:
        plot_histogram_metrics(vertices, elements, stress_tensors, 
                              output_path=hist_plot, show=show)
        plots_generated.append(hist_plot)
    except Exception as e:
        print(f"Warning: Could not generate histograms: {e}")
    
    # 4. Save detailed results
    results_file = os.path.join(report_dir, "diagnostic_results.txt")
    with open(results_file, 'w') as f:
        f.write("DIAGNOSTIC REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        for check_name, result in results.items():
            f.write(f"\n{check_name}:\n")
            f.write(f"  Passed: {result.passed}\n")
            f.write(f"  Severity: {result.severity}\n")
            f.write(f"  Message: {result.message}\n")
            f.write(f"  Details:\n")
            for key, value in result.details.items():
                if not isinstance(value, np.ndarray):
                    f.write(f"    {key}: {value}\n")
    
    print(f"\nReport generated successfully!")
    print(f"Output directory: {report_dir}")
    print(f"Plots generated: {len(plots_generated)}")
    
    return report_dir
