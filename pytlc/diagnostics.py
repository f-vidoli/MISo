"""
Diagnostic utilities for identifying bugs in hyperelastic simulations.

This module provides tools to detect:
- Inverted or degenerate mesh elements
- Material instability issues
- Numerical conditioning problems
- Stress concentrations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    """Container for diagnostic results."""
    passed: bool
    message: str
    details: Dict
    critical_elements: Optional[np.ndarray] = None
    severity: str = "info"  # info, warning, error, critical


def compute_jacobian_determinants(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian determinants for all elements.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        
    Returns:
        (n_elements,) array of Jacobian determinants
    """
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    n_elements = len(elements)
    dim = vertices.shape[1]
    
    jacobians = []
    
    for elem_idx in range(n_elements):
        elem_nodes = elements[elem_idx]
        elem_vertices = vertices[elem_nodes]
        
        if dim == 2:
            # Triangle: compute 2x2 Jacobian
            v0, v1, v2 = elem_vertices
            J = np.column_stack([v1 - v0, v2 - v0])
            det_J = np.linalg.det(J)
        elif dim == 3:
            # Tetrahedron: compute 3x3 Jacobian
            v0, v1, v2, v3 = elem_vertices
            J = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            det_J = np.linalg.det(J)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
        
        jacobians.append(det_J)
    
    return np.array(jacobians)


def compute_element_quality(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute quality metrics for each element.
    
    For triangles/tetrahedra, uses the ratio of actual volume to ideal volume.
    Quality = 1.0 for perfect elements, 0.0 for degenerate, negative for inverted.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        
    Returns:
        (n_elements,) array of quality values
    """
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    n_elements = len(elements)
    dim = vertices.shape[1]
    
    qualities = []
    
    for elem_idx in range(n_elements):
        elem_nodes = elements[elem_idx]
        elem_vertices = vertices[elem_nodes]
        
        if dim == 2:
            # Triangle quality based on aspect ratio
            v0, v1, v2 = elem_vertices
            edges = [np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)]
            area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
            
            # Ideal equilateral triangle with same perimeter
            perimeter = sum(edges)
            ideal_area = (np.sqrt(3) / 36) * perimeter ** 2
            
            if ideal_area > 0:
                quality = (area / ideal_area) * np.sign(area)
            else:
                quality = 0.0
                
        elif dim == 3:
            # Tetrahedron quality
            v0, v1, v2, v3 = elem_vertices
            volume = abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
            
            # Edge lengths
            edges = [
                np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v0), np.linalg.norm(v3 - v0),
                np.linalg.norm(v2 - v1), np.linalg.norm(v3 - v1), np.linalg.norm(v3 - v2)
            ]
            
            # Ideal regular tetrahedron with same total edge length
            total_edge_length = sum(edges)
            ideal_volume = (total_edge_length ** 3) / (216 * np.sqrt(2))
            
            if ideal_volume > 0:
                quality = (volume / ideal_volume) * np.sign(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0))
            else:
                quality = 0.0
        else:
            quality = 0.0
            
        qualities.append(quality)
    
    return np.array(qualities)


def diagnose_mesh_quality(
    vertices: np.ndarray,
    elements: np.ndarray,
    min_quality_threshold: float = 0.1,
    min_volume_threshold: float = 1e-10
) -> DiagnosticResult:
    """
    Diagnose mesh quality issues.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        min_quality_threshold: Minimum acceptable quality (0-1)
        min_volume_threshold: Minimum acceptable element volume
        
    Returns:
        DiagnosticResult with quality analysis
    """
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    
    # Compute metrics
    jacobians = compute_jacobian_determinants(vertices, elements)
    qualities = compute_element_quality(vertices, elements)
    
    # Identify problematic elements
    inverted_mask = jacobians < 0
    degenerate_mask = np.abs(jacobians) < min_volume_threshold
    low_quality_mask = qualities < min_quality_threshold
    
    n_inverted = np.sum(inverted_mask)
    n_degenerate = np.sum(degenerate_mask)
    n_low_quality = np.sum(low_quality_mask)
    
    # Build report
    details = {
        'n_elements': len(elements),
        'min_jacobian': float(np.min(jacobians)),
        'max_jacobian': float(np.max(jacobians)),
        'mean_jacobian': float(np.mean(jacobians)),
        'min_quality': float(np.min(qualities)),
        'max_quality': float(np.max(qualities)),
        'mean_quality': float(np.mean(qualities)),
        'n_inverted': int(n_inverted),
        'n_degenerate': int(n_degenerate),
        'n_low_quality': int(n_low_quality),
        'jacobians': jacobians,
        'qualities': qualities,
        'inverted_indices': np.where(inverted_mask)[0],
        'degenerate_indices': np.where(degenerate_mask)[0],
        'low_quality_indices': np.where(low_quality_mask)[0],
    }
    
    # Determine severity
    if n_inverted > 0:
        severity = "critical"
        message = f"CRITICAL: {n_inverted} inverted elements detected!"
        critical_elements = np.where(inverted_mask)[0]
        passed = False
    elif n_degenerate > 0:
        severity = "error"
        message = f"ERROR: {n_degenerate} degenerate elements detected!"
        critical_elements = np.where(degenerate_mask)[0]
        passed = False
    elif n_low_quality > 0:
        severity = "warning"
        message = f"WARNING: {n_low_quality} elements with low quality (< {min_quality_threshold})"
        critical_elements = np.where(low_quality_mask)[0]
        passed = False
    else:
        severity = "info"
        message = f"Mesh quality OK: {len(elements)} elements, min quality = {np.min(qualities):.4f}"
        critical_elements = None
        passed = True
    
    return DiagnosticResult(
        passed=passed,
        message=message,
        details=details,
        critical_elements=critical_elements,
        severity=severity
    )


def diagnose_material_stability(
    deformation_gradients: np.ndarray,
    shear_modulus: float,
    bulk_modulus: float,
    neo_hookean: bool = True
) -> DiagnosticResult:
    """
    Diagnose material stability issues.
    
    Checks for:
    - Negative shear/bulk modulus
    - Extreme deformations that may cause instability
    - Loss of ellipticity conditions
    
    Args:
        deformation_gradients: (n_elements, dim, dim) array of deformation gradients
        shear_modulus: Shear modulus μ
        bulk_modulus: Bulk modulus K
        neo_hookean: Whether using Neo-Hookean model
        
    Returns:
        DiagnosticResult with stability analysis
    """
    F = np.asarray(deformation_gradients)
    n_elements = F.shape[0]
    dim = F.shape[1]
    
    details = {
        'shear_modulus': shear_modulus,
        'bulk_modulus': bulk_modulus,
        'n_elements': n_elements,
    }
    
    # Check material parameters
    if shear_modulus <= 0:
        return DiagnosticResult(
            passed=False,
            message=f"CRITICAL: Negative or zero shear modulus ({shear_modulus})",
            details=details,
            severity="critical"
        )
    
    if bulk_modulus <= 0:
        return DiagnosticResult(
            passed=False,
            message=f"CRITICAL: Negative or zero bulk modulus ({bulk_modulus})",
            details=details,
            severity="critical"
        )
    
    # Compute determinants (J = det(F))
    J = np.linalg.det(F)
    
    # Check for extreme compression/expansion
    min_J = float(np.min(J))
    max_J = float(np.max(J))
    
    details['min_det_F'] = min_J
    details['max_det_F'] = max_J
    details['mean_det_F'] = float(np.mean(J))
    
    critical_elements = []
    messages = []
    
    # Extreme compression (J < 0.1)
    extreme_compression = J < 0.1
    if np.any(extreme_compression):
        critical_elements.extend(np.where(extreme_compression)[0])
        messages.append(f"{np.sum(extreme_compression)} elements with extreme compression (J < 0.1)")
    
    # Extreme expansion (J > 10)
    extreme_expansion = J > 10
    if np.any(extreme_expansion):
        critical_elements.extend(np.where(extreme_expansion)[0])
        messages.append(f"{np.sum(extreme_expansion)} elements with extreme expansion (J > 10)")
    
    # Near-zero or negative J (inverted)
    invalid_J = J <= 0
    if np.any(invalid_J):
        critical_elements.extend(np.where(invalid_J)[0])
        messages.append(f"{np.sum(invalid_J)} elements with non-positive Jacobian")
    
    if messages:
        severity = "critical" if np.any(invalid_J) else "warning"
        return DiagnosticResult(
            passed=False,
            message="Material stability issues: " + "; ".join(messages),
            details=details,
            critical_elements=np.array(critical_elements),
            severity=severity
        )
    
    return DiagnosticResult(
        passed=True,
        message=f"Material stability OK: J in [{min_J:.4f}, {max_J:.4f}]",
        details=details,
        severity="info"
    )


def diagnose_numerical_conditioning(
    deformation_gradients: np.ndarray,
    condition_number_threshold: float = 1e6
) -> DiagnosticResult:
    """
    Diagnose numerical conditioning issues.
    
    Computes condition numbers of deformation gradients to identify
    elements that may cause numerical instability.
    
    Args:
        deformation_gradients: (n_elements, dim, dim) array of deformation gradients
        condition_number_threshold: Maximum acceptable condition number
        
    Returns:
        DiagnosticResult with conditioning analysis
    """
    F = np.asarray(deformation_gradients)
    n_elements = F.shape[0]
    
    condition_numbers = []
    ill_conditioned_indices = []
    
    for i in range(n_elements):
        try:
            cond = np.linalg.cond(F[i])
            condition_numbers.append(cond)
            if cond > condition_number_threshold:
                ill_conditioned_indices.append(i)
        except:
            condition_numbers.append(np.inf)
            ill_conditioned_indices.append(i)
    
    condition_numbers = np.array(condition_numbers)
    n_ill_conditioned = len(ill_conditioned_indices)
    
    details = {
        'n_elements': n_elements,
        'min_condition_number': float(np.min(condition_numbers[np.isfinite(condition_numbers)])) if np.any(np.isfinite(condition_numbers)) else np.inf,
        'max_condition_number': float(np.max(condition_numbers[np.isfinite(condition_numbers)])) if np.any(np.isfinite(condition_numbers)) else np.inf,
        'mean_condition_number': float(np.mean(condition_numbers[np.isfinite(condition_numbers)])) if np.any(np.isfinite(condition_numbers)) else np.inf,
        'n_ill_conditioned': n_ill_conditioned,
        'condition_numbers': condition_numbers,
        'ill_conditioned_indices': np.array(ill_conditioned_indices),
    }
    
    if n_ill_conditioned > 0:
        return DiagnosticResult(
            passed=False,
            message=f"WARNING: {n_ill_conditioned} elements with poor conditioning (cond > {condition_number_threshold})",
            details=details,
            critical_elements=np.array(ill_conditioned_indices),
            severity="warning"
        )
    
    return DiagnosticResult(
        passed=True,
        message=f"Numerical conditioning OK: max cond = {details['max_condition_number']:.2e}",
        details=details,
        severity="info"
    )


def diagnose_stress_concentration(
    stress_tensors: np.ndarray,
    threshold_factor: float = 3.0
) -> DiagnosticResult:
    """
    Diagnose stress concentration issues.
    
    Identifies elements with stresses significantly higher than average.
    
    Args:
        stress_tensors: (n_elements, dim, dim) array of stress tensors
        threshold_factor: Factor above mean to flag as concentration
        
    Returns:
        DiagnosticResult with stress analysis
    """
    stress = np.asarray(stress_tensors)
    n_elements = stress.shape[0]
    dim = stress.shape[1]
    
    # Compute Von Mises stress for each element
    von_mises = []
    for i in range(n_elements):
        s = stress[i]
        if dim == 3:
            # 3D Von Mises
            s_dev = s - np.trace(s) / 3 * np.eye(3)
            vm = np.sqrt(3/2 * np.sum(s_dev * s_dev))
        else:
            # 2D Von Mises (plane stress/strain approximation)
            s_dev = s - np.trace(s) / 2 * np.eye(2)
            vm = np.sqrt(np.sum(s_dev * s_dev))
        von_mises.append(vm)
    
    von_mises = np.array(von_mises)
    mean_vm = np.mean(von_mises)
    max_vm = np.max(von_mises)
    threshold = mean_vm * threshold_factor
    
    concentrated_mask = von_mises > threshold
    n_concentrated = np.sum(concentrated_mask)
    
    details = {
        'n_elements': n_elements,
        'mean_von_mises': float(mean_vm),
        'max_von_mises': float(max_vm),
        'min_von_mises': float(np.min(von_mises)),
        'threshold': float(threshold),
        'n_concentrated': int(n_concentrated),
        'von_mises': von_mises,
        'concentrated_indices': np.where(concentrated_mask)[0],
    }
    
    if n_concentrated > 0:
        return DiagnosticResult(
            passed=False,
            message=f"WARNING: {n_concentrated} elements with stress concentration (>{threshold_factor}x mean)",
            details=details,
            critical_elements=np.where(concentrated_mask)[0],
            severity="warning"
        )
    
    return DiagnosticResult(
        passed=True,
        message=f"Stress distribution OK: max Von Mises = {max_vm:.4f}",
        details=details,
        severity="info"
    )


def run_full_diagnostic(
    vertices: np.ndarray,
    elements: np.ndarray,
    deformation_gradients: Optional[np.ndarray] = None,
    stress_tensors: Optional[np.ndarray] = None,
    shear_modulus: float = 1.0,
    bulk_modulus: float = 2.0
) -> Dict[str, DiagnosticResult]:
    """
    Run all diagnostic checks and return comprehensive report.
    
    Args:
        vertices: (n_vertices, dim) array of vertex positions
        elements: (n_elements, n_nodes_per_elem) array of element connectivity
        deformation_gradients: Optional (n_elements, dim, dim) array
        stress_tensors: Optional (n_elements, dim, dim) array
        shear_modulus: Shear modulus for material checks
        bulk_modulus: Bulk modulus for material checks
        
    Returns:
        Dictionary with results from all diagnostic checks
    """
    results = {}
    
    # Mesh quality (always run)
    results['mesh_quality'] = diagnose_mesh_quality(vertices, elements)
    
    # Material stability (if F provided)
    if deformation_gradients is not None:
        results['material_stability'] = diagnose_material_stability(
            deformation_gradients, shear_modulus, bulk_modulus
        )
        
        results['numerical_conditioning'] = diagnose_numerical_conditioning(
            deformation_gradients
        )
    
    # Stress concentration (if stress provided)
    if stress_tensors is not None:
        results['stress_concentration'] = diagnose_stress_concentration(stress_tensors)
    
    # Summary
    all_passed = all(r.passed for r in results.values())
    critical_count = sum(1 for r in results.values() if r.severity == "critical")
    error_count = sum(1 for r in results.values() if r.severity == "error")
    warning_count = sum(1 for r in results.values() if r.severity == "warning")
    
    results['_summary'] = DiagnosticResult(
        passed=all_passed,
        message=f"Diagnostic complete: {critical_count} critical, {error_count} errors, {warning_count} warnings",
        details={'all_passed': all_passed, 'critical': critical_count, 'errors': error_count, 'warnings': warning_count},
        severity="critical" if critical_count > 0 else ("error" if error_count > 0 else ("warning" if warning_count > 0 else "info"))
    )
    
    return results
