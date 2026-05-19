#!/usr/bin/env python3
"""
Example: Computing shear stress for hyperelastic materials with large deformations.

This example demonstrates the use of compute_shear_stress_hyperelastic() and
compute_cauchy_stress_hyperelastic() functions for Neo-Hookean materials,
including the effect of transformation strains (eigenstrains).
"""

import numpy as np
import pytlc


def test_simple_shear():
    """Test simple shear deformation without transformation strain."""
    print("=" * 60)
    print("Test 1: Simple Shear Deformation (No Transformation Strain)")
    print("=" * 60)
    
    # Simple shear: F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]]
    gamma = 0.5
    F = np.array([
        [1.0, gamma, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    shear_modulus = 80.0  # MPa (typical for rubber-like materials)
    
    # Compute shear stress (deviatoric part only)
    shear_stress = pytlc.compute_shear_stress_hyperelastic(
        F, 
        shear_modulus=shear_modulus
    )
    
    print(f"\nDeformation gradient F:\n{F}")
    print(f"\nShear modulus: {shear_modulus} MPa")
    print(f"\nShear stress (deviatoric Cauchy stress):\n{shear_stress}")
    print(f"Trace of shear stress (should be ~0): {np.trace(shear_stress):.2e}")
    
    # Verify: For simple shear, sigma_12 should be approximately mu * gamma
    expected_shear = shear_modulus * gamma
    actual_shear = shear_stress[0, 1]
    print(f"\nExpected sigma_12 ≈ mu * gamma = {expected_shear:.2f} MPa")
    print(f"Actual sigma_12 = {actual_shear:.2f} MPa")
    print(f"Relative error: {abs(actual_shear - expected_shear) / expected_shear * 100:.1f}%")


def test_uniaxial_stretch():
    """Test uniaxial stretch deformation."""
    print("\n" + "=" * 60)
    print("Test 2: Uniaxial Stretch (No Transformation Strain)")
    print("=" * 60)
    
    # Uniaxial stretch in z-direction with incompressibility constraint
    lambda_z = 1.5
    lambda_xy = 1.0 / np.sqrt(lambda_z)  # Incompressible: J = 1
    
    F = np.diag([lambda_xy, lambda_xy, lambda_z])
    
    shear_modulus = 80.0
    bulk_modulus = 1000.0 * shear_modulus  # Nearly incompressible
    
    # Compute full Cauchy stress
    cauchy_stress, shear_stress = pytlc.compute_cauchy_stress_hyperelastic(
        F,
        shear_modulus=shear_modulus,
        bulk_modulus=bulk_modulus,
        return_full_stress=True
    )
    
    print(f"\nStretch ratio lambda_z = {lambda_z}")
    print(f"Lateral stretch lambda_xy = {lambda_xy:.4f}")
    print(f"Deformation gradient F:\n{F}")
    print(f"det(F) = {np.linalg.det(F):.4f}")
    
    print(f"\nFull Cauchy stress:\n{cauchy_stress}")
    print(f"Shear stress (deviatoric part):\n{shear_stress}")
    print(f"Trace of shear stress: {np.trace(shear_stress):.2e}")
    
    # Hydrostatic pressure
    pressure = np.trace(cauchy_stress) / 3.0
    print(f"\nHydrostatic pressure: {pressure:.2f} MPa")


def test_thermal_expansion():
    """Test uniaxial stretch with thermal expansion (transformation strain)."""
    print("\n" + "=" * 60)
    print("Test 3: Uniaxial Stretch with Thermal Expansion")
    print("=" * 60)
    
    # Applied deformation
    lambda_z = 1.2
    F = np.diag([1.0, 1.0, lambda_z])
    
    # Thermal expansion (stress-free strain)
    alpha_thermal = 0.05  # 5% thermal expansion
    eps_thermal = alpha_thermal * np.eye(3)
    
    shear_modulus = 80.0
    bulk_modulus = 1000.0 * shear_modulus
    
    # Without thermal expansion
    shear_stress_no_thermal = pytlc.compute_shear_stress_hyperelastic(
        F,
        shear_modulus=shear_modulus
    )
    
    # With thermal expansion
    shear_stress_thermal = pytlc.compute_shear_stress_hyperelastic(
        F,
        transformation_strain=eps_thermal,
        shear_modulus=shear_modulus
    )
    
    print(f"\nApplied stretch: lambda_z = {lambda_z}")
    print(f"Thermal expansion coefficient: alpha = {alpha_thermal * 100:.1f}%")
    print(f"Transformation strain (epsilon_thermal):\n{eps_thermal}")
    
    print(f"\nShear stress WITHOUT thermal expansion:\n{shear_stress_no_thermal}")
    print(f"Shear stress WITH thermal expansion:\n{shear_stress_thermal}")
    
    print(f"\nDifference due to thermal expansion:\n{shear_stress_thermal - shear_stress_no_thermal}")
    print(f"Norm of difference: {np.linalg.norm(shear_stress_thermal - shear_stress_no_thermal):.4f} MPa")


def test_batch_processing():
    """Test batch processing of multiple deformation gradients."""
    print("\n" + "=" * 60)
    print("Test 4: Batch Processing")
    print("=" * 60)
    
    # Create a batch of deformation gradients with varying stretches
    n_samples = 5
    stretches = np.linspace(1.0, 2.0, n_samples)
    
    F_batch = np.zeros((n_samples, 3, 3))
    for i, lam in enumerate(stretches):
        F_batch[i] = np.diag([lam, 1.0, 1.0])
    
    shear_modulus = 80.0
    
    # Compute shear stresses for all samples
    shear_stresses = pytlc.compute_shear_stress_hyperelastic(
        F_batch,
        shear_modulus=shear_modulus
    )
    
    print(f"\nBatch size: {n_samples}")
    print(f"Stretches: {stretches}")
    print(f"\nShear stress norms for each sample:")
    
    for i, (lam, ss) in enumerate(zip(stretches, shear_stresses)):
        norm = np.linalg.norm(ss)
        trace = np.trace(ss)
        print(f"  lambda={lam:.2f}: ||sigma_dev||={norm:.4f} MPa, trace={trace:.2e}")


def test_2d_plane_strain():
    """Test 2D plane strain case."""
    print("\n" + "=" * 60)
    print("Test 5: 2D Plane Strain")
    print("=" * 60)
    
    # 2D deformation gradient (plane strain)
    gamma = 0.3
    F = np.array([
        [1.0, gamma],
        [0.0, 1.0]
    ])
    
    shear_modulus = 80.0
    
    # Compute shear stress
    shear_stress = pytlc.compute_shear_stress_hyperelastic(
        F,
        shear_modulus=shear_modulus
    )
    
    print(f"\n2D deformation gradient F:\n{F}")
    print(f"\nShear stress (2D):\n{shear_stress}")
    print(f"Trace: {np.trace(shear_stress):.2e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HYPERELASTIC SHEAR STRESS COMPUTATION EXAMPLES")
    print("For Neo-Hookean materials with large deformations")
    print("=" * 60)
    
    test_simple_shear()
    test_uniaxial_stretch()
    test_thermal_expansion()
    test_batch_processing()
    test_2d_plane_strain()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
