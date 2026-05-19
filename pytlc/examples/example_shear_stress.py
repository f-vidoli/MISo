"""
Example: Computing shear stress from transformation strain

This example demonstrates how to use the pytlc module to compute
shear stress from transformation strain (eigenstrain) tensors.
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

# Material properties (steel-like)
shear_modulus = 80.0e9  # Pa (80 GPa)

print("=" * 60)
print("Shear Stress from Transformation Strain Examples")
print("=" * 60)

# Example 1: Single 3D strain tensor
print("\n1. Single 3D strain tensor:")
print("-" * 40)
strain = np.array([[0.001, 0.0002, 0.0001],
                   [0.0002, 0.0005, 0.0003],
                   [0.0001, 0.0003, 0.0008]])
print("Input strain tensor:")
print(strain)

shear_stress = pytlc.compute_shear_stress_from_transformation_strain(
    strain, shear_modulus=shear_modulus
)
print("\nComputed shear stress (deviatoric) in MPa:")
print(shear_stress / 1e6)
print("Trace (should be ~0):", np.trace(shear_stress))

# Example 2: Pure shear strain
print("\n\n2. Pure shear strain:")
print("-" * 40)
pure_shear = np.zeros((3, 3))
pure_shear[0, 1] = pure_shear[1, 0] = 0.001
print("Input strain tensor (pure shear):")
print(pure_shear)

shear_pure = pytlc.compute_shear_stress_from_transformation_strain(
    pure_shear, shear_modulus=shear_modulus
)
print("\nComputed shear stress in MPa:")
print(shear_pure / 1e6)
expected = 2*shear_modulus*0.001/1e6
print(f"Expected sigma_xy = 2*mu*eps_xy = {expected:.2f} MPa")
print("Actual sigma_xy =", shear_pure[0, 1]/1e6, "MPa")

# Example 3: Voigt notation
print("\n\n3. Voigt notation input:")
print("-" * 40)
strain_voigt = np.array([0.001, 0.0005, 0.0008, 0.0003, 0.0001, 0.0002])
print("Input strain (Voigt):", strain_voigt)
print("Format: [xx, yy, zz, yz, xz, xy]")

shear_voigt = pytlc.compute_shear_stress_from_transformation_strain(
    strain_voigt, shear_modulus=shear_modulus
)
print("\nComputed shear stress in MPa:")
print(shear_voigt / 1e6)

# Example 4: Batch processing
print("\n\n4. Batch processing (multiple strain states):")
print("-" * 40)
n_samples = 10
strains_batch = np.random.randn(n_samples, 3, 3) * 0.001
strains_batch = (strains_batch + strains_batch.transpose(0, 2, 1)) / 2

shear_batch = pytlc.compute_shear_stress_from_transformation_strain(
    strains_batch, shear_modulus=shear_modulus
)
print("Batch input shape:", strains_batch.shape)
print("Batch output shape:", shear_batch.shape)

# Verify all outputs are traceless (deviatoric)
traces = [np.trace(shear_batch[i]) for i in range(n_samples)]
max_trace = max(abs(t) for t in traces)
print("Maximum |trace| across batch:", max_trace, "(should be ~0)")

# Example 5: Von Mises stress
print("\n\n5. Von Mises equivalent stress:")
print("-" * 40)
stress_tensor = np.array([[100e6, 20e6, 10e6],
                          [20e6, 50e6, 15e6],
                          [10e6, 15e6, 80e6]])
print("Input stress tensor in MPa:")
print(stress_tensor / 1e6)

von_mises = pytlc.compute_von_mises_stress(stress_tensor)
print("\nVon Mises stress:", von_mises / 1e6, "MPa")

# Compute von Mises from shear stress
von_mises_shear = pytlc.compute_von_mises_stress(shear_stress)
print("Von Mises of deviatoric stress:", von_mises_shear / 1e6, "MPa")

# Example 6: Application - thermal expansion
print("\n\n6. Application: Thermal expansion with constraint:")
print("-" * 40)
alpha_thermal = 12e-6  # 1/K (steel)
delta_T = 100  # K
eps_thermal = alpha_thermal * delta_T
print("Thermal expansion coefficient:", alpha_thermal, "/K")
print("Temperature change:", delta_T, "K")
print("Free thermal strain:", eps_thermal)

# If constrained in one direction, creates deviatoric strain
constrained_strain = np.array([
    [0, 0, 0],
    [0, eps_thermal, 0],
    [0, 0, eps_thermal]
])
print("\nConstrained strain (x-direction fixed):")
print(constrained_strain)

shear_thermal = pytlc.compute_shear_stress_from_transformation_strain(
    constrained_strain, shear_modulus=shear_modulus
)
print("\nResulting shear stress in MPa:")
print(shear_thermal / 1e6)

von_mises_thermal = pytlc.compute_von_mises_stress(shear_thermal)
print("Von Mises stress:", von_mises_thermal / 1e6, "MPa")

print("\n" + "=" * 60)
print("Examples completed successfully!")
print("=" * 60)
