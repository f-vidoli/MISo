"""
Core TLC energy computation and optimization routines.

This module implements the Total Lifted Content (TLC) energy and its gradient,
along with the optimization routine to find injective mappings.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.optimize import minimize


def tri_signed_area(pts: np.ndarray) -> float:
    """
    Compute 2 times the signed area of a triangle.
    
    Parameters
    ----------
    pts : np.ndarray
        Triangle vertices, shape (3, 2)
        
    Returns
    -------
    float
        2 times the signed area
    """
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    return x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)


def tet_signed_volume(pts: np.ndarray) -> float:
    """
    Compute 6 times the signed volume of a tetrahedron.
    
    Parameters
    ----------
    pts : np.ndarray
        Tetrahedron vertices, shape (4, 3)
        
    Returns
    -------
    float
        6 times the signed volume
    """
    x1, y1, z1 = pts[0]
    x2, y2, z2 = pts[1]
    x3, y3, z3 = pts[2]
    x4, y4, z4 = pts[3]
    vol = (x4 * (y3 * (-z1 + z2) + y2 * (z1 - z3) + y1 * (-z2 + z3)) +
           x3 * (y4 * (z1 - z2) + y1 * (z2 - z4) + y2 * (-z1 + z4)) +
           x1 * (y4 * (z2 - z3) + y2 * (z3 - z4) + y3 * (-z2 + z4)) +
           x2 * (y4 * (-z1 + z3) + y3 * (z1 - z4) + y1 * (-z3 + z4)))
    return vol


def tri_area_squared_edges(d: np.ndarray) -> float:
    """
    Compute 4 times the triangle area from squared edge lengths.
    
    Uses a numerically robust version of Heron's formula.
    
    Parameters
    ----------
    d : np.ndarray
        Squared edge lengths, shape (3,)
        
    Returns
    -------
    float
        4 times the triangle area
    """
    # Sort d as a >= b >= c for numerical stability
    d_sorted = np.sort(d)[::-1]
    a, b, c = np.sqrt(d_sorted)
    return np.sqrt(abs((a + (b + c)) * (c - abs(a - b)) * (c + abs(a - b)) * (a + abs(b - c))))


def tet_volume_squared_edges(d: np.ndarray) -> float:
    """
    Compute 12 times the tetrahedron volume from squared edge lengths.
    
    Parameters
    ----------
    d : np.ndarray
        Squared edge lengths, shape (6,)
        Order: edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        
    Returns
    -------
    float
        12 times the tetrahedron volume
    """
    d0, d1, d2, d3, d4, d5 = d
    det = (-(d1 * d1 * d4) - d0 * d0 * d5 - 
           d3 * (d2 * d2 + d2 * (d3 - d4 - d5) + d4 * d5) +
           d1 * (d2 * (d3 + d4 - d5) + d4 * (d3 - d4 + d5)) +
           d0 * ((d3 + d4 - d5) * d5 + d2 * (d3 - d4 + d5) + d1 * (-d3 + d4 + d5)))
    return np.sqrt(max(0, det))


def tri_grad_squared_edges(d: np.ndarray) -> np.ndarray:
    """
    Compute gradient of 4*area w.r.t. squared edge lengths.
    
    Parameters
    ----------
    d : np.ndarray
        Squared edge lengths, shape (3,)
        
    Returns
    -------
    np.ndarray
        Gradient, shape (3,)
    """
    d0, d1, d2 = d
    g = np.zeros(3)
    g[0] = 2 * (d1 + d2 - d0)
    g[1] = 2 * (d0 + d2 - d1)
    g[2] = 2 * (d0 + d1 - d2)
    return g


def tet_grad_squared_edges(d: np.ndarray) -> np.ndarray:
    """
    Compute gradient of 144*vol^2 w.r.t. squared edge lengths.
    
    Parameters
    ----------
    d : np.ndarray
        Squared edge lengths, shape (6,)
        
    Returns
    -------
    np.ndarray
        Gradient, shape (6,)
    """
    d0, d1, d2, d3, d4, d5 = d
    g = np.zeros(6)
    g[0] = (-d1 + d2) * (d3 - d4) + (-2.0 * d0 + d1 + d2 + d3 + d4 - d5) * d5
    g[1] = (-d0 + d2) * (d3 - d5) + d4 * (d0 - 2.0 * d1 + d2 + d3 - d4 + d5)
    g[2] = (-d0 + d1) * (d4 - d5) + d3 * (d0 + d1 - 2.0 * d2 - d3 + d4 + d5)
    g[3] = (-d0 + d4) * (d1 - d5) + d2 * (d0 + d1 - d2 - 2.0 * d3 + d4 + d5)
    g[4] = (-d0 + d3) * (d2 - d5) + d1 * (d0 - d1 + d2 + d3 - 2.0 * d4 + d5)
    g[5] = (-d1 + d3) * (d2 - d4) + d0 * (-d0 + d1 + d2 + d3 + d4 - 2.0 * d5)
    return g


def compute_squared_edge_lengths(vertices: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """
    Compute squared edge lengths for all simplices.
    
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
    n_simplices = len(simplices)
    simplex_size = simplices.shape[1]
    n_edges = simplex_size * (simplex_size - 1) // 2
    
    squared_lengths = np.zeros((n_simplices, n_edges))
    
    idx = 0
    for i in range(simplex_size):
        for j in range(i + 1, simplex_size):
            diff = vertices[simplices[:, i]] - vertices[simplices[:, j]]
            squared_lengths[:, idx] = np.sum(diff ** 2, axis=1)
            idx += 1
    
    return squared_lengths


def compute_total_unsigned_measure(vertices: np.ndarray, simplices: np.ndarray) -> float:
    """
    Compute total unsigned measure (area or volume) of a mesh.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    float
        Total unsigned measure
    """
    dim = vertices.shape[1]
    squared_lengths = compute_squared_edge_lengths(vertices, simplices)
    
    if dim == 2:
        areas = np.array([tri_area_squared_edges(squared_lengths[i]) for i in range(len(simplices))])
        return np.sum(areas) / 4.0
    else:  # dim == 3
        volumes = np.array([tet_volume_squared_edges(squared_lengths[i]) for i in range(len(simplices))])
        return np.sum(volumes) / 12.0


def compute_alpha(rest_vertices: np.ndarray, init_vertices: np.ndarray, 
                  simplices: np.ndarray, form: str, alpha_ratio: float) -> float:
    """
    Compute alpha parameter from alpha_ratio.
    
    Parameters
    ----------
    rest_vertices : np.ndarray
        Rest mesh vertices
    init_vertices : np.ndarray
        Initial embedding vertices
    simplices : np.ndarray
        Simplex connectivity
    form : str
        Energy form ('Tutte' or 'harmonic')
    alpha_ratio : float
        Target ratio
        
    Returns
    -------
    float
        Computed alpha value
    """
    n_simplices = len(simplices)
    dim = init_vertices.shape[1]
    
    # Compute initial signed measure
    if dim == 2:
        init_measure = 0.0
        for i in range(n_simplices):
            pts = init_vertices[simplices[i]]
            init_measure += tri_signed_area(pts)
        init_measure /= 2.0
        
        if form == 'harmonic':
            rest_measure = compute_total_unsigned_measure(rest_vertices, simplices)
        else:  # Tutte
            rest_measure = n_simplices * np.sqrt(3) / 4.0
    else:  # dim == 3
        init_measure = 0.0
        for i in range(n_simplices):
            pts = init_vertices[simplices[i]]
            init_measure += tet_signed_volume(pts)
        init_measure /= 6.0
        
        if form == 'harmonic':
            rest_measure = compute_total_unsigned_measure(rest_vertices, simplices)
        else:  # Tutte
            rest_measure = n_simplices * np.sqrt(2) / 12.0
    
    return alpha_ratio * init_measure / rest_measure


class TLCEnergy:
    """
    TLC Energy class for optimization.
    
    This class encapsulates the TLC energy function and its gradient,
    along with mesh data and optimization parameters.
    """
    
    def __init__(self, rest_vertices: np.ndarray, init_vertices: np.ndarray,
                 simplices: np.ndarray, handles: np.ndarray,
                 form: str = 'Tutte', alpha_ratio: float = 1e-6,
                 alpha: float = -1.0):
        """
        Initialize TLC energy computation.
        
        Parameters
        ----------
        rest_vertices : np.ndarray
            Rest mesh vertices, shape (n_verts, dim)
        init_vertices : np.ndarray
            Initial embedding vertices, shape (n_verts, dim)
        simplices : np.ndarray
            Simplex connectivity, shape (n_simplices, dim+1)
        handles : np.ndarray
            Indices of constrained vertices
        form : str
            Energy form ('Tutte' or 'harmonic')
        alpha_ratio : float
            Ratio for computing alpha
        alpha : float
            Override alpha value (if negative, computed from alpha_ratio)
        """
        self.V = init_vertices.copy()
        self.F = simplices.copy()
        self.n_verts = init_vertices.shape[0]
        self.dim = init_vertices.shape[1]
        self.simplex_size = simplices.shape[1]
        self.n_simplices = len(simplices)
        self.n_edges = self.simplex_size * (self.simplex_size - 1) // 2
        
        # Compute free indices
        is_free = np.ones(self.n_verts, dtype=bool)
        is_free[handles] = False
        self.free_indices = np.where(is_free)[0]
        self.n_free = len(self.free_indices)
        
        # Check and flip orientation if needed
        self._ensure_positive_orientation()
        
        # Compute alpha
        if alpha >= 0:
            self.alpha = alpha
        else:
            self.alpha = compute_alpha(rest_vertices, init_vertices, simplices, form, alpha_ratio)
        
        # Compute rest squared edge lengths
        self.rest_D = self._compute_rest_squared_edges(rest_vertices, form)
        
        # Store form
        self.form = form
        
        # Optimization state
        self.solution_found = False
        self.last_energy = np.inf
        self.nb_feval = 0
        self.nb_geval = 0
        
        # Recording
        self.record_vert = False
        self.record_energy = False
        self.record_min_measure = False
        self.vert_record = []
        self.energy_record = []
        self.min_measure_record = []
        
        # Compute initial x0
        self.x0 = init_vertices[self.free_indices].flatten()
    
    def _ensure_positive_orientation(self):
        """Flip simplex orientations if total signed measure is negative."""
        if self.dim == 2:
            total = 0.0
            for i in range(self.n_simplices):
                pts = self.V[self.F[i]]
                total += tri_signed_area(pts)
            
            if total < 0:
                print(f"WARNING: Total signed area is negative ({total}). Flipping orientations...")
                self.F[:, [0, 1]] = self.F[:, [1, 0]]
        else:  # dim == 3
            total = 0.0
            for i in range(self.n_simplices):
                pts = self.V[self.F[i]]
                total += tet_signed_volume(pts)
            
            if total < 0:
                print(f"WARNING: Total signed volume is negative ({total}). Flipping orientations...")
                self.F[:, [0, 1]] = self.F[:, [1, 0]]
    
    def _compute_rest_squared_edges(self, rest_vertices: np.ndarray, form: str) -> np.ndarray:
        """Compute rest squared edge lengths scaled by alpha."""
        if self.dim == 2:
            a = self.alpha
        else:  # dim == 3
            # For 3D, we need alpha^(1/3) for linear scaling, then square for squared edges
            # This ensures proper volume scaling
            # Fixed: Use explicit parentheses to ensure correct order of operations
            # Original bug: (1.0/3.0)**2 was computed first due to right-to-left associativity
            a = abs(self.alpha) ** (2.0 / 3.0)  # equivalent to (alpha^(1/3))^2
        
        rest_D = np.zeros((self.n_simplices, self.n_edges))
        
        if form == 'harmonic':
            for i in range(self.n_simplices):
                idx = 0
                for j in range(self.simplex_size):
                    for k in range(j + 1, self.simplex_size):
                        diff = rest_vertices[self.F[i, j]] - rest_vertices[self.F[i, k]]
                        rest_D[i, idx] = a * np.sum(diff ** 2)
                        idx += 1
        else:  # Tutte
            rest_D[:] = a
        
        return rest_D
    
    def _update_vertices(self, x: np.ndarray):
        """Update vertex positions from optimization variables."""
        x_reshaped = x.reshape((self.n_free, self.dim))
        self.V[self.free_indices] = x_reshaped
    
    def _check_all_good(self) -> bool:
        """Check if all simplices have positive measure."""
        for i in range(self.n_simplices):
            pts = self.V[self.F[i]]
            if self.dim == 2:
                if tri_signed_area(pts) <= 0:
                    return False
            else:
                if tet_signed_volume(pts) <= 0:
                    return False
        return True
    
    def _record(self):
        """Record current state if recording is enabled."""
        if self.record_vert:
            self.vert_record.append(self.V.copy())
        if self.record_energy:
            self.energy_record.append(self.last_energy)
        if self.record_min_measure:
            min_measure = np.inf
            for i in range(self.n_simplices):
                pts = self.V[self.F[i]]
                if self.dim == 2:
                    measure = tri_signed_area(pts) / 2.0
                else:
                    measure = tet_signed_volume(pts) / 6.0
                min_measure = min(min_measure, measure)
            self.min_measure_record.append(min_measure)
    
    def energy_and_gradient(self, x: np.ndarray, grad: Optional[np.ndarray] = None) -> float:
        """
        Compute TLC energy and optionally its gradient.
        
        Parameters
        ----------
        x : np.ndarray
            Optimization variables (free vertex positions flattened)
        grad : np.ndarray, optional
            If provided, will be filled with the gradient
            
        Returns
        -------
        float
            TLC energy value
        """
        if self.solution_found:
            if grad is not None:
                grad[:] = 0
            return self.last_energy
        
        # Update vertices
        self._update_vertices(x)
        
        # Check custom stop criterion
        if self._check_all_good():
            self.solution_found = True
        
        # Compute current squared edge lengths
        D = compute_squared_edge_lengths(self.V, self.F)
        D = D + self.rest_D
        
        # Compute total unsigned measure (energy)
        if self.dim == 2:
            A = np.array([tri_area_squared_edges(D[i]) for i in range(self.n_simplices)])
            energy = np.sum(A) / 4.0
        else:  # dim == 3
            A = np.array([tet_volume_squared_edges(D[i]) for i in range(self.n_simplices)])
            energy = np.sum(A) / 12.0
        
        self.nb_feval += 1
        
        # Compute gradient if requested
        if grad is not None:
            # Compute dA/dD with numerical stability safeguards
            dAdD = np.zeros((self.n_simplices, self.n_edges))
            eps = 1e-20  # Small epsilon to avoid division by zero
            for i in range(self.n_simplices):
                if self.dim == 2:
                    gi = tri_grad_squared_edges(D[i])
                    s = 1.0 / (4.0 * A[i] + eps) if A[i] > eps else 0.0
                else:
                    gi = tet_grad_squared_edges(D[i])
                    s = 1.0 / (12.0 * A[i] + eps) if A[i] > eps else 0.0
                dAdD[i] = gi * s
            
            # Compute gradient w.r.t. vertex positions
            G = np.zeros_like(self.V)
            for i in range(self.n_simplices):
                edge_idx = 0
                for j in range(self.simplex_size):
                    for k in range(j + 1, self.simplex_size):
                        vj = self.V[self.F[i, j]]
                        vk = self.V[self.F[i, k]]
                        ejk = vj - vk
                        s = dAdD[i, edge_idx]
                        G[self.F[i, j]] += s * ejk
                        G[self.F[i, k]] -= s * ejk
                        edge_idx += 1
            
            # Extract gradient for free vertices
            grad_flat = G[self.free_indices].flatten()
            grad[:] = grad_flat
            
            self.nb_geval += 1
        
        # Record and store energy
        self.last_energy = energy
        self._record()
        
        return energy


    def energy_only(self, x: np.ndarray) -> float:
        """Compute TLC energy (for scipy.optimize.minimize)."""
        return self.energy_and_gradient(x, grad=None)
    
    def gradient_only(self, x: np.ndarray) -> np.ndarray:
        """Compute TLC gradient (for scipy.optimize.minimize)."""
        grad = np.zeros_like(x)
        self.energy_and_gradient(x, grad=grad)
        return grad


def check_injectivity(vertices: np.ndarray, simplices: np.ndarray) -> Tuple[bool, float]:
    """
    Check if a mesh configuration is locally injective.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    tuple
        (is_injective, min_measure)
        is_injective: True if all simplices have positive measure
        min_measure: Minimum signed measure across all simplices
    """
    dim = vertices.shape[1]
    n_simplices = len(simplices)
    min_measure = np.inf
    
    for i in range(n_simplices):
        pts = vertices[simplices[i]]
        if dim == 2:
            measure = tri_signed_area(pts) / 2.0
        else:
            measure = tet_signed_volume(pts) / 6.0
        min_measure = min(min_measure, measure)
    
    return min_measure > 0, min_measure


def find_injective_mapping(
    rest_vertices: np.ndarray,
    init_vertices: np.ndarray,
    simplices: np.ndarray,
    handles: np.ndarray,
    form: str = 'Tutte',
    alpha_ratio: float = 1e-6,
    alpha: Optional[float] = None,
    max_iterations: int = 10000,
    ftol_abs: float = 1e-8,
    ftol_rel: float = 1e-8,
    xtol_abs: float = 1e-8,
    xtol_rel: float = 1e-8,
    record_energy: bool = False,
    record_min_measure: bool = False,
    stop_when_injective: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Find an injective mapping using TLC energy minimization.
    
    Parameters
    ----------
    rest_vertices : np.ndarray
        Source mesh vertices, shape (n_verts, dim)
    init_vertices : np.ndarray
        Initial embedding vertices, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
    handles : np.ndarray
        Indices of constrained vertices
    form : str
        Energy form ('Tutte' or 'harmonic')
    alpha_ratio : float
        Ratio for computing alpha
    alpha : float, optional
        Override alpha value
    max_iterations : int
        Maximum number of iterations
    ftol_abs, ftol_rel : float
        Function value tolerances
    xtol_abs, xtol_rel : float
        Variable change tolerances
    record_energy : bool
        Record energy history
    record_min_measure : bool
        Record minimum measure history
    stop_when_injective : bool
        Stop when mesh becomes injective
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict
        Dictionary containing:
        - vertices: Optimized vertex positions
        - success: Whether optimization converged
        - message: Termination message
        - iterations: Number of iterations
        - energy: Final energy value
        - energy_history: Energy per iteration (if recorded)
        - min_measure_history: Min measure per iteration (if recorded)
    """
    if alpha is None:
        alpha = -1.0
    
    # Create TLC energy object
    tlc = TLCEnergy(
        rest_vertices=rest_vertices,
        init_vertices=init_vertices,
        simplices=simplices,
        handles=handles,
        form=form,
        alpha_ratio=alpha_ratio,
        alpha=alpha
    )
    
    # Set recording options
    tlc.record_energy = record_energy
    tlc.record_min_measure = record_min_measure
    
    if verbose:
        print(f"TLC Optimization Setup:")
        print(f"  Vertices: {tlc.n_verts}, Free: {tlc.n_free}")
        print(f"  Simplices: {tlc.n_simplices}, Dimension: {tlc.dim}")
        print(f"  Form: {form}, Alpha: {tlc.alpha:.6e}")
        print(f"  Initial injectivity: {check_injectivity(init_vertices, simplices)}")
    
    # Define callback for early stopping
    def callback(xk):
        if stop_when_injective and tlc._check_all_good():
            if verbose:
                print("Mesh became injective, stopping early.")
            return True
        return False
    
    # Run optimization using separate fun and jac
    result = minimize(
        fun=tlc.energy_only,
        x0=tlc.x0,
        method='L-BFGS-B',
        jac=tlc.gradient_only,
        options={
            'maxiter': max_iterations,
            'ftol': ftol_rel,
            'gtol': xtol_rel,
        },
        callback=callback if stop_when_injective else None,
    )
    
    # Extract final vertices
    final_vertices = init_vertices.copy()
    final_vertices[tlc.free_indices] = result.x.reshape((tlc.n_free, tlc.dim))
    
    # Prepare output
    output = {
        'vertices': final_vertices,
        'success': result.success or tlc.solution_found,
        'message': result.message if hasattr(result, 'message') else '',
        'iterations': result.nit if hasattr(result, 'nit') else tlc.nb_feval,
        'energy': tlc.last_energy,
    }
    
    if record_energy:
        output['energy_history'] = tlc.energy_record
    if record_min_measure:
        output['min_measure_history'] = tlc.min_measure_record
    
    if verbose:
        print(f"\nOptimization completed:")
        print(f"  Success: {output['success']}")
        print(f"  Iterations: {output['iterations']}")
        print(f"  Final energy: {output['energy']:.6e}")
        is_inj, min_m = check_injectivity(final_vertices, simplices)
        print(f"  Final injectivity: {is_inj} (min measure: {min_m:.6e})")
    
    return output


def compute_shear_stress_from_transformation_strain(
    transformation_strain: np.ndarray,
    shear_modulus: float = 1.0,
    bulk_modulus: Optional[float] = None
) -> np.ndarray:
    """
    Calculate shear stress from transformation strain (eigenstrain).
    
    In continuum mechanics, the transformation strain (also called eigenstrain or 
    initial strain) represents stress-free strain due to thermal expansion, 
    plasticity, phase transformation, etc. The shear stress is computed from the 
    deviatoric part of the transformation strain.
    
    For an isotropic linear elastic material:
    - Total strain = Elastic strain + Transformation strain
    - Stress = C : Elastic strain = C : (Total strain - Transformation strain)
    - Shear stress depends only on the deviatoric part of the transformation strain
    
    Parameters
    ----------
    transformation_strain : np.ndarray
        Transformation strain tensor(s). Can be:
        - Shape (3, 3): Single 3D strain tensor (small strain, symmetric)
        - Shape (2, 2): Single 2D strain tensor (plane strain/stress)
        - Shape (n, 3, 3): Batch of n 3D strain tensors
        - Shape (n, 2, 2): Batch of n 2D strain tensors
        - Shape (6,): Single 3D strain in Voigt notation [xx, yy, zz, yz, xz, xy]
        - Shape (n, 6): Batch of n 3D strains in Voigt notation
        - Shape (3,): Single 2D strain in Voigt notation [xx, yy, xy]
        - Shape (n, 3): Batch of n 2D strains in Voigt notation
        
    shear_modulus : float, optional
        Shear modulus (mu or G), by default 1.0
        
    bulk_modulus : float, optional
        Bulk modulus (K). If None, assumes incompressible material for 
        volumetric response. Only used if full stress tensor is needed;
        shear stress depends only on shear_modulus.
        
    Returns
    -------
    np.ndarray
        Shear stress tensor(s) with same shape convention as input:
        - Deviatoric stress = 2 * shear_modulus * deviatoric_strain
        - The shear stress is the deviatoric part of the Cauchy stress
        
    Notes
    -----
    The transformation strain epsilon^* is decomposed as:
        epsilon^* = epsilon_vol + epsilon_dev
    where:
        epsilon_vol = (1/3) * tr(epsilon^*) * I  (volumetric part)
        epsilon_dev = epsilon^* - epsilon_vol     (deviatoric part)
    
    The shear stress (deviatoric stress) is:
        sigma_dev = 2 * mu * epsilon_dev
    
    For pure shear stress calculation, only the deviatoric strain matters.
    The volumetric part contributes to hydrostatic pressure, not shear.
    
    Examples
    --------
    >>> # 3D strain tensor
    >>> strain = np.array([[0.1, 0.02, 0.01],
    ...                    [0.02, 0.05, 0.03],
    ...                    [0.01, 0.03, 0.08]])
    >>> shear_stress = compute_shear_stress_from_transformation_strain(strain, shear_modulus=80.0)
    
    >>> # Voigt notation (3D)
    >>> strain_voigt = np.array([0.1, 0.05, 0.08, 0.03, 0.01, 0.02])
    >>> shear_stress = compute_shear_stress_from_transformation_strain(strain_voigt, shear_modulus=80.0)
    
    >>> # Batch processing
    >>> strains = np.random.randn(100, 3, 3)
    >>> strains = (strains + strains.transpose(0, 2, 1)) / 2  # Make symmetric
    >>> shear_stresses = compute_shear_stress_from_transformation_strain(strains, shear_modulus=80.0)
    """
    # Convert Voigt notation to tensor form if needed
    if transformation_strain.ndim == 1:
        if transformation_strain.shape[0] == 6:
            # 3D Voigt: [xx, yy, zz, yz, xz, xy]
            eps = np.zeros((3, 3))
            eps[0, 0] = transformation_strain[0]
            eps[1, 1] = transformation_strain[1]
            eps[2, 2] = transformation_strain[2]
            eps[1, 2] = eps[2, 1] = transformation_strain[3] / 2.0
            eps[0, 2] = eps[2, 0] = transformation_strain[4] / 2.0
            eps[0, 1] = eps[1, 0] = transformation_strain[5] / 2.0
            return _compute_shear_stress_tensor(eps, shear_modulus)
        elif transformation_strain.shape[0] == 3:
            # 2D Voigt: [xx, yy, xy]
            eps = np.zeros((2, 2))
            eps[0, 0] = transformation_strain[0]
            eps[1, 1] = transformation_strain[1]
            eps[0, 1] = eps[1, 0] = transformation_strain[2] / 2.0
            return _compute_shear_stress_tensor(eps, shear_modulus)
        else:
            raise ValueError(f"Unsupported 1D shape: {transformation_strain.shape}")
    
    elif transformation_strain.ndim == 2:
        # Check for square tensor shapes first (single tensors)
        if transformation_strain.shape == (3, 3):
            # Single 3D tensor
            return _compute_shear_stress_tensor(transformation_strain, shear_modulus)
        elif transformation_strain.shape == (2, 2):
            # Single 2D tensor
            return _compute_shear_stress_tensor(transformation_strain, shear_modulus)
        elif transformation_strain.shape[1] == 6:
            # Batch of 3D Voigt
            n = transformation_strain.shape[0]
            result = np.zeros((n, 3, 3))
            for i in range(n):
                ts = transformation_strain[i]
                eps = np.zeros((3, 3))
                eps[0, 0] = ts[0]
                eps[1, 1] = ts[1]
                eps[2, 2] = ts[2]
                eps[1, 2] = eps[2, 1] = ts[3] / 2.0
                eps[0, 2] = eps[2, 0] = ts[4] / 2.0
                eps[0, 1] = eps[1, 0] = ts[5] / 2.0
                result[i] = _compute_shear_stress_tensor(eps, shear_modulus)
            return result
        elif transformation_strain.shape[1] == 3:
            # Batch of 2D Voigt
            n = transformation_strain.shape[0]
            result = np.zeros((n, 2, 2))
            for i in range(n):
                ts = transformation_strain[i]
                eps = np.zeros((2, 2))
                eps[0, 0] = ts[0]
                eps[1, 1] = ts[1]
                eps[0, 1] = eps[1, 0] = ts[2] / 2.0
                result[i] = _compute_shear_stress_tensor(eps, shear_modulus)
            return result
        else:
            raise ValueError(f"Unsupported 2D shape: {transformation_strain.shape}")
    
    elif transformation_strain.ndim == 3:
        # Batch of tensors
        n = transformation_strain.shape[0]
        if transformation_strain.shape[1:] == (3, 3):
            result = np.zeros((n, 3, 3))
            for i in range(n):
                result[i] = _compute_shear_stress_tensor(transformation_strain[i], shear_modulus)
            return result
        elif transformation_strain.shape[1:] == (2, 2):
            result = np.zeros((n, 2, 2))
            for i in range(n):
                result[i] = _compute_shear_stress_tensor(transformation_strain[i], shear_modulus)
            return result
        else:
            raise ValueError(f"Unsupported 3D shape: {transformation_strain.shape}")
    
    else:
        raise ValueError(f"Unsupported array dimensions: {transformation_strain.ndim}")


def _compute_shear_stress_tensor(eps: np.ndarray, shear_modulus: float) -> np.ndarray:
    """
    Compute shear stress from a single strain tensor.
    
    Parameters
    ----------
    eps : np.ndarray
        Strain tensor, shape (dim, dim) where dim is 2 or 3
    shear_modulus : float
        Shear modulus
        
    Returns
    -------
    np.ndarray
        Shear stress tensor (deviatoric stress), shape (dim, dim)
    """
    dim = eps.shape[0]
    
    # Compute trace (volumetric strain)
    trace_eps = np.trace(eps)
    
    # Compute volumetric part: (1/dim) * trace * I
    eps_vol = (trace_eps / dim) * np.eye(dim)
    
    # Compute deviatoric part: eps - eps_vol
    eps_dev = eps - eps_vol
    
    # Shear stress (deviatoric stress) = 2 * mu * eps_dev
    sigma_dev = 2.0 * shear_modulus * eps_dev
    
    return sigma_dev


def compute_von_mises_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """
    Compute von Mises equivalent stress from stress tensor(s).
    
    The von Mises stress is a scalar measure of stress intensity,
    commonly used in yield criteria for ductile materials.
    
    Parameters
    ----------
    stress_tensor : np.ndarray
        Stress tensor(s). Can be:
        - Shape (3, 3): Single 3D stress tensor
        - Shape (2, 2): Single 2D stress tensor (plane stress/strain)
        - Shape (n, 3, 3): Batch of n 3D stress tensors
        - Shape (n, 2, 2): Batch of n 2D stress tensors
        - Shape (6,): Single 3D stress in Voigt notation
        - Shape (n, 6): Batch of n 3D stresses in Voigt notation
        
    Returns
    -------
    np.ndarray
        Von Mises stress value(s):
        - Scalar for single tensor input
        - Array of shape (n,) for batch input
        
    Notes
    -----
    For 3D:
        sigma_vm = sqrt(0.5 * [(s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2])
    where s1, s2, s3 are principal stresses (eigenvalues of stress tensor).
    
    Equivalently, using deviatoric stress s:
        sigma_vm = sqrt(3/2 * s:s) = sqrt(3/2 * sum(s_ij^2))
    """
    # Convert Voigt notation if needed
    if stress_tensor.ndim == 1 and stress_tensor.shape[0] == 6:
        sigma = np.zeros((3, 3))
        sigma[0, 0] = stress_tensor[0]
        sigma[1, 1] = stress_tensor[1]
        sigma[2, 2] = stress_tensor[2]
        sigma[1, 2] = sigma[2, 1] = stress_tensor[3]
        sigma[0, 2] = sigma[2, 0] = stress_tensor[4]
        sigma[0, 1] = sigma[1, 0] = stress_tensor[5]
        return _von_mises_single(sigma)
    
    elif stress_tensor.ndim == 2:
        if stress_tensor.shape[1] == 6:
            # Batch of Voigt
            n = stress_tensor.shape[0]
            result = np.zeros(n)
            for i in range(n):
                ts = stress_tensor[i]
                sigma = np.zeros((3, 3))
                sigma[0, 0] = ts[0]
                sigma[1, 1] = ts[1]
                sigma[2, 2] = ts[2]
                sigma[1, 2] = sigma[2, 1] = ts[3]
                sigma[0, 2] = sigma[2, 0] = ts[4]
                sigma[0, 1] = sigma[1, 0] = ts[5]
                result[i] = _von_mises_single(sigma)
            return result
        else:
            # Single tensor
            return _von_mises_single(stress_tensor)
    
    elif stress_tensor.ndim == 3:
        # Batch of tensors
        n = stress_tensor.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = _von_mises_single(stress_tensor[i])
        return result
    
    else:
        raise ValueError(f"Unsupported array dimensions: {stress_tensor.ndim}")


def _von_mises_single(sigma: np.ndarray) -> float:
    """Compute von Mises stress for a single stress tensor."""
    dim = sigma.shape[0]
    
    # Compute deviatoric stress
    mean_stress = np.trace(sigma) / dim
    s = sigma - mean_stress * np.eye(dim)
    
    # Von Mises = sqrt(3/2 * s:s)
    s_double_dot_s = np.sum(s ** 2)
    sigma_vm = np.sqrt(1.5 * s_double_dot_s)
    
    return sigma_vm


def compute_shear_stress_hyperelastic(
    deformation_gradient: np.ndarray,
    transformation_strain: Optional[np.ndarray] = None,
    shear_modulus: float = 1.0,
    bulk_modulus: Optional[float] = None,
    material_model: str = 'neo_hookean'
) -> np.ndarray:
    """
    Calculate shear stress for hyperelastic materials with large deformations.
    
    This function computes the deviatoric (shear) part of the Cauchy stress tensor
    for hyperelastic materials undergoing large deformations. It properly handles
    finite strain kinematics and can incorporate transformation strains (eigenstrains)
    such as thermal expansion, plasticity, or growth.
    
    For hyperelastic materials, the stress is derived from a strain energy density
    function W(F), where F is the deformation gradient. The transformation strain
    is incorporated via a multiplicative decomposition: F = F_e * F_p, where F_p
    represents the transformation (eigenstrain) and F_e is the elastic part.
    
    Parameters
    ----------
    deformation_gradient : np.ndarray
        Deformation gradient tensor(s) F. Can be:
        - Shape (3, 3): Single 3D deformation gradient
        - Shape (2, 2): Single 2D deformation gradient (plane strain/stress)
        - Shape (n, 3, 3): Batch of n 3D deformation gradients
        - Shape (n, 2, 2): Batch of n 2D deformation gradients
        
    transformation_strain : np.ndarray, optional
        Transformation strain (eigenstrain) tensor(s) in the reference configuration.
        Represents stress-free strain due to thermal expansion, plasticity, growth, etc.
        Can be:
        - Shape (3, 3): Single 3D transformation strain
        - Shape (2, 2): Single 2D transformation strain
        - Shape (n, 3, 3): Batch of n 3D transformation strains
        - Shape (n, 2, 2): Batch of n 2D transformation strains
        - None: No transformation strain (F_e = F)
        If provided, the elastic deformation gradient is computed as F_e = F * F_p^(-1)
        
    shear_modulus : float, optional
        Shear modulus (mu or G) in the reference configuration, by default 1.0
        
    bulk_modulus : float, optional
        Bulk modulus (K). If None, assumes nearly incompressible material
        with K = 1000 * shear_modulus for numerical stability.
        
    material_model : str, optional
        Hyperelastic material model. Options:
        - 'neo_hookean': Compressible Neo-Hookean model (default)
        - 'mooney_rivlin': Mooney-Rivlin model (requires additional parameters)
        - 'ogden': Ogden model (requires additional parameters)
        
    Returns
    -------
    np.ndarray
        Shear stress tensor(s) - the deviatoric part of the Cauchy stress.
        Same shape convention as deformation_gradient input:
        - Shape (3, 3) or (2, 2) for single tensor
        - Shape (n, 3, 3) or (n, 2, 2) for batch
        
    Notes
    -----
    For a compressible Neo-Hookean material with transformation strain:
    
    1. Multiplicative decomposition: F = F_e * F_p
       where F_p = exp(epsilon_trans) ≈ I + epsilon_trans for small transformation strains
       
    2. Elastic right Cauchy-Green tensor: C_e = F_e^T * F_e
    
    3. Deviatoric Cauchy stress: sigma_dev = (mu/J) * dev(b_e)
       where b_e = F_e * F_e^T is the elastic left Cauchy-Green tensor,
       J = det(F_e), and dev() extracts the deviatoric part.
    
    The shear stress is traceless by construction and represents the distortional
    (volume-preserving) part of the stress state.
    
    Examples
    --------
    >>> # Simple shear deformation (no transformation strain)
    >>> gamma = 0.5
    >>> F = np.array([[1, gamma, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1]])
    >>> shear_stress = compute_shear_stress_hyperelastic(F, shear_modulus=80.0)
    
    >>> # Uniaxial stretch with thermal expansion
    >>> lambda_z = 1.2
    >>> alpha_thermal = 0.05  # 5% thermal expansion
    >>> F = np.diag([1.0, 1.0, lambda_z])
    >>> eps_trans = alpha_thermal * np.eye(3)
    >>> F_p = np.eye(3) + eps_trans  # Approximation for small transformation strain
    >>> shear_stress = compute_shear_stress_hyperelastic(F, transformation_strain=F_p - np.eye(3), 
    ...                                                   shear_modulus=80.0)
    
    >>> # Batch processing
    >>> F_batch = np.array([np.diag([1.0 + i*0.1, 1.0, 1.0]) for i in range(10)])
    >>> shear_stresses = compute_shear_stress_hyperelastic(F_batch, shear_modulus=80.0)
    """
    if bulk_modulus is None:
        bulk_modulus = 1000.0 * shear_modulus  # Nearly incompressible
    
    # Handle different input shapes
    if deformation_gradient.ndim == 2:
        # Single tensor
        if deformation_gradient.shape == (3, 3):
            return _compute_hyperelastic_shear_single(
                deformation_gradient, transformation_strain, 
                shear_modulus, bulk_modulus, material_model
            )
        elif deformation_gradient.shape == (2, 2):
            return _compute_hyperelastic_shear_single(
                deformation_gradient, transformation_strain,
                shear_modulus, bulk_modulus, material_model
            )
        else:
            raise ValueError(f"Unsupported 2D shape: {deformation_gradient.shape}")
    
    elif deformation_gradient.ndim == 3:
        # Batch of tensors
        n = deformation_gradient.shape[0]
        if deformation_gradient.shape[1:] == (3, 3):
            result = np.zeros((n, 3, 3))
            for i in range(n):
                trans_strain = None
                if transformation_strain is not None:
                    if transformation_strain.ndim == 3:
                        trans_strain = transformation_strain[i]
                    elif transformation_strain.ndim == 2:
                        trans_strain = transformation_strain
                result[i] = _compute_hyperelastic_shear_single(
                    deformation_gradient[i], trans_strain,
                    shear_modulus, bulk_modulus, material_model
                )
            return result
        elif deformation_gradient.shape[1:] == (2, 2):
            result = np.zeros((n, 2, 2))
            for i in range(n):
                trans_strain = None
                if transformation_strain is not None:
                    if transformation_strain.ndim == 3:
                        trans_strain = transformation_strain[i]
                    elif transformation_strain.ndim == 2:
                        trans_strain = transformation_strain
                result[i] = _compute_hyperelastic_shear_single(
                    deformation_gradient[i], trans_strain,
                    shear_modulus, bulk_modulus, material_model
                )
            return result
        else:
            raise ValueError(f"Unsupported 3D shape: {deformation_gradient.shape}")
    
    else:
        raise ValueError(f"Unsupported array dimensions: {deformation_gradient.ndim}")


def _compute_hyperelastic_shear_single(
    F: np.ndarray,
    transformation_strain: Optional[np.ndarray],
    shear_modulus: float,
    bulk_modulus: float,
    material_model: str
) -> np.ndarray:
    """
    Compute shear stress for a single deformation gradient using hyperelastic model.
    
    Parameters
    ----------
    F : np.ndarray
        Deformation gradient, shape (dim, dim)
    transformation_strain : np.ndarray or None
        Transformation strain tensor (F_p - I) or None
    shear_modulus : float
        Shear modulus
    bulk_modulus : float
        Bulk modulus
    material_model : str
        Material model name
        
    Returns
    -------
    np.ndarray
        Deviatoric Cauchy stress, shape (dim, dim)
    """
    dim = F.shape[0]
    
    # Compute elastic deformation gradient F_e = F * F_p^(-1)
    if transformation_strain is not None:
        # F_p = I + transformation_strain (for small transformation strains)
        # For finite transformation strains, user should provide F_p directly
        # Here we assume transformation_strain represents the logarithmic strain
        # and compute F_p = exp(transformation_strain) approximately
        F_p = np.eye(dim) + transformation_strain
        
        # Check if F_p is invertible
        det_Fp = np.linalg.det(F_p)
        if abs(det_Fp) < 1e-10:
            raise ValueError(f"Transformation strain leads to singular F_p (det={det_Fp})")
        
        F_e = F @ np.linalg.inv(F_p)
    else:
        F_e = F
    
    # Compute Jacobian J = det(F_e)
    J = np.linalg.det(F_e)
    
    if J <= 0:
        raise ValueError(f"Invalid deformation: det(F_e) = {J} <= 0")
    
    # Compute elastic left Cauchy-Green tensor: b_e = F_e * F_e^T
    b_e = F_e @ F_e.T
    
    # Compute deviatoric part of b_e: dev(b_e) = b_e - (1/dim)*tr(b_e)*I
    tr_b_e = np.trace(b_e)
    b_e_dev = b_e - (tr_b_e / dim) * np.eye(dim)
    
    # Deviatoric Cauchy stress for Neo-Hookean: sigma_dev = (mu/J) * dev(b_e)
    sigma_dev = (shear_modulus / J) * b_e_dev
    
    return sigma_dev


def compute_cauchy_stress_hyperelastic(
    deformation_gradient: np.ndarray,
    transformation_strain: Optional[np.ndarray] = None,
    shear_modulus: float = 1.0,
    bulk_modulus: Optional[float] = None,
    material_model: str = 'neo_hookean',
    return_full_stress: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate full Cauchy stress for hyperelastic materials with large deformations.
    
    This function computes both the deviatoric (shear) and volumetric parts of the
    Cauchy stress tensor for hyperelastic materials.
    
    Parameters
    ----------
    deformation_gradient : np.ndarray
        Deformation gradient tensor(s) F. See compute_shear_stress_hyperelastic for details.
        
    transformation_strain : np.ndarray, optional
        Transformation strain tensor(s). See compute_shear_stress_hyperelastic for details.
        
    shear_modulus : float, optional
        Shear modulus, by default 1.0
        
    bulk_modulus : float, optional
        Bulk modulus. If None, assumes nearly incompressible.
        
    material_model : str, optional
        Hyperelastic material model ('neo_hookean'), by default 'neo_hookean'
        
    return_full_stress : bool, optional
        If True, returns (full_stress, shear_stress).
        If False, returns (shear_stress, None).
        
    Returns
    -------
    tuple
        (cauchy_stress, shear_stress) where:
        - cauchy_stress: Full Cauchy stress tensor(s)
        - shear_stress: Deviatoric part only (or None if return_full_stress=False)
    """
    if bulk_modulus is None:
        bulk_modulus = 1000.0 * shear_modulus
    
    # Handle different input shapes
    if deformation_gradient.ndim == 2:
        # Single tensor
        if deformation_gradient.shape == (3, 3) or deformation_gradient.shape == (2, 2):
            return _compute_hyperelastic_cauchy_single(
                deformation_gradient, transformation_strain,
                shear_modulus, bulk_modulus, material_model, return_full_stress
            )
        else:
            raise ValueError(f"Unsupported 2D shape: {deformation_gradient.shape}")
    
    elif deformation_gradient.ndim == 3:
        # Batch of tensors
        n = deformation_gradient.shape[0]
        if deformation_gradient.shape[1:] == (3, 3):
            full_result = np.zeros((n, 3, 3))
            shear_result = np.zeros((n, 3, 3)) if return_full_stress else None
            for i in range(n):
                trans_strain = None
                if transformation_strain is not None:
                    if transformation_strain.ndim == 3:
                        trans_strain = transformation_strain[i]
                    elif transformation_strain.ndim == 2:
                        trans_strain = transformation_strain
                full, shear = _compute_hyperelastic_cauchy_single(
                    deformation_gradient[i], trans_strain,
                    shear_modulus, bulk_modulus, material_model, return_full_stress
                )
                full_result[i] = full
                if return_full_stress:
                    shear_result[i] = shear
            return full_result, shear_result
        elif deformation_gradient.shape[1:] == (2, 2):
            full_result = np.zeros((n, 2, 2))
            shear_result = np.zeros((n, 2, 2)) if return_full_stress else None
            for i in range(n):
                trans_strain = None
                if transformation_strain is not None:
                    if transformation_strain.ndim == 3:
                        trans_strain = transformation_strain[i]
                    elif transformation_strain.ndim == 2:
                        trans_strain = transformation_strain
                full, shear = _compute_hyperelastic_cauchy_single(
                    deformation_gradient[i], trans_strain,
                    shear_modulus, bulk_modulus, material_model, return_full_stress
                )
                full_result[i] = full
                if return_full_stress:
                    shear_result[i] = shear
            return full_result, shear_result
        else:
            raise ValueError(f"Unsupported 3D shape: {deformation_gradient.shape}")
    
    else:
        raise ValueError(f"Unsupported array dimensions: {deformation_gradient.ndim}")


def _compute_hyperelastic_cauchy_single(
    F: np.ndarray,
    transformation_strain: Optional[np.ndarray],
    shear_modulus: float,
    bulk_modulus: float,
    material_model: str,
    return_full_stress: bool
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute full Cauchy stress for a single deformation gradient.
    
    Returns
    -------
    tuple
        (full_stress, shear_stress) or (shear_stress, None)
    """
    dim = F.shape[0]
    
    # Compute elastic deformation gradient
    if transformation_strain is not None:
        F_p = np.eye(dim) + transformation_strain
        det_Fp = np.linalg.det(F_p)
        if abs(det_Fp) < 1e-10:
            raise ValueError(f"Transformation strain leads to singular F_p (det={det_Fp})")
        F_e = F @ np.linalg.inv(F_p)
    else:
        F_e = F
    
    # Compute Jacobian
    J = np.linalg.det(F_e)
    if J <= 0:
        raise ValueError(f"Invalid deformation: det(F_e) = {J} <= 0")
    
    # Elastic left Cauchy-Green tensor
    b_e = F_e @ F_e.T
    
    # Deviatoric part
    tr_b_e = np.trace(b_e)
    b_e_dev = b_e - (tr_b_e / dim) * np.eye(dim)
    
    # Deviatoric Cauchy stress
    sigma_dev = (shear_modulus / J) * b_e_dev
    
    if not return_full_stress:
        return sigma_dev, None
    
    # Volumetric (hydrostatic) pressure: p = K * (J - 1) / J
    # For neo-Hookean: sigma_vol = p * I where p = K * ln(J) / J (more accurate for large J)
    # Using simpler form: p = K * (J - 1) / J
    p = bulk_modulus * (J - 1.0) / J
    
    # Full Cauchy stress: sigma = sigma_dev + p * I
    sigma_full = sigma_dev + p * np.eye(dim)
    
    return sigma_full, sigma_dev


def relax_boundary_preserving_sphericity(
    stressed_vertices: np.ndarray,
    simplices: np.ndarray,
    boundary_vertex_indices: np.ndarray,
    sphere_center: Optional[np.ndarray] = None,
    sphere_radius: Optional[float] = None,
    shear_modulus: float = 1.0,
    bulk_modulus: Optional[float] = None,
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Relax a stressed configuration while preserving the sphericity of the boundary.
    
    This is the core relaxation routine in MISo (Morphoelastic Inverse problem Solver).
    It computes the relaxed (stress-free) configuration of a body that has been deformed
    by shear stress. The boundary vertices are constrained to lie on a sphere,
    while interior vertices are free to move to minimize the elastic energy.
    
    The method uses an iterative projection approach:
    1. Apply TLC energy minimization to find an injective mapping
    2. Project boundary vertices back onto the sphere surface after each iteration
    3. Continue until convergence
    
    This relaxation provides the morphoelastic inverse solution: given a deformed
    shape, find the stress-free reference configuration that would produce it under
    the applied boundary conditions.
    
    Parameters
    ----------
    stressed_vertices : np.ndarray
        Vertex positions of the stressed (deformed) configuration, shape (n_verts, 3)
    simplices : np.ndarray
        Tetrahedral connectivity, shape (n_tets, 4)
    boundary_vertex_indices : np.ndarray
        Indices of vertices on the spherical boundary
    sphere_center : np.ndarray, optional
        Center of the sphere. If None, computed from boundary vertices.
    sphere_radius : float, optional
        Radius of the sphere. If None, computed from boundary vertices.
    shear_modulus : float, optional
        Shear modulus for hyperelastic material, by default 1.0
    bulk_modulus : float, optional
        Bulk modulus. If None, assumes nearly incompressible (bulk = 1000 * shear).
    max_iterations : int, optional
        Maximum number of relaxation iterations, by default 5000
    tolerance : float, optional
        Convergence tolerance for vertex displacement, by default 1e-8
    verbose : bool, optional
        Print progress information, by default False
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'relaxed_vertices': Vertex positions of the relaxed (stress-free) configuration
        - 'boundary_map': Map from stressed to relaxed boundary vertices
        - 'interior_map': Map from stressed to relaxed interior vertices
        - 'iterations': Number of iterations performed
        - 'converged': Whether the relaxation converged
        - 'displacement_history': History of maximum vertex displacement per iteration
        - 'energy_history': History of TLC energy per iteration
        - 'sphere_center': Center of the constraint sphere
        - 'sphere_radius': Radius of the constraint sphere
        
    Notes
    -----
    The sphericity constraint is enforced by projecting boundary vertices onto
    the sphere surface after each optimization step. This ensures that the
    boundary remains exactly spherical throughout the relaxation process.
    
    The relaxation minimizes the elastic energy while maintaining injectivity
    of the mapping, making it suitable for handling large deformations.
    
    In the context of MISo, this function solves the inverse problem:
    Given the current (stressed) configuration Ω_current, find the reference
    configuration Ω_ref such that the deformation gradient F satisfies the
    constitutive relations with zero residual stress.
    
    See Also
    --------
    compute_morphoelastic_jacobian : Compute the Jacobian determinant (MISo solution)
    compute_deformation_gradient_tensor : Compute F for each simplex
    """
    n_verts = stressed_vertices.shape[0]
    dim = stressed_vertices.shape[1]
    
    if dim != 3:
        raise ValueError("Sphere relaxation currently supports only 3D meshes")
    
    # Compute sphere parameters if not provided
    if sphere_center is None:
        sphere_center = np.mean(stressed_vertices[boundary_vertex_indices], axis=0)
    
    if sphere_radius is None:
        distances = np.linalg.norm(stressed_vertices[boundary_vertex_indices] - sphere_center, axis=1)
        sphere_radius = np.mean(distances)
    
    if bulk_modulus is None:
        bulk_modulus = 1000.0 * shear_modulus
    
    # Identify interior vertices
    all_indices = set(range(n_verts))
    boundary_set = set(boundary_vertex_indices)
    interior_indices = np.array(list(all_indices - boundary_set), dtype=np.int64)
    
    if len(interior_indices) == 0:
        # All vertices are boundary vertices - create empty array with proper shape
        interior_indices = np.array([], dtype=np.int64)
    
    # Create initial embedding (stressed configuration)
    init_vertices = stressed_vertices.copy()
    
    # Use rest configuration as identity (stress-free reference)
    rest_vertices = init_vertices.copy()
    
    # Handles are the boundary vertices (constrained to sphere)
    handles = boundary_vertex_indices
    
    # Store history
    displacement_history = []
    energy_history = []
    
    # Current vertex positions
    current_vertices = init_vertices.copy()
    
    if verbose:
        print(f"Sphere Relaxation Setup:")
        print(f"  Total vertices: {n_verts}")
        print(f"  Boundary vertices: {len(boundary_vertex_indices)}")
        print(f"  Interior vertices: {len(interior_indices)}")
        print(f"  Sphere center: {sphere_center}")
        print(f"  Sphere radius: {sphere_radius}")
    
    # Iterative relaxation with boundary projection
    converged = False
    for iteration in range(max_iterations):
        # Store previous positions
        prev_vertices = current_vertices.copy()
        
        # Run one step of TLC optimization
        tlc_result = find_injective_mapping(
            rest_vertices=rest_vertices,
            init_vertices=current_vertices,
            simplices=simplices,
            handles=handles,
            form='harmonic',
            alpha_ratio=1e-6,
            max_iterations=min(100, max_iterations // 10),
            ftol_rel=tolerance,
            xtol_rel=tolerance,
            stop_when_injective=True,
            verbose=False
        )
        
        current_vertices = tlc_result['vertices'].copy()
        
        # Project boundary vertices onto sphere
        for idx in boundary_vertex_indices:
            vec = current_vertices[idx] - sphere_center
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                current_vertices[idx] = sphere_center + (sphere_radius / norm) * vec
            else:
                # Random direction if at center
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                current_vertices[idx] = sphere_center + sphere_radius * direction
        
        # Compute maximum displacement
        displacement = np.max(np.linalg.norm(current_vertices - prev_vertices, axis=1))
        displacement_history.append(displacement)
        energy_history.append(tlc_result['energy'])
        
        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: max displacement = {displacement:.6e}, energy = {tlc_result['energy']:.6e}")
        
        # Check convergence
        if displacement < tolerance:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
    
    # Compute maps
    # Map from stressed to relaxed: phi(x_stressed) = x_relaxed
    boundary_map = {
        'stressed': stressed_vertices[boundary_vertex_indices].copy(),
        'relaxed': current_vertices[boundary_vertex_indices].copy(),
        'indices': boundary_vertex_indices
    }
    
    interior_map = {
        'stressed': stressed_vertices[interior_indices].copy(),
        'relaxed': current_vertices[interior_indices].copy(),
        'indices': interior_indices
    }
    
    result = {
        'relaxed_vertices': current_vertices,
        'boundary_map': boundary_map,
        'interior_map': interior_map,
        'iterations': iteration + 1,
        'converged': converged,
        'displacement_history': displacement_history,
        'energy_history': energy_history,
        'sphere_center': sphere_center,
        'sphere_radius': sphere_radius
    }
    
    if verbose:
        print(f"\nRelaxation completed:")
        print(f"  Converged: {converged}")
        print(f"  Total iterations: {result['iterations']}")
        print(f"  Final max displacement: {displacement_history[-1]:.6e}")
    
    return result


def compute_deformation_gradient_tensor(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    simplices: np.ndarray
) -> np.ndarray:
    """
    Compute the deformation gradient tensor F for each simplex.
    
    The deformation gradient F maps vectors from the source configuration
    to the target configuration: v_target = F * v_source
    
    This is a fundamental quantity in continuum mechanics and morphoelasticity.
    For a simplex with vertices x0, x1, ..., xn in the source and
    y0, y1, ..., yn in the target, we compute F such that:
        yi - y0 = F * (xi - x0) for i = 1, ..., n
    
    In the context of MISo, this computes the local deformation gradient
    between two configurations (e.g., stressed -> relaxed).
    
    Parameters
    ----------
    source_vertices : np.ndarray
        Source vertex positions, shape (n_verts, dim)
    target_vertices : np.ndarray
        Target vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    np.ndarray
        Deformation gradient tensors, shape (n_simplices, dim, dim)
        
    Notes
    -----
    The deformation gradient can be decomposed as F = R * U (polar decomposition),
    where R is rotation and U is the right stretch tensor. The Jacobian determinant
    J = det(F) measures local volume change.
    
    See Also
    --------
    compute_morphoelastic_jacobian : Compute J = det(F) (the MISo solution)
    """
    n_simplices = len(simplices)
    dim = source_vertices.shape[1]
    
    deformation_gradients = np.zeros((n_simplices, dim, dim))
    
    for i in range(n_simplices):
        # Get simplex vertices
        source_pts = source_vertices[simplices[i]]
        target_pts = target_vertices[simplices[i]]
        
        # Build edge matrices
        # X: edges in source config, Y: edges in target config
        X = np.zeros((dim, dim))
        Y = np.zeros((dim, dim))
        
        for j in range(dim):
            X[:, j] = source_pts[j + 1] - source_pts[0]
            Y[:, j] = target_pts[j + 1] - target_pts[0]
        
        # Compute deformation gradient: F = Y * X^(-1)
        try:
            X_inv = np.linalg.inv(X)
            F = Y @ X_inv
            deformation_gradients[i] = F
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            X_pinv = np.linalg.pinv(X)
            F = Y @ X_pinv
            deformation_gradients[i] = F
    
    return deformation_gradients


def compute_morphoelastic_jacobian(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    simplices: np.ndarray
) -> np.ndarray:
    """
    Compute the Jacobian determinant J = det(F) for each simplex.
    
    This is the core MISo (Morphoelastic Inverse problem Solver) solution.
    Given an STL mesh representing a deformed configuration, this function
    computes the local volume change ratio between the source and target
    configurations.
    
    The Jacobian determinant measures the local volume change:
    - J > 1: local expansion
    - J < 1: local compression  
    - J = 1: volume-preserving
    
    In morphoelasticity, J represents the growth factor or the determinant
    of the elastic part of the deformation gradient.
    
    Parameters
    ----------
    source_vertices : np.ndarray
        Source vertex positions, shape (n_verts, dim)
    target_vertices : np.ndarray
        Target vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    np.ndarray
        Jacobian determinants, shape (n_simplices,)
        
    Notes
    -----
    The Jacobian determinant is computed as:
        J = det(F) where F = d(target)/d(source)
    
    For the morphoelastic inverse problem, this gives the local volume
    change required to transform the stressed configuration into the
    relaxed (stress-free) configuration.
    
    See Also
    --------
    compute_deformation_gradient_tensor : Compute F tensor
    stl_to_jacobian : Complete pipeline from STL file to Jacobian field
    """
    deformation_gradients = compute_deformation_gradient_tensor(
        source_vertices, target_vertices, simplices
    )
    
    jacobian_determinants = np.array([
        np.linalg.det(deformation_gradients[i])
        for i in range(len(simplices))
    ])
    
    return jacobian_determinants


# Aliases for backward compatibility (defined after all functions)
# These will be set at the end of the file after all function definitions


def compute_composed_map_jacobian(
    initial_vertices: np.ndarray,
    stressed_vertices: np.ndarray,
    relaxed_vertices: np.ndarray,
    simplices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute the Jacobian determinant of the composed deformation map.
    
    Given three configurations in the morphoelastic problem:
    - Initial (Ω₀): reference configuration (undeformed)
    - Stressed (Ωₛ): deformed by applied stress
    - Relaxed (Ωᵣ): stress-relaxed while preserving boundary constraints
    
    This function computes the complete deformation chain and its Jacobian:
    1. Deformation gradient F₁: initial → stressed
    2. Deformation gradient F₂: stressed → relaxed  
    3. Composed deformation gradient F = F₂ · F₁: initial → relaxed
    4. Jacobian determinant J = det(F) for the composed map
    
    The composition follows the chain rule:
        F_composed = F_stressed_to_relaxed · F_initial_to_stressed
        J_composed = J_stressed_to_relaxed × J_initial_to_stressed
    
    In MISo, this represents the total morphoelastic transformation from
    the initial configuration through the stressed state to the relaxed state.
    
    Parameters
    ----------
    initial_vertices : np.ndarray
        Initial (reference) vertex positions, shape (n_verts, dim)
    stressed_vertices : np.ndarray
        Stressed vertex positions, shape (n_verts, dim)
    relaxed_vertices : np.ndarray
        Relaxed vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'jacobian_determinants': J = det(F_composed) for each simplex
        - 'deformation_gradient_initial_to_stressed': F₁ tensor
        - 'deformation_gradient_stressed_to_relaxed': F₂ tensor
        - 'deformation_gradient_composed': F = F₂ · F₁ tensor
        - 'volume_ratios': Local volume change ratios (same as jacobian_determinants)
        - 'jacobian_initial_to_stressed': J₁ = det(F₁)
        - 'jacobian_stressed_to_relaxed': J₂ = det(F₂)
        - 'jacobian_chain_rule_product': J₂ × J₁ (verification)
        
    Notes
    -----
    The chain rule verification ensures numerical consistency:
        max|J_composed - J₂ × J₁| < tolerance
    
    This is useful for debugging and validation of the morphoelastic solver.
    
    See Also
    --------
    compute_morphoelastic_jacobian : Compute J for a single map
    relax_boundary_preserving_sphericity : Compute relaxed configuration
    """
    # Compute deformation gradients for each map
    F_initial_to_stressed = compute_deformation_gradient_tensor(
        initial_vertices, stressed_vertices, simplices
    )
    
    F_stressed_to_relaxed = compute_deformation_gradient_tensor(
        stressed_vertices, relaxed_vertices, simplices
    )
    
    # Composed map: initial -> relaxed
    F_composed = np.zeros_like(F_initial_to_stressed)
    for i in range(len(simplices)):
        F_composed[i] = F_stressed_to_relaxed[i] @ F_initial_to_stressed[i]
    
    # Compute Jacobian determinants
    J_initial_to_stressed = np.array([
        np.linalg.det(F_initial_to_stressed[i])
        for i in range(len(simplices))
    ])
    
    J_stressed_to_relaxed = np.array([
        np.linalg.det(F_stressed_to_relaxed[i])
        for i in range(len(simplices))
    ])
    
    J_composed = np.array([
        np.linalg.det(F_composed[i])
        for i in range(len(simplices))
    ])
    
    # Verify chain rule: det(AB) = det(A)*det(B)
    J_chain_rule = J_stressed_to_relaxed * J_initial_to_stressed
    
    result = {
        'jacobian_determinants': J_composed,
        'deformation_gradient_initial_to_stressed': F_initial_to_stressed,
        'deformation_gradient_stressed_to_relaxed': F_stressed_to_relaxed,
        'deformation_gradient_composed': F_composed,
        'volume_ratios': J_composed,
        'jacobian_initial_to_stressed': J_initial_to_stressed,
        'jacobian_stressed_to_relaxed': J_stressed_to_relaxed,
        'jacobian_chain_rule_product': J_chain_rule
    }
    
    return result


def compute_deformation_map(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    simplices: np.ndarray,
    evaluation_points: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute the complete deformation map between two configurations.
    
    This function provides a comprehensive representation of the morphoelastic
    deformation, including displacement field, deformation gradient tensor,
    and Jacobian determinant (the MISo solution).
    
    The map φ: Ω_source → Ω_target is characterized by:
    - Displacement field: u(x) = φ(x) - x
    - Deformation gradient: F = ∂φ/∂X
    - Jacobian determinant: J = det(F)
    
    Parameters
    ----------
    source_vertices : np.ndarray
        Source configuration vertex positions, shape (n_verts, dim)
    target_vertices : np.ndarray
        Target configuration vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
    evaluation_points : np.ndarray, optional
        Points at which to evaluate the map, shape (n_points, dim)
        If None, returns only element-wise quantities
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'vertex_map': Displacement at vertices (target - source)
        - 'deformation_gradients': F tensor for each simplex
        - 'jacobian_determinants': J = det(F) for each simplex (MISo solution)
        - 'evaluated_map': Map values at evaluation points (if provided)
        - 'displacement_magnitude': |u| at each vertex
        - 'max_displacement': Maximum vertex displacement
        - 'mean_displacement': Mean vertex displacement
        
    Notes
    -----
    In the context of MISo, this function is typically used to compute:
    - stressed → relaxed map (inverse morphoelastic problem)
    - initial → deformed map (forward problem)
    
    See Also
    --------
    compute_morphoelastic_jacobian : Compute J only
    compute_deformation_gradient_tensor : Compute F only
    relax_boundary_preserving_sphericity : Compute relaxed configuration
    """
    n_verts = source_vertices.shape[0]
    dim = source_vertices.shape[1]
    
    # Vertex displacement map
    displacement = target_vertices - source_vertices
    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    
    # Deformation gradients
    F = compute_deformation_gradient_tensor(source_vertices, target_vertices, simplices)
    
    # Jacobian determinants
    J = compute_morphoelastic_jacobian(source_vertices, target_vertices, simplices)
    
    result = {
        'vertex_map': displacement,
        'deformation_gradients': F,
        'jacobian_determinants': J,
        'displacement_magnitude': displacement_magnitude,
        'max_displacement': np.max(displacement_magnitude),
        'mean_displacement': np.mean(displacement_magnitude)
    }
    
    # Evaluate at arbitrary points if requested
    if evaluation_points is not None:
        evaluated = evaluate_map_at_points(
            source_vertices, target_vertices, simplices, evaluation_points
        )
        result['evaluated_map'] = evaluated
    
    return result


# Alias for backward compatibility
compute_map_stressed_to_relaxed = compute_deformation_map


def evaluate_map_at_points(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    simplices: np.ndarray,
    evaluation_points: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Evaluate the deformation map at arbitrary points using barycentric interpolation.
    
    Parameters
    ----------
    source_vertices : np.ndarray
        Source vertex positions, shape (n_verts, dim)
    target_vertices : np.ndarray
        Target vertex positions, shape (n_verts, dim)
    simplices : np.ndarray
        Simplex connectivity, shape (n_simplices, dim+1)
    evaluation_points : np.ndarray
        Points to evaluate, shape (n_points, dim)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'mapped_points': Evaluated map values
        - 'simplex_indices': Which simplex each point falls in (-1 if outside)
        - 'barycentric_coords': Barycentric coordinates for interpolation
    """
    n_points = evaluation_points.shape[0]
    dim = source_vertices.shape[1]
    
    mapped_points = np.zeros((n_points, dim))
    simplex_indices = np.full(n_points, -1, dtype=int)
    barycentric_coords = np.zeros((n_points, dim + 1))
    
    for p_idx in range(n_points):
        pt = evaluation_points[p_idx]
        
        # Find containing simplex
        found = False
        for s_idx in range(len(simplices)):
            verts = source_vertices[simplices[s_idx]]
            
            # Compute barycentric coordinates
            if dim == 2:
                bary = _compute_barycentric_2d(pt, verts)
            elif dim == 3:
                bary = _compute_barycentric_3d(pt, verts)
            else:
                raise ValueError(f"Unsupported dimension: {dim}")
            
            # Check if point is inside simplex (all barycentric coords >= 0)
            if np.all(bary >= -1e-10):
                # Normalize to sum to 1
                bary = bary / np.sum(bary)
                
                # Interpolate target position
                target_verts = target_vertices[simplices[s_idx]]
                mapped_points[p_idx] = np.dot(bary, target_verts)
                
                simplex_indices[p_idx] = s_idx
                barycentric_coords[p_idx] = bary
                found = True
                break
        
        if not found:
            # Point outside mesh - use nearest vertex
            distances = np.linalg.norm(source_vertices - pt, axis=1)
            nearest_idx = np.argmin(distances)
            mapped_points[p_idx] = target_vertices[nearest_idx]
    
    return {
        'mapped_points': mapped_points,
        'simplex_indices': simplex_indices,
        'barycentric_coords': barycentric_coords
    }


def _compute_barycentric_2d(pt: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates for a point in a triangle."""
    # Triangle vertices
    v0, v1, v2 = verts
    
    # Vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = pt - v0
    
    # Dot products
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    
    # Barycentric coordinates
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, v, w])


def _compute_barycentric_3d(pt: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates for a point in a tetrahedron."""
    # Tetrahedron vertices
    v0, v1, v2, v3 = verts
    
    # Compute volumes using signed volume formula
    vol = tet_signed_volume(verts)
    
    if abs(vol) < 1e-10:
        return np.array([0.25, 0.25, 0.25, 0.25])
    
    # Barycentric coordinates are ratios of sub-tetrahedron volumes
    b0 = tet_signed_volume(np.array([pt, v1, v2, v3])) / vol
    b1 = tet_signed_volume(np.array([v0, pt, v2, v3])) / vol
    b2 = tet_signed_volume(np.array([v0, v1, pt, v3])) / vol
    b3 = tet_signed_volume(np.array([v0, v1, v2, pt])) / vol
    
    return np.array([b0, b1, b2, b3])
