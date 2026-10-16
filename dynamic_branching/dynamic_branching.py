#!/usr/bin/env python3
"""
Phase-Field Fracture with AMRC and Condensation of Linear Inactive Parts (CLIP)
===============================================================================
Dynamic crack branching under HHT-alpha time integration (paper Section 4.2).
Plate [0, W] × [−L/2, L/2] (100 × 40 mm), edge crack of length W/2 along its mid-plane,
tensile traction p(t) on the top and bottom edges (the paper's σ_yy).

Authors
-------
Aarosh Dahal¹ and Aditya Kumar¹

¹ School of Civil and Environmental Engineering,
  Georgia Institute of Technology, Atlanta, GA 30332, USA

Contact: adahal8@gatech.edu, aditya.kumar@ce.gatech.edu

References
----------
Method   A. Dahal, A. Kumar, "Adaptive Crack-Tip Localization of Phase-Field Fracture
         Computations" (AMRC-CLIP).
AMRC     A. Dahal, A. Gupta, A. Kumar, "Adaptive mesh coarsening for efficient phase-field
         fracture simulations", arXiv:2609.21201 (2026).
Model    A. Kumar, B. Bourdin, G. A. Francfort, O. Lopez-Pamies, "Revisiting nucleation in
         the phase-field approach to brittle fracture", JMPS 142 (2020) 104027.
         Dynamics: Y. Liu, O. Lopez-Pamies, J. E. Dolbow, "On the effects of material
         strength in dynamic fracture: A phase-field study", arXiv:2411.16393 (2024);
         A. Dahal, U. Khayaz, R. Duddu, A. Kumar, "Dynamic phase-field model for brittle
         fracture in grounded glaciers", arXiv:2607.26274 (2026).

Physics
-------
- 2D plane-stress, dynamic mechanics with degraded stiffness and mass (paper Section 2.2)
- Complete phase-field fracture with Drucker-Prager strength surface
- HHT-alpha / Newmark time integration (paper Section 2.2.1)
- Tensile traction p(t) n on the top and bottom edges, ramped up linearly from 0 to p_max
  over t_ramp_up (= dt) and then held constant (evaluated at the HHT time t - alpha_f dt);
  the left and right edges and the crack faces are traction-free; no Dirichlet condition and
  no condition on z (the crack is a slit in the mesh)
- At rest (u = v = a = 0) and intact (z = 1) at t = 0

Units: consistent (mm, tonne, s; N, MPa)
"""

# =============================================================================
# SECTION 0: Imports
# =============================================================================

import sys, os, time, json, ctypes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sps
import scipy.linalg as sla
import basix
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import dolfinx.fem.petsc
import dolfinx.cpp as _cpp
from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import (compute_midpoints, create_submesh, exterior_facet_indices,
                          locate_entities_boundary, meshtags, refine, RefinementOption)
import ufl
from ufl import (CellDiameter, FacetNormal, Identity, Measure, TestFunction,
                 TrialFunction, conditional, derivative, grad, inner, lt, sqrt, sym, tr)

comm = MPI.COMM_WORLD

# =============================================================================
# SECTION 1: Method Selection
# =============================================================================

METHOD = "amrc_clip"   # "amrc" | "amrc_clip" | "amrc_clip_tw"

if METHOD not in ("amrc", "amrc_clip", "amrc_clip_tw"):
    raise SystemExit('METHOD must be "amrc", "amrc_clip" or "amrc_clip_tw"')
LOCAL = METHOD != "amrc"   # solve on a patch (AMRC-CLIP, AMRC-CLIP-TW)

# =============================================================================
# SECTION 2: Geometry
# =============================================================================

W  = 100.0         # plate width  (mm), x in [0, W]
L  = 40.0          # plate height (mm), y in [-L/2, L/2]
cw = 0.1           # crack mouth width on the left edge (mm); the crack tip is at (W/2, 0)

# =============================================================================
# SECTION 3: Material Properties
# =============================================================================

# --- Elastic constants and density ---
E, nu = 72000.0, 0.22                          # Young's modulus (MPa), Poisson's ratio
rho_s = 2.44e-9                                # mass density (tonne/mm^3)
mu    = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
kappa = E / (3 * (1 - 2 * nu))

# --- Strength surface (Drucker-Prager) ---
sts, scs = 2 * 3.08, 2 * 3 * 3.08              # uniaxial tensile / compressive strength (MPa)
shs = 2 * scs * sts / (3 * (scs - sts))        # hydrostatic strength
Wts = 0.5 * sts**2 / E
Whs = 0.5 * shs**2 / kappa

# --- Fracture parameters ---
Gc  = 0.0038       # critical energy release rate (N/mm)
eps = 0.625        # regularisation length (mm)
eta = 1e-7         # residual stiffness

# =============================================================================
# SECTION 4: AMR Parameters
# =============================================================================

H0_EPS  = 1.0                  # coarse element size / eps
HMIN_H0 = 0.25                 # refined element size / h0
h0      = H0_EPS * eps         # coarse (base) mesh element size
h_min   = HMIN_H0 * h0         # element size after full refinement
amr_refine_radius = 3 * eps    # refinement radius R around the effective cells
pf_flag_threshold = -0.1       # indicator threshold
hysteresis_steps  = 50         # effective while flagged within the last this many steps (0 = off)

# =============================================================================
# SECTION 5: Phase-Field Control
# =============================================================================

v_th = 0.05        # irreversibility threshold
z_ub_offset = 0.0  # z_ub = z + this where z <= v_th (no margin)

# --- Strength-surface coefficients (depend on h_min) ---
corr  = 1 / (1 + 3 * h_min / (8 * eps))
delta = corr**2 * ((sts + (1 + 2 * 3**0.5) * shs) / ((8 + 3 * 3**0.5) * shs)) \
    * 3 * Gc / (16 * Wts * eps) + corr * (2 / 5)
beta1 = -(delta * Gc) / (shs * 8 * eps) + (2 * Whs) / (3 * shs)
beta2 = -(3**0.5 * (3 * shs - sts) * delta * Gc) / (shs * sts * 8 * eps) \
    - (2 * Whs) / (3**0.5 * shs) + (2 * 3**0.5 * Wts) / sts
G_eps = 3 * delta * Gc / 8 / eps

# =============================================================================
# SECTION 6: Patch Parameters (AMRC-CLIP, AMRC-CLIP-TW)
# =============================================================================

PATCH_R_EPS   = 4.0           # patch = coarse cells within PATCH_R_EPS * eps of A (Rp)
DAMAGE_Z      = 1 - 1e-5      # D: cells with a node z below this
RETIRE_STEPS  = 1             # damaged cells stay in the patch this many refinement events
                              # after they were last refined, so a coarsened band re-damages
                              # in the patch before it freezes

# =============================================================================
# SECTION 7: Loading, Time Stepping & Stagger Control
# =============================================================================

# --- Time stepping (HHT-alpha / Newmark) ---
dt        = 2.5e-8                    # time step (s)
t_max     = 80e-6                     # end time (s): steps while t - dt < t_max ...
max_steps = 2000                      # ... and at most this many steps (2000 steps = 50 us)
MAX_STEPS = 0                         # stop after this many steps (0 = run to max_steps / t_max)
alpha_m, alpha_f = 0.0, 0.1           # HHT-alpha parameters
gamma_nm  = 0.5 + alpha_f             # Newmark gamma
beta_nm   = 0.25 * (1 + alpha_f)**2   # Newmark beta

# --- Traction on the top and bottom edges ---
p_max     = 2.0                       # traction (MPa) ...
t_ramp_up = dt                        # ... ramped up linearly over the first step

# --- Stagger control ---
tol_stag_u, tol_stag_z = 1e-10, 1e-10   # stagger tolerances on |du| and |dz| (L2)
max_stag  = 500                         # at most max_stag - 1 stagger iterations per step

# =============================================================================
# SECTION 8: Linear Solvers
# =============================================================================

# Factorizations, the same for every method (u: the CG preconditioner and the exterior A_EE;
# z: the Newton linear solves).  The z Jacobian is SPD for these constants; should it ever
# not be, z falls back to LU for the rest of the run.  (The ring Schur complement of
# CONDENSE = "schur" always uses MUMPS.)
U_FACTOR = ("cholesky", "cholmod")
Z_FACTOR = ("cholesky", "cholmod")
U_CG_RTOL  = 1e-12  # u: CG tolerance (relative, preconditioned residual)
U_CG_MAXIT = 200    # u: CG iterations before falling back to a fresh factorization
Z_SNES = dict(snes_atol=1e-8, snes_rtol=1e-9, snes_stol=0.0, snes_max_it=10)  # z: Newton tolerances

# --- Implementation (no effect on the results beyond round-off) ---
FAST_ASSEMBLY = True     # per-iteration u/z assembly from a per-cell tabulation (else dolfinx).
                         # False works only with METHOD = "amrc": update_q() has no dolfinx
                         # branch and recurses without end after a patch move inside a step
CONDENSE      = "schur"  # exterior condensation: "schur" (MUMPS ring Schur) | "solve" (column solves)

# =============================================================================
# SECTION 9: I/O & Output Control
# =============================================================================

RESULTS_DIR  = "results"
output_every = 20      # XDMF every n steps (0 = never) ...
output_t_max = 60e-6   # ... up to this time
vel_interval = 40      # crack tip and crack velocity every n steps
LOG_FLUSH    = 50      # flush the per-step logs every n steps
RUN_TAG      = ""      # appended to the output folder name
verbose      = 1       # 0: summary only, 1: one line per step, 2: every stagger iteration
CLOCK        = "wall"  # "wall" | "cpu"

TAG = METHOD + (f"_h{H0_EPS:g}-{HMIN_H0:g}" if (H0_EPS, HMIN_H0) != (1.0, 0.25) else "") + RUN_TAG
output_dir = os.path.join(RESULTS_DIR, TAG)

_QMD = {"quadrature_degree": 2, "quadrature_scheme": "default"}
_USTAT = dict(n_factor=0, n_cg=0)                  # u: factorizations, CG iterations (per run)
VFIELDS = ("u", "u_prev", "v_prev", "a_prev")      # displacement and the Newmark history
WAKE_LAYERS, max_escape_retry, max_rim_moves = 3, 8, 20
ACTIVE_DZ, ACTIVE_STEPS, USE_DELTA, DELTA_PAD_EPS = 0.01, 100, False, 3.0


def clock():
    return time.process_time() if CLOCK == "cpu" else time.perf_counter()

# =============================================================================
# SECTION 10: Constitutive Model
# =============================================================================

def epsilon(v):  return sym(grad(v))
def dgd(z):      return z**2 + eta


def sigma(em):
    """Undegraded plane-stress stress."""
    return 2.0 * mu * em + lmbda * (1 - 2 * nu) / (1 - nu) * tr(em) * Identity(2)


def energy(em):
    """Plane-stress elastic energy density (= sigma:em / 2)."""
    return mu * (inner(em, em) + (nu / (1 - nu))**2 * tr(em)**2) \
        + 0.5 * lmbda * ((1 - 2 * nu) / (1 - nu))**2 * tr(em)**2


def cell_terms(u):
    """The u-dependent parts of the driving force: (psi1, beta2 svm + beta1 I1, I1 < 0).
    For P1 u the strain is constant per cell, so these are exactly DG0."""
    em = epsilon(u)
    psi1, sig = energy(em), sigma(em)
    I1 = tr(sig)
    dev = sig - (1 / 3) * I1 * Identity(2)
    svm = sqrt(0.5 * (inner(dev, dev) + (1 / 9) * I1**2) + 1e-30)
    return psi1, beta2 * svm + beta1 * I1, conditional(lt(I1, 0.0), 1.0, 0.0)


def driving_force(z, p, a, s):
    """Drucker-Prager driving force ce from z and the cell terms."""
    return dgd(z) * a + s * 2.0 * z * p


def z_residual(z, y, p, a, s, dx):
    return (y * 2 * z * p - y * driving_force(z, p, a, s)
            + 3 * delta * Gc / 8 * (-y / eps + 2 * eps * inner(grad(z), grad(y)))) * dx


def indicator(z, p, a, s):
    """1 where the refinement flag fires.  Cells with z < 0.6 are gated out (coarsening)."""
    gate = conditional(lt(z, 0.6), 0.0, 1.0)
    return conditional((gate * (2 * p - driving_force(z, p, a, s) / dgd(z)) - G_eps) / G_eps
                       > pf_flag_threshold, 1, 0)


def load(t):
    """Traction on the top and bottom edges at the HHT time t - alpha_f dt."""
    return p_max * min((t - alpha_f * dt) / t_ramp_up, 1.0)


def newmark_a(u, up, vp, ap):
    """Newmark acceleration of the new u (UFL or numpy)."""
    return (u - up - dt * vp) / (beta_nm * dt**2) - ((1 - 2 * beta_nm) / (2 * beta_nm)) * ap


def newmark_v(a, vp, ap):
    return vp + dt * ((1 - gamma_nm) * ap + gamma_nm * a)


def u_bilinear(V, z, dx):
    """HHT-alpha step matrix (1 - a_f) K(z) + (1 - a_m) M(z) / (beta dt^2)."""
    v, du = TestFunction(V), TrialFunction(V)
    return ((1 - alpha_f) * inner(dgd(z) * sigma(epsilon(du)), epsilon(v))
            + (1 - alpha_m) / (beta_nm * dt**2) * dgd(z) * rho_s * inner(du, v)) * dx


def stiffness_mass(V, z, dx):
    """Degraded stiffness K(z) and mass M(z) forms (the exterior builds its step matrix,
    load, energies and reactions from these two)."""
    v, du = TestFunction(V), TrialFunction(V)
    return (inner(dgd(z) * sigma(epsilon(du)), epsilon(v)) * dx,
            dgd(z) * rho_s * inner(du, v) * dx)


def u_linear(V, z, F, press, dx, ds):
    """HHT-alpha step right-hand side: traction and the Newmark history in F."""
    v, n = TestFunction(V), FacetNormal(V.mesh)
    up, vp, ap = F["u_prev"], F["v_prev"], F["a_prev"]
    hist = (up + dt * vp) / (beta_nm * dt**2) + ((1 - 2 * beta_nm) / (2 * beta_nm)) * ap
    return (press * inner(v, n) * ds(3) + press * inner(v, n) * ds(4)
            - alpha_f * inner(dgd(z) * sigma(epsilon(up)), epsilon(v)) * dx
            + dgd(z) * rho_s * inner((1 - alpha_m) * hist - alpha_m * ap, v) * dx)


def post_forms(F, dx):
    """Elastic, kinetic and surface energy, z-integral (after the Newmark update)."""
    u, v, z = F["u"], F["v_prev"], F["z"]
    return tuple(fem.form(f) for f in (
        0.5 * inner(dgd(z) * sigma(epsilon(u)), epsilon(u)) * dx,
        0.5 * dgd(z) * rho_s * inner(v, v) * dx,
        (3.0 * Gc / 8.0) * ((1 - z) / eps + eps * inner(grad(z), grad(z))) * dx,
        (1 - z) * dx))


def react_form(F, vr, dx):
    """The alpha-averaged dynamic residual (no traction) applied to vr."""
    u, up, vp, ap, z = F["u"], F["u_prev"], F["v_prev"], F["a_prev"], F["z"]
    ua = alpha_f * up + (1 - alpha_f) * u
    aa = alpha_m * ap + (1 - alpha_m) * newmark_a(u, up, vp, ap)
    return fem.form((inner(dgd(z) * sigma(epsilon(ua)), epsilon(vr))
                     + dgd(z) * rho_s * inner(aa, vr)) * dx)

# =============================================================================
# SECTION 11: Mesh Generation & Mesh Helpers
# =============================================================================

def generate_mesh():
    """Coarse cracked plate, element size h0."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    geo = gmsh.model.geo
    # points: the 4 corners, the upper / lower crack mouth, the crack tip
    for i, (x, y) in enumerate([(0, L / 2), (W, L / 2), (W, -L / 2), (0, -L / 2),
                                (0, cw / 2), (0, -cw / 2), (W / 2, 0)], 1):
        geo.addPoint(x, y, 0.0, h0, i)
    # lines: top, right, bottom, left below / above the mouth, upper / lower crack face
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 6), (5, 1), (5, 7), (6, 7)], 1):
        geo.addLine(a, b, i)
    geo.addCurveLoop([1, 2, 3, 4, 7, -6, 5], 10)
    geo.addPlaneSurface([10], 11)
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [11], tag=100)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h0)
    gmsh.model.mesh.generate(2)
    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return msh


def ncells(msh):
    return msh.topology.index_map(2).size_local


def nnodes(msh):
    return msh.topology.index_map(0).size_local


def refine_near(base, pts):
    """Bisect edges of *base* within amr_refine_radius of *pts* down to h_min.
    Returns (mesh, parent) with parent[c] = cell of *base* containing cell c."""
    m, parent = base, np.arange(ncells(base), dtype=np.int32)
    if pts is None or len(pts) == 0:
        return m, parent
    tree = cKDTree(pts[:, :2])
    for lev in range(max(1, int(np.ceil(np.log2(h0 / h_min))))):
        m.topology.create_entities(1)
        m.topology.create_connectivity(1, 2)
        e = np.arange(m.topology.index_map(1).size_local, dtype=np.int32)
        e = e[m.h(1, e) > 1.05 * h0 / 2**(lev + 1)]
        if e.size:
            d, _ = tree.query(compute_midpoints(m, 1, e)[:, :2],
                              distance_upper_bound=amr_refine_radius)
            e = e[np.isfinite(d)]
        if e.size == 0:
            break
        m, pc, _ = refine(m, e, option=RefinementOption.parent_cell)
        # dolfinx 0.9: parent_cell is in the refined mesh's pre-reordering cell order
        parent = parent[pc[np.asarray(m.topology.original_cell_index)]]
    return m, parent


def midpoints(msh, cells):
    return compute_midpoints(msh, 2, cells.astype(np.int32)) if cells.size \
        else np.empty((0, 3))


def cg_ksp(A, kind):
    """CG on A, preconditioned by a Cholesky factor that is kept between solves."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=U_CG_RTOL, atol=0.0, max_it=U_CG_MAXIT)
    ksp.setInitialGuessNonzero(True)
    ksp.getPC().setType(kind[0])
    ksp.getPC().setFactorSolverType(kind[1])
    return ksp


def factor_ksp(A, kind):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType(kind[0])
    ksp.getPC().setFactorSolverType(kind[1])
    return ksp


def dense_block(A, rows, cols):
    """A[rows, cols] as a numpy array.  The submatrix must stay referenced until the
    copy is taken: getDenseArray() is a view into memory the Mat owns."""
    sub = A.createSubMatrix(rows, cols)
    out = sub.convert("dense").getDenseArray().copy()
    sub.destroy()
    return out


def boundary_tags(msh):
    """Facets and tags 1 right, 2 left, 3 top, 4 bottom (outer edges only)."""
    marks = [(1, lambda x: np.isclose(x[0], W, atol=1e-8)),
             (2, lambda x: np.isclose(x[0], 0.0, atol=1e-8)),
             (3, lambda x: np.isclose(x[1], L / 2, atol=1e-8)),
             (4, lambda x: np.isclose(x[1], -L / 2, atol=1e-8))]
    msh.topology.create_connectivity(1, 2)
    f = [locate_entities_boundary(msh, 1, m) for _, m in marks]
    v = np.hstack([np.full(a.size, k, np.int32) for (k, _), a in zip(marks, f)])
    f = np.hstack(f).astype(np.int32)
    o = np.argsort(f)
    return f[o], v[o]


def edge_dofs(V, bf, bv):
    """u_y dofs on the top and bottom edges (the reaction-force dofs)."""
    return tuple(fem.locate_dofs_topological(V.sub(1), 1, bf[bv == k]) for k in (3, 4))

# =============================================================================
# SECTION 12: Solvers on the Solve Mesh (the patch, or the whole mesh for AMRC)
# =============================================================================

class USolver:
    """(A(z) + S) u = b(z, history) + g, A the HHT-alpha step matrix.  S (the condensed
    exterior) is written once into a template matrix; each solve copies the template and
    adds the patch matrix, so the dense block is never re-inserted."""

    def __init__(self, P, S=None, pdG=None):
        self.P, self.u, self.q, self.q_expr = P, P["u"], P["q"], P["q_expr"]
        self.a, self.L, self.pdG, self.g = P["a_u"], P["L_u"], pdG, None
        self.tab = P.get("tab")
        sp = fem.create_sparsity_pattern(self.a)
        if pdG is not None:
            nodes = np.unique(pdG // 2).astype(np.int32)
            sp.insert(nodes, nodes)
        sp.finalize()
        self.A = _cpp.la.petsc.create_matrix(comm, sp)
        # touch every pattern slot before the first final assembly, so none is dropped
        fem.petsc.assemble_matrix(self.A, self.a)
        if pdG is not None:
            self.A.setValues(pdG, pdG, np.zeros((pdG.size, pdG.size)),
                             addv=PETSc.InsertMode.ADD_VALUES)
        self.A.assemble()
        self.A0 = None
        if self.tab is not None:
            # the matrix values live in a numpy array written from the tabulation
            ptr, ind, _ = self.A.getValuesCSR()
            self.A.destroy()
            self.A, ptr, ind, self.vals = aij_with_arrays(ptr, ind)
            self.tab.u_setup(ptr, ind)
            self.Sfull = None                      # S scattered into the value array
            if pdG is not None:
                self.Sfull = np.zeros(ind.size)
                self.Sfull[csr_positions(ptr, ind, np.repeat(pdG, pdG.size), np.tile(pdG, pdG.size))] = \
                    np.ascontiguousarray(S).ravel()
        elif pdG is not None:
            self.A0 = self.A.duplicate(copy=False)
            self.A0.setValues(pdG, pdG, S, addv=PETSc.InsertMode.INSERT_VALUES)
            self.A0.assemble()
        self.b = self.u.x.petsc_vec.duplicate()
        self.ksp = cg_ksp(self.A, U_FACTOR)
        self.refactor, self.t_fresh = True, np.inf  # the first solve factorizes

    def solve(self):
        if self.tab is None:
            if self.A0 is None:
                self.A.zeroEntries()
            else:
                self.A0.copy(self.A, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            fem.petsc.assemble_matrix(self.A, self.a)
            self.A.assemble()
            with self.b.localForm() as bl:
                bl.set(0.0)
            fem.petsc.assemble_vector(self.b, self.L)
        else:
            vK, vM = self.tab.u_values(self.P["z"].x.array)
            np.multiply(vK, 1 - alpha_f, out=self.vals)
            self.vals += ((1 - alpha_m) / (beta_nm * dt**2)) * vM
            rhs = self.tab.u_rhs(self.P)
            if not self.tab.checked["u"]:
                self._check(rhs)
            if self.Sfull is not None:
                self.vals += self.Sfull
            self.A.assemble()                      # new values: bump the state for the PC
            self.b.getArray()[:] = rhs
        if self.g is not None:
            with self.b.localForm() as bl:
                bl.array[self.pdG] += self.g
        self._ksp_solve()
        if self.tab is None:
            self.q.interpolate(self.q_expr)
        else:
            self.tab.set_q(self.P)
            if not self.tab.checked["u"]:
                qf = self.tab.get_q(self.P).copy()
                self.q.interpolate(self.q_expr)
                e = _rel(qf, self.tab.read_q(self.P))
                self.tab.set_q(self.P)
                if e > 1e-10:
                    raise RuntimeError(f"fast cell terms differ from UFL ({e:.1e})")
                self.tab.checked["u"] = True

    def _check(self, rhs):
        """Tabulated matrix and right-hand side against dolfinx (once per solve mesh)."""
        Ad = fem.petsc.assemble_matrix(self.a)
        Ad.assemble()
        ptr, ind, v = Ad.getValuesCSR()
        Ad.destroy()
        n = ptr.size - 1
        ref = sps.csr_matrix((v, ind, ptr), shape=(n, n))
        mine = sps.csr_matrix((self.vals.copy(), self.A.getAttr("__csr__")[1],
                               self.A.getAttr("__csr__")[0]), shape=(n, n))
        e1 = abs(ref - mine).max() / abs(ref).max()
        bd = fem.assemble_vector(self.L).array
        e2 = _rel(rhs, bd)
        if max(e1, e2) > 1e-10:
            raise RuntimeError(f"fast u assembly differs from dolfinx (A {e1:.1e}, b {e2:.1e})")

    def _ksp_solve(self):
        """CG on the current matrix, preconditioned by the factor of an earlier one.
        Refactor when the last solve with the old factor took longer than the last
        factorization, or when CG does not converge (then redo from the old u)."""
        pc, x = self.ksp.getPC(), self.u.x.petsc_vec
        if self.b.norm() == 0.0:
            x.set(0.0)
            return
        x0 = x.copy()
        for fresh in ((True,) if self.refactor else (False, True)):
            pc.setReusePreconditioner(not fresh)
            if fresh:
                x0.copy(x)
            t0 = time.perf_counter()
            self.ksp.solve(self.b, x)
            dt_ = time.perf_counter() - t0
            _USTAT["n_cg"] += self.ksp.getIterationNumber()
            if fresh:
                _USTAT["n_factor"] += 1
                self.t_fresh = dt_
            if self.ksp.getConvergedReason() > 0:
                self.refactor = (not fresh) and dt_ > self.t_fresh
                x0.destroy()
                return
        x0.destroy()
        raise RuntimeError(f"u solve failed ({self.ksp.getConvergedReason()})")

    def destroy(self):
        for o in (self.ksp, self.A, self.A0, self.b):
            if o is not None:
                o.destroy()
        self.P = None


class ZSolver:
    """The phase-field VI posed on the free dofs alone.  Frozen dofs (patch boundary) are
    not constrained inside the system but deleted from it; their values enter through z
    as a coefficient of the residual."""

    def __init__(self, z, zub, free, F, J, P=None):
        self.z, self.ub, self.free, self.F, self.J = z, zub, free, F, J
        self.P = P
        self.tab = P.get("tab") if P is not None else None
        n = int(free.size)
        self.bF = fem.petsc.create_vector(F)
        self.AF = fem.petsc.create_matrix(J)
        if self.tab is not None:
            fem.petsc.assemble_matrix(self.AF, J)     # touch the pattern, then read it
            self.AF.assemble()
            ptr, ind, _ = self.AF.getValuesCSR()
            self.AF.destroy()
            self.AF, ptr, ind, self.jvals = aij_with_arrays(ptr, ind)
            self.tab.z_setup(ptr, ind)
        self.isf = PETSc.IS().createGeneral(free, comm=comm)
        self.vec = [PETSc.Vec().createSeq(n, comm=comm) for _ in range(4)]  # b, x, lb, ub
        self._assemble_J()
        self.Ared = self.AF.createSubMatrix(self.isf, self.isf)
        opts = PETSc.Options()
        for k, v in dict(snes_type="vinewtonrsls", snes_linesearch_type="basic", **Z_SNES,
                         ksp_type="preonly", pc_type=Z_FACTOR[0],
                         pc_factor_mat_solver_type=Z_FACTOR[1]).items():
            opts[f"z_{k}"] = v
        s = self.snes = PETSc.SNES().create(comm)
        s.setOptionsPrefix("z_")
        s.setFromOptions()
        s.setFunction(self._F, self.vec[0])
        s.setJacobian(self._J, self.Ared)

    def _assemble_J(self):
        if self.tab is not None:
            self.jvals[:] = self.tab.z_jac(self.z.x.array, self.tab.get_q(self.P))
            self.AF.assemble()
            return
        self.AF.zeroEntries()
        fem.petsc.assemble_matrix(self.AF, self.J)
        self.AF.assemble()

    def _F(self, snes, x, b):
        self.z.x.array[self.free] = x.getArray(readonly=True)
        if self.tab is not None:
            r = self.tab.z_res(self.z.x.array, self.tab.get_q(self.P))
            if not self.tab.checked["z"]:
                self._check(r)
            b.getArray()[:] = r[self.free]
            return
        with self.bF.localForm() as bl:
            bl.set(0.0)
        fem.petsc.assemble_vector(self.bF, self.F)
        b.getArray()[:] = self.bF.getArray(readonly=True)[self.free]

    def _check(self, r):
        """Tabulated residual and Jacobian against dolfinx (once per solve mesh)."""
        e1 = _rel(r, fem.assemble_vector(self.F).array)
        Jd = fem.petsc.assemble_matrix(self.J)
        Jd.assemble()
        ptr, ind, v = Jd.getValuesCSR()
        Jd.destroy()
        n = ptr.size - 1
        mine = sps.csr_matrix((self.tab.z_jac(self.z.x.array, self.tab.get_q(self.P)),
                               self.AF.getAttr("__csr__")[1], self.AF.getAttr("__csr__")[0]), shape=(n, n))
        ref = sps.csr_matrix((v, ind, ptr), shape=(n, n))
        e2 = abs(ref - mine).max() / abs(ref).max()
        if max(e1, e2) > 1e-10:
            raise RuntimeError(f"fast z assembly differs from dolfinx (R {e1:.1e}, J {e2:.1e})")
        self.tab.checked["z"] = True

    def _J(self, snes, x, A, Pm):
        self._assemble_J()
        self.AF.createSubMatrix(self.isf, self.isf, A)

    def solve(self):
        global Z_FACTOR
        _, x, lb, ub = self.vec
        z0 = self.z.x.array[self.free]
        lb.set(0.0)
        ub.getArray()[:] = self.ub.x.array[self.free]
        self.snes.setVariableBounds(lb, ub)
        for attempt in (0, 1):
            x.getArray()[:] = z0
            try:
                self.snes.solve(None, x)
                failed = self.snes.getConvergedReason() == \
                    PETSc.SNES.ConvergedReason.DIVERGED_LINEAR_SOLVE
            except PETSc.Error as e:
                if e.ierr != 81:                      # 81: CHOLMOD found it indefinite
                    raise
                failed = True
            if not failed:
                break
            if attempt or Z_FACTOR[0] != "cholesky":
                raise RuntimeError("z solve: linear solver failed")
            # vinewtonrsls rebuilds its PC from the options database, so switch there
            Z_FACTOR = ("lu", "mumps")
            PETSc.Options()["z_pc_type"], PETSc.Options()["z_pc_factor_mat_solver_type"] = Z_FACTOR
            self.snes.setFromOptions()
            print("    [z] Jacobian not SPD: LU from now on", flush=True)
        self.z.x.array[self.free] = x.getArray(readonly=True)

    def destroy(self):
        for o in (self.snes, self.bF, self.AF, self.Ared, self.isf, *self.vec):
            o.destroy()
        self.P = None

# =============================================================================
# SECTION 13: Fast Assembly & Ring-Schur Condensation
# =============================================================================
# FAST_ASSEMBLY: the per-iteration u and z systems from a per-cell tabulation (Tab).
# CONDENSE = "schur": the exterior Schur complement through the first exterior cell ring
# (ring_schur, used by build_exterior).

def csr_positions(indptr, indices, rows, cols):
    """Positions of the entries (rows, cols) in a CSR structure with sorted columns."""
    n = np.int64(indptr.size - 1)
    keys = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr)) * n + indices
    k = rows.astype(np.int64) * n + cols
    pos = np.searchsorted(keys, k)
    if not (keys[np.minimum(pos, keys.size - 1)] == k).all():
        raise RuntimeError("entry missing from the sparsity pattern")
    return pos


def aij_with_arrays(indptr, indices):
    """A SeqAIJ matrix on (indptr, indices) whose value array is a numpy array we write."""
    n = indptr.size - 1
    ptr = np.ascontiguousarray(indptr, dtype=PETSc.IntType)
    ind = np.ascontiguousarray(indices, dtype=PETSc.IntType)
    A = PETSc.Mat().createAIJWithArrays((n, n), (ptr, ind, np.zeros(ind.size, dtype=PETSc.ScalarType)),
                                        comm=comm)
    vals = A.getAttr("__csr__")[2]               # the buffer PETSc uses
    vals[:] = 1.0
    A.assemble()
    assert A.getValuesCSR()[2].sum() == ind.size, "AIJ value buffer is not shared"
    vals[:] = 0.0
    return A, ptr, ind, vals


class Tab:
    """Per-cell tabulation of a P1 solve mesh.  The u matrix (1 - a_f) K(z) + (1 - a_m)
    M(z) / (beta dt^2), the u right-hand side, the z residual and Jacobian, the cell terms
    q and the L2 norms of the stagger check are sparse products of fixed per-cell data with
    z at the quadrature points.  The quadrature is the degree-2 rule of the UFL forms, so
    the values are those of dolfinx assembly up to round-off (checked on the first solve of
    every solve mesh)."""

    def __init__(self, P):
        msh, V, Y = P["msh"], P["V"], P["Y"]
        x = msh.geometry.x[:, :2]
        gd = msh.geometry.dofmap
        nc = self.nc = gd.shape[0]
        e1, e2 = x[gd[:, 1]] - x[gd[:, 0]], x[gd[:, 2]] - x[gd[:, 0]]
        det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        inv = np.empty((nc, 2, 2))                 # J^-1, J = [e1 e2]
        inv[:, 0, 0], inv[:, 0, 1] = e2[:, 1] / det, -e2[:, 0] / det
        inv[:, 1, 0], inv[:, 1, 1] = -e1[:, 1] / det, e1[:, 0] / det
        G = np.einsum("kj,cji->cki", np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), inv)
        qp, qw = basix.make_quadrature(basix.CellType.triangle, 2)
        Phi = self.Phi = np.c_[1 - qp[:, 0] - qp[:, 1], qp[:, 0], qp[:, 1]]    # (nq, 3)
        nq = self.nq = Phi.shape[0]
        W = self.W = np.abs(det)[:, None] * qw[None, :]                       # (nc, nq)
        self.Yd, self.Vd = Y.dofmap.list, V.dofmap.list
        self.gl = (2 * self.Vd[:, :, None] + np.arange(2)).reshape(nc, 6)
        B = self.B = np.zeros((nc, 3, 6))          # Voigt strain (xx, yy, 2xy) of the 6 dofs
        B[:, 0, 0::2], B[:, 1, 1::2] = G[:, :, 0], G[:, :, 1]
        B[:, 2, 0::2], B[:, 2, 1::2] = G[:, :, 1], G[:, :, 0]
        self.lp = lmbda * (1 - 2 * nu) / (1 - nu)
        D = np.array([[2 * mu + self.lp, self.lp, 0.0], [self.lp, 2 * mu + self.lp, 0.0], [0.0, 0.0, mu]])
        self.K0 = np.einsum("cai,ab,cbj->cij", B, D, B)                        # (nc, 6, 6)
        self.G, self.area = G, np.abs(det) / 2
        cq = (np.arange(nc)[:, None] * nq + np.arange(nq)[None, :])            # (nc, nq)
        self.cq = cq
        # z: vector part of the residual, P1 mass (for the norm) in the V block numbering
        PP = Phi[:, :, None] * Phi[:, None, :]                                 # (nq, 3, 3)
        self.PP = PP
        nY = Y.dofmap.index_map.size_local
        self.BZV = sps.csr_matrix((np.broadcast_to(Phi[None], (nc, nq, 3)).ravel(),
                                   (np.broadcast_to(self.Yd[:, None, :], (nc, nq, 3)).ravel(),
                                    np.broadcast_to(cq[:, :, None], (nc, nq, 3)).ravel())),
                                  shape=(nY, nc * nq))
        self.cvec = np.bincount(self.Yd.ravel(), np.repeat(self.area / 3, 3), minlength=nY)
        m0 = np.einsum("cq,qkl->ckl", W, PP)
        nV = V.dofmap.index_map.size_local
        self.M0V = sps.csr_matrix((m0.ravel(), (np.broadcast_to(self.Vd[:, :, None], (nc, 3, 3)).ravel(),
                                                np.broadcast_to(self.Vd[:, None, :], (nc, 3, 3)).ravel())),
                                  shape=(nV, nV))
        self.M0Y = sps.csr_matrix((m0.ravel(), (np.broadcast_to(self.Yd[:, :, None], (nc, 3, 3)).ravel(),
                                                np.broadcast_to(self.Yd[:, None, :], (nc, 3, 3)).ravel())),
                                  shape=(nY, nY))
        self.qd = P["q"].function_space.dofmap.list[:, 0]
        self.checked = dict(u=False, z=False)

    # --- u
    def u_setup(self, ptr, ind):
        """Sparse maps from the per-cell weights to the values of the u CSR structure."""
        nc, nq = self.nc, self.nq
        pos = csr_positions(ptr, ind, np.broadcast_to(self.gl[:, :, None], (nc, 6, 6)),
                            np.broadcast_to(self.gl[:, None, :], (nc, 6, 6)))
        nnz, n = ind.size, ptr.size - 1
        self.BK = sps.csr_matrix((self.K0.ravel(), (pos.ravel(), np.repeat(np.arange(nc), 36))),
                                 shape=(nnz, nc))
        comp, kk = np.arange(6) % 2, np.arange(6) // 2
        l1, l2 = np.nonzero(comp[:, None] == comp[None, :])                    # 18 pairs
        Mq = rho_s * self.Phi[:, kk[l1]] * self.Phi[:, kk[l2]]                 # (nq, 18)
        self.BM = sps.csr_matrix((np.broadcast_to(Mq[None], (nc, nq, 18)).ravel(),
                                  (np.broadcast_to(pos[:, None, l1, l2], (nc, nq, 18)).ravel(),
                                   np.broadcast_to(self.cq[:, :, None], (nc, nq, 18)).ravel())),
                                 shape=(nnz, nc * nq))
        self.Ks = sps.csr_matrix((np.zeros(nnz), ind, ptr), shape=(n, n))
        self.Ms = sps.csr_matrix((np.zeros(nnz), ind, ptr), shape=(n, n))

    def zq(self, z):
        return z[self.Yd] @ self.Phi.T                                         # (nc, nq)

    def u_values(self, z):
        """Values of K(z) and M(z) in the u CSR structure."""
        d = self.W * (self.zq(z) ** 2 + eta)
        self.Ks.data = self.BK @ d.sum(axis=1)
        self.Ms.data = self.BM @ d.ravel()
        return self.Ks.data, self.Ms.data

    def u_rhs(self, P):
        """Traction and Newmark history (uses the K, M of the last u_values call)."""
        up, vp, ap = (P[f].x.array for f in ("u_prev", "v_prev", "a_prev"))
        hw = (1 - alpha_m) * ((up + dt * vp) / (beta_nm * dt**2)
                              + ((1 - 2 * beta_nm) / (2 * beta_nm)) * ap) - alpha_m * ap
        return float(P["press"].value) * P["ftr"] - alpha_f * (self.Ks @ up) + self.Ms @ hw

    def cell_terms(self, u):
        """(psi1, beta2 svm + beta1 I1, I1 < 0) per cell, as cell_terms()."""
        exx, eyy, gam = np.einsum("cij,cj->ic", self.B, u[self.gl])
        tr = exx + eyy
        psi = mu * (exx**2 + eyy**2 + 0.5 * gam**2 + (nu / (1 - nu))**2 * tr**2) \
            + 0.5 * lmbda * ((1 - 2 * nu) / (1 - nu))**2 * tr**2
        sxx, syy, sxy = 2 * mu * exx + self.lp * tr, 2 * mu * eyy + self.lp * tr, mu * gam
        I1 = sxx + syy
        svm = np.sqrt(0.5 * ((sxx - I1 / 3)**2 + (syy - I1 / 3)**2 + 2 * sxy**2 + I1**2 / 9) + 1e-30)
        return np.c_[psi, beta2 * svm + beta1 * I1, (I1 < 0.0).astype(float)]

    def set_q(self, P):
        self.qc = self.cell_terms(P["u"].x.array)
        P["q"].x.array.reshape(-1, 3)[self.qd] = self.qc

    def get_q(self, P):
        """The cell terms of the last set_q (all q updates go through set_q)."""
        if not hasattr(self, "qc"):
            self.qc = P["q"].x.array.reshape(-1, 3)[self.qd].copy()
        return self.qc

    def read_q(self, P):
        return P["q"].x.array.reshape(-1, 3)[self.qd]

    def unorm(self, du):
        d = du.reshape(-1, 2)
        return np.sqrt(sum(d[:, k] @ (self.M0V @ d[:, k]) for k in range(2)))

    def znorm(self, dz):
        return np.sqrt(dz @ (self.M0Y @ dz))

    # --- z
    def z_setup(self, ptr, ind):
        nc, nq = self.nc, self.nq
        pos = csr_positions(ptr, ind, np.broadcast_to(self.Yd[:, :, None], (nc, 3, 3)),
                            np.broadcast_to(self.Yd[:, None, :], (nc, 3, 3)))
        nnz, n = ind.size, ptr.size - 1
        Lc = self.area[:, None, None] * np.einsum("cki,cli->ckl", self.G, self.G)
        self.lap = np.bincount(pos.ravel(), Lc.ravel(), minlength=nnz)
        self.LapS = sps.csr_matrix((self.lap, ind, ptr), shape=(n, n))
        self.BZM = sps.csr_matrix((np.broadcast_to(self.PP[None], (nc, nq, 3, 3)).ravel(),
                                   (np.broadcast_to(pos[:, None], (nc, nq, 3, 3)).ravel(),
                                    np.broadcast_to(self.cq[:, :, None, None], (nc, nq, 3, 3)).ravel())),
                                  shape=(nnz, nc * nq))

    def z_res(self, z, q):
        p, a, s = q[:, 0], q[:, 1], q[:, 2]
        zq = self.zq(z)
        g = self.W * (2 * zq * (p * (1 - s))[:, None] - (zq**2 + eta) * a[:, None])
        return self.BZV @ g.ravel() + (3 * delta * Gc / 8) * (-self.cvec / eps + 2 * eps * (self.LapS @ z))

    def z_jac(self, z, q):
        p, a, s = q[:, 0], q[:, 1], q[:, 2]
        h = self.W * (2 * (p * (1 - s))[:, None] - 2 * self.zq(z) * a[:, None])
        return self.BZM @ h.ravel() + (3 * delta * Gc / 8) * (2 * eps) * self.lap


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


_LIBPETSC = None


def ring_schur(AEE, ridx):
    """Dense Schur complement of AEE onto its rows/columns ridx, from one MUMPS Cholesky
    factorization (MatFactorSetSchurIS through ctypes: petsc4py does not wrap it)."""
    global _LIBPETSC
    if _LIBPETSC is None:
        _LIBPETSC = ctypes.CDLL(next(l.split()[-1] for l in open("/proc/self/maps") if "libpetsc.so" in l))
    lib, vp = _LIBPETSC, (lambda o: ctypes.c_void_p(o.handle))
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(AEE)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    isR = PETSc.IS().createGeneral(ridx.astype(PETSc.IntType), comm=comm)
    if lib.MatFactorSetSchurIS(vp(pc.getFactorMatrix()), vp(isR)):
        raise RuntimeError("MatFactorSetSchurIS failed")
    ksp.setUp()
    Sp, st = ctypes.c_void_p(), ctypes.c_int()
    if lib.MatFactorCreateSchurComplement(vp(pc.getFactorMatrix()), ctypes.byref(Sp), ctypes.byref(st)):
        raise RuntimeError("MatFactorCreateSchurComplement failed")
    arr = ctypes.POINTER(ctypes.c_double)()
    lib.MatDenseGetArrayRead(Sp, ctypes.byref(arr))
    n = ridx.size
    S = np.ctypeslib.as_array(arr, shape=(n, n)).T.copy()                     # column-major
    lib.MatDenseRestoreArrayRead(Sp, ctypes.byref(arr))
    lib.MatDestroy(ctypes.byref(Sp))
    ksp.destroy()
    isR.destroy()
    up, lo = np.abs(np.triu(S, 1)).max(initial=0.0), np.abs(np.tril(S, -1)).max(initial=0.0)
    if (up == 0.0) != (lo == 0.0):                 # one triangle only: mirror it
        T = np.tril(S) if up == 0.0 else np.triu(S)
        S = T + T.T - np.diag(np.diag(T))
    return 0.5 * (S + S.T)

# =============================================================================
# SECTION 14: Coarse (Base) Mesh Data — global and fixed
# =============================================================================

def coarse_setup(msh):
    C = dict(msh=msh)
    for a, b in ((1, 2), (2, 1), (0, 2), (2, 0)):
        msh.topology.create_connectivity(a, b)
    V = C["V"] = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    Y = C["Y"] = fem.functionspace(msh, ("Lagrange", 1))
    C["Yv"] = fem.functionspace(msh, ("DG", 0))
    nc = C["nc"] = ncells(msh)
    C["c2n"], C["xn"] = Y.dofmap.list, Y.tabulate_dof_coordinates()
    C["mid"] = compute_midpoints(msh, 2, np.arange(nc, dtype=np.int32))[:, :2]
    C["tree"], C["ntree"] = cKDTree(C["mid"]), cKDTree(C["xn"][:, :2])
    for f in VFIELDS:
        C[f] = fem.Function(V, name="displacement" if f == "u" else f)
    C["z"], C["zub"] = fem.Function(Y, name="phasefield"), fem.Function(Y)
    C["z"].x.array[:] = C["zub"].x.array[:] = 1.0
    if not LOCAL:
        return C
    f2c = msh.topology.connectivity(1, 2)          # interior facets and their two cells
    cnt = np.diff(f2c.offsets)[:msh.topology.index_map(1).size_local]
    fi = np.where(cnt == 2)[0].astype(np.int32)
    C["f2c"] = f2c.array[f2c.offsets[fi][:, None] + np.arange(2)]
    C["fmid"] = compute_midpoints(msh, 1, fi)[:, :2]
    C["bf"], C["bv"] = boundary_tags(msh)
    C["bcell"] = f2c.array[f2c.offsets[C["bf"]]]
    C["dtop"], C["dbot"] = edge_dofs(V, C["bf"], C["bv"])
    C["ind_expr"] = fem.Expression(indicator(C["z"], *cell_terms(C["u"])),
                                   C["Yv"].element.interpolation_points())
    C["ind"] = fem.Function(C["Yv"], name="pf_flag")
    return C


def mask(C, cells):
    m = np.zeros(C["nc"], bool)
    m[cells] = True
    return m


def grow(C, cells, layers=1):
    """*cells* plus *layers* vertex-connected layers of coarse cells (bool mask in/out)."""
    for _ in range(layers):
        mark = np.zeros(C["xn"].shape[0], bool)
        mark[C["c2n"][cells]] = True
        cells = mark[C["c2n"]].any(axis=1)
    return cells


def damaged_cells(C):
    """D: coarse cells with a node z < DAMAGE_Z."""
    return (C["z"].x.array < DAMAGE_Z)[C["c2n"]].any(axis=1)


def delta_cells(C):
    """Delta: coarse cells with a node whose z dropped by more than ACTIVE_DZ since zref."""
    if "zref" not in C:
        return np.zeros(C["nc"], bool)
    return (C["z"].x.array < C["zref"] - ACTIVE_DZ)[C["c2n"]].any(axis=1)

# =============================================================================
# SECTION 15: Patch Selection & Exterior Condensation (AMRC-CLIP, AMRC-CLIP-TW)
# =============================================================================

def near(C, pts, r):
    """Coarse cells whose midpoint lies within r of a point of pts (bool mask)."""
    if len(pts) == 0:
        return np.zeros(C["nc"], bool)
    d, _ = cKDTree(pts[:, :2]).query(C["mid"], distance_upper_bound=r * (1 + 1e-12))
    return np.isfinite(d)


def tip_patch(C, pts):
    """The AMRC-CLIP patch around the active points *pts* (bool mask): B(A, Rp) [u the padded
    Delta cells, USE_DELTA] u N(D n Wp).  AMRC-CLIP and AMRC-CLIP-TW both use it."""
    cells = near(C, pts, PATCH_R_EPS * eps)
    if USE_DELTA:
        de = delta_cells(C)
        if de.any():
            # Delta padded by DELTA_PAD_EPS eps, but only through damaged cells (along the
            # crack, not into intact material), plus one layer so every node is solved
            pad = near(C, C["mid"][de], DELTA_PAD_EPS * eps)
            cells |= grow(C, pad & damaged_cells(C))
    if RETIRE_STEPS and "refage" in C:
        # damaged cells coarsened within the last RETIRE_STEPS refinement events (plus one layer): the
        # coarse copy of a crack band is partly healed and must re-damage before it freezes
        cells |= grow(C, damaged_cells(C) & (C["refage"] < RETIRE_STEPS))
    return cells


def select_patch(C, pts):
    """Coarse cells of the patch around the active points *pts*: AMRC-CLIP -- the tip_patch()
    cells; AMRC-CLIP-TW -- those u (D + WAKE_LAYERS cell layers)."""
    cells = tip_patch(C, pts)
    if METHOD == "amrc_clip_tw":
        cells |= grow(C, damaged_cells(C), WAKE_LAYERS)   # every node of D solved
    return np.where(cells)[0].astype(np.int32)


def split_facets(msh, gmid):
    """For each interface-facet midpoint *gmid*: is that facet split in the refined *msh*?"""
    msh.topology.create_connectivity(1, 2)
    fm = compute_midpoints(msh, 1, exterior_facet_indices(msh.topology))[:, :2]
    return ~np.isfinite(cKDTree(fm).query(gmid, distance_upper_bound=1e-8 * W)[0])


def place_patch(C, pcells, refine_pts):
    """Submesh of *pcells* refined near *refine_pts*.  Where the refinement splits a
    patch-boundary facet, the cells around that facet join the patch, until the boundary
    is made of coarse facets only."""
    for _ in range(10):
        inP = mask(C, pcells)
        gf = inP[C["f2c"][:, 0]] != inP[C["f2c"][:, 1]]
        base, bmap, _, _ = create_submesh(C["msh"], 2, pcells)
        msh, parent = refine_near(base, refine_pts)
        split = split_facets(msh, C["fmid"][gf])
        if not split.any():
            return pcells, base, bmap, msh, parent
        pcells = np.where(inP | grow(C, mask(C, C["f2c"][gf][split].ravel())))[0].astype(np.int32)
    raise RuntimeError("the patch could not be made to contain the refinement")


def build_exterior(C, pcells):
    """Condense the exterior onto the patch boundary: S, the factor of A_EE, the rows of
    A_EE^-1 A_EG on the first exterior cell layer, and the exterior forms (load, energies,
    reactions).  None if nothing is left outside the patch."""
    t0 = clock()
    msh, nc, c2n = C["msh"], C["nc"], C["c2n"]
    inP = mask(C, pcells)
    nn = C["xn"].shape[0]
    tp, te = np.zeros(nn, bool), np.zeros(nn, bool)
    tp[c2n[inP].ravel()] = True
    te[c2n[~inP].ravel()] = True
    if not (te & ~tp).any():
        return None
    tags = meshtags(msh, 2, np.arange(nc, dtype=np.int32), np.where(inP, 1, 2).astype(np.int32))
    dxe = Measure("dx", domain=msh, subdomain_data=tags, metadata=_QMD)(2)
    # z is frozen outside the patch, so the exterior stiffness and mass are fixed until the
    # next move: the step matrix, the exterior load, energies and reactions come from them
    Km, Mm = (fem.petsc.assemble_matrix(fem.form(f)) for f in stiffness_mass(C["V"], C["z"], dxe))
    Km.assemble()
    Mm.assemble()
    A = Km.copy()
    A.scale(1 - alpha_f)
    A.axpy((1 - alpha_m) / (beta_nm * dt**2), Mm, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

    blk = lambda n: (2 * n[:, None] + np.arange(2)).ravel().astype(np.int32)
    dG, dE = blk(np.where(tp & te)[0]), blk(np.where(te & ~tp)[0])
    iE, iG = (PETSc.IS().createGeneral(d, comm=comm) for d in (dE, dG))
    AEE, AEG, AGE = A.createSubMatrix(iE, iE), A.createSubMatrix(iE, iG), A.createSubMatrix(iG, iE)
    S = dense_block(A, iG, iG)
    kspE = factor_ksp(AEE, U_FACTOR)
    kspE.setUp()
    F = kspE.getPC().getFactorMatrix()
    nE, nG = dE.size, dG.size
    ring = grow(C, inP) & ~inP                     # first exterior cell layer
    rn = np.unique(c2n[ring].ravel())
    rdofs = blk(rn[~tp[rn]])
    ridx = np.searchsorted(dE, rdofs)
    isR = PETSc.IS().createStride(nE, first=0, step=1, comm=comm)
    if CONDENSE == "schur":
        # A_EG vanishes off the ring r (every exterior node coupled to G is a ring node), so
        # S = A_GG - A_Gr S_rr^-1 A_rG and the ring rows Yr = S_rr^-1 A_rG, with S_rr the
        # Schur complement of A_EE onto r from one MUMPS factorization
        iRd = PETSc.IS().createGeneral(rdofs, comm=comm)
        ARG = dense_block(A, iRd, iG)
        iRd.destroy()
        cf = sla.cho_factor(ring_schur(AEE, ridx))
        Yr = sla.cho_solve(cf, ARG)
        S -= ARG.T @ Yr
    Yr = np.zeros((rdofs.size, nG)) if CONDENSE != "schur" else Yr
    for j0 in (range(0, nG, 128) if CONDENSE != "schur" else ()):   # column solves
        w = min(128, nG - j0)
        isC = PETSc.IS().createStride(w, first=j0, step=1, comm=comm)
        B = AEG.createSubMatrix(isR, isC).convert("dense")
        Xc = PETSc.Mat().createDense([nE, w], comm=comm)
        Xc.setUp()
        F.matSolve(B, Xc)
        Yr[:, j0:j0 + w] = Xc.getDenseArray()[ridx, :]
        GX = AGE.matMult(Xc)
        S[:, j0:j0 + w] -= GX.getDenseArray()
        for o in (isC, B, Xc, GX):
            o.destroy()
    S = 0.5 * (S + S.T)

    # Self-check of S on a random vector (one extra solve): condensation errors would
    # otherwise show up only as a wrong response.
    AGG = dense_block(A, iG, iG)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(nG)
    ve, vv, tg = AEE.createVecLeft(), AEG.createVecLeft(), AGE.createVecLeft()
    AEG.mult(PETSc.Vec().createWithArray(v, comm=comm), vv)
    kspE.solve(vv, ve)
    AGE.mult(ve, tg)
    ref = AGG @ v - tg.getArray()
    err = np.linalg.norm(S @ v - ref) / max(np.linalg.norm(ref), 1e-300)

    # exterior traction vector (unit load) and the frozen surface energy and z-integral
    ft = meshtags(msh, 1, C["bf"][~inP[C["bcell"]]], C["bv"][~inP[C["bcell"]]])
    dse = Measure("ds", domain=msh, subdomain_data=ft)
    v, n, z = TestFunction(C["V"]), FacetNormal(msh), C["z"]
    fext = fem.petsc.assemble_vector(fem.form(inner(v, n) * dse(3) + inner(v, n) * dse(4)))
    const = [fem.assemble_scalar(fem.form(f)) for f in (
        (3.0 * Gc / 8.0) * ((1 - z) / eps + eps * inner(grad(z), grad(z))) * dxe, (1 - z) * dxe)]
    X = dict(dG=dG, dE=dE, S=S, AEG=AEG, AGE=AGE, kspE=kspE, K=Km, M=Mm, fext=fext,
             const=np.array(const), w1=Km.createVecRight(), w2=Km.createVecLeft(),
             bE=AEE.createVecLeft(), xE=AEE.createVecLeft(), yE=AEE.createVecLeft(),
             tG=AGE.createVecLeft(),
             xG=C["xn"][dG // 2, :2],
             gmid=C["fmid"][inP[C["f2c"][:, 0]] != inP[C["f2c"][:, 1]]],
             ecells=np.where(~inP)[0].astype(np.int32), enodes=np.where(te)[0],
             onodes=np.where(te & ~tp)[0],
             ring=np.where(ring)[0].astype(np.int32), rdofs=rdofs, ridx=ridx, Yr=Yr,
             vr=np.zeros(rdofs.size), err=err, t=clock() - t0)
    for o in (A, AEE, iE, iG, isR, ve, vv, tg):
        o.destroy()
    return X


def destroy_exterior(X):
    for k in ("AEG", "AGE", "kspE", "K", "M", "fext", "w1", "w2", "bE", "xE", "yE", "tG"):
        X[k].destroy()


def matvec(X, name, x):
    """X[name] @ x for a coarse-size numpy vector x (exterior K or M)."""
    X["w1"].getArray()[:] = x
    X[name].mult(X["w1"], X["w2"])
    return X["w2"].getArray(readonly=True).copy()


def exterior_post(C, X):
    """Elastic and kinetic energy of the exterior (after the Newmark update), and its
    frozen surface energy and z-integral.  Rows and columns of coarse dofs inside the patch
    are empty in K and M, so their (stale) coarse values do not enter."""
    u, v = C["u"].x.array, C["v_prev"].x.array
    return np.array([0.5 * u @ matvec(X, "K", u), 0.5 * v @ matvec(X, "M", v), *X["const"]])


def recover(C, P, X):
    """u on the exterior from u on the interface: A_EE u_E = b_E - A_EG u_G."""
    uG = P["u"].x.array[P["pdG"]]
    X["AEG"].mult(PETSc.Vec().createWithArray(uG, comm=comm), X["xE"])
    X["xE"].aypx(-1.0, X["bE"])                    # xE = bE - A_EG u_G
    X["kspE"].solve(X["xE"], X["yE"])
    uc = C["u"].x.array
    uc[X["dE"]] = X["yE"].getArray(readonly=True)
    uc[X["dG"]] = uG


def lookahead(C, P, X):
    """Flagged cells of the first exterior layer, from the exact exterior u there:
    u_E = A_EE^-1 b_E - A_EE^-1 A_EG u_G."""
    uG = P["u"].x.array[P["pdG"]]
    C["u"].x.array[X["dG"]] = uG
    C["u"].x.array[X["rdofs"]] = -(X["Yr"] @ uG + X["vr"])
    C["ind"].interpolate(C["ind_expr"], X["ring"])
    return X["ring"][C["ind"].x.array[X["ring"]] > 0.5]

# =============================================================================
# SECTION 16: The Solve Mesh (the patch, or the whole domain for AMRC)
# =============================================================================

def build_patch(C, base, bmap, msh, parent, pts, X=None):
    """Set up both problems on *msh*, a refinement of *base* (a coarse submesh with cell
    map *bmap*, or the coarse mesh itself)."""
    msh.name = "solve"
    msh.topology.create_connectivity(1, 2)
    msh.topology.create_connectivity(2, 1)
    P = dict(msh=msh, base=base, bmap=bmap, cmap=bmap[parent], pts=pts)
    V = P["V"] = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    Y = P["Y"] = fem.functionspace(msh, ("Lagrange", 1))
    Yv = P["Yv"] = fem.functionspace(msh, ("DG", 0))
    for f in VFIELDS:
        P[f] = fem.Function(V, name="displacement" if f == "u" else f)
    P["z"] = fem.Function(Y, name="phasefield")
    u, z = P["u"], P["z"]
    P["zub"], P["zd"], P["ud"] = fem.Function(Y), fem.Function(Y), fem.Function(V)
    P["cnt"] = fem.Function(Yv, name="inactive_count")   # hysteresis counter (DG0)
    P["cnt"].x.array[:] = hysteresis_steps + 1           # start beyond the grace period
    xn = P["xn"] = Y.tabulate_dof_coordinates()
    c2n = P["c2n"] = Y.dofmap.list                 # (DG0 dofs are in cell order in serial)

    # z: the frozen patch boundary
    frozen, P["pdG"] = np.empty(0, np.int32), None
    if X is not None:
        dist, pn = cKDTree(xn[:, :2]).query(X["xG"], distance_upper_bound=1e-8 * W)
        assert np.isfinite(dist).all(), "interface node missing from the patch mesh"
        P["pdG"] = (2 * pn + X["dG"] % 2).astype(np.int32)
        P["gtree"] = cKDTree(X["xG"])
        frozen = np.unique(pn).astype(np.int32)
        # rim: free nodes sharing a cell with the frozen boundary.  z is P1, so damage
        # cannot reach the boundary without first showing up here.
        on = np.zeros(xn.shape[0], bool)
        on[pn] = True
        rim = np.unique(c2n[on[c2n].any(axis=1)])
        P["rim"] = rim[~on[rim]]
    P["free"] = np.setdiff1d(np.arange(xn.shape[0], dtype=np.int32), frozen).astype(np.int32)
    dist, cn = C["ntree"].query(xn[:, :2], distance_upper_bound=1e-8 * W)
    P["pn"] = np.where(np.isfinite(dist))[0]       # patch dofs that are coarse nodes ...
    P["cn"] = cn[P["pn"]]                          # ... and those coarse nodes

    # Forms.  Per stagger iteration the u-dependent cell terms are evaluated once (DG0,
    # exact for P1 u), so no kernel inside the z Newton loop recomputes stresses.
    dx_ = Measure("dx", domain=msh, metadata=_QMD)
    bf, bv = boundary_tags(msh)
    ds_ = Measure("ds", domain=msh, subdomain_data=meshtags(msh, 1, bf, bv))
    ip = Yv.element.interpolation_points()
    P["press"] = fem.Constant(msh, PETSc.ScalarType(0.0))
    P["a_u"] = fem.form(u_bilinear(V, z, dx_))
    P["L_u"] = fem.form(u_linear(V, z, P, P["press"], dx_, ds_))
    P["q"] = fem.Function(fem.functionspace(msh, ("DG", 0, (3,))))
    P["q_expr"] = fem.Expression(ufl.as_vector(cell_terms(u)), ip)
    p, a, s = P["q"][0], P["q"][1], P["q"][2]
    R = z_residual(z, TestFunction(Y), p, a, s, dx_)
    P["ind_expr"] = fem.Expression(indicator(z, p, a, s), ip)
    P["ind"] = fem.Function(Yv, name="pf_flag")
    P["h"] = fem.Function(Yv)
    P["h"].interpolate(fem.Expression(CellDiameter(msh), ip))
    P["f_dz"] = fem.form(P["zd"] * P["zd"] * dx_)
    P["f_du"] = fem.form(inner(P["ud"], P["ud"]) * dx_)
    P["post"] = post_forms(P, dx_)
    P["vr"] = fem.Function(V)
    rc = np.where(np.isclose(np.abs(xn[:, 1]), L / 2, atol=1e-8)[c2n].any(axis=1))[0].astype(np.int32)
    P["react"] = react_form(P, P["vr"], Measure("dx", domain=msh, metadata=_QMD, subdomain_data=meshtags(
        msh, 2, rc, np.ones(rc.size, np.int32)))(1))  # vr lives on the cells touching those edges
    P["dtop"], P["dbot"] = edge_dofs(V, bf, bv)

    if FAST_ASSEMBLY:
        P["tab"] = Tab(P)
        v_, n_ = TestFunction(V), FacetNormal(msh)
        P["ftr"] = fem.assemble_vector(fem.form(inner(v_, n_) * ds_(3) + inner(v_, n_) * ds_(4))).array.copy()
    P["us"] = USolver(P, X["S"] if X else None, P["pdG"])
    P["zs"] = ZSolver(z, P["zub"], P["free"], fem.form(R),
                      fem.form(derivative(R, z, TrialFunction(Y))), P)
    return P


def update_q(P):
    if "tab" in P:
        P["tab"].set_q(P)
    else:
        update_q(P)


def unorm(P):
    return P["tab"].unorm(P["ud"].x.array) if "tab" in P else np.sqrt(fem.assemble_scalar(P["f_du"]))


def znorm(P):
    return P["tab"].znorm(P["zd"].x.array) if "tab" in P else np.sqrt(fem.assemble_scalar(P["f_dz"]))


def destroy_patch(P):
    P["us"].destroy()
    P["zs"].destroy()
    P.clear()                                      # drop the refined mesh now, not at GC


def transfer(P_new, P_old, cells=None):
    """Interpolate u and its history, z, zub and the hysteresis counter from P_old onto
    P_new (on *cells* of P_new, default all); fields P_old lacks are left alone."""
    if cells is None:
        cells = np.arange(ncells(P_new["msh"]), dtype=np.int32)
    if cells.size == 0:
        return
    for V, names in (("V", VFIELDS), ("Y", ("z", "zub")), ("Yv", ("cnt",))):
        names = [f for f in names if f in P_old]
        if not names:
            continue
        idata = fem.create_interpolation_data(P_new[V], P_old[V], cells, padding=1e-8)
        bs = P_new[V].dofmap.index_map_bs
        dofs = np.unique(P_new[V].dofmap.list[cells].ravel())
        dofs = (bs * dofs[:, None] + np.arange(bs)).ravel()
        for f in names:
            tmp = fem.Function(P_new[V])
            tmp.interpolate_nonmatching(P_old[f], cells, idata)
            P_new[f].x.array[dofs] = tmp.x.array[dofs]


def from_coarse(C, P):
    """Fill P's u (and history), z, zub from the coarse fields (coarse -> base submesh by
    cell map, base -> refined patch by interpolation)."""
    base = P["base"]
    Q = dict(V=fem.functionspace(base, ("Lagrange", 1, (2,))), Y=fem.functionspace(base, ("Lagrange", 1)))
    for f, sp in [(f, "V") for f in VFIELDS] + [("z", "Y"), ("zub", "Y")]:
        Q[f] = fem.Function(Q[sp])
        Q[f].interpolate(C[f], cells0=P["bmap"], cells1=np.arange(ncells(base), dtype=np.int32))
    transfer(P, Q)


def to_coarse(C, P):
    """Write the patch values at coarse nodes back into the coarse fields."""
    pn, cn = P["pn"], P["cn"]
    C["z"].x.array[cn] = P["z"].x.array[pn]
    C["zub"].x.array[cn] = P["zub"].x.array[pn]
    for f in VFIELDS:
        for k in range(2):
            C[f].x.array[2 * cn + k] = P[f].x.array[2 * pn + k]


def coarse_z(C, P):
    """z at every coarse node, taking the patch value where the patch has that node."""
    z = C["z"].x.array.copy()
    z[P["cn"]] = P["z"].x.array[P["pn"]]
    return z


def new_patch(C, P_old, X_old, pts, refine_pts=None):
    """(Re)place the patch around *pts*, refine it near *refine_pts*, condense the new
    exterior and carry the fields over.  O(N) -- only when the patch moves.  Returns
    (P, X); X is None if the patch had to become the whole domain."""
    if P_old is not None:
        recover(C, P_old, X_old)
        to_coarse(C, P_old)
    pcells, base, bmap, msh, parent = place_patch(C, select_patch(C, pts), refine_pts)
    if P_old is not None and METHOD == "amrc_clip":
        relax_wake(C, pcells)
    X = build_exterior(C, pcells)
    if X is None:                                  # nothing left outside: go whole-domain
        print("    [patch] covers the whole domain -- whole-domain solve from here", flush=True)
        base, bmap = C["msh"], np.arange(C["nc"], dtype=np.int32)
        msh, parent = refine_near(base, refine_pts)
    P = build_patch(C, base, bmap, msh, parent, refine_pts, X)
    from_coarse(C, P)
    if P_old is not None:
        old = mask(C, P_old["bmap"])
        transfer(P, P_old, np.where(old[P["cmap"]])[0].astype(np.int32))
        destroy_patch(P_old)
        destroy_exterior(X_old)
    arm_rim(P)
    return P, X


def relax_wake(C, pcells):
    """AMRC-CLIP: one z solve on the damage outside the patch, on the coarse mesh, with the
    exact exterior u.  The crack-centre z keeps decaying as the opening grows; refreshing it
    at every move bounds its lag to one move.  The exterior is condensed right after, so u
    sees the refreshed wake."""
    c2n, inP = C["c2n"], mask(C, pcells)
    band = damaged_cells(C) & ~inP
    if not band.any():
        return
    band = grow(C, band) & ~inP                    # one layer of intact cells around it
    cells = np.where(band)[0].astype(np.int32)
    fixed = np.zeros(C["xn"].shape[0], bool)
    fixed[c2n[~band].ravel()] = True               # touched by the patch or the rest
    free = np.setdiff1d(np.unique(c2n[cells]), np.where(fixed)[0])
    if free.size == 0:
        return
    tags = meshtags(C["msh"], 2, np.arange(C["nc"], dtype=np.int32), band.astype(np.int32))
    dxw = Measure("dx", domain=C["msh"], subdomain_data=tags, metadata=_QMD)(1)
    q = fem.Function(fem.functionspace(C["msh"], ("DG", 0, (3,))))
    q.interpolate(fem.Expression(ufl.as_vector(cell_terms(C["u"])),
                                 C["Yv"].element.interpolation_points()), cells)
    z = C["z"]
    R = z_residual(z, TestFunction(C["Y"]), q[0], q[1], q[2], dxw)
    zs = ZSolver(z, C["zub"], free.astype(np.int32), fem.form(R),
                 fem.form(derivative(R, z, TrialFunction(C["Y"]))))
    zs.solve()
    zs.destroy()


def arm_rim(P):
    """Remember which rim nodes are still undamaged; only those can signal a spread."""
    if "rim" in P:
        P["rim0"] = P["rim"][P["z"].x.array[P["rim"]] > DAMAGE_Z]


def set_step_data(C, P, X, t):
    """Load of this step: the traction, and the condensed exterior load g = b_G - A_GE
    A_EE^-1 b_E (b_E from the exterior Newmark history) with the part of the exterior u
    that does not depend on u_G, on the first exterior layer."""
    p = load(t)
    P["press"].value = p
    if X is None:
        return
    up, vp, ap = (C[f].x.array for f in ("u_prev", "v_prev", "a_prev"))
    hist = (up + dt * vp) / (beta_nm * dt**2) + ((1 - 2 * beta_nm) / (2 * beta_nm)) * ap
    ba = p * X["fext"].getArray(readonly=True) - alpha_f * matvec(X, "K", up) \
        + matvec(X, "M", (1 - alpha_m) * hist - alpha_m * ap)
    X["bE"].getArray()[:] = ba[X["dE"]]
    X["kspE"].solve(X["bE"], X["yE"])              # yE = A_EE^-1 b_E
    X["AGE"].mult(X["yE"], X["tG"])
    P["us"].g = ba[X["dG"]] - X["tG"].getArray(readonly=True)
    X["vr"] = -X["yE"].getArray(readonly=True)[X["ridx"]]


def irreversibility(C, P, X):
    for z, ub in ((P["z"], P["zub"]),) + (((C["z"], C["zub"]),) if X is not None else ()):
        m = z.x.array <= v_th
        ub.x.array[m] = z.x.array[m] + z_ub_offset


def newmark_update(C, P, X):
    """Advance a_prev, v_prev, u_prev on the solve mesh and at the exterior coarse dofs
    (the interface dofs take the patch values)."""
    parts = [(P, slice(None))] + ([(C, X["dE"])] if X is not None else [])
    for F, d in parts:
        u, up, vp, ap = (F[f].x.array for f in VFIELDS)
        a = newmark_a(u[d], up[d], vp[d], ap[d])
        vp[d] = newmark_v(a, vp[d], ap[d])
        ap[d] = a
        up[d] = u[d]
    if X is not None:
        for f in VFIELDS:
            C[f].x.array[X["dG"]] = P[f].x.array[P["pdG"]]


def reactions(C, P, X):
    """F_top, F_bot and the external-work increment, from the alpha-averaged dynamic
    residual, summed over the solve mesh and the exterior cells."""
    out = np.zeros(3)
    vr, dtop, dbot = P["vr"], P["dtop"], P["dbot"]
    if dtop.size or dbot.size:
        du = P["u"].x.array - P["u_prev"].x.array
        for k, fill in enumerate(((dtop, 1.0), (dbot, 1.0), None)):
            vr.x.array[:] = 0.0
            if fill is None:
                vr.x.array[dtop], vr.x.array[dbot] = du[dtop], du[dbot]
            else:
                vr.x.array[fill[0]] = fill[1]
            out[k] += fem.assemble_scalar(P["react"])
    if X is not None:                              # residual rows K u_avg + M a_avg
        u, up, vp, ap = (C[f].x.array for f in VFIELDS)
        r = matvec(X, "K", alpha_f * up + (1 - alpha_f) * u) \
            + matvec(X, "M", alpha_m * ap + (1 - alpha_m) * newmark_a(u, up, vp, ap))
        dtop, dbot = C["dtop"], C["dbot"]
        du = u - up
        out += [r[dtop].sum(), r[dbot].sum(), r[dtop] @ du[dtop] + r[dbot] @ du[dbot]]
    return out


def crack_tip(C, P, X):
    """The broken node (z <= v_th, y >= 0) farthest from the initial tip."""
    xs, zs = [P["xn"]], [P["z"].x.array]
    if X is not None:
        xs.append(C["xn"][X["onodes"]])
        zs.append(C["z"].x.array[X["onodes"]])
    x, z = np.vstack(xs), np.concatenate(zs)
    x = x[(z <= v_th) & (x[:, 1] >= 0.0)]
    if x.shape[0] == 0:
        return None
    return x[np.argmax(np.linalg.norm(x - np.array([W / 2, 0.0, 0.0]), axis=1))]

# =============================================================================
# SECTION 17: Output
# =============================================================================

class Output:
    """xdmf/patch_N.xdmf: the solve mesh with u, z and the flags; xdmf/exterior_N.xdmf: the
    exterior coarse cells with u, z (AMRC-CLIP, AMRC-CLIP-TW).
    xdmf/coarse.xdmf: a time series on the fixed coarse mesh with u, z at the coarse nodes
    and the regions as DG0 fields (1 inside, 0 outside):
      solved     the coarse cells of the patch (all cells for AMRC)
      refined    the coarse cells that are refined
      flagged    the coarse cells holding a flagged cell
      effective  the coarse cells holding an effective cell (flagged or in hysteresis, A)
      delta      AMRC-CLIP: the cells whose z is still dropping (Delta)
      damaged    AMRC-CLIP-TW: the cells with a node z < DAMAGE_Z (D)"""

    def __init__(self, C):
        self.dir = os.path.join(output_dir, "xdmf")
        os.makedirs(self.dir, exist_ok=True)
        C["msh"].name = "coarse"
        self.xf = XDMFFile(comm, os.path.join(self.dir, "coarse.xdmf"), "w")
        self.xf.write_mesh(C["msh"])
        names = ["solved", "refined", "flagged", "effective"] \
            + {"amrc_clip": ["delta"], "amrc_clip_tw": ["damaged"]}.get(METHOD, [])
        self.reg = {n: fem.Function(C["Yv"], name=n) for n in names}

    def write(self, C, P, X, step, t):
        P["ind"].interpolate(P["ind_expr"])
        self.write_exterior(C, X, step, t)
        with XDMFFile(comm, os.path.join(self.dir, f"patch_{step:06d}.xdmf"), "w") as xf:
            xf.write_mesh(P["msh"])
            for f in (P["u"], P["z"], P["ind"]):
                xf.write_function(f, t)
        to_coarse(C, P)
        nc, cmap = C["nc"], P["cmap"]
        region = dict(solved=mask(C, P["bmap"]),
                      refined=np.bincount(cmap, minlength=nc) > 1,
                      flagged=mask(C, cmap[P["ind"].x.array > 0.5]),
                      effective=mask(C, cmap[(P["ind"].x.array > 0.5)
                                             | (P["cnt"].x.array < hysteresis_steps)]))
        if METHOD == "amrc_clip":
            region["delta"] = delta_cells(C)
        if METHOD == "amrc_clip_tw":
            region["damaged"] = damaged_cells(C)
        for n, f in self.reg.items():
            f.x.array[:] = region[n]
        for f in (C["u"], C["z"], *self.reg.values()):
            self.xf.write_function(f, t)

    def write_exterior(self, C, X, step, t):
        """xdmf/exterior_N.xdmf: the coarse cells outside the patch with the exact u (recovered
        at the end of the step) and the frozen z.  Shown together with patch_N.xdmf it covers
        the whole domain.  AMRC (no exterior) writes none."""
        if X is None:
            return
        if "_exo" not in X:                        # the exterior submesh, once per patch
            tdim = C["msh"].topology.dim
            sub, emap, _, _ = create_submesh(C["msh"], tdim, X["ecells"])
            sub.name = "exterior"
            X["_exo"] = (sub, np.asarray(emap, dtype=np.int32),
                         fem.Function(fem.functionspace(sub, ("Lagrange", 1, (tdim,))), name="displacement"),
                         fem.Function(fem.functionspace(sub, ("Lagrange", 1)), name="phasefield"))
        sub, emap, ue, ze = X["_exo"]
        cells = np.arange(emap.size, dtype=np.int32)
        ue.interpolate(C["u"], cells0=emap, cells1=cells)
        ze.interpolate(C["z"], cells0=emap, cells1=cells)
        with XDMFFile(comm, os.path.join(self.dir, f"exterior_{step:06d}.xdmf"), "w") as xf:
            xf.write_mesh(sub)
            xf.write_function(ue, t)
            xf.write_function(ze, t)

    def close(self):
        self.xf.close()

# =============================================================================
# SECTION 18: Print Summary
# =============================================================================

METHOD_NAME = {"amrc": "AMRC", "amrc_clip": "AMRC-CLIP", "amrc_clip_tw": "AMRC-CLIP-TW"}[METHOD]
_bw = 70  # box inner width


def _box(title, rows):
    print(f"\n╔{'═' * _bw}╗")
    print(f"║{title:^{_bw}}║")
    print(f"╠{'═' * _bw}╣")
    for row in rows:
        print(f"║  {row:<{_bw - 2}}║")
    print(f"╚{'═' * _bw}╝", flush=True)


def print_summary():
    last = min(max_steps, MAX_STEPS) if MAX_STEPS else max_steps   # the step count of main()
    last = min(last, int(np.ceil(t_max / dt - 1e-9)))
    _box("PHASE-FIELD FRACTURE — PARAMETER SUMMARY", [
        "Problem   : dynamic crack branching (HHT-alpha), plane stress",
        f"Method    : {METHOD_NAME}",
        f"Geometry  : W = {W:g} mm   L = {L:g} mm   crack tip at ({W / 2:g}, 0)",
        f"Elastic   : E = {E:.4e}   nu = {nu:.4f}   rho_s = {rho_s:.4e}",
        f"Strength  : sts = {sts:.4e}   scs = {scs:.4e}   shs = {shs:.4e}",
        f"Fracture  : Gc = {Gc:.6f}   eps = {eps:.4e}   delta = {delta:.4e}",
        f"AMR       : h0 = {h0:.4g}   h_min = {h_min:.4g}   R = {amr_refine_radius:.4g}",
        f"Hysteresis: {hysteresis_steps} steps",
        (f"Patch     : Rp = {PATCH_R_EPS:g} eps   retire = {RETIRE_STEPS} refinement"
         + (f"   wake layers = {WAKE_LAYERS}" if METHOD == "amrc_clip_tw" else "")
         if LOCAL else "Patch     : none (AMRC: the whole mesh is solved)"),
        f"Loading   : p = {p_max:g} MPa on the top / bottom edges, ramp {t_ramp_up:g} s",
        f"Time      : {last} steps   dt = {dt:g} s   (t_max = {t_max:g} s)",
        f"HHT-alpha : alpha_f = {alpha_f:g}   alpha_m = {alpha_m:g}   "
        f"(beta {beta_nm:.4g}, gamma {gamma_nm:.4g})",
        f"Stagger   : tol_u = {tol_stag_u:.1e}   tol_z = {tol_stag_z:.1e}   max_stag = {max_stag}",
        f"Solvers   : u {'/'.join(U_FACTOR)} (CG)   z {'/'.join(Z_FACTOR)}",
        f"Assembly  : fast = {FAST_ASSEMBLY}" + (f"   condensation = {CONDENSE}" if LOCAL else ""),
        f"Output    : {output_dir}",
    ])

# =============================================================================
# SECTION 19: Main Solve Loop
# =============================================================================

def main():
    t_start = clock()
    say = print if verbose > 1 else (lambda *a, **k: None)
    last = min(max_steps, MAX_STEPS) if MAX_STEPS else max_steps
    last = min(last, int(np.ceil(t_max / dt - 1e-9)))   # steps while t - dt < t_max
    print_summary()
    C = coarse_setup(generate_mesh())
    os.makedirs(output_dir, exist_ok=True)
    _USTAT.update(n_factor=0, n_cg=0)
    TT = dict(u=0.0, z=0.0, amr=0.0, ext=0.0, exs=0.0, post=0.0, io=0.0, n_u=0, n_z=0,
              n_amr=0, n_move=0, n_lookahead=0, n_escape=0, n_rim=0)
    nonconverged, unresolved = [], []
    H = float(hysteresis_steps)

    # --- Initial solve mesh: the patch around the crack tip, or the whole domain ---
    tip0 = np.array([[W / 2, 0.0, 0.0]])
    if LOCAL:
        P, X = new_patch(C, None, None, tip0)
        TT["ext"] += X["t"] if X else 0.0
    else:
        X = None
        cells = np.arange(C["nc"], dtype=np.int32)
        P = build_patch(C, C["msh"], cells, C["msh"], cells, None)
        from_coarse(C, P)
    # pre-solve of the phase field (z = 1 stays)
    P["zs"].solve()
    out = Output(C) if output_every else None
    log_file, timing_file = f"{output_dir}/log.txt", f"{output_dir}/timing.csv"
    with open(log_file, "w") as f:
        f.write("step time E_elastic E_kinetic W_ext E_surface z_integral min_z F_top F_bot "
                "iterations converged du_last dz_last wall_step wall_total n_nodes n_solve_nodes\n")
    with open(timing_file, "w") as f:
        f.write("step,t,iterations,converged,n_cells,n_solve_cells,n_G,n_amr,n_move,"
                "t_u,t_z,t_amr,t_post,t_io,t_step,t_cum,n_nodes,n_solve_nodes,t_ext\n")
    for name, head in (("Energies.txt", "time(s)  E_elastic  E_kinetic  W_ext  E_surface"),
                       ("Traction_action.txt", "time(s)  F_top  F_bot"),
                       ("CrackVelocity.txt", "time(s)  crack_inc  crack_length_total  crack_velocity"),
                       ("CrackTipCoords.txt", "time(s)  tip_x  tip_y")):
        with open(f"{output_dir}/{name}", "w") as f:
            f.write(head + "\n")

    def seeds(P, *extra):
        """Points the patch must cover: the last refinement points (or the initial tip),
        the effective cells, the currently flagged cells and *extra*."""
        pts = [P["pts"] if P["pts"] is not None else tip0]
        P["ind"].interpolate(P["ind_expr"])
        eff = np.where((P["ind"].x.array > 0.5) | (P["cnt"].x.array < H))[0]
        pts += [midpoints(P["msh"], eff)] + [e for e in extra if len(e)]
        return np.vstack(pts)

    def move(P, X, pts, refine_pts):
        """Re-select the patch around *pts* and refine it near *refine_pts*."""
        t0 = clock()
        Pn, Xn = new_patch(C, P, X, pts, refine_pts)
        TT["n_move"] += 1
        if Xn is not None:
            TT["ext"] += Xn["t"]
            say(f"    [patch] moved: {ncells(Pn['msh']):,} cells, interface {Xn['dG'].size} "
                f"dofs, exterior {Xn['dE'].size:,} dofs ({clock() - t0:.2f} s, check {Xn['err']:.1e})")
        return Pn, Xn

    def remesh(P, X, pts):
        """Re-refine the solve mesh around *pts* (the effective cells); move the patch instead
        if one of them lies within R of the patch boundary or the refinement would split it."""
        if X is None or not np.isfinite(P["gtree"].query(
                pts[:, :2], distance_upper_bound=amr_refine_radius)[0]).any():
            msh, parent = refine_near(P["base"], pts)
            if X is None or not split_facets(msh, X["gmid"]).any():
                Pn = build_patch(C, P["base"], P["bmap"], msh, parent, pts, X)
                transfer(Pn, P)
                arm_rim(Pn)
                destroy_patch(P)
                say(f"    [AMR] refined -> {ncells(Pn['msh']):,} cells")
                return Pn, X
        return move(P, X, pts, pts)

    logs = {k: open(p, "a") for k, p in (("log", log_file), ("timing", timing_file),
            ("en", f"{output_dir}/Energies.txt"), ("tr", f"{output_dir}/Traction_action.txt"),
            ("cv", f"{output_dir}/CrackVelocity.txt"), ("ct", f"{output_dir}/CrackTipCoords.txt"))}
    crack_prev, crack_total = np.array([W / 2, 0.0, 0.0]), 0.0
    W_ext = 0.0
    for step in range(1, last + 1):
        t = step * dt
        ts = clock()
        su = sz = sa = sp = se = 0.0
        n_amr0, n_move0 = TT["n_amr"], TT["n_move"]
        say(f"\n-- step {step}  t = {t:.6e}")

        # --- A: Loading and irreversibility bounds ---
        tk = clock()
        set_step_data(C, P, X, t)
        se += clock() - tk
        irreversibility(C, P, X)                   # once per step, from the last step's z
        if (METHOD == "amrc_clip" or USE_DELTA) and X is not None:   # coarse-node z at the step start (Delta)
            C["zhist"] = (C.get("zhist", []) + [coarse_z(C, P)])[-max(1, ACTIVE_STEPS):]
            C["zref"] = C["zhist"][0]

        # --- B: Staggered u-z solve (with lookahead, rim and escape patch moves) ---
        it, ures, zres, esc = 0, np.inf, np.inf, 0
        while True:
            while it < max_stag - 1 and (ures > tol_stag_u or zres > tol_stag_z):
                it += 1
                tk = clock()
                P["ud"].x.array[:] = P["u"].x.array
                P["us"].solve()
                P["ud"].x.array[:] -= P["u"].x.array
                ures = unorm(P)
                su += clock() - tk
                TT["n_u"] += 1

                # a flagged cell in the first exterior layer: grow the patch over it (it
                # stays coarse until the step-end refinement, as in AMRC)
                if X is not None:
                    tk = clock()
                    ring = lookahead(C, P, X)
                    if ring.size:
                        TT["n_lookahead"] += 1
                        P, X = move(P, X, seeds(P, midpoints(C["msh"], ring)), P["pts"])
                        set_step_data(C, P, X, t)
                        update_q(P)
                        ures = np.inf              # iterate at least once more on the new patch
                    sa += clock() - tk

                tk = clock()
                zprev = P["z"].x.array.copy()
                P["zs"].solve()
                TT["n_z"] += 1
                sz += clock() - tk
                # AMRC-CLIP-TW: damage reached a free node next to the frozen boundary.
                # Undo this z solve, grow the patch around those nodes and redo it.  The
                # patch is grown around every node hit so far in this z solve: the undo
                # erases the damage, so neither D nor the latest hits alone would keep the
                # earlier ones.
                hits = np.empty((0, 3))
                for _ in range(max_rim_moves):
                    if METHOD != "amrc_clip_tw" or X is None:
                        break
                    hit = P["rim0"][P["z"].x.array[P["rim0"]] < DAMAGE_Z]
                    if hit.size == 0:
                        break
                    tk = clock()
                    hits = np.vstack([hits, P["xn"][hit]])
                    P["z"].x.array[:] = zprev
                    P, X = move(P, X, seeds(P, hits), P["pts"])
                    set_step_data(C, P, X, t)
                    update_q(P)
                    zprev = P["z"].x.array.copy()
                    ures = np.inf
                    TT["n_rim"] += 1
                    sa += clock() - tk
                    tk = clock()
                    P["zs"].solve()
                    TT["n_z"] += 1
                    sz += clock() - tk
                P["zd"].x.array[:] = P["z"].x.array - zprev
                zres = znorm(P)
                say(f"    iter {it:>3d}  |du| = {ures:.7e}  |dz| = {zres:.7e}")

            if ures > tol_stag_u or zres > tol_stag_z:
                # stopped at max_stag: u for the final z
                tk = clock()
                P["us"].solve()
                su += clock() - tk
                TT["n_u"] += 1
            if X is None:
                break
            # nothing outside the patch may be flagged (exterior u recovered exactly)
            tk = clock()
            recover(C, P, X)
            C["ind"].interpolate(C["ind_expr"], X["ecells"])
            outside = X["ecells"][C["ind"].x.array[X["ecells"]] > 0.5]
            se += clock() - tk
            if outside.size == 0:
                break
            esc += 1
            TT["n_escape"] += 1
            print(f"    step {step}: {outside.size} flagged cells outside the patch -- absorbing",
                  flush=True)
            tk = clock()
            P, X = move(P, X, seeds(P, midpoints(C["msh"], outside)), P["pts"])
            set_step_data(C, P, X, t)
            update_q(P)
            sa += clock() - tk
            if esc > max_escape_retry or it >= max_stag - 1:
                unresolved.append(step)
                tk = clock()
                P["us"].solve()
                su += clock() - tk
                TT["n_u"] += 1
                if X is not None:
                    recover(C, P, X)
                    # cells still flagged outside join the patch (no more iterations), so
                    # they get their hysteresis counter at the step-end AMR like in AMRC
                    C["ind"].interpolate(C["ind_expr"], X["ecells"])
                    outside = X["ecells"][C["ind"].x.array[X["ecells"]] > 0.5]
                    if outside.size:
                        tk = clock()
                        P, X = move(P, X, seeds(P, midpoints(C["msh"], outside)), P["pts"])
                        set_step_data(C, P, X, t)
                        update_q(P)
                        sa += clock() - tk
                break
            ures = zres = np.inf                   # iterate again on the enlarged patch
        converged = ures <= tol_stag_u and zres <= tol_stag_z
        if not converged:
            nonconverged.append(step)

        # --- C: Reactions and external work, Newmark update, energies ---
        tk = clock()
        F_top, F_bot, dW = reactions(C, P, X)
        W_ext += dW
        newmark_update(C, P, X)
        vals = np.array([fem.assemble_scalar(f) for f in P["post"]])
        min_z = P["z"].x.array.min()
        n_cells = n_solve_cells = ncells(P["msh"])
        n_G = X["dG"].size if X is not None else 0
        n_nodes = n_solve_nodes = nnodes(P["msh"])
        if X is not None:
            vals += exterior_post(C, X)
            min_z = min(min_z, C["z"].x.array[X["enodes"]].min())
            n_cells += X["ecells"].size
            n_nodes += X["onodes"].size            # the interface nodes are patch nodes
        E_el, E_kin, E_surf, Zint = vals
        sp += clock() - tk

        # --- D: Output (before the refinement) ---
        tk = clock()
        if out is not None and step % output_every == 0 and t <= output_t_max + 1e-15:
            out.write(C, P, X, step, t)
        tio = clock() - tk

        # --- E: refinement events since each coarse cell was last refined (counted at the
        #     AMR below; reset where the solve mesh refines the cell; retirement rule) ---
        if LOCAL and RETIRE_STEPS:
            ref = np.bincount(P["cmap"], minlength=C["nc"]) > 1
            C["refage"] = np.where(ref, 0, C.get("refage", np.full(C["nc"], 10**9)))

        # --- F: AMR with hysteresis (end of step) ---
        tk = clock()
        P["ind"].interpolate(P["ind_expr"])
        act = P["ind"].x.array > 0.5
        cnt = P["cnt"].x.array
        cnt[act] = 0.0
        cnt[~act] = np.minimum(cnt[~act] + 1.0, H + 1.0)
        eff = act | (cnt < H)
        if (eff & (P["h"].x.array > 1.5 * h_min)).any():
            if LOCAL and RETIRE_STEPS:
                # a refinement event: one more for the cells not refined now
                C["refage"][~ref] += 1
            P, X = remesh(P, X, midpoints(P["msh"], np.where(eff)[0]))
            TT["n_amr"] += 1
        sa += clock() - tk

        # --- G: Crack tip and velocity (after the refinement) ---
        tk = clock()
        if step % vel_interval == 0:
            tipx = crack_tip(C, P, X)
            tipx = crack_prev if tipx is None else tipx
            inc = np.linalg.norm(tipx - crack_prev)
            crack_total += inc
            print(t, inc, crack_total, inc / (vel_interval * dt), file=logs["cv"])
            print(t, tipx[0], tipx[1], file=logs["ct"])
            crack_prev = tipx
        sp += clock() - tk

        # --- H: Timing and per-step logs ---
        for k, v in (("u", su), ("z", sz), ("amr", sa), ("exs", se), ("post", sp), ("io", tio)):
            TT[k] += v
        dt_step, t_cum = clock() - ts, clock() - t_start
        print(t, E_el, E_kin, W_ext, E_surf, file=logs["en"])
        print(t, F_top, F_bot, file=logs["tr"])
        f = logs["log"]
        f.write(f"{step} {t:.6e} {E_el:.9e} {E_kin:.9e} {W_ext:.9e} {E_surf:.9e} {Zint:.6e} "
                f"{min_z:.6f} {F_top:.6e} {F_bot:.6e} {it} {int(converged)} {ures:.3e} "
                f"{zres:.3e} {dt_step:.3f} {t_cum:.3f} {n_nodes} {n_solve_nodes}\n")
        f = logs["timing"]
        f.write(f"{step},{t:.8e},{it},{int(converged)},{n_cells},{n_solve_cells},"
                f"{n_G},{TT['n_amr'] - n_amr0},{TT['n_move'] - n_move0},"
                f"{su:.6f},{sz:.6f},{sa:.6f},{sp:.6f},{tio:.6f},{dt_step:.6f},"
                f"{t_cum:.3f},{n_nodes},{n_solve_nodes},{se:.6f}\n")
        if step % LOG_FLUSH == 0:
            for f in logs.values():
                f.flush()
        if verbose:
            print(f"  step {step:>4d}  iters {it:>3d}{'' if converged else ' (max)'}  "
                  f"E_s {E_surf:.5e}  min_z {min_z:.3f}  nodes {n_solve_nodes:,} solved / {n_nodes:,}  "
                  f"| u {su:.2f}s  z {sz:.2f}s  amr {sa:.2f}s  ext {se:.2f}s  post {sp:.2f}s  "
                  f"io {tio:.2f}s  | step {dt_step:.2f}s  total {t_cum:.1f}s", flush=True)

    # --- Finalisation ---
    for f in logs.values():
        f.close()
    if out is not None:
        out.close()
    elapsed = clock() - t_start
    _box("SIMULATION COMPLETE", [
        f"Method          : {METHOD_NAME}",
        f"Total time      : {elapsed:.1f} s ({CLOCK} clock)",
        f"Steps completed : {last}   ({len(nonconverged)} not converged in {max_stag - 1} iterations)",
    ] + [f"  {lab:<14}: {TT[k]:9.1f} s ({100 * TT[k] / max(elapsed, 1e-9):5.1f} %)"
         for k, lab in (("u", "displacement"), ("z", "phase field"), ("amr", "AMR + patch"),
                        ("exs", "exterior/step"), ("post", "post/check"), ("io", "XDMF output"))] + [
        f"Solves          : u {TT['n_u']:,}   z {TT['n_z']:,}",
        f"Refinements     : {TT['n_amr']:,}",
    ] + ([
        f"Patch moves     : {TT['n_move']:,}   ({TT['ext']:.1f} s condensing)",
        f"Escapes         : {TT['n_escape']}   lookahead moves {TT['n_lookahead']}   "
        f"rim retries {TT['n_rim']}",
    ] if LOCAL else []) + [
        f"u solver        : {_USTAT['n_factor']:,} factorizations, {_USTAT['n_cg']:,} CG iterations",
    ] + ([f"UNRESOLVED steps: {unresolved}"] if unresolved else []))
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(dict(method=METHOD, use_delta=USE_DELTA, run_tag=RUN_TAG, steps=last,
                       elapsed=elapsed, clock=CLOCK, eps=eps, h0=h0, h_min=h_min, W=W, L=L,
                       dt=dt, max_stag=max_stag, tol_stag_u=tol_stag_u, tol_stag_z=tol_stag_z,
                       hysteresis_steps=hysteresis_steps, active_dz=ACTIVE_DZ,
                       active_steps=ACTIVE_STEPS, damage_z=DAMAGE_Z, patch_r_eps=PATCH_R_EPS,
                       delta_pad_eps=DELTA_PAD_EPS, z_ub_margin=z_ub_offset,
                       nonconverged_steps=nonconverged, unresolved_steps=unresolved,
                       u_factor=U_FACTOR, z_factor=Z_FACTOR, u_cg_rtol=U_CG_RTOL,
                       wake_layers=WAKE_LAYERS, retire_steps=RETIRE_STEPS,
                       fast_assembly=FAST_ASSEMBLY, condense=CONDENSE,
                       u_refactors=_USTAT["n_factor"], u_cg_iterations=_USTAT["n_cg"], **TT), f, indent=2)
    return elapsed


if __name__ == "__main__":
    main()
