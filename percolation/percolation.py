#!/usr/bin/env python3
"""
Phase-Field Fracture with AMRC and Condensation of Linear Inactive Parts (CLIP)
===============================================================================
Coalescence of multiple cracks ("percolation") — growth and coalescence of 16 cracks in a
unit cell under outward edge displacement (paper Section 4.4).
Unit cell [0, W] × [0, L] = [0, 1] × [0, 1] mm with 16 crack slits of length CRACK_LEN,
loaded by the outward edge displacement t (the paper's ū).

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
Problem  R. Geelen, J. Plews, M. Tupek, J. Dolbow, "An extended/generalized phase-field
         finite element method for crack growth with global-local enrichment", Int. J.
         Numer. Methods Eng. 121 (2020) 2534-2557 (adapted from Sec. 4.2, Figs. 10-11).

Physics
-------
- 2D plane-strain, quasi-static mechanics
- Complete phase-field fracture with Drucker-Prager strength surface
- 16 straight cracks centred on the grid
- Each crack is a thin diamond slit in the mesh (traction-free faces), cw = CRACK_W_EPS eps =
  eps / 6 wide at the centre and sharp at both tips
- u_x = -t / +t on the left / right edge and u_y = -t / +t on the bottom / top edge (the
  outward normal displacement u.n = t, the paper's ū), tangential components free (the
  corner nodes get both); t = U_MAX = 0.008 mm at the end (paper Section 4.4 reports up
  to 0.005 mm), in NSTEPS = 160 load steps; z = 0 at the 32 crack-tip nodes
- Material as in Geelen et al.: E = 210 GPa, nu = 0.3, Gc = 2.7 N/mm, eps = 0.006 mm, sts = 4209 MPa, and scs = 5 sts.

Units: consistent (mm, N, MPa); the load parameter t is the edge displacement (mm)
"""

# =============================================================================
# SECTION 0: Imports
# =============================================================================

import sys, os, time, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import dolfinx.fem.petsc
import dolfinx.cpp as _cpp
from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import (compute_midpoints, create_submesh, exterior_facet_indices,
                          locate_entities, locate_entities_boundary, meshtags, refine,
                          RefinementOption)
import ufl
from ufl import (CellDiameter, FacetNormal, Identity, Measure, TestFunction,
                 TrialFunction, conditional, derivative, dot, grad, inner, lt, sqrt,
                 sym, tr)

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

W = 1.0            # unit cell width  (mm), x in [0, W]
L = 1.0            # unit cell height (mm), y in [0, L]

# --- The 16 cracks ---
CRACK_LEN   = 0.05     # crack length (mm), between Geelen et al.'s text (0.025) and Fig. 11A (~0.08)
CRACK_W_EPS = 1 / 6    # slit width at the crack centre / eps, sharp at both tips (cw: SECTION 3)

# Geelen et al.'s 16 seeds (Figs. 10/11A): centres (0.2 (j + 1), 0.2 (4 - i)), rotation from the
# x axis in degrees (counter-clockwise); row i = 0 is the top row (y = 0.8)
CRACK_ANGLES = ((120.0, 18.0, 135.0, 36.0),
                (36.0, 120.0, 22.5, 0.0),
                (157.5, 36.0, 135.0, 180 / 7),
                (22.5, 90.0, 36.0, 18.0))
SHIFTED_SEEDS = ((2, 2), (2, 3))   # seeds (i, j) moved up by SEED_SHIFT: (0.6, 0.4) and (0.8, 0.4)
SEED_SHIFT    = 0.05               # upward shift of those seeds (mm): to y = 0.45

# crack centres and rotations (rad), and the 32 crack tips
CRACKS = [((0.2 * (j + 1), 0.2 * (4 - i) + (SEED_SHIFT if (i, j) in SHIFTED_SEEDS else 0.0)), np.radians(a))
          for i, row in enumerate(CRACK_ANGLES) for j, a in enumerate(row)]
TIPS = np.array([[cx + s * 0.5 * CRACK_LEN * np.cos(th), cy + s * 0.5 * CRACK_LEN * np.sin(th), 0.0]
                 for (cx, cy), th in CRACKS for s in (1.0, -1.0)])

# =============================================================================
# SECTION 3: Material Properties
# =============================================================================

# --- Elastic constants (as in Geelen et al.) ---
E, nu = 210000.0, 0.3                          # Young's modulus (MPa), Poisson's ratio
mu    = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
kappa = E / (3 * (1 - 2 * nu))

# --- Fracture parameters ---
Gc  = 2.7                 # critical energy release rate (N/mm), as in Geelen et al.
eps = 0.006               # regularisation length (mm): Geelen et al.'s l
eta = 1e-5                # residual stiffness
cw  = CRACK_W_EPS * eps   # crack slit width at the centre (mm)

# --- Strength surface (Drucker-Prager); Geelen et al. give no strength ---
LCH_EPS = 2.0                                  # sets sts: lch = 3 Gc E / (8 sts^2) = LCH_EPS eps
SCS_STS = 5.0                                  # scs = SCS_STS sts
sts = (3 * Gc * E / (8 * LCH_EPS * eps))**0.5  # uniaxial tensile strength (MPa)
scs = SCS_STS * sts                            # uniaxial compressive strength (MPa)
shs = (2 / 3) * sts * scs / (scs - sts)        # hydrostatic strength
Wts = sts**2 / (2 * E)
Whs = shs**2 / (2 * kappa)

# =============================================================================
# SECTION 4: AMR Parameters
# =============================================================================

H0_EPS  = 1.0                  # coarse element size / eps
HMIN_H0 = 0.25                 # refined element size / h0
h0      = H0_EPS * eps         # coarse (base) mesh element size
h_min   = HMIN_H0 * h0         # element size after full refinement
amr_refine_radius = 3 * eps    # refinement radius R around the flagged cells
pf_flag_threshold = -0.1       # indicator threshold
COARSEN_IDLE = True            # nothing flagged but the mesh refined: coarsen it (<= once per step)

# =============================================================================
# SECTION 5: Phase-Field Control
# =============================================================================

v_th = 0.05        # irreversibility threshold

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

U_MAX     = 0.008                     # outward normal displacement of every edge at the end (mm)
NSTEPS    = 160                       # number of load steps
T_total   = U_MAX                     # final value of the load parameter t (edge displacement)
nsteps    = NSTEPS                    # = NSTEPS (the name the main loop uses)
stepsize  = U_MAX / NSTEPS            # load (edge displacement) increment per step (mm)
MAX_STEPS = 0                         # stop after this many steps (0 = run to T_total)
tol_stag, max_stag = 1e-8, 400        # stagger tolerance and maximum iterations per step

# =============================================================================
# SECTION 8: Linear Solvers
# =============================================================================

# Factorizations, the same for every method (u: the CG preconditioner and the exterior A_EE;
# z: the Newton linear solves).  The z Jacobian is SPD for these constants; should it ever
# not be, z falls back to LU for the rest of the run.
U_FACTOR = ("cholesky", "cholmod")
Z_FACTOR = ("cholesky", "cholmod")
U_CG_RTOL  = 1e-12  # u: CG tolerance (relative, preconditioned residual)
U_CG_MAXIT = 200    # u: CG iterations before falling back to a fresh factorization

# =============================================================================
# SECTION 9: I/O & Output Control
# =============================================================================

RESULTS_DIR  = "results"
output_every = 1       # XDMF every n steps (0 = never)
RUN_TAG      = ""      # appended to the output folder name
verbose      = 1       # 0: summary only, 1: one line per step, 2: every stagger iteration
CLOCK        = "wall"  # "wall" | "cpu"

TAG = METHOD + (f"_h{H0_EPS:g}-{HMIN_H0:g}" if (H0_EPS, HMIN_H0) != (1.0, 0.25) else "") + RUN_TAG
output_dir = os.path.join(RESULTS_DIR, TAG)

_QMD = {"quadrature_degree": 2, "quadrature_scheme": "default"}
_USTAT = dict(n_factor=0, n_cg=0)                  # u: factorizations, CG iterations (per run)
WAKE_LAYERS, max_escape_retry, max_rim_moves = 1, 8, 20
ACTIVE_DZ, ACTIVE_T, USE_DELTA, DELTA_PAD_EPS = 0.01, 1e-4, False, 3.0


def clock():
    return time.process_time() if CLOCK == "cpu" else time.perf_counter()

# =============================================================================
# SECTION 10: Constitutive Model
# =============================================================================

def epsilon(v):  return sym(grad(v))
def dgd(z):      return z**2 + eta


def sigma(em):
    """Undegraded plane-strain stress (in-plane components)."""
    return 2.0 * mu * em + lmbda * tr(em) * Identity(2)


def energy(em):
    """Plane-strain elastic energy density (= sigma:em / 2)."""
    return mu * inner(em, em) + 0.5 * lmbda * tr(em)**2


def cell_terms(u):
    """The u-dependent parts of the driving force: (psi1, beta2 svm + beta1 I1, I1 < 0).
    Plane strain: sigma_zz = lambda tr(em) enters I1 and the deviator.
    For P1 u the strain is constant per cell, so these are exactly DG0."""
    em = epsilon(u)
    psi1, sig = energy(em), sigma(em)
    szz = lmbda * tr(em)
    I1 = tr(sig) + szz
    dev = sig - (1 / 3) * I1 * Identity(2)
    svm = sqrt(0.5 * (inner(dev, dev) + (szz - I1 / 3)**2) + 1e-30)
    return psi1, beta2 * svm + beta1 * I1, conditional(lt(I1, 0.0), 1.0, 0.0)


def driving_force(z, p, a, s):
    """Drucker-Prager driving force ce from z and the cell terms."""
    return dgd(z) * a + s * 2.0 * z * p


def z_residual(z, y, p, a, s, dx):
    return (y * 2 * z * p - y * driving_force(z, p, a, s)
            + 3 * delta * Gc / 8 * (-y / eps + 2 * eps * inner(grad(z), grad(y)))) * dx


def u_bilinear(u, z, dx):
    v, du = TestFunction(u.function_space), TrialFunction(u.function_space)
    return inner(dgd(z) * sigma(epsilon(du)), epsilon(v)) * dx


def indicator(z, p, a, s):
    """1 where the refinement flag fires.  Cells with z < 0.6 are gated out (coarsening)."""
    gate = conditional(lt(z, 0.6), 0.0, 1.0)
    return conditional((gate * (2 * p - driving_force(z, p, a, s) / dgd(z)) - G_eps) / G_eps
                       > pf_flag_threshold, 1, 0)


def post_forms(u, z, dx):
    """Elastic energy, z-integral.  (The edge reactions are sums of the residual K(z) u.)"""
    psi_d = dgd(z) * energy(epsilon(u))
    return fem.form(psi_d * dx), fem.form((1 - z) * dx)

# =============================================================================
# SECTION 11: Mesh Generation & Mesh Helpers
# =============================================================================

def edge_dofs(xn):
    """(D, d0): the Dirichlet dofs of the P1 vector space whose nodes are at *xn* (dof 2 n + k):
    u_x on the left/right edges, u_y on the bottom/top edges; d0 = the outward normal
    component (-1 left/bottom, +1 right/top).  The load is d(t) = t d0."""
    D, d0 = [], []
    for k, hi in ((0, W), (1, L)):
        for val, side in ((-1.0, 0.0), (1.0, hi)):
            n = np.where(np.isclose(xn[:, k], side, atol=1e-8))[0]
            D.append(2 * n + k)
            d0.append(np.full(n.size, val))
    D, d0 = np.concatenate(D), np.concatenate(d0)
    o = np.argsort(D)
    return D[o].astype(np.int32), d0[o]


def generate_mesh():
    """Coarse unit cell with the 16 crack slits (diamonds: sharp tips, cw wide at the centre),
    element size h0."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    geo = gmsh.model.geo
    # points: the 4 corners; lines: the bottom, right, top and left edge (the outer loop)
    sq = [geo.addPoint(x, y, 0.0, h0) for x, y in ((0, 0), (W, 0), (W, L), (0, L))]
    loops = [geo.addCurveLoop([geo.addLine(sq[i], sq[(i + 1) % 4]) for i in range(4)])]
    for (cx, cy), th in CRACKS:
        # one slit (a hole): points tip, side, other tip, side (t along the crack, n across);
        # lines: its 4 faces
        c, t, n = np.array([cx, cy]), np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        q = [c + 0.5 * CRACK_LEN * t, c + 0.5 * cw * n, c - 0.5 * CRACK_LEN * t, c - 0.5 * cw * n]
        p = [geo.addPoint(x, y, 0.0, h0) for x, y in q]
        loops.append(geo.addCurveLoop([geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]))
    # surface: the unit cell minus the 16 slits
    geo.addPlaneSurface(loops, 1)
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], tag=100)
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


def tip_nodes(xn):
    """The nodes within h_min / 4 of a crack tip (the crack-tip condition z = 0)."""
    dist, _ = cKDTree(TIPS[:, :2]).query(xn[:, :2], distance_upper_bound=0.25 * h_min)
    return np.where(np.isfinite(dist))[0].astype(np.int32)


def reaction_dofs(xn):
    """(u_x dofs on the right edge, u_y dofs on the top edge) of the P1 vector space whose
    nodes are at *xn*: the edge reactions Fx, Fy are the sums of the residual K(z) u there."""
    return ((2 * np.where(np.isclose(xn[:, 0], W, atol=1e-8))[0]).astype(np.int32),
            (2 * np.where(np.isclose(xn[:, 1], L, atol=1e-8))[0] + 1).astype(np.int32))


def dirichlet_dofs(V):
    """[(k, Vk, dofs)], k = 0, 1: the u_k dofs on the two edges normal to x_k, as pairs of dofs
    in V and in the collapsed Vk.  Located by coordinates, so that a node where a patch touches
    an edge with a vertex only (no facet) is found too."""
    out = []
    for k, hi in ((0, W), (1, L)):
        Vk, _ = V.sub(k).collapse()
        dofs = fem.locate_dofs_geometrical(
            (V.sub(k), Vk), lambda x, k=k, hi=hi: np.isclose(x[k], 0.0, atol=1e-8) | np.isclose(x[k], hi, atol=1e-8))
        out.append((k, Vk, dofs))
    return out

# =============================================================================
# SECTION 12: Solvers on the Solve Mesh (the patch, or the whole mesh for AMRC)
# =============================================================================

class USolver:
    """(K(z) + S) u = g + lifting.  S (the condensed exterior) is written once into a
    template matrix; each solve copies the template and adds the stiffness, so the dense
    block is never re-inserted."""

    def __init__(self, P, S=None, pdG=None):
        self.u, self.z, self.w, self.c2n = P["u"], P["z"], P["w"], P["c2n"]
        self.q, self.q_expr = P["q"], P["q_expr"]
        self.a, self.bcs, self.pdG, self.g = P["a_u"], P["bcs_u"], pdG, None
        sp = fem.create_sparsity_pattern(self.a)
        if pdG is not None:
            nodes = np.unique(pdG // 2).astype(np.int32)
            sp.insert(nodes, nodes)
        sp.finalize()
        self.A = _cpp.la.petsc.create_matrix(comm, sp)
        # touch every pattern slot before the first final assembly, so none is dropped
        fem.petsc.assemble_matrix(self.A, self.a, self.bcs)
        if pdG is not None:
            # a patch with Dirichlet dofs had its bc diagonal INSERTed: flush before ADDing
            self.A.assemble(PETSc.Mat.AssemblyType.FLUSH)
            self.A.setValues(pdG, pdG, np.zeros((pdG.size, pdG.size)),
                             addv=PETSc.InsertMode.ADD_VALUES)
        self.A.assemble()
        self.A0 = None
        if pdG is not None:
            self.A0 = self.A.duplicate(copy=False)
            self.A0.setValues(pdG, pdG, S, addv=PETSc.InsertMode.INSERT_VALUES)
            self.A0.assemble()
        self.b = self.u.x.petsc_vec.duplicate()
        self.ksp = cg_ksp(self.A, U_FACTOR)
        self.refactor, self.t_fresh = True, np.inf  # the first solve factorizes

    def solve(self):
        zc = self.z.x.array[self.c2n]                       # cell average of dgd(z)
        self.w.x.array[:] = eta + (zc.sum(1)**2 + (zc**2).sum(1)) / 12.0
        if self.A0 is None:
            self.A.zeroEntries()
        else:
            self.A0.copy(self.A, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        fem.petsc.assemble_matrix(self.A, self.a, self.bcs)
        self.A.assemble()
        with self.b.localForm() as bl:
            bl.set(0.0)
            if self.g is not None:
                bl.array[self.pdG] = self.g
        if self.bcs:
            fem.petsc.apply_lifting(self.b, [self.a], [self.bcs])
            fem.petsc.set_bc(self.b, self.bcs)
        self._ksp_solve()
        self.q.interpolate(self.q_expr)

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
            dt = time.perf_counter() - t0
            _USTAT["n_cg"] += self.ksp.getIterationNumber()
            if fresh:
                _USTAT["n_factor"] += 1
                self.t_fresh = dt
            if self.ksp.getConvergedReason() > 0:
                self.refactor = (not fresh) and dt > self.t_fresh
                x0.destroy()
                return
        x0.destroy()
        raise RuntimeError(f"u solve failed ({self.ksp.getConvergedReason()})")

    def destroy(self):
        for o in (self.ksp, self.A, self.A0, self.b):
            if o is not None:
                o.destroy()


class ZSolver:
    """The phase-field VI posed on the free dofs alone.  Frozen dofs (patch boundary,
    crack-tip condition) are not constrained inside the system but deleted from it;
    their values enter through z as a coefficient of the residual."""

    def __init__(self, z, zub, free, F, J):
        self.z, self.ub, self.free, self.F, self.J = z, zub, free, F, J
        n = int(free.size)
        self.bF = fem.petsc.create_vector(F)
        self.AF = fem.petsc.create_matrix(J)
        self.isf = PETSc.IS().createGeneral(free, comm=comm)
        self.vec = [PETSc.Vec().createSeq(n, comm=comm) for _ in range(4)]  # b, x, lb, ub
        self._assemble_J()
        self.Ared = self.AF.createSubMatrix(self.isf, self.isf)
        opts = PETSc.Options()
        for k, v in dict(snes_type="vinewtonrsls", snes_linesearch_type="basic",
                         snes_atol=1e-8, snes_rtol=1e-9, snes_stol=1e-9, snes_max_it=10,
                         ksp_type="preonly", pc_type=Z_FACTOR[0],
                         pc_factor_mat_solver_type=Z_FACTOR[1]).items():
            opts[f"z_{k}"] = v
        s = self.snes = PETSc.SNES().create(comm)
        s.setOptionsPrefix("z_")
        s.setFromOptions()
        s.setFunction(self._F, self.vec[0])
        s.setJacobian(self._J, self.Ared)

    def _assemble_J(self):
        self.AF.zeroEntries()
        fem.petsc.assemble_matrix(self.AF, self.J)
        self.AF.assemble()

    def _F(self, snes, x, b):
        self.z.x.array[self.free] = x.getArray(readonly=True)
        with self.bF.localForm() as bl:
            bl.set(0.0)
        fem.petsc.assemble_vector(self.bF, self.F)
        b.getArray()[:] = self.bF.getArray(readonly=True)[self.free]

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

# =============================================================================
# SECTION 13: Coarse (Base) Mesh Data — global and fixed
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
    C["u"] = fem.Function(V, name="displacement")
    C["z"], C["zub"] = fem.Function(Y, name="phasefield"), fem.Function(Y)
    C["z"].x.array[:] = C["zub"].x.array[:] = 1.0
    tip = C["tipn"] = tip_nodes(C["xn"])
    if tip.size != len(TIPS):
        raise RuntimeError(f"{tip.size} crack-tip nodes for {len(TIPS)} tips")
    C["z"].x.array[tip] = 0.0
    C["rx"], C["ry"] = reaction_dofs(C["xn"])
    if not LOCAL:
        return C
    C["D"], C["d0"] = edge_dofs(C["xn"])           # d(t) = t d0
    f2c = msh.topology.connectivity(1, 2)          # interior facets and their two cells
    cnt = np.diff(f2c.offsets)[:msh.topology.index_map(1).size_local]
    fi = np.where(cnt == 2)[0].astype(np.int32)
    C["f2c"] = f2c.array[f2c.offsets[fi][:, None] + np.arange(2)]
    C["fmid"] = compute_midpoints(msh, 1, fi)[:, :2]
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
# SECTION 14: Patch Selection & Exterior Condensation (AMRC-CLIP, AMRC-CLIP-TW)
# =============================================================================

def tip_patch(C, pts):
    """The AMRC-CLIP patch around the active points *pts* (bool mask): B(A, Rp) [u the padded
    Delta cells, USE_DELTA] u N(D n Wp).  AMRC-CLIP and AMRC-CLIP-TW both use it."""
    hit = C["tree"].query_ball_point(pts[:, :2], PATCH_R_EPS * eps)
    cells = mask(C, np.concatenate([np.asarray(h, np.int64) for h in hit] + [np.empty(0, np.int64)]))
    if USE_DELTA:
        de = delta_cells(C)
        if de.any():
            # Delta padded by DELTA_PAD_EPS eps, but only through damaged cells (along the
            # crack, not into intact material), plus one layer so every node is solved
            hit = C["tree"].query_ball_point(C["mid"][de], DELTA_PAD_EPS * eps)
            pad = mask(C, np.concatenate([np.asarray(h, np.int64) for h in hit]))
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
    """Condense the exterior onto the patch boundary: S, T, the factor of A_EE, the rows of
    A_EE^-1 A_EG on the first exterior cell layer and the exterior stiffness A (reactions).
    The patch may reach the loaded edges: the Dirichlet dofs are neither G nor E unknowns;
    the patch keeps those of its nodes as boundary conditions and D (in T) is every
    Dirichlet dof on a node of an exterior cell.  None only if the patch covers every cell."""
    t0 = clock()
    msh, nc, c2n = C["msh"], C["nc"], C["c2n"]
    inP = mask(C, pcells)
    if inP.all():
        return None
    nn = C["xn"].shape[0]
    tp, te = np.zeros(nn, bool), np.zeros(nn, bool)
    tp[c2n[inP].ravel()] = True
    te[c2n[~inP].ravel()] = True
    tags = meshtags(msh, 2, np.arange(nc, dtype=np.int32), np.where(inP, 1, 2).astype(np.int32))
    dxe = Measure("dx", domain=msh, subdomain_data=tags, metadata=_QMD)(2)
    A = fem.petsc.assemble_matrix(fem.form(u_bilinear(C["u"], C["z"], dxe)))
    A.assemble()

    blk = lambda n: (2 * n[:, None] + np.arange(2)).ravel().astype(np.int32)
    isD = np.zeros(2 * nn, bool)
    isD[C["D"]] = True
    ext = te[C["D"] // 2]                          # Dirichlet dofs on nodes of exterior cells
    D, d0 = C["D"][ext], C["d0"][ext]
    gn = np.where(tp & te)[0]                      # interface nodes
    dG = blk(gn)
    dG = dG[~isD[dG]]
    dE = blk(np.where(te & ~tp)[0])
    dE = dE[~isD[dE]]
    iE, iG, iD = (PETSc.IS().createGeneral(d, comm=comm) for d in (dE, dG, D))
    AEE, AEG, AED = (A.createSubMatrix(iE, j) for j in (iE, iG, iD))
    AGE, ADE = A.createSubMatrix(iG, iE), A.createSubMatrix(iD, iE)
    AGG, AGD = dense_block(A, iG, iG), dense_block(A, iG, iD)
    S, T = AGG.copy(), -AGD
    kspE = factor_ksp(AEE, U_FACTOR)
    kspE.setUp()
    F = kspE.getPC().getFactorMatrix()
    nE, nG = dE.size, dG.size
    ring = grow(C, inP) & ~inP                     # first exterior cell layer
    rn = np.unique(c2n[ring].ravel())
    rdofs = blk(rn[~tp[rn]])
    rdofs = rdofs[~isD[rdofs]]
    ridx = np.searchsorted(dE, rdofs)
    Yr = np.zeros((rdofs.size, nG))                # rows of A_EE^-1 A_EG on the ring
    isR = PETSc.IS().createStride(nE, first=0, step=1, comm=comm)
    for j0 in range(0, nG, 128):                   # Y = A_EE^-1 A_EG, in column chunks
        w = min(128, nG - j0)
        isC = PETSc.IS().createStride(w, first=j0, step=1, comm=comm)
        B = AEG.createSubMatrix(isR, isC).convert("dense")
        Xc = PETSc.Mat().createDense([nE, w], comm=comm)
        Xc.setUp()
        F.matSolve(B, Xc)
        Yr[:, j0:j0 + w] = Xc.getDenseArray()[ridx, :]
        GX, DX = AGE.matMult(Xc), ADE.matMult(Xc)
        S[:, j0:j0 + w] -= GX.getDenseArray()
        T[j0:j0 + w, :] += DX.getDenseArray().T
        for o in (isC, B, Xc, GX, DX):
            o.destroy()
    S = 0.5 * (S + S.T)

    # Self-check of S and T on random vectors (two extra solves): condensation errors
    # would otherwise show up only as wrong edge reactions Fx, Fy.
    rng = np.random.default_rng(0)
    ve, tg = AEE.createVecLeft(), AGE.createVecLeft()
    errs = []
    for M, v, dense, lin in ((AEG, rng.standard_normal(nG), AGG, S),
                             (AED, rng.standard_normal(D.size), -AGD, T)):
        vv = M.createVecLeft()
        M.mult(PETSc.Vec().createWithArray(v, comm=comm), vv)
        kspE.solve(vv, ve)
        AGE.mult(ve, tg)
        ref = dense @ v - tg.getArray() if lin is S else tg.getArray() - AGD @ v
        errs.append(np.linalg.norm(lin @ v - ref) / max(np.linalg.norm(ref), 1e-300))
        vv.destroy()

    X = dict(dG=dG, dE=dE, D=D, d0=d0, S=S, T=T, A=A, AEG=AEG, AED=AED, kspE=kspE,
             xE=AEE.createVecLeft(), yE=AEE.createVecLeft(), xG=C["xn"][dG // 2, :2],
             xGn=C["xn"][gn, :2], nGn=gn.size,
             gmid=C["fmid"][inP[C["f2c"][:, 0]] != inP[C["f2c"][:, 1]]],
             ecells=np.where(~inP)[0].astype(np.int32), enodes=np.where(te)[0],
             ring=np.where(ring)[0].astype(np.int32), rdofs=rdofs, ridx=ridx, Yr=Yr,
             vr=np.zeros(rdofs.size),
             post=post_forms(C["u"], C["z"], dxe), err=max(errs), t=clock() - t0)
    for o in (AEE, AGE, ADE, iE, iG, iD, isR, ve, tg):
        o.destroy()
    return X


def destroy_exterior(X):
    for k in ("A", "AEG", "AED", "kspE", "xE", "yE"):
        X[k].destroy()


def recover(C, P, X, t):
    """u on the exterior from u on the interface: A_EE u_E = -A_EG u_G - A_ED d."""
    uG = P["u"].x.array[P["pdG"]]
    d = t * X["d0"]
    rhs, tmp = X["xE"], X["AEG"].createVecLeft()
    X["AEG"].mult(PETSc.Vec().createWithArray(uG, comm=comm), rhs)
    X["AED"].mult(PETSc.Vec().createWithArray(d, comm=comm), tmp)
    rhs.axpy(1.0, tmp)
    rhs.scale(-1.0)
    X["kspE"].solve(rhs, X["yE"])
    tmp.destroy()
    uc = C["u"].x.array
    uc[X["dE"]] = X["yE"].getArray(readonly=True)
    uc[X["dG"]] = uG
    uc[X["D"]] = d


def lookahead(C, P, X):
    """Flagged cells of the first exterior layer, from the exact exterior u there:
    u_E = -(A_EE^-1 A_EG u_G + A_EE^-1 A_ED d)."""
    uG = P["u"].x.array[P["pdG"]]
    C["u"].x.array[X["dG"]] = uG
    C["u"].x.array[X["rdofs"]] = -(X["Yr"] @ uG + X["vr"])
    C["ind"].interpolate(C["ind_expr"], X["ring"])
    return X["ring"][C["ind"].x.array[X["ring"]] > 0.5]

# =============================================================================
# SECTION 15: The Solve Mesh (the patch, or the whole domain for AMRC)
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
    u, z = P["u"], P["z"] = fem.Function(V, name="displacement"), fem.Function(Y, name="phasefield")
    P["zub"], P["zd"] = fem.Function(Y), fem.Function(Y)
    xn = Y.tabulate_dof_coordinates()
    c2n = P["c2n"] = Y.dofmap.list                 # (DG0 dofs are in cell order in serial)

    # u: the edge Dirichlet dofs that fall inside the solve mesh (all of them for AMRC), and
    # the dofs whose residual sums to the edge reactions
    P["r"], P["bcs_u"] = [], []
    for k, Vk, dofs in dirichlet_dofs(V):
        if dofs[0].size:
            P["r"].append((k, fem.Function(Vk)))
            P["bcs_u"].append(fem.dirichletbc(P["r"][-1][1], dofs, V.sub(k)))
    P["rx"], P["ry"] = reaction_dofs(xn)

    # z: crack-tip condition, and the frozen patch boundary: every interface node (the
    # interface dofs G leave out the Dirichlet dofs, so a corner node may have none)
    P["tip"] = tip_nodes(xn)
    frozen, P["pdG"] = P["tip"], None
    if X is not None:
        tree = cKDTree(xn[:, :2])
        dist, pg = tree.query(X["xG"], distance_upper_bound=1e-8 * W)
        assert np.isfinite(dist).all(), "interface node missing from the patch mesh"
        P["pdG"] = (2 * pg + X["dG"] % 2).astype(np.int32)
        dist, pn = tree.query(X["xGn"], distance_upper_bound=1e-8 * W)
        assert np.isfinite(dist).all(), "interface node missing from the patch mesh"
        P["gtree"] = cKDTree(X["xGn"])
        frozen = np.union1d(frozen, pn)
        # rim: free nodes sharing a cell with the frozen boundary.  z is P1, so damage
        # cannot reach the boundary without first showing up here.
        on = np.zeros(xn.shape[0], bool)
        on[pn] = True
        rim = np.unique(c2n[on[c2n].any(axis=1)])
        P["rim"] = rim[~on[rim]]
    P["free"] = np.setdiff1d(np.arange(xn.shape[0], dtype=np.int32), frozen).astype(np.int32)

    # Forms.  Per stagger iteration the u-dependent cell terms are evaluated once (DG0,
    # exact for P1 u) and the stiffness weight is the cell average of dgd(z) (what
    # degree-2 quadrature integrates exactly), so no kernel inside the Newton loop
    # recomputes stresses.
    dx_ = Measure("dx", domain=msh, metadata=_QMD)
    ip = Yv.element.interpolation_points()
    P["w"] = fem.Function(Yv)
    P["a_u"] = fem.form(inner(P["w"] * sigma(epsilon(TrialFunction(V))), epsilon(TestFunction(V)))
                        * Measure("dx", domain=msh, metadata={"quadrature_degree": 0}))
    P["q"] = fem.Function(fem.functionspace(msh, ("DG", 0, (3,))))
    P["q_expr"] = fem.Expression(ufl.as_vector(cell_terms(u)), ip)
    p, a, s = P["q"][0], P["q"][1], P["q"][2]
    R = z_residual(z, TestFunction(Y), p, a, s, dx_)
    P["ind_expr"] = fem.Expression(indicator(z, p, a, s), ip)
    P["ind"] = fem.Function(Yv, name="pf_flag")
    P["h"] = fem.Function(Yv)
    P["h"].interpolate(fem.Expression(CellDiameter(msh), ip))
    P["f_dz"] = fem.form(P["zd"] * P["zd"] * dx_)
    P["post"] = post_forms(u, z, dx_)
    P["f_react"] = fem.form(inner(P["w"] * sigma(epsilon(u)), epsilon(TestFunction(V)))
                            * Measure("dx", domain=msh, metadata={"quadrature_degree": 0}))

    if LOCAL and RETIRE_STEPS:                     # cells refined in this solve mesh: refined now
        C.setdefault("refage", np.full(C["nc"], 10**9))[np.bincount(P["cmap"], minlength=C["nc"]) > 1] = 0
    P["us"] = USolver(P, X["S"] if X else None, P["pdG"])
    P["zs"] = ZSolver(z, P["zub"], P["free"], fem.form(R),
                      fem.form(derivative(R, z, TrialFunction(Y))))
    return P


def destroy_patch(P):
    P["us"].destroy()
    P["zs"].destroy()
    P.clear()                                      # drop the refined mesh now, not at GC


def transfer(P_new, P_old, cells=None):
    """Interpolate u, z, zub from P_old onto P_new (on *cells* of P_new, default all)."""
    if cells is None:
        cells = np.arange(ncells(P_new["msh"]), dtype=np.int32)
    if cells.size == 0:
        return
    for f, V in (("u", "V"), ("z", "Y"), ("zub", "Y")):
        idata = fem.create_interpolation_data(P_new[V], P_old[V], cells, padding=1e-8)
        tmp = fem.Function(P_new[V])
        tmp.interpolate_nonmatching(P_old[f], cells, idata)
        bs = P_new[V].dofmap.index_map_bs
        dofs = np.unique(P_new[V].dofmap.list[cells].ravel())
        dofs = (bs * dofs[:, None] + np.arange(bs)).ravel()
        P_new[f].x.array[dofs] = tmp.x.array[dofs]
    P_new["z"].x.array[P_new["tip"]] = 0.0         # interpolation is only approximate there


def from_coarse(C, P):
    """Fill P's u, z, zub from the coarse fields (coarse -> base submesh by cell map,
    base -> refined patch by interpolation)."""
    base = P["base"]
    Q = dict(V=fem.functionspace(base, ("Lagrange", 1, (2,))), Y=fem.functionspace(base, ("Lagrange", 1)))
    for f, sp in (("u", "V"), ("z", "Y"), ("zub", "Y")):
        Q[f] = fem.Function(Q[sp])
        Q[f].interpolate(C[f], cells0=P["bmap"], cells1=np.arange(ncells(base), dtype=np.int32))
    transfer(P, Q)


def coarse_nodes(C, P):
    """(patch dofs, coarse nodes) of the patch nodes that are coarse nodes."""
    dist, cn = C["ntree"].query(P["Y"].tabulate_dof_coordinates()[:, :2],
                                distance_upper_bound=1e-8 * W)
    pn = np.where(np.isfinite(dist))[0]
    return pn, cn[pn]


def to_coarse(C, P):
    """Write the patch values at coarse nodes back into the coarse fields."""
    pn, cn = coarse_nodes(C, P)
    C["z"].x.array[cn] = P["z"].x.array[pn]
    C["zub"].x.array[cn] = P["zub"].x.array[pn]
    for k in range(2):
        C["u"].x.array[2 * cn + k] = P["u"].x.array[2 * pn + k]


def coarse_z(C, P):
    """z at every coarse node, taking the patch value where the patch has that node."""
    z = C["z"].x.array.copy()
    pn, cn = coarse_nodes(C, P)
    z[cn] = P["z"].x.array[pn]
    return z


def new_patch(C, P_old, X_old, pts, t, refine_pts=None):
    """(Re)place the patch around *pts*, refine it near *refine_pts*, condense the new
    exterior and carry the fields over.  O(N) -- only when the patch moves.  Returns
    (P, X); X is None if the patch had to become the whole domain."""
    if P_old is not None:
        recover(C, P_old, X_old, t)
        to_coarse(C, P_old)
    pcells, base, bmap, msh, parent = place_patch(C, select_patch(C, pts), refine_pts)
    if P_old is not None and METHOD == "amrc_clip":
        relax_wake(C, pcells)
    X = build_exterior(C, pcells)
    if X is None:                                  # the patch is every cell: go whole-domain
        print("    [patch] covers the whole domain -- whole-domain solve from here",
              flush=True)
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
    free = np.setdiff1d(free, C["tipn"])          # the crack-tip condition stays
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
    """Load for this time: Dirichlet values on the solve mesh (u_k = t (2 x_k / W_k - 1), i.e.
    -t / +t on the two edges normal to x_k), the condensed load, and the Dirichlet part of
    the exterior u on the first exterior layer."""
    for k, r in P["r"]:
        r.interpolate(lambda x, k=k: t * (2.0 * x[k] / (W, L)[k] - 1.0))
    if X is not None:
        d = t * X["d0"]
        P["us"].g = X["T"] @ d
        X["AED"].mult(PETSc.Vec().createWithArray(d, comm=comm), X["xE"])
        X["kspE"].solve(X["xE"], X["yE"])
        X["vr"] = X["yE"].getArray(readonly=True)[X["ridx"]].copy()
        C["u"].x.array[X["D"]] = d


def irreversibility(C, P, X):
    for z, ub in ((P["z"], P["zub"]),) + (((C["z"], C["zub"]),) if X is not None else ()):
        m = z.x.array <= v_th
        ub.x.array[m] = z.x.array[m]

# =============================================================================
# SECTION 16: Output
# =============================================================================

class Output:
    """xdmf/patch_N.xdmf: the solve mesh with u, z and the flags; xdmf/exterior_N.xdmf: the
    exterior coarse cells with u, z (AMRC-CLIP, AMRC-CLIP-TW).
    xdmf/coarse.xdmf: a time series on the fixed coarse mesh with u, z at the coarse nodes
    and the regions as DG0 fields (1 inside, 0 outside):
      solved   the coarse cells of the patch (all cells for AMRC)
      refined  the coarse cells that are refined
      flagged  the coarse cells holding a flagged cell (A)
      delta    AMRC-CLIP: the cells whose z is still dropping (Delta)
      damaged  AMRC-CLIP-TW: the cells with a node z < DAMAGE_Z (D)"""

    def __init__(self, C):
        self.dir = os.path.join(output_dir, "xdmf")
        os.makedirs(self.dir, exist_ok=True)
        C["msh"].name = "coarse"
        self.xf = XDMFFile(comm, os.path.join(self.dir, "coarse.xdmf"), "w")
        self.xf.write_mesh(C["msh"])
        names = ["solved", "refined", "flagged"] \
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
                      flagged=mask(C, cmap[P["ind"].x.array > 0.5]))
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
# SECTION 17: Print Summary
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
    _box("PHASE-FIELD FRACTURE — PARAMETER SUMMARY", [
        f"Problem   : percolation ({len(CRACKS)} cracks in a unit cell), plane strain",
        f"Method    : {METHOD_NAME}",
        f"Geometry  : W = {W:g} mm   L = {L:g} mm   {len(CRACKS)} cracks of {CRACK_LEN:g} mm",
        f"Elastic   : E = {E:.4e}   nu = {nu:.4f}",
        f"Strength  : sts = {sts:.4e}   scs = {scs:.4e}   shs = {shs:.4e}",
        f"Fracture  : Gc = {Gc:.6f}   eps = {eps:.4e}   delta = {delta:.4e}",
        f"AMR       : h0 = {h0:.4g}   h_min = {h_min:.4g}   R = {amr_refine_radius:.4g}"
        + ("   COARSEN_IDLE" if COARSEN_IDLE else ""),
        (f"Patch     : Rp = {PATCH_R_EPS:g} eps   retire = {RETIRE_STEPS} refinement"
         + (f"   wake layers = {WAKE_LAYERS}" if METHOD == "amrc_clip_tw" else "")
         if LOCAL else "Patch     : none (AMRC: the whole mesh is solved)"),
        f"Loading   : {nsteps} steps   du = {stepsize:.4g} mm   to U_MAX = {U_MAX:g} mm",
        f"Stagger   : tol = {tol_stag:.1e}   max_stag = {max_stag}",
        f"Solvers   : u {'/'.join(U_FACTOR)} (CG)   z {'/'.join(Z_FACTOR)}",
        f"Output    : {output_dir}",
    ])

# =============================================================================
# SECTION 18: Main Solve Loop
# =============================================================================

def main():
    t_start = clock()
    say = print if verbose > 1 else (lambda *a, **k: None)
    print_summary()
    C = coarse_setup(generate_mesh())
    os.makedirs(output_dir, exist_ok=True)
    _USTAT.update(n_factor=0, n_cg=0)
    TT = dict(u=0.0, z=0.0, amr=0.0, ext=0.0, post=0.0, io=0.0, n_u=0, n_z=0, n_amr=0,
              n_move=0, n_lookahead=0, n_escape=0, n_rim=0)
    nonconverged, unresolved = [], []

    # --- Initial solve mesh: the patch around every crack tip, or the whole domain ---
    tip0 = TIPS.copy()
    if LOCAL:
        P, X = new_patch(C, None, None, tip0, 0.0)
        TT["ext"] += X["t"] if X else 0.0
    else:
        X = None
        cells = np.arange(C["nc"], dtype=np.int32)
        P = build_patch(C, C["msh"], cells, C["msh"], cells, None)
        from_coarse(C, P)
    out = Output(C) if output_every else None
    log_file, timing_file = f"{output_dir}/log.txt", f"{output_dir}/timing.csv"
    with open(log_file, "w") as f:
        f.write("step time Fx Fy z_integral elastic_energy min_z iterations converged dz_last "
                "wall_step wall_total n_nodes n_solve_nodes\n")
    with open(timing_file, "w") as f:
        f.write("step,t,iterations,converged,n_cells,n_solve_cells,n_G,n_amr,n_move,"
                "t_u,t_z,t_amr,t_post,t_io,t_step,t_cum,n_nodes,n_solve_nodes\n")

    def seeds(P):
        return P["pts"] if P["pts"] is not None else tip0

    def move(P, X, pts, refine_pts, t):
        """Re-select the patch around *pts* and refine it near *refine_pts*."""
        t0 = clock()
        Pn, Xn = new_patch(C, P, X, pts, t, refine_pts)
        TT["n_move"] += 1
        if Xn is not None:
            TT["ext"] += Xn["t"]
            say(f"    [patch] moved: {ncells(Pn['msh']):,} cells, interface {Xn['dG'].size} "
                f"dofs, exterior {Xn['dE'].size:,} dofs ({clock() - t0:.2f} s, check {Xn['err']:.1e})")
        return Pn, Xn

    def remesh(P, X, pts, t):
        """Re-refine the solve mesh around *pts*; move the patch instead if a flagged point
        lies within R of the patch boundary or the refinement would split it."""
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
        return move(P, X, pts, pts, t)

    last = min(nsteps, MAX_STEPS) if MAX_STEPS else nsteps
    for step in range(1, last + 1):
        t = step * stepsize
        ts = clock()
        su = sz = sa = sp = 0.0
        n_amr0, n_move0 = TT["n_amr"], TT["n_move"]
        say(f"\n-- step {step}  t = {t:.6e}")

        # --- A: Loading and irreversibility bounds ---
        set_step_data(C, P, X, t)
        irreversibility(C, P, X)                   # once per step, from the last step's z
        if (METHOD == "amrc_clip" or USE_DELTA) and X is not None:   # coarse-node z at the step start (Delta)
            C["zhist"] = (C.get("zhist", []) + [coarse_z(C, P)])[-max(1, round(ACTIVE_T / stepsize)):]
            C["zref"] = C["zhist"][0]

        # --- B: Staggered u-z solve (with mid-stagger AMR and patch moves) ---
        it, zres, esc, coarsened = 0, np.inf, 0, False
        while True:
            while it < max_stag and zres > tol_stag:
                it += 1
                tk = clock()
                P["us"].solve()
                su += clock() - tk
                TT["n_u"] += 1

                # refinement: flags on the solve mesh and on the first exterior layer
                tk = clock()
                P["ind"].interpolate(P["ind_expr"])
                hot = P["ind"].x.array > 0.5
                ring = lookahead(C, P, X) if X is not None else np.empty(0, np.int32)
                # COARSEN_IDLE: nothing flagged, but the solve mesh still refined -- one
                # refinement event around no cells (once per step): coarse everywhere
                idle = (COARSEN_IDLE and not coarsened and not ring.size and not hot.any()
                        and (np.bincount(P["cmap"], minlength=C["nc"]) > 1).any())
                if ring.size or (hot & (P["h"].x.array > 1.5 * h_min)).any() or idle:
                    if idle:
                        coarsened = True
                        print(f"    step {step}: nothing flagged -- solve mesh coarsened", flush=True)
                    if LOCAL and RETIRE_STEPS and "refage" in C:
                        # a refinement event: one more for the cells not refined now
                        C["refage"][np.bincount(P["cmap"], minlength=C["nc"]) <= 1] += 1
                    pts = midpoints(P["msh"], np.where(hot)[0])
                    if ring.size:                  # flagged just outside: move over them
                        TT["n_lookahead"] += 1
                        pts = np.vstack([pts, midpoints(C["msh"], ring)])
                        P, X = move(P, X, pts, pts, t)
                    else:
                        P, X = remesh(P, X, pts, t)
                    set_step_data(C, P, X, t)
                    P["q"].interpolate(P["q_expr"])
                    TT["n_amr"] += 1
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
                    hits = np.vstack([hits, P["Y"].tabulate_dof_coordinates()[hit]])
                    P["z"].x.array[:] = zprev
                    P, X = move(P, X, np.vstack([seeds(P), hits]), P["pts"], t)
                    set_step_data(C, P, X, t)
                    P["q"].interpolate(P["q_expr"])
                    zprev = P["z"].x.array.copy()
                    TT["n_rim"] += 1
                    sa += clock() - tk
                    tk = clock()
                    P["zs"].solve()
                    TT["n_z"] += 1
                    sz += clock() - tk
                P["zd"].x.array[:] = P["z"].x.array - zprev
                zres = np.sqrt(fem.assemble_scalar(P["f_dz"]))
                say(f"    iter {it:>3d}  |dz| = {zres:.7e}")

            # u for the final z (also when the iterations stopped at max_stag)
            tk = clock()
            P["us"].solve()
            su += clock() - tk
            TT["n_u"] += 1
            if X is None:
                break
            # nothing outside the patch may be flagged (exterior u recovered exactly)
            tk = clock()
            recover(C, P, X, t)
            C["ind"].interpolate(C["ind_expr"], X["ecells"])
            outside = X["ecells"][C["ind"].x.array[X["ecells"]] > 0.5]
            sp += clock() - tk
            if outside.size == 0:
                break
            esc += 1
            TT["n_escape"] += 1
            print(f"    step {step}: {outside.size} flagged cells outside the patch -- absorbing",
                  flush=True)
            tk = clock()
            P, X = move(P, X, np.vstack([seeds(P), midpoints(C["msh"], outside)]), P["pts"], t)
            set_step_data(C, P, X, t)
            sa += clock() - tk
            if esc > max_escape_retry or it >= max_stag:
                unresolved.append(step)
                tk = clock()
                P["us"].solve()
                su += clock() - tk
                TT["n_u"] += 1
                if X is not None:
                    recover(C, P, X, t)
                break
            zres = np.inf                          # iterate again on the enlarged patch
        converged = zres <= tol_stag
        if not converged:
            nonconverged.append(step)

        # --- C: refinement events since each coarse cell was last refined (counted at the
        #     refinement in the stagger loop; reset where the solve mesh refines the cell) ---
        if LOCAL and RETIRE_STEPS and "refage" in C:
            C["refage"][np.bincount(P["cmap"], minlength=C["nc"]) > 1] = 0

        # --- D: Post-processing (u is consistent with z; the exterior u is recovered): edge
        #     reactions Fx (right) and Fy (top) as sums of the residual K(z) u, energy, damage ---
        tk = clock()
        Rv = fem.petsc.assemble_vector(P["f_react"])
        Ra = Rv.getArray(readonly=True)
        Fx, Fy = Ra[P["rx"]].sum(), Ra[P["ry"]].sum()
        Rv.destroy()
        Eel, Zint = (fem.assemble_scalar(f) for f in P["post"])
        min_z = P["z"].x.array.min()
        n_cells = ncells(P["msh"])
        n_nodes = n_solve_nodes = nnodes(P["msh"])
        if X is not None:
            uv = PETSc.Vec().createWithArray(C["u"].x.array.copy(), comm=comm)
            Ru = X["A"].createVecLeft()
            # exterior cells only (patch-interior columns are 0; the interface u is from recover())
            X["A"].mult(uv, Ru)
            Rc = Ru.getArray(readonly=True)
            Fx, Fy = Fx + Rc[C["rx"]].sum(), Fy + Rc[C["ry"]].sum()
            Ru.destroy()
            uv.destroy()
            Eel, Zint = (v + fem.assemble_scalar(f) for v, f in zip((Eel, Zint), X["post"]))
            min_z = min(min_z, C["z"].x.array[X["enodes"]].min())
            n_cells += X["ecells"].size
            n_nodes += X["enodes"].size - X["nGn"]    # the interface nodes are patch nodes
        sp += clock() - tk

        # --- E: Output ---
        tk = clock()
        if out is not None and step % output_every == 0:
            out.write(C, P, X, step, t)
        tio = clock() - tk

        for k, v in (("u", su), ("z", sz), ("amr", sa), ("post", sp), ("io", tio)):
            TT[k] += v
        dt_step, t_cum = clock() - ts, clock() - t_start
        with open(log_file, "a") as f:
            f.write(f"{step} {t:.6e} {Fx:.8e} {Fy:.8e} {Zint:.8e} {Eel:.8e} {min_z:.6f} {it} "
                    f"{int(converged)} {zres:.3e} {dt_step:.3f} {t_cum:.3f} {n_nodes} {n_solve_nodes}\n")
        with open(timing_file, "a") as f:
            f.write(f"{step},{t:.8e},{it},{int(converged)},{n_cells},{ncells(P['msh'])},"
                    f"{X['dG'].size if X else 0},{TT['n_amr'] - n_amr0},{TT['n_move'] - n_move0},"
                    f"{su:.6f},{sz:.6f},{sa:.6f},{sp:.6f},{tio:.6f},{dt_step:.6f},"
                    f"{t_cum:.3f},{n_nodes},{n_solve_nodes}\n")
        if verbose:
            print(f"  step {step:>4d}  u {t:.3e}  iters {it:>3d}{'' if converged else ' (max)'}  "
                  f"Fx {Fx:.3f}  Fy {Fy:.3f}  "
                  f"nodes {n_solve_nodes:,} solved / {n_nodes:,}  "
                  f"| u {su:.2f}s  z {sz:.2f}s  amr {sa:.2f}s  post {sp:.2f}s  io {tio:.2f}s  "
                  f"| step {dt_step:.2f}s  total {t_cum:.1f}s", flush=True)

    # --- Finalisation ---
    if out is not None:
        out.close()
    elapsed = clock() - t_start
    _box("SIMULATION COMPLETE", [
        f"Method          : {METHOD_NAME}",
        f"Total time      : {elapsed:.1f} s ({CLOCK} clock)",
        f"Steps completed : {last}   ({len(nonconverged)} not converged in {max_stag} iterations)",
    ] + [f"  {lab:<14}: {TT[k]:9.1f} s ({100 * TT[k] / max(elapsed, 1e-9):5.1f} %)"
         for k, lab in (("u", "displacement"), ("z", "phase field"), ("amr", "AMR + patch"),
                        ("post", "post/check"), ("io", "XDMF output"))] + [
        f"Solves          : u {TT['n_u']:,}   z {TT['n_z']:,}",
        f"Refinements     : {TT['n_amr']:,}",
    ] + ([
        f"Patch moves     : {TT['n_move']:,}   ({TT['ext']:.1f} s condensing)",
        f"Escapes         : {TT['n_escape']}   rim retries {TT['n_rim']}",
    ] if LOCAL else []) + [
        f"u solver        : {_USTAT['n_factor']:,} factorizations, {_USTAT['n_cg']:,} CG iterations",
    ] + ([f"UNRESOLVED steps: {unresolved}"] if unresolved else []))
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(dict(method=METHOD, problem="percolation", steps=last, elapsed=elapsed, clock=CLOCK,
                       E=E, nu=nu, Gc=Gc, eps=eps, sts=sts, scs=scs, crack_len=CRACK_LEN,
                       crack_angles=CRACK_ANGLES, u_max=U_MAX, nsteps=nsteps,
                       h0=h0, h_min=h_min, W=W, L=L, max_stag=max_stag, tol_stag=tol_stag,
                       active_dz=ACTIVE_DZ, active_t=ACTIVE_T, damage_z=DAMAGE_Z,
                       patch_r_eps=PATCH_R_EPS, delta_pad_eps=DELTA_PAD_EPS, use_delta=USE_DELTA,
                       retire_steps=RETIRE_STEPS, coarsen_idle=COARSEN_IDLE,
                       wake_layers=WAKE_LAYERS, z_ub_margin=0.0,
                       nonconverged_steps=nonconverged, unresolved_steps=unresolved,
                       u_factor=U_FACTOR, z_factor=Z_FACTOR, u_cg_rtol=U_CG_RTOL,
                       u_refactors=_USTAT["n_factor"], u_cg_iterations=_USTAT["n_cg"], **TT), f, indent=2)
    return elapsed


if __name__ == "__main__":
    main()
