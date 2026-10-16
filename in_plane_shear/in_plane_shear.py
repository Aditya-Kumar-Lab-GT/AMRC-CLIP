#!/usr/bin/env python3
"""
Phase-Field Fracture with AMRC and Condensation of Linear Inactive Parts (CLIP)
===============================================================================
Three-dimensional crack growth under in-plane shear (paper Section 4.3).
Plate [0, W] × [0, L] × [0, Lz] (25 × 25 × 3 mm) with an edge crack from the left face to
(ac, L/2), ac = 12.5 mm, through the whole thickness (the paper's top-face displacement u_x
is U_MAX t here).

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

Physics
-------
- 3D linear elasticity, quasi-static mechanics
- Complete phase-field fracture with Drucker-Prager strength surface
- u = 0 on the bottom face (y = 0); u = (U_MAX t, 0, 0) on the top face (y = L), t in
  [0, 1]; the other faces and the crack faces are traction-free; the crack is a geometric
  slit (no z = 0 condition)
- The crack front is loaded in in-plane shear; the crack kinks and grows downward

Units: consistent (mm, N, MPa); t in [0, 1] is the load parameter
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
                          locate_entities_boundary, meshtags, refine, RefinementOption)
import ufl
from ufl import (MinCellEdgeLength, Identity, Measure, TestFunction, TrialFunction,
                 conditional, derivative, grad, inner, lt, sqrt, sym, tr)

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

W  = 25.0          # plate width  (mm), x in [0, W]
L  = 25.0          # plate height (mm), y in [0, L]
Lz = 3.0           # plate thickness (mm), z in [0, Lz]
ac = 12.5          # initial crack length (mm), from the left face; crack plane y = L/2
cw = 0.05          # crack mouth width on the left face (mm)

# =============================================================================
# SECTION 3: Material Properties
# =============================================================================

# --- Elastic constants ---
E, nu = 9800.0, 0.3                            # Young's modulus (MPa), Poisson's ratio
mu    = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
kappa = E / (3 * (1 - 2 * nu))

# --- Strength surface (Drucker-Prager) ---
sts, scs = 27.0, 77.0                          # uniaxial tensile / compressive strength (MPa)
shs = (2 / 3) * sts * scs / (scs - sts)        # hydrostatic strength
Wts = sts**2 / (2 * E)
Whs = shs**2 / (2 * kappa)

# --- Fracture parameters ---
Gc  = 0.091125     # critical energy release rate (N/mm)
eps = 0.3          # regularisation length (mm)
eta = 1e-5         # residual stiffness

# =============================================================================
# SECTION 4: AMR Parameters
# =============================================================================

H0_EPS  = 1.0                  # coarse element size / eps
HMIN_H0 = 0.25                 # refined element size / h0
h0      = H0_EPS * eps         # coarse (base) mesh element size
h_min   = HMIN_H0 * h0         # element size after full refinement
amr_refine_radius = 3 * eps    # refinement radius R around the flagged cells
pf_flag_threshold = -0.1       # indicator threshold
H_COARSE = 1.8                 # flagged cell still coarse: shortest edge > H_COARSE * h_min

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


ZONE_EPS = 5.0     # frozen zone y > L/2 + ZONE_EPS * eps (None: no zone)
ZONE_Y   = (L / 2 + ZONE_EPS * eps) if ZONE_EPS else 1e30

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

U_MAX     = 0.08                      # top-face u_x at t = 1 (mm): u = (U_MAX t, 0, 0)
NSTEPS    = 400                       # number of load steps (u_x increment U_MAX / NSTEPS)
stepsize  = 1.0 / NSTEPS              # load (t) increment per step
MAX_STEPS = 0                         # stop after this many steps (0 = all NSTEPS)
tol_stag, max_stag = 1e-8, 10         # stagger tolerance; at most max_stag - 1 iterations/step

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
U_SKIP_ATOL = 1e-8  # u: skip the solve when the residual of the current u is below this

# =============================================================================
# SECTION 9: I/O & Output Control
# =============================================================================

RESULTS_DIR  = "results"
output_every = 8       # XDMF every n steps (0 = never; 50 files over 400 steps)
RUN_TAG      = ""      # appended to the output folder name
verbose      = 1       # 0: summary only, 1: one line per step, 2: every stagger iteration
CLOCK        = "wall"  # "wall" | "cpu"

TAG = METHOD + (f"_h{H0_EPS:g}-{HMIN_H0:g}" if (H0_EPS, HMIN_H0) != (1.0, 0.25) else "") + RUN_TAG
output_dir = os.path.join(RESULTS_DIR, TAG)

_QMD = {"quadrature_degree": 2, "quadrature_scheme": "default"}
_USTAT = dict(n_factor=0, n_cg=0)                  # u: factorizations, CG iterations (per run)
WAKE_LAYERS, max_escape_retry, max_rim_moves = 1, 8, 20
ACTIVE_DZ, ACTIVE_T, USE_DELTA, DELTA_PAD_EPS = 0.01, 2.0 / NSTEPS, False, 3.0
STOP_ON_DIRICHLET = True


def clock():
    return time.process_time() if CLOCK == "cpu" else time.perf_counter()

# =============================================================================
# SECTION 10: Constitutive Model
# =============================================================================

def epsilon(v):  return sym(grad(v))
def dgd(z):      return z**2 + eta


def sigma(em):
    """Undegraded 3D stress."""
    return 2.0 * mu * em + lmbda * tr(em) * Identity(3)


def energy(em):
    """Elastic energy density (= sigma:em / 2)."""
    return mu * inner(em, em) + 0.5 * lmbda * tr(em)**2


def cell_terms(u):
    """The u-dependent parts of the driving force: (psi1, beta2 sqrt(J2) + beta1 I1, I1 < 0).
    For P1 u the strain is constant per cell, so these are exactly DG0."""
    em = epsilon(u)
    psi1, sig = energy(em), sigma(em)
    I1 = tr(sig)
    dev = sig - (1 / 3) * I1 * Identity(3)
    sj2 = sqrt(0.5 * inner(dev, dev) + 1e-30)
    return psi1, beta2 * sj2 + beta1 * I1, conditional(lt(I1, 0.0), 1.0, 0.0)


def driving_force(z, p, a, s):
    """Drucker-Prager driving force ce from z and the cell terms."""
    return dgd(z) * a + s * 2.0 * z * p


def z_residual(z, y, p, a, s, dx):
    return (y * 2 * z * p - y * driving_force(z, p, a, s)
            + 3 * delta * Gc / 8 * (-y / eps + 2 * eps * inner(grad(z), grad(y)))) * dx


def indicator(z, p, a, s):
    """1 where the refinement flag fires.  Cells with z < 0.6 are gated out (coarsening)."""
    gate = conditional(lt(z, 0.6), 0.0, 1.0)
    flag = conditional((gate * (2 * p - driving_force(z, p, a, s) / dgd(z)) - G_eps) / G_eps
                       > pf_flag_threshold, 1, 0)
    y = ufl.SpatialCoordinate(z.function_space.mesh)[1]
    return conditional(ufl.gt(y, ZONE_Y), 0, flag)   # frozen zone


def k_form(u, v, z, dx):
    return inner(dgd(z) * sigma(epsilon(u)), epsilon(v)) * dx


def top_disp(t):
    return U_MAX * t

# =============================================================================
# SECTION 11: Mesh Generation & Mesh Helpers
# =============================================================================

def generate_mesh():
    """Plate with an edge crack from the left face to (ac, L/2), extruded through Lz."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    geo = gmsh.model.geo
    # points (on the face z = 0): the 4 corners, the upper / lower crack mouth, the crack tip
    for i, (x, y) in enumerate([(0, L), (W, L), (W, 0), (0, 0), (0, L / 2 + cw / 2),
                                (0, L / 2 - cw / 2), (ac, L / 2)], 1):
        geo.addPoint(x, y, 0.0, h0, i)
    # lines: top, right, bottom, left below the mouth, lower / upper crack face, left above
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 6), (6, 7), (7, 5), (5, 1)], 1):
        geo.addLine(a, b, i)
    # the cracked face z = 0, extruded through the thickness Lz into the plate volume
    geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7], 10)
    geo.addPlaneSurface([10], 11)
    ext = geo.extrude([(2, 11)], 0, 0, Lz)
    geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [e[1] for e in ext if e[0] == 3], tag=100)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h0)
    gmsh.model.mesh.generate(3)
    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    gmsh.finalize()
    return msh


def ncells(msh):
    return msh.topology.index_map(3).size_local


def nnodes(msh):
    return msh.topology.index_map(0).size_local


def refine_near(base, pts):
    """Bisect edges of *base* within amr_refine_radius of *pts* down to h_min.
    Returns (mesh, parent) with parent[c] = cell of *base* containing cell c."""
    m, parent = base, np.arange(ncells(base), dtype=np.int32)
    if pts is None or len(pts) == 0:
        return m, parent
    tree = cKDTree(pts)
    for lev in range(max(1, int(np.ceil(np.log2(h0 / h_min))))):
        m.topology.create_entities(1)
        m.topology.create_connectivity(1, 3)
        e = np.arange(m.topology.index_map(1).size_local, dtype=np.int32)
        e = e[m.h(1, e) > 1.05 * h0 / 2**(lev + 1)]
        if e.size:
            me = compute_midpoints(m, 1, e)
            d, _ = tree.query(me, distance_upper_bound=amr_refine_radius)
            e = e[np.isfinite(d) & (me[:, 1] <= ZONE_Y)]   # none in the frozen zone
        if e.size == 0:
            break
        m, pc, _ = refine(m, e, option=RefinementOption.parent_cell)
        # dolfinx 0.9: parent_cell is in the refined mesh's pre-reordering cell order
        parent = parent[pc[np.asarray(m.topology.original_cell_index)]]
    return m, parent


def midpoints(msh, cells):
    return compute_midpoints(msh, 3, cells.astype(np.int32)) if cells.size \
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


def face_facets(msh, which):
    tol = 1e-8 * W
    f = {"top": lambda x: np.isclose(x[1], L, atol=tol),
         "bottom": lambda x: np.isclose(x[1], 0.0, atol=tol)}[which]
    msh.topology.create_connectivity(2, 3)
    return locate_entities_boundary(msh, 2, f)


def dirichlet(V, t_const):
    """Boundary conditions on the faces of V's mesh: bottom fixed, top (t_const, 0, 0)."""
    top, bot = face_facets(V.mesh, "top"), face_facets(V.mesh, "bottom")
    bcs = []
    for k in range(3):
        db = fem.locate_dofs_topological(V.sub(k), 2, bot)
        dt_ = fem.locate_dofs_topological(V.sub(k), 2, top)
        if db.size:
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0), db, V.sub(k)))
        if dt_.size:
            bcs.append(fem.dirichletbc(t_const if k == 0 else PETSc.ScalarType(0), dt_, V.sub(k)))
    return bcs


def top_dofs(V):
    """Top-face dofs of each component (indices into the vector array)."""
    top = face_facets(V.mesh, "top")
    return [fem.locate_dofs_topological(V.sub(k), 2, top) for k in range(3)]

# =============================================================================
# SECTION 12: Solvers on the Solve Mesh (the patch, or the whole mesh for AMRC)
# =============================================================================

class USolver:
    """(K(z) + S) u = U(t) g0 + lifting.  S (the condensed exterior) is written once into a
    template matrix; each solve copies the template and adds the patch stiffness.  The solve
    is skipped when the residual of the current u is below U_SKIP_ATOL."""

    def __init__(self, P, S=None, pdG=None):
        self.u, self.a, self.L = P["u"], P["a_u"], P["L_u"]
        self.q, self.q_expr, self.bcs = P["q"], P["q_expr"], P["bcs_u"]
        self.pdG, self.g = pdG, None
        sp = fem.create_sparsity_pattern(self.a)
        if pdG is not None:
            nodes = np.unique(pdG // 3).astype(np.int32)
            sp.insert(nodes, nodes)
        sp.finalize()
        self.A = _cpp.la.petsc.create_matrix(comm, sp)
        # touch every pattern slot before the first final assembly, so none is dropped
        fem.petsc.assemble_matrix(self.A, self.a, self.bcs)
        if pdG is not None:
            self.A.setValues(pdG, pdG, np.zeros((pdG.size, pdG.size)),
                             addv=PETSc.InsertMode.ADD_VALUES)
        self.A.assemble()
        self.A0 = None
        if pdG is not None:
            self.A0 = self.A.duplicate(copy=False)
            self.A0.setValues(pdG, pdG, S, addv=PETSc.InsertMode.INSERT_VALUES)
            self.A0.assemble()
        self.b = fem.petsc.create_vector(self.L)
        self.r = self.b.duplicate()
        self.ksp = cg_ksp(self.A, U_FACTOR)
        self.refactor, self.t_fresh = True, np.inf  # the first solve factorizes

    def assemble(self):
        if self.A0 is None:
            self.A.zeroEntries()
        else:
            self.A0.copy(self.A, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        fem.petsc.assemble_matrix(self.A, self.a, self.bcs)
        self.A.assemble()
        with self.b.localForm() as bl:
            bl.set(0.0)
        fem.petsc.assemble_vector(self.b, self.L)
        if self.g is not None:
            self.b.getArray()[self.pdG] += self.g
        if self.bcs:
            fem.petsc.apply_lifting(self.b, [self.a], [self.bcs])
            fem.petsc.set_bc(self.b, self.bcs)

    def residual_norm(self):
        self.A.mult(self.u.x.petsc_vec, self.r)
        self.r.axpy(-1.0, self.b)
        return self.r.norm()

    def solve(self, force=False):
        """Returns False if the solve was skipped."""
        self.assemble()
        if not force and self.residual_norm() < U_SKIP_ATOL:
            return False
        self._ksp_solve()
        self.q.interpolate(self.q_expr)
        return True

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
        for o in (self.ksp, self.A, self.A0, self.b, self.r):
            if o is not None:
                o.destroy()


class ZSolver:
    """The phase-field VI posed on the free dofs alone.  Frozen dofs (patch boundary,
    frozen zone) are not constrained inside the system but deleted from it; their values
    enter through z as a coefficient of the residual."""

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
    for a, b in ((2, 3), (3, 2), (0, 3), (3, 0), (1, 3)):
        msh.topology.create_connectivity(a, b)
    V = C["V"] = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    Y = C["Y"] = fem.functionspace(msh, ("Lagrange", 1))
    C["Yv"] = fem.functionspace(msh, ("DG", 0))
    nc = C["nc"] = ncells(msh)
    C["c2n"], C["xn"] = Y.dofmap.list, Y.tabulate_dof_coordinates()
    C["mid"] = compute_midpoints(msh, 3, np.arange(nc, dtype=np.int32))
    C["tree"], C["ntree"] = cKDTree(C["mid"]), cKDTree(C["xn"])
    C["u"] = fem.Function(V, name="displacement")
    C["z"], C["zub"] = fem.Function(Y, name="phasefield"), fem.Function(Y)
    C["z"].x.array[:] = C["zub"].x.array[:] = 1.0
    if not LOCAL:
        return C
    # Dirichlet dofs (top and bottom faces, all components) and the load pattern d0
    D, d0 = [], []
    for which, val in (("bottom", (0.0, 0.0, 0.0)), ("top", (1.0, 0.0, 0.0))):
        f = face_facets(msh, which)
        for k in range(3):
            dk = fem.locate_dofs_topological(V.sub(k), 2, f)
            D.append(dk)
            d0.append(np.full(dk.size, val[k]))
    D, d0 = np.concatenate(D).astype(np.int32), np.concatenate(d0)
    D, i = np.unique(D, return_index=True)
    C["D"], C["d0"] = D.astype(np.int32), d0[i]
    f2c = msh.topology.connectivity(2, 3)          # interior facets and their two cells
    cnt = np.diff(f2c.offsets)[:msh.topology.index_map(2).size_local]
    fi = np.where(cnt == 2)[0].astype(np.int32)
    C["f2c"] = f2c.array[f2c.offsets[fi][:, None] + np.arange(2)]
    C["fmid"] = compute_midpoints(msh, 2, fi)
    C["top"] = top_dofs(V)
    C["ind_expr"] = fem.Expression(indicator(C["z"], *cell_terms(C["u"])),
                                   C["Yv"].element.interpolation_points())
    C["q_expr"] = fem.Expression(ufl.as_vector(cell_terms(C["u"])), C["Yv"].element.interpolation_points())
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


_VOL = {}   # coarse cell volumes for the exterior z-integral, computed on first use


def _cell_volumes(C):
    """Coarse cell volumes (cached)."""
    if "v" not in _VOL:
        x = C["xn"][C["c2n"]]
        _VOL["v"] = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                                     x[:, 3] - x[:, 0])) / 6.0
    return _VOL["v"]

# =============================================================================
# SECTION 14: Patch Selection & Exterior Condensation (AMRC-CLIP, AMRC-CLIP-TW)
# =============================================================================

def tip_patch(C, pts):
    """The AMRC-CLIP patch around the active points *pts* (bool mask): B(A, Rp) [u the padded
    Delta cells, USE_DELTA] u N(D n Wp).  AMRC-CLIP and AMRC-CLIP-TW both use it."""
    hit = C["tree"].query_ball_point(pts, PATCH_R_EPS * eps)
    cells = mask(C, np.concatenate([np.asarray(h, np.int64) for h in hit] + [np.empty(0, np.int64)]))
    if USE_DELTA:
        de = delta_cells(C)
        if de.any():
            # Delta padded by DELTA_PAD_EPS eps, but only through damaged cells (along the
            # crack, not into intact material), plus one layer so every node is solved
            hit = C["tree"].query_ball_point(C["mid"][de], DELTA_PAD_EPS * eps)
            pad = mask(C, np.concatenate([np.asarray(h, np.int64) for h in hit]))
            cells |= grow(C, pad & damaged_cells(C))
    cells &= C["mid"][:, 1] <= ZONE_Y       # no patch in the frozen zone
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
        # every node of D solved (outside the frozen zone)
        cells |= grow(C, damaged_cells(C), WAKE_LAYERS) & (C["mid"][:, 1] <= ZONE_Y)
    return np.where(cells)[0].astype(np.int32)


def split_facets(msh, gmid):
    """For each interface-facet midpoint *gmid*: is that facet split in the refined *msh*?"""
    msh.topology.create_connectivity(2, 3)
    fm = compute_midpoints(msh, 2, exterior_facet_indices(msh.topology))
    return ~np.isfinite(cKDTree(fm).query(gmid, distance_upper_bound=1e-8 * W)[0])


def place_patch(C, pcells, refine_pts):
    """Submesh of *pcells* refined near *refine_pts*.  Where the refinement splits a
    patch-boundary facet, the cells around that facet join the patch, until the boundary
    is made of coarse facets only."""
    for _ in range(10):
        inP = mask(C, pcells)
        gf = inP[C["f2c"][:, 0]] != inP[C["f2c"][:, 1]]
        base, bmap, _, _ = create_submesh(C["msh"], 3, pcells)
        msh, parent = refine_near(base, refine_pts)
        split = split_facets(msh, C["fmid"][gf])
        if not split.any():
            return pcells, base, bmap, msh, parent
        pcells = np.where(inP | grow(C, mask(C, C["f2c"][gf][split].ravel())))[0].astype(np.int32)
    raise RuntimeError("the patch could not be made to contain the refinement")


def build_exterior(C, pcells):
    """Condense the exterior onto the patch boundary: S, g0, the factor of A_EE, the
    exterior stiffness (for the reaction force and the energy) and the rows of
    A_EE^-1 A_EG on the first exterior cell layer.  None if the patch touches a Dirichlet
    face."""
    t0 = clock()
    msh, nc, c2n = C["msh"], C["nc"], C["c2n"]
    inP = mask(C, pcells)
    nn = C["xn"].shape[0]
    tp, te = np.zeros(nn, bool), np.zeros(nn, bool)
    tp[c2n[inP].ravel()] = True
    te[c2n[~inP].ravel()] = True
    if tp[C["D"] // 3].any() or inP.all():
        return None
    tags = meshtags(msh, 3, np.arange(nc, dtype=np.int32), np.where(inP, 1, 2).astype(np.int32))
    dxe = Measure("dx", domain=msh, subdomain_data=tags, metadata=_QMD)(2)
    V = C["V"]
    A = fem.petsc.assemble_matrix(fem.form(k_form(TrialFunction(V), TestFunction(V), C["z"], dxe)))
    A.assemble()

    blk = lambda n: (3 * n[:, None] + np.arange(3)).ravel().astype(np.int32)
    D, dG = C["D"], blk(np.where(tp & te)[0])
    dE = np.setdiff1d(blk(np.where(te & ~tp)[0]), D).astype(np.int32)
    iE, iG, iD = (PETSc.IS().createGeneral(d, comm=comm) for d in (dE, dG, D))
    AEE, AEG, AED = (A.createSubMatrix(iE, j) for j in (iE, iG, iD))
    AGE, AGD = A.createSubMatrix(iG, iE), A.createSubMatrix(iG, iD)
    AGG = dense_block(A, iG, iG)
    S = AGG.copy()
    kspE = factor_ksp(AEE, U_FACTOR)
    kspE.setUp()
    F = kspE.getPC().getFactorMatrix()
    nE, nG = dE.size, dG.size
    # the Dirichlet load for U = 1: b0 = A_ED d0, y0 = A_EE^-1 b0, g0 = A_GE y0 - A_GD d0
    b0, y0 = AEE.createVecLeft(), AEE.createVecLeft()
    AED.mult(PETSc.Vec().createWithArray(C["d0"].copy(), comm=comm), b0)
    kspE.solve(b0, y0)
    g0v, tmpG = AGE.createVecLeft(), AGE.createVecLeft()
    AGE.mult(y0, g0v)
    AGD.mult(PETSc.Vec().createWithArray(C["d0"].copy(), comm=comm), tmpG)
    g0 = g0v.getArray(readonly=True) - tmpG.getArray(readonly=True)
    ring = grow(C, inP) & ~inP                     # first exterior cell layer
    rn = np.unique(c2n[ring].ravel())
    rdofs = np.setdiff1d(blk(rn[~tp[rn]]), D).astype(np.int32)
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
        GX = AGE.matMult(Xc)
        S[:, j0:j0 + w] -= GX.getDenseArray()
        for o in (isC, B, Xc, GX):
            o.destroy()
    S = 0.5 * (S + S.T)

    # Self-check of S on a random vector (one extra solve): condensation errors would
    # otherwise show up only as a wrong reaction force.
    rng = np.random.default_rng(0)
    vG = rng.standard_normal(nG)
    ve, tg, vv = AEE.createVecLeft(), AGE.createVecLeft(), AEG.createVecLeft()
    AEG.mult(PETSc.Vec().createWithArray(vG, comm=comm), vv)
    kspE.solve(vv, ve)
    AGE.mult(ve, tg)
    ref = AGG @ vG - tg.getArray()
    err = np.linalg.norm(S @ vG - ref) / max(np.linalg.norm(ref), 1e-300)

    X = dict(dG=dG, dE=dE, D=D, S=S, g0=g0, y0=y0.getArray(readonly=True).copy(), A=A,
             AEG=AEG, kspE=kspE, xE=AEE.createVecLeft(), yE=AEE.createVecLeft(),
             xG=C["xn"][dG // 3], gmid=C["fmid"][inP[C["f2c"][:, 0]] != inP[C["f2c"][:, 1]]],
             ecells=np.where(~inP)[0].astype(np.int32), enodes=np.where(te)[0],
             ring=np.where(ring)[0].astype(np.int32), rdofs=rdofs, ridx=ridx, Yr=Yr,
             zmin=C["z"].x.array[np.where(te)[0]].min(), err=err, t=clock() - t0)
    for o in (AEE, AED, AGE, AGD, iE, iG, iD, isR, ve, tg, vv, b0, y0, g0v, tmpG):
        o.destroy()
    return X


def destroy_exterior(X):
    for k in ("A", "AEG", "kspE", "xE", "yE"):
        X[k].destroy()


def recover(C, P, X, t):
    """u on the exterior from u on the interface: u_E = -(A_EE^-1 A_EG u_G + U(t) y0)."""
    uG = P["u"].x.array[P["pdG"]]
    X["AEG"].mult(PETSc.Vec().createWithArray(uG.copy(), comm=comm), X["xE"])
    X["kspE"].solve(X["xE"], X["yE"])
    uc = C["u"].x.array
    uc[X["dE"]] = -(X["yE"].getArray(readonly=True) + top_disp(t) * X["y0"])
    uc[X["dG"]] = uG
    uc[X["D"]] = top_disp(t) * C["d0"]


def lookahead(C, P, X, t):
    """Flagged cells of the first exterior layer, from the exact exterior u there:
    u_E = -(A_EE^-1 A_EG u_G + U(t) y0)."""
    uG = P["u"].x.array[P["pdG"]]
    C["u"].x.array[X["dG"]] = uG
    C["u"].x.array[X["rdofs"]] = -(X["Yr"] @ uG + top_disp(t) * X["y0"][X["ridx"]])
    C["u"].x.array[X["D"]] = top_disp(t) * C["d0"]
    C["ind"].interpolate(C["ind_expr"], X["ring"])
    return X["ring"][C["ind"].x.array[X["ring"]] > 0.5]

# =============================================================================
# SECTION 15: The Solve Mesh (the patch, or the whole domain for AMRC)
# =============================================================================

def build_patch(C, base, bmap, msh, parent, pts, X=None):
    """Set up both problems on *msh*, a refinement of *base* (a coarse submesh with cell
    map *bmap*, or the coarse mesh itself)."""
    msh.name = "solve"
    for a, b in ((2, 3), (3, 2), (0, 3)):
        msh.topology.create_connectivity(a, b)
    P = dict(msh=msh, base=base, bmap=bmap, cmap=bmap[parent], pts=pts)
    V = P["V"] = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    Y = P["Y"] = fem.functionspace(msh, ("Lagrange", 1))
    Yv = P["Yv"] = fem.functionspace(msh, ("DG", 0))
    u, z = fem.Function(V, name="displacement"), fem.Function(Y, name="phasefield")
    P["u"], P["z"], P["zub"], P["zd"] = u, z, fem.Function(Y), fem.Function(Y)
    xn = Y.tabulate_dof_coordinates()
    c2n = P["c2n"] = Y.dofmap.list                 # (DG0 dofs are in cell order in serial)

    # u: Dirichlet dofs on the solve mesh (only for the whole-domain solve)
    P["U"] = fem.Constant(msh, PETSc.ScalarType(0.0))
    P["bcs_u"] = dirichlet(V, P["U"]) if X is None else []
    P["top"] = top_dofs(V)

    # z: the frozen patch boundary
    frozen, P["pdG"] = np.empty(0, np.int32), None
    if X is not None:
        dist, pn = cKDTree(xn).query(X["xG"], distance_upper_bound=1e-8 * W)
        assert np.isfinite(dist).all(), "interface node missing from the patch mesh"
        P["pdG"] = (3 * pn + X["dG"] % 3).astype(np.int32)
        P["gtree"] = cKDTree(X["xG"])
        frozen = np.unique(pn).astype(np.int32)
        # rim: free nodes sharing a cell with the frozen boundary.  z is P1, so damage
        # cannot reach the boundary without first showing up here.
        on = np.zeros(xn.shape[0], bool)
        on[frozen] = True
        rim = np.unique(c2n[on[c2n].any(axis=1)])
        P["rim"] = rim[~on[rim]]
    zone = np.where(xn[:, 1] > ZONE_Y)[0]     # frozen zone: z stays 1
    P["free"] = np.setdiff1d(np.arange(xn.shape[0], dtype=np.int32),
                             np.union1d(frozen, zone)).astype(np.int32)

    # Forms.  Per stagger iteration the u-dependent cell terms are evaluated once (DG0,
    # exact for P1 u), so no kernel inside the z Newton loop recomputes stresses.
    dx_ = Measure("dx", domain=msh, metadata=_QMD)
    ip = Yv.element.interpolation_points()
    du_, v = TrialFunction(V), TestFunction(V)
    P["a_u"] = fem.form(k_form(du_, v, z, dx_))
    P["L_u"] = fem.form(inner(fem.Constant(msh, np.zeros(3)), v) * dx_)
    P["q"] = fem.Function(fem.functionspace(msh, ("DG", 0, (3,))))
    P["q_expr"] = fem.Expression(ufl.as_vector(cell_terms(u)), ip)
    p, a, s = P["q"][0], P["q"][1], P["q"][2]
    R = z_residual(z, TestFunction(Y), p, a, s, dx_)
    P["ind_expr"] = fem.Expression(indicator(z, p, a, s), ip)
    P["ind"] = fem.Function(Yv, name="pf_flag")
    P["h"] = fem.Function(Yv)
    P["h"].interpolate(fem.Expression(MinCellEdgeLength(msh), ip))
    P["f_dz"] = fem.form(P["zd"] * P["zd"] * dx_)
    P["f_react"] = fem.form(k_form(u, v, z, dx_))
    P["f_post"] = (fem.form(0.5 * k_form(u, u, z, dx_)), fem.form((1 - z) * dx_))

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
    idata = {}
    for f, V in (("u", "V"), ("z", "Y"), ("zub", "Y")):
        if V not in idata:
            idata[V] = fem.create_interpolation_data(P_new[V], P_old[V], cells, padding=1e-8)
        tmp = fem.Function(P_new[V])
        tmp.interpolate_nonmatching(P_old[f], cells, idata[V])
        bs = P_new[V].dofmap.index_map_bs
        dofs = np.unique(P_new[V].dofmap.list[cells].ravel())
        dofs = (bs * dofs[:, None] + np.arange(bs)).ravel()
        P_new[f].x.array[dofs] = tmp.x.array[dofs]


def from_coarse(C, P):
    """Fill P's u, z, zub from the coarse fields (coarse -> base submesh by cell map,
    base -> refined patch by interpolation)."""
    base = P["base"]
    Q = dict(V=fem.functionspace(base, ("Lagrange", 1, (3,))), Y=fem.functionspace(base, ("Lagrange", 1)))
    cb = np.arange(ncells(base), dtype=np.int32)
    for f, sp in (("u", "V"), ("z", "Y"), ("zub", "Y")):
        Q[f] = fem.Function(Q[sp])
        Q[f].interpolate(C[f], cells0=P["bmap"], cells1=cb)
    transfer(P, Q)


def coarse_nodes(C, P):
    """(patch dofs, coarse nodes) of the patch nodes that are coarse nodes."""
    dist, cn = C["ntree"].query(P["Y"].tabulate_dof_coordinates(), distance_upper_bound=1e-8 * W)
    pn = np.where(np.isfinite(dist))[0]
    return pn, cn[pn]


def to_coarse(C, P):
    """Write the patch values at coarse nodes back into the coarse fields."""
    pn, cn = coarse_nodes(C, P)
    C["z"].x.array[cn] = P["z"].x.array[pn]
    C["zub"].x.array[cn] = P["zub"].x.array[pn]
    for k in range(3):
        C["u"].x.array[3 * cn + k] = P["u"].x.array[3 * pn + k]


def coarse_z(C, P):
    """z at every coarse node, taking the patch value where the patch has that node."""
    z = C["z"].x.array.copy()
    pn, cn = coarse_nodes(C, P)
    z[cn] = P["z"].x.array[pn]
    return z


class PatchAtDirichlet(Exception):
    """The patch reached the top/bottom face (STOP_ON_DIRICHLET): the run ends at the step before."""


def new_patch(C, P_old, X_old, pts, t, refine_pts=None):
    """(Re)place the patch around *pts*, refine it near *refine_pts*, condense the new
    exterior and carry the fields over.  O(N) -- only when the patch moves.  Returns
    (P, X); X is None if the patch had to become the whole domain.
    With STOP_ON_DIRICHLET it raises PatchAtDirichlet instead."""
    if P_old is not None:
        if X_old is not None:
            recover(C, P_old, X_old, t)
        to_coarse(C, P_old)
        # free the old factorizations before building the new ones (in 3D each can take
        # several GB); the old fields stay for the transfer below
        P_old["us"].destroy()
        P_old["zs"].destroy()
        if X_old is not None:
            destroy_exterior(X_old)
            X_old.clear()
    pcells, base, bmap, msh, parent = place_patch(C, select_patch(C, pts), refine_pts)
    if P_old is not None and METHOD == "amrc_clip":
        relax_wake(C, pcells)
    X = build_exterior(C, pcells)
    if X is None and STOP_ON_DIRICHLET:            # reached a Dirichlet face: stop the run
        raise PatchAtDirichlet()
    if X is None:                                  # reached a Dirichlet face: go whole-domain
        print("    [patch] reached the top/bottom face -- whole-domain solve from here", flush=True)
        base, bmap = C["msh"], np.arange(C["nc"], dtype=np.int32)
        msh, parent = refine_near(base, refine_pts)
    P = build_patch(C, base, bmap, msh, parent, refine_pts, X)
    from_coarse(C, P)
    if P_old is not None:
        old = mask(C, P_old["bmap"])
        transfer(P, P_old, np.where(old[P["cmap"]])[0].astype(np.int32))
        P_old.clear()
    P["q"].interpolate(P["q_expr"])                # the z problem reads the cell terms of u
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
    free = free[C["xn"][free, 1] <= ZONE_Y]   # frozen zone
    if free.size == 0:
        return
    tags = meshtags(C["msh"], 3, np.arange(C["nc"], dtype=np.int32), band.astype(np.int32))
    dxw = Measure("dx", domain=C["msh"], subdomain_data=tags, metadata=_QMD)(1)
    q = fem.Function(fem.functionspace(C["msh"], ("DG", 0, (3,))))
    q.interpolate(C["q_expr"], cells)
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
    """The load of this time: the top-face displacement, or the condensed load."""
    P["U"].value = top_disp(t)
    if X is not None:
        P["us"].g = top_disp(t) * X["g0"]


def irreversibility(C, P, X):
    for z, ub in ((P["z"], P["zub"]),) + (((C["z"], C["zub"]),) if X is not None else ()):
        m = z.x.array <= v_th
        ub.x.array[m] = z.x.array[m] + z_ub_offset

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
        pn, cn = coarse_nodes(C, P)
        zc, uc = C["z"].x.array.copy(), C["u"].x.array.copy()
        zc[cn] = P["z"].x.array[pn]
        for k in range(3):
            uc[3 * cn + k] = P["u"].x.array[3 * pn + k]
        nc, cmap = C["nc"], P["cmap"]
        region = dict(solved=mask(C, P["bmap"]), refined=np.bincount(cmap, minlength=nc) > 1,
                      flagged=mask(C, cmap[P["ind"].x.array > 0.5]))
        if METHOD == "amrc_clip":
            zkeep = C["z"].x.array.copy()
            C["z"].x.array[:] = zc
            region["delta"] = delta_cells(C)
            C["z"].x.array[:] = zkeep
        if METHOD == "amrc_clip_tw":
            region["damaged"] = (zc < DAMAGE_Z)[C["c2n"]].any(axis=1)
        for n, f in self.reg.items():
            f.x.array[:] = region[n]
        fu, fz = fem.Function(C["V"], name="displacement"), fem.Function(C["Y"], name="phasefield")
        fu.x.array[:], fz.x.array[:] = uc, zc
        for f in (fu, fz, *self.reg.values()):
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
        "Problem   : in-plane shear (edge-cracked plate), 3D",
        f"Method    : {METHOD_NAME}",
        f"Geometry  : W = {W:g} mm   L = {L:g} mm   Lz = {Lz:g} mm   ac = {ac:g} mm",
        f"Elastic   : E = {E:.4e}   nu = {nu:.4f}",
        f"Strength  : sts = {sts:.4e}   scs = {scs:.4e}   shs = {shs:.4e}",
        f"Fracture  : Gc = {Gc:.6f}   eps = {eps:.4e}   delta = {delta:.4e}",
        f"AMR       : h0 = {h0:.4g}   h_min = {h_min:.4g}   R = {amr_refine_radius:.4g}",
        (f"Frozen    : y > {ZONE_Y:g} mm  (L/2 + {ZONE_EPS:g} eps): z = 1, no refinement"
         if ZONE_EPS else "Frozen    : none"),
        (f"Patch     : Rp = {PATCH_R_EPS:g} eps   retire = {RETIRE_STEPS} refinement"
         + (f"   wake layers = {WAKE_LAYERS}" if METHOD == "amrc_clip_tw" else "")
         if LOCAL else "Patch     : none (AMRC: the whole mesh is solved)"),
    ] + ([
        "Dirichlet : patch at the top/bottom face -> "
        + ("stop the run" if STOP_ON_DIRICHLET else "whole-domain solve"),
    ] if LOCAL else []) + [
        f"Loading   : {NSTEPS} steps   dt = {stepsize:.4g}   top u_x = {U_MAX:g} t mm",
        f"Stagger   : tol = {tol_stag:.1e}   max_stag = {max_stag}",
        f"Solvers   : u {'/'.join(U_FACTOR)} (CG)   z {'/'.join(Z_FACTOR)}",
        f"Output    : {output_dir}",
    ])

# =============================================================================
# SECTION 18: Main Solve Loop
# =============================================================================

def main():
    _VOL.clear()
    t_start = clock()
    say = print if verbose > 1 else (lambda *a, **k: None)
    print_summary()
    C = coarse_setup(generate_mesh())
    print(f"    coarse mesh: {C['nc']:,} tets, {C['xn'].shape[0]:,} nodes", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    _USTAT.update(n_factor=0, n_cg=0)
    TT = dict(u=0.0, z=0.0, amr=0.0, ext=0.0, post=0.0, io=0.0, n_u=0, n_u_skip=0, n_z=0,
              n_amr=0, n_move=0, n_lookahead=0, n_escape=0, n_rim=0)
    nonconverged, unresolved = [], []

    # --- Initial solve mesh: the patch around the crack front, or the whole domain ---
    front0 = np.array([[ac, L / 2, zz] for zz in np.linspace(0.0, Lz, max(2, int(round(Lz / h0)) + 1))])
    if LOCAL:
        P, X = new_patch(C, None, None, front0, 0.0)
        TT["ext"] += X["t"] if X else 0.0
    else:
        X = None
        cells = np.arange(C["nc"], dtype=np.int32)
        P = build_patch(C, C["msh"], cells, C["msh"], cells, None)
        from_coarse(C, P)
        P["q"].interpolate(P["q_expr"])
    out = Output(C) if output_every else None
    log_file, timing_file = f"{output_dir}/log.txt", f"{output_dir}/timing.csv"
    with open(log_file, "w") as f:
        f.write("step time disp_top_x Fx Fy Fz elastic_energy z_integral min_z iterations converged dz_last "
                "wall_step wall_total n_nodes n_solve_nodes\n")
    with open(timing_file, "w") as f:
        f.write("step,t,iterations,converged,n_cells,n_solve_cells,n_G,n_amr,n_move,"
                "t_u,t_z,t_amr,t_post,t_io,t_step,t_cum,n_nodes,n_solve_nodes\n")

    def seeds(P):
        return P["pts"] if P["pts"] is not None else front0

    def move(P, X, pts, refine_pts, t, why):
        """Re-select the patch around *pts* and refine it near *refine_pts*."""
        t0 = clock()
        Pn, Xn = new_patch(C, P, X, pts, t, refine_pts)
        TT["n_move"] += 1
        set_step_data(C, Pn, Xn, t)
        if Xn is not None:
            TT["ext"] += Xn["t"]
            say(f"    [patch] moved ({why}): {ncells(Pn['msh']):,} cells, interface {Xn['dG'].size} "
                f"dofs, exterior {Xn['dE'].size:,} dofs ({clock() - t0:.2f} s, check {Xn['err']:.1e})")
        return Pn, Xn

    def remesh(P, X, pts, t):
        """Re-refine the solve mesh around *pts*; move the patch instead if a flagged point
        lies within R of the patch boundary or the refinement would split it."""
        if X is None or not np.isfinite(P["gtree"].query(
                pts, distance_upper_bound=amr_refine_radius)[0]).any():
            msh, parent = refine_near(P["base"], pts)
            if X is None or not split_facets(msh, X["gmid"]).any():
                Pn = build_patch(C, P["base"], P["bmap"], msh, parent, pts, X)
                transfer(Pn, P)
                Pn["q"].interpolate(Pn["q_expr"])
                set_step_data(C, Pn, X, t)
                arm_rim(Pn)
                destroy_patch(P)
                say(f"    [AMR] refined -> {ncells(Pn['msh']):,} cells")
                return Pn, X
        return move(P, X, pts, pts, t, "refinement")

    last = min(NSTEPS, MAX_STEPS) if MAX_STEPS else NSTEPS
    step = 0
    stopped_at = None
    try:
        for step in range(1, last + 1):
            t = step * stepsize
            ts = clock()
            su = sz = sa = sp = 0.0
            n_amr0, n_move0 = TT["n_amr"], TT["n_move"]

            # --- A: Loading and irreversibility bounds ---
            set_step_data(C, P, X, t)
            irreversibility(C, P, X)                   # once per step, from the last step's z
            # coarse-node z at the step start (Delta)
            if (METHOD == "amrc_clip" or USE_DELTA) and X is not None:
                C["zhist"] = (C.get("zhist", []) + [coarse_z(C, P)])[-max(1, round(ACTIVE_T / stepsize)):]
                C["zref"] = C["zhist"][0]

            # --- B: Staggered u-z solve (with mid-stagger AMR and patch moves) ---
            stag, zres, esc = 1, 1e7, 0
            while True:
                while stag < max_stag and zres > tol_stag:
                    tk = clock()
                    if P["us"].solve():
                        TT["n_u"] += 1
                    else:
                        TT["n_u_skip"] += 1
                    su += clock() - tk

                    # refinement: flags on the solve mesh and on the first exterior layer
                    tk = clock()
                    P["ind"].interpolate(P["ind_expr"])
                    hot = P["ind"].x.array > 0.5
                    ring = lookahead(C, P, X, t) if X is not None else np.empty(0, np.int32)
                    if ring.size or (hot & (P["h"].x.array > H_COARSE * h_min)).any():
                        if LOCAL and RETIRE_STEPS and "refage" in C:
                            # a refinement event: one more for the cells not refined now
                            C["refage"][np.bincount(P["cmap"], minlength=C["nc"]) <= 1] += 1
                        pts = midpoints(P["msh"], np.where(hot)[0])
                        if ring.size:                  # flagged just outside: move over them
                            TT["n_lookahead"] += 1
                            pts = np.vstack([pts, midpoints(C["msh"], ring)])
                            P, X = move(P, X, pts, pts, t, "exterior flag")
                        else:
                            P, X = remesh(P, X, pts, t)
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
                        P, X = move(P, X, np.vstack([seeds(P), hits]), P["pts"], t, "rim")
                        zprev = P["z"].x.array.copy()
                        TT["n_rim"] += 1
                        sa += clock() - tk
                        tk = clock()
                        P["zs"].solve()
                        TT["n_z"] += 1
                        sz += clock() - tk
                    P["zd"].x.array[:] = P["z"].x.array - zprev
                    zres = np.sqrt(fem.assemble_scalar(P["f_dz"]))
                    say(f"    iter {stag:>3d}  |dz| = {zres:.7e}")
                    stag += 1

                # u for the final z (also when the iterations stopped at max_stag)
                tk = clock()
                P["us"].solve(force=True)
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
                print(f"    step {step}: {outside.size} flagged cells outside the patch -- absorbing", flush=True)
                tk = clock()
                P, X = move(P, X, np.vstack([seeds(P), midpoints(C["msh"], outside)]), P["pts"], t, "escape")
                sa += clock() - tk
                if esc > max_escape_retry or stag >= max_stag:
                    unresolved.append(step)
                    tk = clock()
                    P["us"].solve(force=True)
                    su += clock() - tk
                    if X is not None:
                        recover(C, P, X, t)
                    break
                zres = 1e7                             # iterate again on the enlarged patch
            iters = stag - 1
            converged = zres <= tol_stag
            if not converged:
                nonconverged.append(step)

            # --- C: refinement events since each coarse cell was last refined (counted at the
            #     refinement in the stagger loop; reset where the solve mesh refines the cell) ---
            if LOCAL and RETIRE_STEPS and "refage" in C:
                C["refage"][np.bincount(P["cmap"], minlength=C["nc"]) > 1] = 0

            # --- D: Post-processing (u is consistent with z; the exterior u is recovered):
            #     top-face reaction force (virtual work), energy, damage ---
            tk = clock()
            Rv = fem.petsc.assemble_vector(P["f_react"])
            Ra = Rv.getArray(readonly=True)
            Fk = np.array([Ra[P["top"][k]].sum() for k in range(3)])
            Rv.destroy()
            E_el, Zint = (fem.assemble_scalar(f) for f in P["f_post"])
            min_z = P["z"].x.array.min()
            n_cells = ncells(P["msh"])
            n_nodes = n_solve_nodes = nnodes(P["msh"])
            if X is not None:
                uv = PETSc.Vec().createWithArray(C["u"].x.array.copy(), comm=comm)
                Ru = X["A"].createVecLeft()
                X["A"].mult(uv, Ru)
                Rc = Ru.getArray(readonly=True)
                Fk += np.array([Rc[C["top"][k]].sum() for k in range(3)])
                E_el += 0.5 * Ru.dot(uv)
                Zint += float(((1.0 - C["z"].x.array[C["c2n"][X["ecells"]]]).mean(axis=1)
                               * _cell_volumes(C)[X["ecells"]]).sum())
                min_z = min(min_z, X["zmin"])
                n_cells += X["ecells"].size
                n_nodes += X["enodes"].size - X["dG"].size // 3    # the interface nodes are patch nodes
                Ru.destroy()
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
                f.write(f"{step} {t:.6f} {top_disp(t):.6e} {Fk[0]:.6e} {Fk[1]:.6e} {Fk[2]:.6e} {E_el:.6e} "
                        f"{Zint:.6e} {min_z:.6f} {iters} {int(converged)} {zres:.3e} "
                        f"{dt_step:.3f} {t_cum:.3f} {n_nodes} {n_solve_nodes}\n")
            with open(timing_file, "a") as f:
                f.write(f"{step},{t:.8e},{iters},{int(converged)},{n_cells},{ncells(P['msh'])},"
                        f"{X['dG'].size if X else 0},{TT['n_amr'] - n_amr0},{TT['n_move'] - n_move0},"
                        f"{su:.6f},{sz:.6f},{sa:.6f},{sp:.6f},{tio:.6f},{dt_step:.6f},"
                        f"{t_cum:.3f},{n_nodes},{n_solve_nodes}\n")
            if verbose:
                print(f"  step {step:>4d}  u {top_disp(t):.4e}  iters {iters:>3d}{'' if converged else ' (max)'}  "
                      f"Fx {Fk[0]:9.3f}  min_z {min_z:.4f}  nodes {n_solve_nodes:,} solved / {n_nodes:,}  "
                      f"| u {su:.1f}s  z {sz:.1f}s  amr {sa:.1f}s  io {tio:.1f}s  "
                      f"| step {dt_step:.1f}s  total {t_cum:.1f}s", flush=True)
    except PatchAtDirichlet:
        stopped_at, step = step, step - 1
        print(f"\n    [patch] reached the top/bottom face at step {stopped_at} -- "
              f"stopping (results up to step {step})", flush=True)

    # --- Finalisation ---
    if out is not None:
        out.close()
    elapsed = clock() - t_start
    _box("SIMULATION COMPLETE", [
        f"Method          : {METHOD_NAME}",
        f"Total time      : {elapsed:.1f} s ({CLOCK} clock)",
        f"Steps completed : {step}   ({len(nonconverged)} not converged in {max_stag - 1} iterations)",
    ] + [f"  {lab:<14}: {TT[k]:9.1f} s ({100 * TT[k] / max(elapsed, 1e-9):5.1f} %)"
         for k, lab in (("u", "displacement"), ("z", "phase field"), ("amr", "AMR + patch"),
                        ("post", "post/check"), ("io", "XDMF output"))] + [
        f"Solves          : u {TT['n_u']:,} (+{TT['n_u_skip']:,} skipped)   z {TT['n_z']:,}",
        f"Refinements     : {TT['n_amr']:,}",
    ] + ([
        f"Patch moves     : {TT['n_move']:,}   ({TT['ext']:.1f} s condensing)",
        f"Escapes         : {TT['n_escape']}   rim retries {TT['n_rim']}",
    ] if LOCAL else []) + [
        f"u solver        : {_USTAT['n_factor']:,} factorizations, {_USTAT['n_cg']:,} CG iterations",
    ] + ([f"UNRESOLVED steps: {unresolved}"] if unresolved else []))
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(dict(method=METHOD, steps=step, elapsed=elapsed, clock=CLOCK, eps=eps, h0=h0,
                       h_min=h_min, nu=nu, W=W, L=L, Lz=Lz, U_MAX=U_MAX, max_stag=max_stag,
                       tol_stag=tol_stag, active_dz=ACTIVE_DZ, active_t=ACTIVE_T, damage_z=DAMAGE_Z,
                       patch_r_eps=PATCH_R_EPS, delta_pad_eps=DELTA_PAD_EPS, zone_eps=ZONE_EPS,
                       use_delta=USE_DELTA, retire_steps=RETIRE_STEPS,
                       wake_layers=WAKE_LAYERS, stop_on_dirichlet=STOP_ON_DIRICHLET,
                       stopped_at=stopped_at, z_ub_offset=z_ub_offset,
                       nonconverged_steps=nonconverged, unresolved_steps=unresolved,
                       u_factor=U_FACTOR, z_factor=Z_FACTOR, u_cg_rtol=U_CG_RTOL,
                       u_refactors=_USTAT["n_factor"], u_cg_iterations=_USTAT["n_cg"], **TT), f, indent=2)
    return elapsed


if __name__ == "__main__":
    main()
