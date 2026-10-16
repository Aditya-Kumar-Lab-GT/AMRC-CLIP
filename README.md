# Adaptive Crack-Tip Localization of Phase-Field Fracture Computations

FEniCSx implementation of **AMRC-CLIP**, adaptive mesh refinement and coarsening (AMRC) with
condensation of linear inactive parts (CLIP), and of the four benchmark problems reported in:

> A. Dahal, A. Kumar, *Adaptive Crack-Tip Localization of Phase-Field Fracture Computations*.

AMRC-CLIP solves the coupled displacement / phase-field problem only on a small patch of cells around the crack tip. The exterior, whose stiffness does not change, is
condensed onto the patch boundary through a Schur complement that is factorized once and reused
for as long as the patch stays in place; the phase field is frozen there. AMRC-CLIP-TW (tip and
wake) also keeps the damaged wake inside the patch. The AMRC framework it builds on is available
at [Aditya-Kumar-Lab-GT/AMRC](https://github.com/Aditya-Kumar-Lab-GT/AMRC).

Contact: adahal8@gatech.edu, aditya.kumar@ce.gatech.edu

## Requirements

DOLFINx 0.9 with PETSc built with SuiteSparse (CHOLMOD) and MUMPS, plus `mpi4py`, `gmsh`,
`numpy` and `scipy`:

```bash
conda env create -f environment.yml
conda activate amrc-clip
```

The scripts were tested with DOLFINx 0.9.0, PETSc 3.21.

## Layout

One folder per benchmark problem, each holding one self-contained script that builds its own
mesh with Gmsh, so no mesh files are needed:

```
surfing/surfing.py                        §4.1  Surfing problem (original and doubled domain)
dynamic_branching/dynamic_branching.py    §4.2  Dynamic crack branching
in_plane_shear/in_plane_shear.py          §4.3  Three-dimensional crack growth under in-plane shear
percolation/percolation.py                §4.4  Growth and coalescence of 16 cracks in a unit cell
```

## Choosing the method

Every script implements the three methods compared in the paper. Select one with `METHOD` in
SECTION 1 at the top of the script:

| Method | `METHOD` | Region where u and the phase field are solved |
| --- | --- | --- |
| AMRC (reference) | `"amrc"` | the whole adaptively refined and coarsened mesh |
| AMRC-CLIP | `"amrc_clip"` (default) | a patch around the crack tip(s); the exterior is condensed onto the patch boundary and its phase field is frozen between patch moves |
| AMRC-CLIP-TW | `"amrc_clip_tw"` | a patch around the crack tip(s) and the whole damaged wake |

For the surfing problem, `SCALE` (also in SECTION 1) selects the domain: `1.0` is the original
30 mm × 10 mm strip (Fig. 2), `2.0` the doubled 60 mm × 20 mm strip (Figs. 3 and 4).

## Running

```bash
cd surfing
python surfing.py
```

## Output

Results are written to `results/<METHOD>/` in the working directory (surfing with
`SCALE = 2.0`: `results/<METHOD>_x2/`):

| File | Content |
| --- | --- |
| `log.txt` | one row per step: the global response (J-integral, reaction forces or energies), min z, stagger iterations, convergence, wall time and node counts (`n_nodes`: whole current mesh; `n_solve_nodes`: active nodes) |
| `timing.csv` | per-step timing breakdown and counters (refinements, patch moves) |
| `summary.json` | parameters and totals of the run |
| `xdmf/patch_N.xdmf` | the solve mesh at step N (the patch, or the whole mesh for AMRC): u, z, refinement indicator |
| `xdmf/exterior_N.xdmf` | AMRC-CLIP(-TW): the coarse cells outside the patch: u, z |
| `xdmf/coarse.xdmf` | time series on the base mesh: u, z and the regions solved / refined / flagged |

Open `patch_N.xdmf` together with `exterior_N.xdmf` in ParaView to see the whole domain.
The dynamic branching script also writes `Energies.txt`, `Traction_action.txt`,
`CrackVelocity.txt` and `CrackTipCoords.txt`.

## Problems

| Problem | Script | E (MPa) | ν | G<sub>c</sub> (N/mm) | σ<sub>ts</sub> (MPa) | σ<sub>cs</sub> (MPa) | ε (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Surfing (§4.1) | `surfing/surfing.py` | 9800 | 0.13 | 0.091125 | 27 | 77 | 0.175 |
| Dynamic branching (§4.2) | `dynamic_branching/dynamic_branching.py` | 72000 | 0.22 | 0.0038 | 6.16 | 18.48 | 0.625 |
| 3D in-plane shear (§4.3) | `in_plane_shear/in_plane_shear.py` | 9800 | 0.3 | 0.091125 | 27 | 77 | 0.3 |
| Coalescence of multiple cracks (§4.4) | `percolation/percolation.py` | 210000 | 0.3 | 2.7 | 4209 | 21047 | 0.006 |

Unless stated otherwise in a script, h<sub>0</sub> = ε, h<sub>min</sub> = ε/4, the refinement
radius is 3ε and the patch radius R<sub>p</sub> = 4ε.

## License

MIT — see [LICENSE](LICENSE).
