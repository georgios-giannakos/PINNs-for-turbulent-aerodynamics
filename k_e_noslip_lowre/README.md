# Low-Re k–ε replacement for the failed no-slip baseline

This is an independently runnable PyTorch experiment for the **fixed, symmetric NASA TMR NACA0012, α=0, Re=6,000,000** case in the supplied archive. It replaces the high-Re closure with a Launder–Sharma low-Re closure and makes boundary-condition errors, raw PDE residuals and aerodynamic observables visible. It is not a parameterized airfoil surrogate and cannot resume the old PhysicsNeMo checkpoints.

**Status:** equation/manufactured-solution tests, automatic differentiation, a two-step CPU training smoke test, checkpoint resume and VTP read-back passed. **No converged aerodynamic result from this new solver is claimed.** The old model's failure and the proposed repair are documented in [the audit](../analysis/k_e_noslip_20261128/REPORT.md).

## Run

Use a separate environment. Install a PyTorch build appropriate for the training machine, then:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --out runs/smoke --smoke
python train.py --out runs/lowre_seed1701
python train.py --out runs/lowre_seed1701 --resume runs/lowre_seed1701/checkpoint_latest.pt
```

The symbolic manufactured-solution test additionally needs `sympy>=1.13`. The smoke test uses a tiny network, two training steps and CPU. It checks execution only; its physical outputs are deliberately not validation results. Training refuses to overwrite a populated run directory unless a checkpoint is supplied. Changing physical, architecture or geometry settings on resume is rejected. For a new case, use a new output directory.

All defaults are in `config.yaml`; `--config other.yaml` selects another configuration. Default precision is float64; CUDA is selected if available. A full high-Re run is substantially more expensive than the CPU smoke test. Losses are evaluated and backpropagated region by region to limit graph memory. Old PhysicsNeMo runs are unaffected.

## Governing equations and pressure convention

Coordinates, velocity and density are nondimensionalized with c=U∞=ρ=1; ν=1/Re. `p` is mean static pressure. Let q=√k and e=`ep_tilde`. With summation over spatial indices:

```text
Re_t = k² / (ν e)
f_mu = exp[-3.4 / (1 + Re_t/50)²]
f_2  = 1 - 0.3 exp(-Re_t²)
ν_t  = 0.09 f_mu k²/e
P_k  = 2 ν_t S_ij S_ij
D    = 2 ν |grad q|²
E    = 2 ν ν_t Σ_i,j,l (∂²u_i / ∂x_j∂x_l)²
ε    = e + D

div u = 0
u_j ∂_j u_i + ∂_i p + (2/3)∂_i k - ∂_j[2(ν+ν_t) S_ij] = 0
u·grad k - div[(ν+ν_t) grad k] - P_k + e + D = 0
u·grad e - div[(ν+ν_t/1.3) grad e] - 1.44 P_k e/k
    + 1.92 f_2 e²/k - E = 0
```

The near-wall closure follows [Launder & Sharma (1974)](https://doi.org/10.1016/0094-4548(74)90150-7); the full Hessian contraction for E follows the tensor extension used by [OpenFOAM](https://cpp.openfoam.org/v12/LaunderSharmaKE_8C_source.html). The code is an independent implementation of the equations, not copied OpenFOAM source. Denominators contain a small numerical guard (1e-20), which must be included in sensitivity studies if the solution approaches that scale.

**`ep_tilde` is the transported modified dissipation. `ep` is physical ε=e+D.** At a smooth no-slip wall k→0 and e→0, while physical ε generally remains finite. Setting physical ε=0 would be a different, incorrect boundary condition. The exported `D`, `E`, `production`, `f_mu`, `f_2` and raw residuals make this distinction inspectable.

The standard high-Re stress in the original repository already included the symmetric stress and isotropic +2/3 grad(k) correction. Those terms are retained, not presented as newly discovered omissions.

## Changes tied to the observed failure

| Observed problem | Implemented response |
|---|---|
| Inlet k RMS error 97%, ε RMS error 89%, with tiny absolute losses | Logarithmic relative inlet and upper/lower boundary residuals for k and e |
| Standard high-Re closure evaluated to the no-slip surface; no ε wall condition | Launder–Sharma damping and D/E terms; normalized q=0 and √e=0 soft wall constraints |
| Inlet-scaled softplus requires raw outputs about 462 for k=1e-3 and 57,000 for ε=1e-2 | Squared-softplus fields scaled to 1e-3 and 1e-2, with biases that preserve inlet initialization |
| Near-wall residuals multiplied by 2·SDF | No vanishing distance weight; separate near, far, wall-layer, leading-edge and wake losses |
| Almost no points inside 1e-4 chord | Log-uniform wall offsets from 1e-6 to 0.02 chord and leading-edge samples |
| BCs suppressed during original initialization | All BCs active from step one; only PDE multiplier ramps from 0.1 to 1 |
| Tiny residuals can hide a collapsed turbulence field | Raw and relative PDE residuals, inlet relative errors and per-network gradient norms logged |
| 512 random surface samples corrupt force integration | One midpoint per polygon segment, exact segment lengths and closed integrated normals |
| Archive lacks exact LE point and physical ε distinction | Exact (0,0) probe; separate `ep_tilde`, `D`, `ep`; independent holdout diagnostics |

The positive fields are q=√k_scale·softplus(k_raw+b_k), k=q², e=ep_scale·softplus(ep_raw+b_e)². Small numerical positive offsets protect log residuals. These are **soft** wall constraints; monitor their errors rather than assuming exact zeros. For the chosen low-Re closure, the freestream e is solved so that ν_t/ν=10 including f_mu. It is about 9.64e-8, not the old high-Re value 1.215e-7.

The turbulence PDE residuals are divided by detached sums of absolute local terms plus a positive freestream scale. This is an optimization preconditioner, not a changed PDE or a learned physical source. The zero set is unchanged for finite denominators. It can still influence optimization, so both raw and scaled residuals are exported and fixed-scale ablations remain necessary. No empirical clipping of ν_t or production is used to make the plots look plausible.

The five interior regions have equal aggregate importance, regardless of their physical area. Wall-layer samples are deliberately distributed in log distance and their loss is a sampling objective, **not** an unbiased physical volume integral. Resampling happens every 100 steps. No detached SDF enters a network, avoiding missing chain-rule derivatives. The plain tanh architecture remains an approximation limitation for a very thin boundary layer; dense samples alone do not prove adequate representation.

## Outputs

Each recorded step contains ParaView-readable ASCII VTP files for near field, far field, deterministic boundary quadrature, exact LE/freestream probes, an independent wall-layer residual set, and independent inlet checks. `boundary_1.vtp` carries Cp, signed Cf and AD derivatives. Normals in exported files point outward from the **fluid**, matching the old archive; the force routine internally uses solid-outward normals.

```text
Cp = (p - p∞)/(0.5 ρ U∞²) = 2(p-p∞)
τ_w = t · [ν(grad u + grad uᵀ)] · n_solid
Cf = τ_w/(0.5 U∞²)
F/(0.5 ρ U∞² c) = ∮ [-Cp n_solid + 2ν(grad u+grad uᵀ)n_solid] ds
```

Both upper/lower tangents point from LE toward TE. Surface viscous force uses molecular viscosity, appropriate to the wall limit; the reported wall ν_t/ν must also become small. Pressure-force invariance to a constant gauge and stress signs are regression tested. `metrics.json` contains CL/CD pressure and viscous parts, LE pressure, freestream pressure probes, boundary errors and holdout residuals. Cp uses the configured zero-pressure gauge; the separate `p_LE_minus_p_inf` metric exposes any finite-domain shift.

`losses.jsonl` includes individual objectives, raw turbulence residual RMS, each network's pre-clipping gradient norm, learning rate and step. `run_manifest.json` records software versions, configuration and source hashes. `checkpoint_latest.pt` contains model, optimizer, step and configuration.

## Validation plan before accepting a result

1. Re-run the supplied unit and smoke tests on the training environment; check CUDA/float64 execution separately.
2. Check normalized inlet errors, wall velocity, k and e conditions. A cheap collapse of the turbulence fields must fail these checks.
3. Compare both raw and scaled residual distributions in independent holdout points, especially within 1e-4 chord and around LE/TE. Do not judge by a single summed training loss.
4. At α=0 seek CL near zero, symmetric upper/lower Cp and Cf, LE pressure recovery near 0.5 relative to freestream, positive drag, and plausible turbulent skin friction. These are validation observables, not imposed Cp/Cf targets.
5. Compare with [NASA TMR](https://tmbwg.github.io/turbmodels/naca0012_val_sa.html), explicitly accounting for SA versus low-Re k–ε, M=0.15 versus incompressible, turbulence inlet conditions and the different farfield domain. NASA's curves are representative CFD references, not exact truth and not direct k/ε validation data.
6. Repeat seeds 1701/1702/1703; double wall-layer sampling, double polygon resolution, change dmin to 5e-7, and expand bounds to [-10,20,-10,10]. Require stable Cp/Cf/forces before acceptance. Repeat with a larger network and compare convergence. Targeting y+≲1 is only a spacing guide; recompute it from the **new** wall shear.
7. Test the low-Re implementation on a fully developed channel or a flat plate with a trusted reference before interpreting airfoil turbulence fields quantitatively. This physical validation has not been completed here.

The original I=0.1% and ν_t/ν=10 are preserved as explicit inputs. A fully turbulent NASA reference is not a transition-model benchmark. Low-Re damping is not a validated transition model; boundary-layer state and any experimental tripping must be matched before making a quantitative validation claim. The default domain is deliberately retained for a controlled comparison; its finite size remains an uncertainty.

No leading-edge pressure of 0.5 is imposed as an extra BC. For the stated nondimensionalization it is the incompressible stagnation-pressure scale, with pressure reference and viscous/finite-domain effects considered. Forcing it would hide the failure being tested.
