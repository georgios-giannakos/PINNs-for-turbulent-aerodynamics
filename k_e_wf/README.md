# k_e_wf — thesis wall-function comparator

Steady 2-D incompressible standard high-Re k-epsilon RANS PINN with corrected equilibrium Launder–Spalding wall treatment. Built from [k_e_noslip](../k_e_noslip/) at a40d7e1aefa28fec04a848884e6b09a238003b4f. The [standalone NS baseline](../N-S_noslip_corrected/) is unchanged. **Research candidate, not a validated solution.**

## Reproduction

Same single-GPU legacy PhysicsNeMo Sym environment (e.g. 25.08), Warp, PyTorch, NumPy and VTK as the baselines. Use a fresh checkpoint; no historical checkpoint is compatible.

```bash
cd k_e_wf
python -m compileall -q .
python preflight_geometry.py --device cpu
python preflight_wf.py --device cpu
python preflight_geometry.py --device cuda:0
python preflight_wf.py --device cuda:0
python airfoil_ke_wf.py training.max_steps=1100 training.rec_constraint_freq=1000 network_dir=smoke_wf
python ../tools/archive_training.py --self-test
python ../tools/archive_training.py --interval 1000 -- airfoil_ke_wf.py training.max_steps=1100 training.rec_constraint_freq=1000 network_dir=smoke_archive_wf
# Pilot: allow the initialization transition to finish
python ../tools/archive_training.py -- airfoil_ke_wf.py training.max_steps=50000 network_dir=pilot_wf
# Production only after acceptable pilot diagnostics
python ../tools/archive_training.py -- airfoil_ke_wf.py
# Alternative historical Reynolds number (fresh checkpoint)
python ../tools/archive_training.py -- airfoil_ke_wf.py custom.reynolds_number=1000000.0 network_dir=network_checkpoint_ke_wf_re1m
```

Smoke tests verify plumbing, not convergence. The common initialization schedule is centered at step 20,000. Check the step-0/1000 archives, new constraint coordinates after refresh, finite losses, VTP fields and monitor CSVs. To continue the same pilot rather than start a fresh production run, retain the pilot directory and all settings except max_steps; verify its optimizer checkpoint step. The shared archive launcher saves snapshots every 50k including step 0, not optimizer/model weights.

## Matched numerical specification

| Setting | Value |
|---|---|
| Benchmark | NASA TMR NACA 0012, alpha=0, thickness=0.12, camber=0, Re=6e6, U=c=rho=1 |
| Outer rectangle | x [-2.25,5.25], y [-2.5,2.5] |
| Near rectangle | x [-0.32,1.28], y [-0.5,0.5] |
| Networks | Four decoupled fully connected (u,v),p,k_star,ep_star; width 512, 6 layers, SiLU |
| Optimizer | Adam 1e-4, decay 0.95/8000 |
| Training | 1M steps; CUDA graphs off, zero workers |
| Batches | inlet/outlet/top_bot 640, airfoil 512, near 7000, far 6000 |
| Sampling | Same per-constraint seed and cache refresh every 1000 steps after inference |
| Recording | inference and monitors 1k, constraints 10k, optional history 50k |
| Outer BC | u=1,v=0,k_inf,ep_inf at inlet/top/bottom; p=0 at outlet |
| RANS | Byte-identical complete symmetric stress divergence, mean static p, 2/3 grad(k), fixed k_ref=1e-3, ep_ref=1e-2 |
| PDE weights | Same far weights and near 2*max(physical SDF,0) |

At Re=6e6, nu=1/6e6, k_inf=1.5e-6, ep_inf=1.215e-7 (I=0.001, viscosity ratio 10). The geometry, RANS PDE, original aggregator, cache and solver files are copied unchanged. Near and wall coordinates necessarily differ because of the unresolved layer.

![Physical geometry](../k_e_noslip/figures/geometry.svg)

![Domain](../k_e_noslip/figures/domain.svg)

![Resampling](../k_e_noslip/figures/resampling.svg)

## Offset geometry

Underlying physical polygon: same NASA TMR 501 cosine points per side, 1,000 segments, signed segment SDF and normals into solid. The fixed nominal convex polygon is offset to P=W-d*n, d/c=target_yplus/(Re*utau_ref) = 0.000333333 at Re=6e6 (80, 0.04), 0.002 at Re=1e6. This is a selection heuristic, not a prescribed k or y+ constraint. No y+ penalties and no adaptive weighting.

Wall training samples offset segments and round vertex arcs with offset-perimeter weights. Interior rejects true wall distance <=d (not strips at top/bottom), with resolved measure 1.5192284 - perimeter*d - pi*d^2 by the convex parallel-body formula, subject to float32 polygon tolerance. Far area remains 35.9. The original SDF remains the PDE weight. The boundary inferencer retains the physical-wall arc-length samples/weights mapped to P; actual distance is checked. Only the fixed nominal profile is supported in this WF benchmark; general nonconvex/cambered offsets require new verification. Normal at P is an approximate impermeability condition, not exact boundary-layer mass conservation.

## Wall conditions

At P: U_t=U dot t, u_tau=C_mu^0.25 sqrt(k), y+=u_tau*d/nu. t points downstream, n from fluid into solid. U+=y+ below the continuous linear/log join at 11.224708076; above, U+=log(E*y+)/kappa with E=9.793 and kappa=0.4187. Epsilon uses u_tau^3/(kappa*d) with an exponential blend to 2*nu*k/d^2 at low y+. These safeguards do NOT make standard high-Re k-epsilon a validated low-Re closure.

| Residual at P | Fixed lambda |
|---|---:|
| U dot n = 0 | 1 |
| abs(U_t)-u_tau*U+(y+) = 0 | 10 |
| (tau_RANS-tau_model)/utau_ref^2 = 0 | 1 |
| (ep-ep_wall)/(utau_ref^3/(kappa*d_ref)) = 0 | 1 |

tau_model = nu*U_t/d * y+/U+ (ratio exactly 1 on linear branch). tau_RANS = t dot [(nu+nu_t)(grad U + grad U^T)] dot (-n). Both are signed kinematic traction on solid. The tangential velocity wall law closes equilibrium k jointly with velocity; stress closes tangential momentum. No additional k-gradient or independent velocity Dirichlet condition is imposed. When the equilibrium residual vanishes, abs(tau_model)=u_tau^2. No physical-wall no-slip or k=0. Relative tangential emphasis is inspired by 22D, not raw copied weights: residual scales differ. Fixed weights are pilot choices, not tuned or validated.

Unlike the inlet-scaled positive mapping in the baseline, shifted stable softplus uses k_scale=1e-3, ep_scale=1e-2, with raw zero still equal to inlet values and floor 1e-12. This allows realistic wall magnitudes without excessively large raw outputs. Initialization aggregator and far initialization are unchanged; near initialization seeds equilibrium-scale k/epsilon and approximately projected tangential velocity, decaying over 0.02c. This is a warm-start heuristic only, not data supervision or a persistent constraint. These two explicit WF-specific numerical changes target the collapse seen in the historical V4; their effectiveness is unverified.

## Constraints

| Name | Outputs |
|---|---|
| inlet, top_bot | u,v,k,ep |
| outlet | p |
| airfoil | velocity_wall_normal_wf, velocity_wall_parallel_wf, wall_shear_stress_wf, ep_wf |
| interior_init | u_near_init,v_near_init,p_near_init,k_near_init,ep_near_init |
| interior_init_far_field | u_init,v_init,p_init,k_init,ep_init |
| interior_near_field, interior_far_field | continuity,momentum_x,momentum_y,k_equation,ep_equation |

Same far criterion for training/inference, same independent named caches for all constraints. No PDE sampling inside excluded strip.

## Inference, monitors and force integration

Same three inferencer names: near_field_1, boundary_1, far_field_1. Near/far record u,v,p,k,ep; near also wall_distance. Boundary VTP coordinates x,y are **P**, not W; physical x_wall,y_wall, physical area weights, normals, downstream tangents and normal_distance are retained. Fields include u,v,p,k,ep and four AD velocity derivatives, signed wall_tau_model and wall_tau_rans, u_tau (from k), u_tau_shear, y_plus_wall, u_plus_wall, cf_wall, nu_t_wall, ep_wall_target, four residuals. Wall monitors report y+ range/fractions, k,ep,u_tau, signed Cf, reverse-flow fraction and RMS residuals every 1k; they do not change the loss.

```bash
python surface_coefficients.py /path/to/snapshot/inferencers --output surface_50k.csv
```

Use the actual archive directory. The script estimates p_inf from upstream far points (x<-1.5) or accepts --p-inf. Cp=2*(p(P)-p_inf), Cf=2*tau_model; integrates Cp*n + Cf*t on **physical** contour quadrature, not offset weights, to report separate CD and CL pressure/shear contributions. Reports normal closure errors. Pressure at P approximates wall pressure. Converge surface quadrature before citing absolute drag; 512 randomly sampled points are diagnostic. The postprocessor is WF-specific.

## Verification and limitations

- New Warp and PhysicsNeMo runtime tests are **pending** in the target environment. No validated or trained WF results are claimed. The shared baseline preflight does not certify the new wrapper or graph.
- Before 1M production, pass CPU/CUDA preflight and 1100-step smoke (including archive integration); inspect the 50k pilot after the 20k transition. Check inlet, individual PDE/BC residuals, k/ep and y+, pressure peak/recovery, velocity/wake, symmetry, Cp/Cf/CD/CL. Low total loss is insufficient.
- Local-equilibrium wall treatment, stagnation, sharp TE, curvature, pressure gradients and separation limit accuracy. Standard high-Re k-epsilon remains invalid as a low-Re closure; linear/blended safeguards only avoid singularity. Domain smaller than NASA and incompressible rather than low-Mach compressible. NASA SA and k-epsilon variants are context, not exact equivalence.
- Record commit, software versions, seed, actual steps, and pressure reference. No adaptive weighting, historical checkpoint, y+ penalty or trained results included.

## Sources

- [Launder & Spalding (1974)](https://doi.org/10.1016/0045-7825%2874%2990029-2).
- [OpenFOAM epsilonWallFunction](https://doc.openfoam.com/2312/tools/processing/boundary-conditions/rtm/derived/wall/epsilonWallFunction/) (exponential blend).
- [NASA TMR NACA 0012 validation](https://tmbwg.github.io/turbmodels/naca0012_val.html) and [grid convergence](https://tmbwg.github.io/turbmodels/naca0012numerics_val.html).
- [PhysicsNeMo Sym 25.08 monitors](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/api/physicsnemo.sym.domain.monitor.html) and [inferencers](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/api/physicsnemo.sym.domain.inferencer.html).
- [Baseline README](../k_e_noslip/README.md). Historical 22D and V4 motivated the design, not validation.

Applicable NVIDIA Apache-2.0 notices retained; new modules use Apache-2.0 SPDX identifiers.
