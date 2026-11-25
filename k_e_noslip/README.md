# k_e_noslip — thesis study baseline

Steady, two-dimensional incompressible standard k-epsilon RANS PINN with physical no-slip and a matched Navier-Stokes (NS) comparator. **No wall functions, wall offset or prescribed wall epsilon.** This is intentionally a high-Re no-slip baseline, not a validated low-Re near-wall closure.

## Model-specific changes

See the [main README](../README.md) for the full NVIDIA and Ver1 change tables, geometry, quadrature, dynamic sampling and step-indexed history, including the original figures. Relative to NVIDIA NS, this version adds the standard high-Re k–epsilon closure, four decoupled networks, full RANS stress divergence with mean static pressure, fixed transport scales, stable positive output mapping, normalized freestream initialization and tanh aggregation. Relative to WF, it imposes physical no-slip and k=0 (positive-floor approximation), with no offset or wall-epsilon prescription.

The RANS equations are implemented in [custom_k_ep_corrected.py](custom_k_ep_corrected.py). With U=(u,v), S=(grad U+grad Uᵀ)/2:

$
\nabla\cdot\mathbf U=0
$

$
(\mathbf U\cdot\nabla)\mathbf U+\nabla p/\rho+\tfrac23\nabla k-\nabla\cdot[2(\nu+\nu_t)\mathbf S]=0
$

$
\mathbf U\cdot\nabla k-\nabla\cdot[(\nu+\nu_t/\sigma_k)\nabla k]-P_k+\varepsilon=0
$

$
\mathbf U\cdot\nabla\varepsilon-\nabla\cdot[(\nu+\nu_t/\sigma_\varepsilon)\nabla\varepsilon]-(C_{\varepsilon1}P_k-C_{\varepsilon2}\varepsilon)\varepsilon/(k+10^{-20})=0
$

Here ν_t=C_mu k²/(ε+1e-20), P_k=2ν_t S:S; C_mu=0.09, sigma_k=1, sigma_ep=1.3, C_ep1=1.44, C_ep2=1.92. Fixed transport reference scales are k_ref=1e-3 and ep_ref=1e-2. The output mapping uses floor 1e-12 and stable softplus scaled so raw zero maps to the inlet value. Initialization uses the same freestream targets in both near and far regions, with normalized residuals. The fixed tanh schedule is centered at 20k steps (smoothness 0.0005); no adaptive loss weighting.

The physical boundary inferencer records p,u,v,k,ep, four AD velocity derivatives, physical normals and area; no modeled wall shear or y+ is provided. See [surface data](../README.md#surface-data-for-drag-analysis). Use the original archive path override if required: `python ../tools/archive_training.py --archive-dir /path/to/history/ke_run -- airfoil_ke_corrected.py`.

## Reproduction

Both executables use the same Warp geometry, sampling settings, domain, Reynolds number and training budget. NS retains a single u,v,p network and sum loss. The k-epsilon model retains four decoupled networks and initialization aggregation.

Use a single-GPU NVIDIA PhysicsNeMo Sym environment compatible with the legacy constraint API (e.g. 25.08), NVIDIA Warp, PyTorch and NumPy. The original geometry and cache preflight passed on the reported CPU and RTX 4500 Ada (Warp 1.10.0). The k-epsilon smoke test reached step 1000 with a finite loss and the expected cache reset. The new archive launcher and augmented boundary inferencer still require the short integration smoke test below.

```bash
cd k_e_noslip
python -m compileall -q .
python preflight_geometry.py --device cpu
python preflight_geometry.py --device cuda:0
python airfoil_ke_corrected.py training.max_steps=1100 training.rec_constraint_freq=1000 network_dir=smoke_ke
python airfoil_ns.py training.max_steps=1100 training.rec_constraint_freq=1000 network_dir=smoke_ns
# Confirm the step-1000 refresh and inspect both VTP outputs.
python ../tools/archive_training.py --self-test
python ../tools/archive_training.py --interval 1000 -- airfoil_ke_corrected.py training.max_steps=1100 training.rec_constraint_freq=1000 network_dir=smoke_archive_ke
# Production training with step-indexed history:
python ../tools/archive_training.py -- airfoil_ke_corrected.py
python ../tools/archive_training.py -- airfoil_ns.py
```

Default Re=6,000,000, alpha=0, nominal thickness=0.12 and camber=0. Both configurations set max_steps=1,000,000. To compare at the original NS Reynolds number, pass custom.reynolds_number=1000000.0 to **both** executables. Start new checkpoint directories; do not reuse incompatible historical checkpoints. A checkpoint restart is not bitwise reproducible before the next scheduled sampling reset.


 


## Constraint inventory

| Constraint | NS | k-epsilon |
|---|---|---|
| inlet | u=1, v=0 | u=1, v=0, k_inf, ep_inf |
| outlet | p=0 | p=0 |
| top_bot | u=1, v=0 | u=1, v=0, k_inf, ep_inf |
| airfoil | u=0, v=0 | u=0, v=0, k=0 (positive-floor approximation) |
| interior_init | — | Five initialization residuals |
| interior_init_far_field | — | Five initialization residuals in far region |
| interior_far_field | continuity, momentum_x, momentum_y | Same plus k_equation, ep_equation |
| interior_near_field | Same three PDEs with SDF weighting | Same five PDEs with SDF weighting |

At Re=6e6, U=c=rho=1, I=0.001 and nu_t/nu=10, nu=1/6e6, k_inf=1.5e-6, ep_inf=1.215e-7. The k-epsilon transition remains centered at 20,000 steps. NS is a steady no-closure comparator, not DNS of turbulence.

## Verification status and limitations

- Reported server verification (Warp 1.10.0, CUDA 12.8, RTX 4500 Ada): full geometry and cache preflight PASS on CPU and CUDA; k-epsilon training reached step 1000, loss 0.002610, and logged the scheduled cache reset. The log does not independently establish that step-1001 coordinates changed or that the 1100-step run finished. NS smoke test and the new archive launcher/boundary-output integration remain to be confirmed.
- A million-step configuration is not proof of convergence or successful completion. Record software versions, seed, commit hash and actual final step.
- The original finite domain is much smaller than the NASA reference domain. The incompressible PDE formulation also differs from the low-Mach compressible reference. Geometry matching is not full benchmark reproduction.
- Standard high-Re k-epsilon without wall treatment is not a physically complete near-wall closure. Treat Cf and drag predictions as baseline results, not verified wall-resolved RANS.
- Before thesis conclusions, check residual convergence, mass conservation, symmetry, domain independence and geometry resolution sensitivity; compare Cp, Cf, total CD and separate viscous/pressure components to reference data.

## References

1. [NASA TMR: NACA 0012 validation case](https://tmbwg.github.io/turbmodels/naca0012_val.html).
2. [NASA TMR: NACA 0012 numerical analysis and grid convergence](https://tmbwg.github.io/turbmodels/naca0012numerics_val.html).
3. Launder and Spalding (1974), *The numerical computation of turbulent flows*, CMAME 3(2), 269–289. [DOI](https://doi.org/10.1016/0045-7825%2874%2990029-2).
4. [NVIDIA PhysicsNeMo Sym constraint API](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/api/physicsnemo.sym.domain.constraint.html).
5. [NVIDIA Warp documentation](https://nvidia.github.io/warp/).
6. [Original PINNs project](https://github.com/georgios-giannakos/PINNs).

Applicable NVIDIA Apache-2.0 source notices are retained. New modules use Apache-2.0 SPDX identifiers.
