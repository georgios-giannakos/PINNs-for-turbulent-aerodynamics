# PINNs for turbulent aerodynamics

Thesis research code for steady, two-dimensional incompressible external aerodynamics.

- [k_e_noslip](k_e_noslip/): standard k-epsilon no-slip baseline and a matched Navier-Stokes comparator with shared NASA TMR geometry.

- [N-S_noslip_corrected](N-S_noslip_corrected/): standalone corrected NVIDIA Navier–Stokes no-slip baseline, matched to the k-epsilon study settings.

- [k_e_wf](k_e_wf/): matched standard k-epsilon wall-function candidate with offset geometry, signed traction and fixed diagnostics.

All three model variants are configured for 1,000,000 steps. Runtime preflight and smoke tests are required before production training; no trained or validated results are claimed.
