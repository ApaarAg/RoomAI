# ROOMAI — PHASE 5: STAGE 2 3D GAUSSIAN SPLATTING NOVEL-VIEW AUDIT

**Document ID:** `research/3d_reconstruction/reports/PHASE5_STAGE2_GENERALIZATION.md`  
**Date:** 2026-11-08  
**Auditor:** Senior 3D Computer Vision Research Scientist & Independent Model Evaluation Lead  
**Scope:** Forensic evaluation of the 3D Gaussian Splatting (3DGS) pipeline, camera pose validity, and novel-view generalization claims.

---

## 1. Executive Summary

In Phase 4, Stage 2 3D Gaussian Splatting training was executed on local GPU hardware, verifying:
* Instantiation of the 32.97M parameter `gs_head` and `gs_renderer`.
* Forward generation of 3D Gaussian Splat primitives (`means`, `scales`, `quats`, `opacities`, `sh`).
* Backpropagation through opacity and scale regularization losses.
* Successful checkpoint serialization (`stage2_real_verified.ckpt`, 2,644,926,673 bytes).

However, a rigorous scientific evaluation demonstrates that **Stage 2 cannot claim novel-view radiance generalization**.  
The primary bottleneck is not the neural architecture or rasterizer, but **the absence of calibrated multi-view camera poses in the training dataset**.

---

## 2. Technical Audit of the 3DGS Pipeline

### 2.1 Multi-View Correspondence vs Camera Poses

| Requirement for Valid 3DGS NVS | Audited Status in RoomAI | Scientific Finding |
| :--- | :--- | :--- |
| **Physical Multi-View Overlap** | **PARTIALLY VERIFIED** | 5 of 11 scenes (`Bright_Room`, `Dining_Table`, `Hall`, `Living_Room`, `Workspace`) have genuine visual overlap (>190 good ORB feature matches). 2 scenes are disjoint, and 2 are single images. |
| **Calibrated Camera Intrinsics ($K$)** | **UNVERIFIED (ASSUMED)** | Intrinsics were manually assumed as a standard $54^\circ$ pinhole ($f=518$), rather than calibrated from device lens distortion. |
| **Calibrated Camera Extrinsics ($SE(3)$)**| **INVALID (SYNTHETIC)** | Extrinsics were generated via a synthetic equation ($R_y(0.05 \cdot v), t_x = 0.1\cdot v$). The actual handheld capture trajectory was unmeasured. |
| **Novel-View Target Supervision** | **UNSUPPORTED** | Pixel-accurate novel-view rendering loss ($\mathcal{L}_{\text{rgb}} = \|I_{\text{rendered}} - I_{\text{target}}\|_1$) requires the virtual camera pose to exactly match the target camera's real viewpoint. With synthetic poses, novel-view photometric rendering is unanchored. |

---

## 3. Photometric Quality Metrics (PSNR, SSIM, LPIPS)

In literature (e.g., Mildenhall et al., Kerbl et al.), novel-view synthesis quality is evaluated by rendering from a known held-out camera pose $P_{\text{target}}$ and computing:
$$\text{PSNR} = 10 \log_{10}\left(\frac{1}{\text{MSE}(I_{\text{pred}}, I_{\text{gt}})}\right)$$
$$\text{SSIM}(I_{\text{pred}}, I_{\text{gt}}), \quad \text{LPIPS}(I_{\text{pred}}, I_{\text{gt}})$$

### Why PSNR/SSIM Cannot Be Claimed on the Current Dataset:
1. If the model renders a view at virtual pose $P_{\text{synth}} = [R_y(0.1) | t_x=0.2]$, but the real camera was tilted down and translated along Z ($P_{\text{actual}} \ne P_{\text{synth}}$), comparing pixels directly measures camera trajectory mismatch, NOT radiance field quality.
2. Training loss on training views converged to zero ($1.797\times 10^{-5}$), which proves that the model memorized the training viewpoints.
3. **Crucial Scientific Distinction:** Memorizing training views does NOT equal novel-view radiance generalization.
4. Until calibrated COLMAP bundle adjustment or ScanNet/ARKitScenes poses are integrated, novel-view PSNR/SSIM metrics cannot be scientifically substantiated.

---

## 4. Geometric Consistency of 3DGS Representations

Inspection of `test_out["splats"]`:
* **Splat Centers (`means`):** Verified 100% finite coordinates, consistent with the predicted depth and point map from Stage 1.
* **Splat Opacities (`opacities`):** Regularized to prevent opacity saturation.
* **Splat Scales (`scales`):** Bounded and finite, avoiding giant splat artifacts.
* **Computational Overhead:** Adding 3DGS rendering increases forward pass latency from **248.9 ms to 482.5 ms** (+93.8% compute overhead) and increases model parameter count from 1.223B to 1.256B (+32.97M parameters).

---

## 5. Audit Conclusion

1. **Stage 2 3DGS execution mechanics are verified and functional.**
2. **Novel-view synthesis generalization is UNVERIFIED due to synthetic camera pose limitations.**
3. **Stage 2 must remain classified as EXPERIMENTAL / PROOF-OF-CONCEPT.**
4. Stage 2 must NOT be deployed to production as a novel-view synthesizer until real calibrated multi-view datasets (ScanNet / Hypersim) are acquired.

