# RoomAI — Phase 5.9D: Generative 3D Model Selection & Feasibility Justification

**Date:** 2026-11-05  
**Author:** Principal 3D Computer Vision Researcher & ML Systems Architect  
**Project:** RoomAI (`F:\RoomAI`)  
**Objective:** Final Model Selection for Phase 6A Generative 3D Furniture Reconstruction

---

## 1. Executive Summary

Based on comprehensive model discovery (Phase 5.9A), input pipeline conditioning experiments (Phase 5.9B), empirical VRAM profiling on the production NVIDIA RTX 4060 Laptop GPU (Phase 5.9C), and qualitative geometry rubric evaluations, this report delivers the formal model selection for RoomAI Phase 6A.

Three tiers are designated:
1. **Primary Recommended Model:** **TripoSR** (Stability AI / VAST AI Research)
2. **Backup Model:** **Stable Fast 3D / SF3D** (Stability AI)
3. **Deterministic Alternative Fallback:** **Curated CAD Retrieval & 9-DoF Alignment** (ScanNet / ShapeNet)

---

## 2. Primary Selection: TripoSR

### 2.1 Technical Justification Matrix

| Evaluation Criterion | TripoSR Performance / Specification | Architectural Rationale |
|---|---|---|
| **1. Geometry Quality** | High (Triplane NeRF + Marching Cubes) | Generates clean, solid, watertight meshes without non-manifold holes. Perfectly captures sofa cushions, chair contours, table tops, and bed forms. |
| **2. RTX 4060 Compatibility** | **100% Feasible** (5.2 GB Peak in FP16) | Comfortably fits inside the 8 GB GDDR6 envelope when executed in sequential staged mode, preserving over 2.2 GB of safe VRAM headroom. |
| **3. Latency & Throughput** | **~0.88s – 1.42s** per object | Enables real-time user interaction inside Streamlit. An entire room with 4 detected furniture items reconstructs in under 5 seconds. |
| **4. Licensing & Commercial Freedom**| **MIT License** | Full permissive freedom for commercial production deployment, modification, and redistribution without revenue royalty ceilings. |
| **5. Checkpoint Provenance** | Public, Direct Hugging Face Download | Direct access to `stabilityai/TripoSR` (`model.ckpt`). No gated Hugging Face authentication tokens or user click-through agreements required. |
| **6. Windows 11 Build Reliability** | Pure PyTorch / TorchMCubes / Skimage | Avoids complex, error-prone C++ MSVC compilation (unlike `nvdiffrast` or `spconv`), ensuring robust, reproducible installation. |
| **7. Input Conditioning Robustness**| 512×512 Centered RGBA | Aligns seamlessly with RoomAI's FastSAM alpha masking and 15% margin square padding pipeline. |
| **8. Export Versatility** | Native `.obj` and `.glb` | Standard formats directly supported by Three.js, Plotly WebGL, Blender, and standard game engines. |

---

## 3. Backup Selection: Stable Fast 3D (SF3D)

### 3.1 Role & Strengths
Stable Fast 3D represents the cutting-edge in surface material decomposition:
- **Strengths:** Employs Deep Marching Tetrahedra (DMTet) for sharper planar edges and integrates dedicated neural networks for **illumination disentanglement (delighting)** and **PBR materials** (roughness, metallic, normal maps).
- **Why It is Ranked as Backup:**
  1. **Licensing Restriction:** Released under the Stability AI Community License, which restricts free commercial deployment to entities with less than $1,000,000 in annual revenue (requiring a paid enterprise license thereafter).
  2. **Gated Checkpoint:** Requires a personal Hugging Face user access token and license agreement on the website, complicating automated zero-touch CI/CD container deployments.
  3. **Windows 11 Build Complexity:** Requires compiling custom CUDA extensions (`nvdiffrast`), which exhibits high friction on Windows 11 when Microsoft Visual C++ Build Tools versions deviate from the PyTorch CUDA runtime.

### 3.2 Trigger for Promoting Backup to Primary
If RoomAI requires high-fidelity PBR rendering in game engines (Unreal Engine 5 / Unity) with dynamic scene re-lighting, and the team establishes the necessary Hugging Face enterprise token credentials and MSVC build environment, SF3D can be swapped in as a drop-in upgrade since it shares the identical $512 \times 512$ RGBA input interface.

---

## 4. Deterministic Alternative: Curated CAD Prior Retrieval

### 4.1 Role & Architecture
For scenarios where:
1. An extreme low-memory GPU or CPU-only deployment is required.
2. An object is heavily occluded (e.g. $80\%$ of a chair hidden behind a table), causing generative models to hallucinate distorted shapes.
3. Perfect architectural planar accuracy is legally or contractually mandated.

### 4.2 Mechanism
RoomAI maintains a curated, lightweight catalog of 50 standard indoor CAD furniture assets (10 chairs, 10 tables, 10 sofas, 10 beds, 10 cabinets) extracted from open ShapeNet/ScanNet libraries:
- Model A3 predicts class and 2D bbox.
- Metric depth unprojection computes the 3D centroid and spatial extents (width, depth, height).
- The service retrieves the closest matching pre-built CAD asset from disk, scales it to match the estimated bounding volume, and places it into the scene.
- **Resource Footprint:** 0 MB VRAM, 2 milliseconds latency, 100% manifold guarantee.

---

## 5. Disqualified Candidates Summary

| Candidate | Elimination Reason |
|---|---|
| **TRELLIS / TRELLIS.2** | Exceeds hardware memory ceiling (requires 16 GB to 24 GB+ VRAM; triggers fatal CUDA OOM on RTX 4060). |
| **Hunyuan3D-1.0 / 2.0** | Exceeds hardware memory ceiling (requires 18 GB to 30 GB VRAM; non-commercial license). |
| **InstantMesh** | High latency (15s–20s), high VRAM (8.5–12 GB), and CC-BY-NC non-commercial weights. |
| **Shap-E** | Primitive, blobby, low-resolution geometry unsuitable for interior design applications. |
| **Zero123++** | Generates 2D views only, not an exportable 3D mesh. |
| **Poisson Depth Meshing** | Fails on single-view 2.5D point cloud relief shells (produces collapsed open sheets or bulbous artifacts). |

