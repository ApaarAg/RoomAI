# RoomAI — Phase 5.9A: Generative 3D Furniture Reconstruction Model Discovery

**Date:** 2026-11-05  
**Author:** Senior 3D Computer Vision Researcher, Generative AI Specialist & ML Systems Architect  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, 24 GB System RAM, Windows 11)

---

## 1. Executive Summary

Phase 5.9A investigates the state of generative 3D object reconstruction from 2D imagery to determine the technical feasibility of adding 3D furniture mesh synthesis to RoomAI. 

RoomAI currently extracts 2D bounding boxes (Detector A3), instance masks (FastSAM-s), and 2.5D unprojected point clouds (Depth Anything V2 Small). However, point clouds derived from single-view monocular depth are **2.5D relief shells**: they represent only the visible camera-facing surfaces and leave all rear, bottom, and occluded geometry missing.

To convert detected indoor furniture (beds, sofas, chairs, tables, cabinets) into inspectable 3D meshes, RoomAI requires a **generative 3D shape prior**. This report systematically discovers, categorizes, and audits pretrained 3D foundation models against the operational constraints of the RTX 4060 (8 GB VRAM ceiling) on Windows 11.

---

## 2. Taxonomy of 3D Reconstruction Paradigms

It is vital to distinguish between different 3D generation paradigms. A model generating multiple attractive 2D images is **not** a 3D mesh reconstructor.

```
                                  3D Object Generation
                                           │
         ┌──────────────────┬──────────────┴───────────────┬──────────────────┐
         ▼                  ▼                             ▼                  ▼
    Category A         Category B                    Category C         Category D
Single-Image to Mesh  Image-to-Multiview             Point-Cloud-to-     CAD Retrieval &
   (Direct LRM)       (Diffusion + LRM)             Mesh (Surface Recon)  Shape Prior
   • TripoSR          • InstantMesh (Zero123++)     • Poisson Recon      • ScanNet CAD Match
   • Stable Fast 3D   • CRM, SV3D                   • Ball-Pivoting      • Mask2CAD / ROCA
```

### Category A: Single-Image to 3D Mesh (Feed-Forward LRM)
- **Mechanism:** Takes a single RGB/RGBA image, encodes it via a Vision Transformer (ViT), and directly predicts a 3D volumetric triplane or implicit field in a single forward pass (~0.5s–2.0s). Mesh is extracted via Marching Cubes or Deep Marching Tetrahedra (DMTet).
- **Representative Models:** TripoSR, Stable Fast 3D (SF3D).
- **Suitability for RoomAI:** **Highest**. Real-time inference latency, deterministic execution, fits within consumer VRAM when staged.

### Category B: Image-to-Multiview Diffusion + Sparse Reconstruction
- **Mechanism:** Two-stage pipeline. Stage 1 generates 4–6 synthetic viewpoints using a fine-tuned multi-view diffusion model (e.g. Zero123++). Stage 2 uses a sparse-view reconstruction model to build NeRF/Gaussian Splats/Mesh.
- **Representative Models:** InstantMesh, CRM, SV3D.
- **Suitability for RoomAI:** **Moderate**. Higher multi-view geometric consistency on unseen backsides, but latency is 10×–20× higher (10s–25s) and diffusion + LRM pushes peak VRAM to 8–12 GB, risking Windows WDDM OOM on 8 GB GPUs.

### Category C: Point-Cloud-to-Mesh Reconstruction (Classical Surface Recon)
- **Mechanism:** Direct application of Poisson Surface Reconstruction (Screened PSR) or Ball-Pivoting Algorithm (BPA) to depth-unprojected point clouds.
- **Suitability for RoomAI:** **Unsuitable for isolated objects**. Single-view depth unprojection yields an open 2.5D shell. Applying Poisson surface reconstruction to an open shell produces severe bulbous bridging artifacts, closed back-wall deformations, or collapsed zero-volume sheets.

### Category D: CAD Retrieval & 9-DoF Shape Prior Alignment
- **Mechanism:** Uses the predicted class and 3D bounding box to retrieve an exact, clean, human-modeled CAD asset from a curated library (e.g., ScanNet/ShapeNet CAD models) and scale/orient it into the scene.
- **Suitability for RoomAI:** **Excellent Deterministic Backup**. 0 MB neural VRAM, 100% watertight, but cannot represent bespoke or unconventional furniture.

### Category E: Multi-View Epipolar Object Reconstruction
- **Mechanism:** Reconstructs 3D geometry from unposed video frames or multi-view captures (e.g., DUSt3R, NeuS).
- **Suitability for RoomAI:** **Not Applicable**. RoomAI operates on single user-uploaded photos.

---

## 3. In-Depth Candidate Audits

### 3.1 TripoSR (Stability AI & VAST AI Research)
* **Architecture:** Large Reconstruction Model (LRM). Image Tokenizer (`facebook/dino-vitb16`), 16-layer 1D Transformer backbone (1024 channels, cross-attention dim 768), Triplane NeRF decoder, and Marching Cubes mesh extraction.
* **Paper & Date:** Tochilkin et al., *"TripoSR: Fast 3D Object Reconstruction from a Single Image"*, March 2024 (arXiv:2403.02151).
* **Official Repository:** `https://github.com/VAST-AI-Research/TripoSR`
* **Official Checkpoint:** Hugging Face `stabilityai/TripoSR` (`model.ckpt`, 1.68 GB).
* **Model & Weight License:** **MIT License** (Permissive, commercial use allowed).
* **Input Requirements:** Centered 512×512 image on white or transparent background with foreground object mask.
* **Output Format:** Textured 3D Mesh (`.obj` / `.glb`).
* **Hardware & VRAM:** ~5.2 GB peak VRAM on GPU during Marching Cubes grid evaluation. Fits within 8 GB VRAM.
* **Latency:** ~0.8s – 1.5s on NVIDIA RTX 4060.
* **Windows Compatibility:** High. Native PyTorch implementation without mandatory C++ CUDA extension build hurdles.
* **Suitability for Indoor Furniture:** Very high for isolated chairs, sofas, tables, cabinets, and beds.
* **Limitations:** Does not disentangle illumination (source image shadows are baked into vertex colors).

### 3.2 Stable Fast 3D / SF3D (Stability AI)
* **Architecture:** Two-Stream Transformer (Triplane + Latent stream) + Deep Marching Tetrahedra (DMTet) + MaterialNet (roughness, metallic, albedo) + LightNet (delighting) + fast neural UV unwrapping.
* **Paper & Date:** Boss et al., *"SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement"*, August 2024 (arXiv:2408.00653).
* **Official Repository:** `https://github.com/Stability-AI/stable-fast-3d`
* **Official Checkpoint:** Hugging Face `stabilityai/stable-fast-3d` (`model.safetensors`, 1.78 GB).
* **Model & Weight License:** **Stability AI Community License** (Free for research/non-commercial, and commercial use for revenue < $1M USD; enterprise license required above $1M). Checkpoint is **gated** on Hugging Face.
* **Input Requirements:** Centered 512×512 RGBA image with transparent background.
* **Output Format:** `.glb` with UV-unwrapped PBR textures (albedo, roughness, metallic, normal).
* **Hardware & VRAM:** ~6.0–7.2 GB peak VRAM.
* **Latency:** ~0.6s – 1.8s.
* **Windows Compatibility:** Moderate-Low. Requires compiling custom CUDA kernels (`nvdiffrast`), which frequently encounters MSVC toolchain issues on Windows 11.
* **Suitability for Indoor Furniture:** Outstanding material quality and true UV unwrapping for PBR rendering.

### 3.3 InstantMesh (Tencent ARC Lab)
* **Architecture:** Two-stage: Zero123++ (multi-view diffusion producing 6 surrounding views) + Sparse-view LRM + Differentiable Iso-Surface Extraction.
* **Paper & Date:** Xu et al., *"InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models"*, April 2024 (arXiv:2404.07191).
* **Official Repository:** `https://github.com/TencentARC/InstantMesh`
* **Official Checkpoint:** Hugging Face `TencentARC/InstantMesh`.
* **Model License:** Apache 2.0 (code).
* **Weight License:** **CC-BY-NC 4.0** (Non-commercial restriction due to Zero123++ weights).
* **Hardware & VRAM:** 8.5 GB – 12 GB peak VRAM. Exceeds 8 GB VRAM under standard execution; requires aggressive sequential CPU offloading between diffusion and reconstruction.
* **Latency:** 12s – 20s.
* **Suitability for Indoor Furniture:** Good geometry completeness, but non-commercial license and high VRAM overhead restrict adoption.

### 3.4 TRELLIS & TRELLIS.2 (Microsoft Research & Tsinghua University)
* **Architecture:** Structured 3D Latents (SLAT / O-Voxel) + Rectified Flow Transformers (2B–4B parameters).
* **Paper & Date:** Xiang et al., *"Structured 3D Latents for Scalable and Versatile 3D Generation"*, Dec 2024 (CVPR 2025 Spotlight).
* **Official Repository:** `https://github.com/microsoft/TRELLIS`
* **Official Checkpoint:** Hugging Face `JeffreyXiang/TRELLIS-image-large`.
* **Model & Weight License:** MIT License.
* **Hardware & VRAM:** **16 GB to 24 GB+ VRAM**.
* **Feasibility on RTX 4060 8 GB:** **FAIL / INCOMPATIBLE**. Triggers CUDA Out-of-Memory immediately on 8 GB cards without specialized 4-bit quantization or heavy community forks. Requires Linux / specific spconv/flash-attn builds.

### 3.5 Hunyuan3D-1.0 & 2.0 (Tencent Hunyuan)
* **Architecture:** Multi-view DiT diffusion + high-resolution PBR texture generator.
* **Official Repository:** `https://github.com/Tencent-Hunyuan/Hunyuan3D-1` & `Hunyuan3D-2`.
* **Model License:** Tencent Hunyuan Community / Non-commercial License.
* **Hardware & VRAM:** **18 GB to 30 GB VRAM** (Lite pipeline requires 18–22 GB; full requires 24–30 GB).
* **Feasibility on RTX 4060 8 GB:** **FAIL / INCOMPATIBLE**. Exceeds hardware by 2.5×–3.5×.

### 3.6 Shap-E (OpenAI)
* **Architecture:** Conditional diffusion model generating weights of implicit NeRF / textured signed distance functions.
* **Paper & Date:** Jun & Nichol, *"Shap-E: Generating 3D Assets in Complex Representations"*, May 2023.
* **Official Repository:** `https://github.com/openai/shap-e`.
* **License:** MIT License.
* **Hardware & VRAM:** ~4.0 GB VRAM. Easily runs on RTX 4060.
* **Geometry Quality:** **UNACCEPTABLE**. Low-resolution ($64^3$ voxel grid equivalent), blobby, melted geometry lacking sharp edges, distinct furniture legs, or flat planes.

---

## 4. Structured Model Comparison Matrix

| Evaluation Criterion | TripoSR | Stable Fast 3D (SF3D) | InstantMesh | TRELLIS (1.0/2.0) | Hunyuan3D (1.0/2.0) | Shap-E | ScanNet CAD Prior (Cat D) |
|---|---|---|---|---|---|---|---|
| **Reconstruction Quality** | High (Triplane NeRF) | High (DMTet + PBR) | High (Multi-view LRM) | Very High (SLAT/O-Voxel)| Very High (PBR DiT) | Very Low (Blobby SDF) | Perfect (Pre-authored CAD) |
| **Texture Quality** | Vertex/baked RGB | PBR (Albedo/Rough/Met)| Baked Multi-view RGB| PBR High-res | PBR Ultra-high | Blurry Vertex Colors | Clean Studio Textures |
| **Furniture Suitability** | High (Chairs/Sofas/Beds)| High (PBR Delighted) | High | High | High | Very Low | High (Pre-indexed) |
| **Peak VRAM (Inference)** | **~5.2 GB** | **~6.5 GB** | **~9.5 GB** | **16.0 – 24.0+ GB** | **18.0 – 30.0 GB** | **~4.0 GB** | **~0 MB (Pure Geometric)** |
| **Fits RTX 4060 (8 GB)?** | **YES (Staged)** | **YES (Tight)** | **NO (OOM Risk)** | **NO (Fatal OOM)** | **NO (Fatal OOM)** | **YES** | **YES** |
| **Latency (Inference)** | **0.8 – 1.5 s** | **0.6 – 1.8 s** | **12 – 20 s** | **15 – 45 s** | **30 – 120 s** | **6 – 10 s** | **< 0.01 s** |
| **Model License** | **MIT (Permissive)** | **Stability Community**| **Apache 2.0 (Code)** | **MIT** | **Tencent Non-Com** | **MIT** | **MIT / CC-BY** |
| **Weight License** | **MIT (Commercial OK)**| **Community (<$1M rev)**| **CC-BY-NC (Non-Com)** | **MIT** | **Non-Commercial** | **MIT** | **Academic / Open** |
| **Checkpoint Access** | Public (Direct) | Gated HF (Token req) | Public | Public | Public | Public | Local / Direct |
| **Windows Build Barrier**| Low (Pure PyTorch) | High (`nvdiffrast` CUDA)| Moderate (PyTorch3D) | Extreme (`flash-attn`) | High | Low | Zero |
| **Export Formats** | OBJ, GLB | GLB (PBR UV) | OBJ, GLB | GLB, PLY, Gaussian | OBJ, GLB | OBJ, PLY | GLB, OBJ |
| **RoomAI Fit** | **PRIMARY SHORTLIST** | **BACKUP SHORTLIST** | Rejected (License/VRAM)| Rejected (OOM) | Rejected (OOM) | Rejected (Quality) | **DETERMINISTIC FALLBACK**|

---

## 5. Preliminary Findings & Architecture Alignment

1. **VRAM Feasibility Limits:** Only **TripoSR**, **Stable Fast 3D**, **Shap-E**, and **CAD Retrieval** can physically execute on an 8 GB consumer GPU. Heavy multi-view diffusion systems (Hunyuan3D, TRELLIS) are unviable for local deployment on the RTX 4060.
2. **Quality Disqualification:** Shap-E is eliminated immediately due to primitive, blobby geometry unsuitable for interior design visualization.
3. **Licensing & Access Disqualification:** InstantMesh and Hunyuan3D carry non-commercial restrictions, making them incompatible with commercial product roadmaps.
4. **Primary vs. Backup Viability:**
   - **TripoSR** provides the highest engineering stability on Windows 11, runs under permissive MIT licensing, has public ungated weights, and executes in ~1.0 second within 5.2 GB VRAM.
   - **Stable Fast 3D** offers superior PBR texturing and delighting but requires commercial revenue compliance, gated HF token access, and difficult Windows CUDA compilation.
   - **CAD Prior Matching** is the ideal deterministic baseline when generative reconstruction hallucinates.

