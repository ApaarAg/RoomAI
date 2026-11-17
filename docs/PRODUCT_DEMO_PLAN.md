# RoomAI — LinkedIn Demonstration & Screen Recording Plan

**Document Version:** 1.0.0 (Phase 7.2A Product Showcase)  
**Product:** **RoomAI — Understand Your Space. Visualize It in 3D.**  
**Target Platform:** LinkedIn Video Post, Recruiter Evaluation, Technical Portfolio  
**Target Duration:** Exactly 2 minutes (120 seconds)  
**Hardware Setup:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) running Windows 11  
**Recording Resolution:** 1920 × 1080 (16:9 Full HD), 60 FPS, browser zoom at 100%  
**Audio Configuration:** Clean voiceover, neutral pace, professional engineering tone  

---

## 1. Executive Demonstration Narrative

RoomAI is a local, privacy-first computer vision studio that elevates monocular room photographs into interactive 3D spatial reconstructions. Unlike generative AI models that hallucinate imaginary meshes, RoomAI performs mathematically grounded pinhole backprojection from physically calibrated metric depth, paired with custom YOLO11s object detection.

This demo demonstrates:
1. Immediate product identity and real-mode GPU execution.
2. Fast, high-resolution indoor furniture detection (Model A3 800px).
3. Honest system governance (GPU memory preservation in Vision Q&A).
4. Physically grounded 3D scene reconstruction and interactive inspection.
5. Zero-dependency export to universal glTF 2.0 (GLB) and Stanford PLY formats.

---

## 2. Step-by-Step Recording Timeline & Narration Script

```
┌─────────────────┬────────────────────────────────────────────────┬──────────────┐
│ Time Interval   │ Visual Action & Screen Display                 │ Screen Asset │
├─────────────────┼────────────────────────────────────────────────┼──────────────┤
│ 0:00 – 0:15     │ RoomAI Branding, Tagline & Live GPU Status     │ Screen 1     │
│ 0:15 – 0:35     │ Image Ingestion & Model A3 Object Detection    │ Screen 2     │
│ 0:35 – 0:55     │ Vision Q&A & Transparent VRAM Governance       │ Screen 3     │
│ 0:55 – 1:25     │ Metric Depth & Interactive 3D Reconstruction   │ Screen 4 & 5 │
│ 1:25 – 1:45     │ 3D Orbit, Bounding Boxes & Spatial Graph       │ Screen 5     │
│ 1:45 – 2:00     │ glTF 2.0 GLB / PLY Export & Closing Call       │ Screen 6     │
└─────────────────┴────────────────────────────────────────────────┴──────────────┘
```

---

### [0:00 – 0:15] Section 1: Product Identity & Architecture
**Visual Action:**
- Open browser at `http://127.0.0.1:8501`.
- Display top application header: RoomAI logo, tagline *"Understand Your Space. Visualize It in 3D."*, and live badges (*Real Inference*, *NVIDIA RTX 4060*, *319 MB VRAM*, *Checkpoints Active*).
- Highlight that the system is running 100% locally with zero cloud dependencies.

**Narration Script:**
> *"Welcome to RoomAI — an AI studio built to understand indoor spaces and reconstruct them in true metric 3D from a single photograph.*  
> *Notice our live hardware telemetry: RoomAI runs fully local on an NVIDIA RTX 4060 GPU with strict Real Mode inference, consuming under 400 megabytes of VRAM."*

---

### [0:15 – 0:35] Section 2: Ingestion & Model A3 Object Detection
**Visual Action:**
- Drag and drop an ADE20K living room photo into the upload dropzone.
- Point out automatic image validation, resolution detection (e.g. 683 × 512 px), and SHA-256 session caching.
- Click `🔍 Object Detection` tab and click `Run Detection`.
- Show detected bounding boxes, confidence badges (e.g. Table 94%, Sofa 89%), and latency breakdown (~45 ms).

**Narration Script:**
> *"Starting with a standard room photo, RoomAI validates resolution and caches the image fingerprint. In our Object Detection workspace, our custom Model A3 YOLO11s detector processes the scene at 800-pixel resolution.*  
> *In under 50 milliseconds, it segments key furniture elements with calibrated confidences and pinpoint bounding coordinates."*

---

### [0:35 – 0:55] Section 3: Vision Q&A & Transparent System Governance
**Visual Action:**
- Switch to `💬 Vision Q&A` workspace.
- Point out the active room preview on the left.
- Select a quick suggestion: *"How can I optimize the furniture arrangement in this room?"*
- Hover over the transparent VLM status card explaining GPU VRAM preservation.

**Narration Script:**
> *"Next, the Vision Q&A workspace provides spatial reasoning assistance. But notice our commitment to engineering integrity:*  
> *Rather than hallucinating ungrounded mock answers, RoomAI explicitly informs the user that heavy multimodal VLM weights are kept offline by default to guarantee dedicated GPU memory for real-time spatial reconstruction."*

---

### [0:55 – 1:25] Section 4: Metric Depth & 3D Scene Reconstruction
**Visual Action:**
- Switch to `🧊 3D Scene Reconstruction` workspace.
- Click `Reconstruct 3D Room Scene` (processing spinner appears for ~1.8 seconds).
- Show Step 1: Input image side-by-side with the Metric Depth Heatmap. Explain depth range (e.g. 1.30 m – 15.05 m).
- Show Step 2: Estimated Room Architecture metrics (Width, Length, Ceiling Height, Camera Elevation).

**Narration Script:**
> *"Now for the core capability: 3D Scene Reconstruction. In a single GPU pass, Depth Anything V2 Metric Small predicts true Euclidean distance in calibrated meters—not unscaled relative disparity.*  
> *Our fusion engine fits a horizontal floor plane via RANSAC with strict verticality constraints, deriving real room dimensions, ceiling height, and camera elevation."*

---

### [1:25 – 1:45] Section 5: Interactive 3D Visualizer & Spatial Graph
**Visual Action:**
- Scroll to Step 3: Interactive Plotly 3D WebGL viewer.
- Click and drag to orbit smoothly around the room point cloud.
- Zoom in on furniture clusters and highlight 3D bounding boxes.
- Hover over spatial relationship cards (*Table near Chair*, *Sofa adjacent to Wall*).

**Narration Script:**
> *"In the interactive WebGL viewer, we can freely rotate, orbit, and pan through 350,000 spatial points. Furniture items are enclosed in ground-aligned 3D bounding envelopes, and pairwise spatial relationships are dynamically computed based on physical proximity."*

---

### [1:45 – 2:00] Section 6: Standard 3D Exports & Closing
**Visual Action:**
- Scroll to Step 4: Export 3D Scene Geometry.
- Hover over the download cards for Binary glTF 2.0 (`.glb`) and Stanford PLY (`.ply`).
- Highlight the honest technical disclosure: point clouds and bounding boxes, not generative mesh hallucinations.
- Conclude video on application header.

**Narration Script:**
> *"Finally, RoomAI exports the complete scene in industry-standard glTF 2.0 GLB and Stanford PLY formats for instant workflow integration into Blender, Three.js, or CAD suites.*  
> *Production models, real GPU performance, and honest geometry.*  
> *RoomAI: Understand your space. Visualize it in 3D."*

---

## 3. LinkedIn Post Copy (Ready-to-Publish)

```markdown
🚀 Excited to showcase RoomAI: Real-Time 3D Room Scene Reconstruction from a Single Photo!

Most monocular 3D vision demos either produce unscaled relative disparity or hallucinate distorted generative meshes. 

With RoomAI, we set out to build an engineering-grade, production-ready vision studio:
✅ Real-Time Object Detection: Custom YOLO11s 800px detector segmenting indoor furniture in <50ms.
✅ Grounded Metric Depth: Single-pass depth inference predicting true physical distances in meters.
✅ RANSAC Room Layout: Automated floor plane estimation and room envelope bounding.
✅ Interactive 3D WebGL Viewer: Orbit, pan, and inspect dense point clouds and 3D bounding boxes.
✅ Universal 3D Export: Native glTF 2.0 (.GLB) and Stanford PLY (.PLY) downloads for Blender/CAD.

All running 100% locally on a single NVIDIA GeForce RTX 4060 Laptop GPU, consuming under 400 MB of VRAM.

Watch the 2-minute walkthrough below! 👇

#ComputerVision #DeepLearning #PyTorch #3DReconstruction #YOLO11 #AIProduct #MachineLearning #Streamlit #Robotics
```

