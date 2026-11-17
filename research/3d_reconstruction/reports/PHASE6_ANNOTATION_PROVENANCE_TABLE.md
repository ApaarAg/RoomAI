# ROOMAI — PHASE 6: ANNOTATION PROVENANCE & GEOMETRIC SPECIFICATION TABLE

**Document ID:** `research/3d_reconstruction/reports/PHASE6_ANNOTATION_PROVENANCE_TABLE.md`  
**Date:** 2026-11-08  
**Author:** Senior 3D Computer Vision Research Scientist & ML Infrastructure Lead  
**Scope:** Forensic provenance classification, sensor modalities, mathematical definitions, coordinate conventions, and units across all training and evaluation datasets in RoomAI.

---

## 1. Executive Summary

In Phase 5, the RoomAI 3D reconstruction system concluded **PSEUDO-LABEL DISTILLATION ONLY** because all supervision signals originated from neural network predictions (`Depth Anything V2`) and synthetic circular trajectories.

Phase 6 establishes an authoritative, scientifically verified dataset containing **genuine geometric ground truth** (Hypersim synthetic CAD path tracing) and **direct physical sensor measurements** (ScanNet IR structured light). To maintain strict documentation integrity and prevent conflating sensor measurements with model predictions or CAD geometry, every field is classified under standard provenance taxonomies.

---

## 2. Field-by-Field Annotation Provenance Matrix

| Dataset | Modality | Field / File | Provenance Category | Physical Sensor / Generator | Native Units & Representation | Standardized Target Schema | Coordinate System & Conventions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hypersim** | RGB Image | `color.jpg` | **Synthetic Ground Truth** | V-Ray photorealistic path tracer | Tonemapped 8-bit sRGB JPEG ($1024 \times 768$) | $518 \times 518 \times 3$ uint8 PNG | Image plane $(u, v)$ |
| **Hypersim** | Ray Distance | `depth_meters.hdf5` | **Synthetic Ground Truth** | V-Ray primary ray intersection | Float16 Euclidean distance $R = \|\mathbf{X}_c\|_2$ (meters) | Converted to Planar Z-depth | Ray origin at camera focal center |
| **Hypersim** | Planar Z-Depth | `frame_XXXX_depth.npy` | **Derived Annotation** | Converted via $Z_c = R \cdot \cos(\theta)$ using $M_{\text{cam\_from\_uv}}$ | Float32 orthogonal metric depth $Z_c > 0$ (meters) | $518 \times 518$ float32 NPY ($0.05\text{m} \le Z \le 50.0\text{m}$) | OpenCV camera frame ($+Z$ forward) |
| **Hypersim** | Surface Normal | `normal_cam.hdf5` | **Synthetic Ground Truth** | CAD surface mesh differential geometry | Float16 unit vector $[N_x, N_y, N_z]$ in OpenGL camera | $518 \times 518 \times 3$ float32 NPY in OpenCV frame | Transformed via $\text{diag}(1, -1, -1)$ to OpenCV |
| **Hypersim** | Camera Intrinsics | `metadata_camera_parameters.csv` | **Dataset-Provided Calibration** | V-Ray camera definition & tilt-shift parameters | $M_{\text{cam\_from\_uv}}$ $3\times3$ projection matrix | $3\times3$ Pinhole Matrix $K$ scaled to $518\times518$ | Standard pinhole model |
| **Hypersim** | Camera Trajectory | `camera_keyframe_*.hdf5` | **Synthetic Ground Truth** | Render camera keyframe animation path | Position $t_{\text{world}}$ (cm), Rotation $R_{\text{C2W}}$ (OpenGL) | $4\times4$ Float32 C2W matrix (meters, OpenCV) | OpenCV camera frame in metric world |
| **Hypersim** | Validity Mask | `frame_XXXX_mask.png` | **Derived Annotation** | Non-finite / sky / window thresholding | Binary mask (0 or 255) | $518 \times 518$ uint8 PNG | 255 = Valid surface, 0 = Non-surface/sky |
| **ScanNet** | RGB Image | `color/XXXX.jpg` | **Direct Sensor Measurement** | iPad Air 2 color camera | 8-bit sRGB JPEG ($640 \times 480$) | $518 \times 518 \times 3$ uint8 PNG | Image plane $(u, v)$ |
| **ScanNet** | Sensor Depth | `depth/XXXX.npy` | **Direct Sensor Measurement** | Structure Sensor (IR structured light) | 16-bit unsigned integer ($d_{\text{mm}}$, millimeters) | $518 \times 518$ float32 NPY ($Z = d / 1000.0$) | OpenCV camera frame ($+X$ right, $+Y$ down, $+Z$ fwd) |
| **ScanNet** | Missing Sensor Depth | `depth == 0` | **Physical Sensor Artifact** | Optical absorption, specular reflection, occlusion | Value 0 (unmeasured infrared light) | Masked as 0 in `mask.png` | Out-of-range or unreflected regions |
| **ScanNet** | Surface Normal | `frame_XXXX_normal.npy` | **Derived Annotation** | Local finite-difference gradient on sensor depth | Float32 unit vector $[N_x, N_y, N_z]$ in OpenCV frame | $518 \times 518 \times 3$ float32 NPY | OpenCV camera frame ($+Z$ forward) |
| **ScanNet** | Camera Intrinsics | `intrinsic_depth.txt` | **Dataset-Provided Calibration** | Pre-calibrated factory pinhole calibration | $4\times4$ Intrinsic Matrix ($640 \times 480$ frame) | $3\times3$ Pinhole Matrix $K$ scaled to $518\times518$ | Standard pinhole model |
| **ScanNet** | Camera Trajectory | `pose/XXXX.txt` | **Reconstructed Geometry** | BundleFusion real-time SLAM tracking | $4\times4$ Float32 C2W matrix | $4\times4$ Float32 C2W matrix | Stanford CAD world frame ($+Z$ up) |
| **ScanNet** | Surface Mesh | `sceneXXXX_vh_clean_2.ply` | **Reconstructed Geometry** | TSDF volumetric integration / marching cubes | Dense triangular 3D surface mesh | Reference 3D CAD mesh | ScanNet world coordinate frame |
| **ARKitScenes** | RGB Image | `lowres_wide/*.png` | **Direct Sensor Measurement** | Apple iPad / iPhone wide camera | 8-bit sRGB PNG | Optional evaluation modality | Image plane $(u, v)$ |
| **ARKitScenes** | LiDAR Depth | `lowres_depth/*.png` | **Direct Sensor Measurement** | Apple direct Time-of-Flight (dToF) LiDAR | 16-bit uint16 depth in millimeters | Millimeter to meter ($d / 1000.0$) | Apple ARKit camera frame |
| **ARKitScenes** | Camera Pose | `color.traj` / `_pose.txt` | **Dataset-Provided Calibration / VIO** | Apple ARKit Visual-Inertial Odometry | 6-DoF axis-angle + translation in meters | $4\times4$ transformation matrix | Right-handed (+Y up, +Z back) |
| **RoomAI Old** | Depth (Phase 4/5) | `frame_XXXX_depth.npy` | **Model-Generated Pseudo-Label** | Depth Anything V2 Metric Indoor Small | Float32 neural network metric prediction | Deprecated (Audit benchmark only) | OpenCV camera frame |
| **RoomAI Old** | Camera Pose (Phase 4/5) | `frame_XXXX_cam.npz` | **Synthetic Heuristic** | Assumed circular trajectory ($R_y(0.05v), t_x=0.1v$) | Heuristic float32 pose | Deprecated (Synthetic heuristic) | Arbitrary camera coordinates |

---

## 3. Strict Provenance Guidelines Enforced

1. **Sensor Measurement vs. LiDAR Clarity:**  
   ScanNet's depth sensor is explicitly documented as an **infrared structured-light camera** (Structure Sensor with PrimeSense technology), NOT LiDAR. ARKitScenes is documented as **direct time-of-flight (dToF) mobile LiDAR**.
2. **CAD Ground Truth vs. Sensor Depth:**  
   Hypersim depth and normals are classified as **Synthetic Ground Truth** because they represent exact mathematical ray-traced intersections of high-poly 3D indoor scenes with zero sensor noise. ScanNet depth is classified as **Direct Sensor Measurement** subject to physical reflection dropouts, sensor noise, and boundary artifacts.
3. **Derived Normal Transparency:**  
   Normals computed from ScanNet depth maps are explicitly categorized as **Derived Annotations** rather than ground truth, preventing false claims of high-precision surface normal training on ScanNet data.
4. **Zero Model-in-the-Loop Supervision:**  
   Zero pseudo-labels from Depth Anything V2, YOLO, or other neural estimators exist within the standardized training pipeline for Phase 6. Supervision is 100% anchored to verified physical sensor captures and exact CAD path tracing.

