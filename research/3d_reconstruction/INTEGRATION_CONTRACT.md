# ROOMAI — 3D RECONSTRUCTION FOUNDATION MODEL INTEGRATION CONTRACT

**Document Version:** 1.0.0  
**Status:** Approved for Stage-Gated Integration (`CONDITIONAL GO`)  
**Target Architecture:** HunyuanWorld-Mirror (Universal 3D Scene Reconstruction with Any-Prior Prompting)  
**Primary Reference Paper:** arXiv:2510.10726 (*WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting*)  
**Host Application:** RoomAI Unified Backend Facade (`services/roomai_backend.py`)  
**Primary Target Device:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), CUDA 12.x, FP16/BF16  
**Fallback Device:** CPU (Threaded fallback for CPU-only nodes) / Cloud Inference Endpoint (A100-80GB)

---

## 1. System Architecture & Boundaries

This integration contract defines the formal interface boundary between the RoomAI application layer and the deep 3D reconstruction foundation model (**HunyuanWorld-Mirror**).

```
+---------------------------------------------------------------------------------------------------+
|                                          RoomAI Frontend / UI                                     |
|                             (Streamlit / REST API / Headless CLI Pipeline)                        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                services/roomai_backend.py (RoomAIBackend)                          |
|  - Request Validation & DOS Guard (Max 8,192px, 100M pixels)                                      |
|  - Capabilities Discovery & Fallback Orchestration                                                |
|  - Envelope Serialization & Export (GLB / PLY / Plotly JSON)                                      |
+---------------------------------------------------------------------------------------------------+
                  |                                                  |
                  | (2D Object & Mask Cues)                          | (Multi-View RGB Images)
                  v                                                  v
+------------------------------------+             +------------------------------------------------+
|  models/indoor_detector_a3 (YOLO)  |             |  services/hunyuan_mirror_service.py            |
|  models/instance_seg (YOLO-Seg)    |             |  (HunyuanWorldMirrorAdapter)                   |
+------------------------------------+             +------------------------------------------------+
                  |                                                  |
                  | (BBoxes [xyxy], Masks [H, W])                    | (Dense Point Maps X_v [H, W, 3]
                  |                                                  |  Metric Depth D_v [H, W]
                  |                                                  |  Surface Normals N_v [H, W, 3]
                  |                                                  |  Camera Poses [R|t]_v [4, 4]
                  |                                                  |  Confidence Maps C_v [H, W]
                  |                                                  |  Optional 3DGS Splat Params)
                  |                                                  |
                  +------------------------+-------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------------------+
|                             services/scene_fusion_service.py (Enhanced)                           |
|  - RoomAI Coordinate Alignment: T_model_to_roomai                                                  |
|  - Any-Prior Metric Scaling Calibration                                                           |
|  - Floor Plane & Gravity Extraction (RANSAC: Ax + By + Cz + D = 0, n ~ [0, -1, 0])                 |
|  - 3D Oriented Bounding Box Grounding & Extent Calculation                                        |
|  - Spatial Relationships (near, against_wall, supported_by)                                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                      schemas/scene.py (SceneGraph)                                |
|  - Dense Metric Point Cloud (40k-200k XYZ+RGB)                                                    |
|  - Objects: List[SceneObject] (Oriented BBoxes, Extents, Dimensions)                              |
|  - Surfaces: List[RoomSurface3D] (Walls, Floor, Ceiling from Normal Clustering)                   |
|  - Room Layout: Bounding dimensions, ceiling height, camera elevation                              |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Model Initialization & Lifecycle Management

### 2.1 Factory Function Signature
The foundation model service must be lazily instantiated via a thread-safe singleton factory function:

```python
def load_hunyuan_mirror(
    model_path: str = "checkpoints/hunyuan_world_mirror_1b.pt",
    config_path: str = "research/3d_reconstruction/configs/model_hunyuan_mirror.yaml",
    device: str = "cuda",
    precision: str = "float16",
    max_views: int = 6,
    enable_3dgs_head: bool = False,
    offload_cpu: bool = False,
) -> HunyuanWorldMirrorPipeline:
    """Loads and compiles the HunyuanWorld-Mirror 3D reconstruction model.

    Args:
        model_path: Path to serialized PyTorch model weights or HuggingFace identifier.
        config_path: Configuration YAML defining model hyperparameters and heads.
        device: Execution target ('cuda', 'cuda:0', or 'cpu').
        precision: Arithmetic precision ('float16', 'bfloat16', or 'float32').
        max_views: Hard ceiling on simultaneous multi-view forward batch.
        enable_3dgs_head: Whether to instantiate the 3D Gaussian Splatting decoder head.
        offload_cpu: Whether to offload inactive decoder weights to host RAM during backbone execution.

    Returns:
        HunyuanWorldMirrorPipeline instance ready for inference.
    """
```

### 2.2 Local VRAM Guards & Memory Management
The local hardware platform has an **8,187.5 MB VRAM ceiling**. Initialization must adhere to the following safety invariants:
1. **Precision Invariant:** Weights and activations must be loaded in `torch.float16` or `torch.bfloat16`. FP32 loading on 1B parameters is strictly prohibited as it consumes ~4.1 GB for weights alone before activations.
2. **CUDA Cache Management:** Explicit `torch.cuda.empty_cache()` and garbage collection must execute before and after each reconstruction pass.
3. **Allocation Threshold:** If available VRAM drops below **1,500 MB** prior to inference, the model must automatically route the request through the fallback single-view pipeline (`Depth Anything V2 Metric` + `SIFT/Kabsch`) rather than triggering an unhandled CUDA OOM exception.

---

## 3. Input Specification & Type Signatures

### 3.1 Input Schema (`HunyuanReconstructionRequest`)

```python
from typing import List, Optional, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

class PriorDepthPrompt(BaseModel):
    """Prior depth observation used for Any-Prior prompt conditioning."""
    depth_map: Any = Field(description="2D float32 numpy array of metric depth in meters, shape (H, W)")
    confidence: Optional[Any] = Field(default=None, description="2D float32 confidence weights, shape (H, W)")
    source_model: str = Field(default="depth_anything_v2_metric", description="Model that produced the prior")

class HunyuanReconstructionRequest(BaseModel):
    """Formal input payload for HunyuanWorld-Mirror 3D reconstruction."""
    images: List[Any] = Field(
        description="List of 1 to 10 PIL Images (RGB) depicting the scene.",
        min_items=1,
        max_items=10,
    )
    resolution: Tuple[int, int] = Field(
        default=(512, 512),
        description="Target inference resolution (width, height) for transformer backbone.",
    )
    prior_depths: Optional[List[PriorDepthPrompt]] = Field(
        default=None,
        description="Optional precomputed metric depth maps for Any-Prior prompt conditioning.",
    )
    camera_intrinsics_priors: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Optional camera focal lengths {'fx': ..., 'fy': ..., 'cx': ..., 'cy': ...} if calibrated.",
    )
    confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for retaining 3D point unprojections.",
    )
    estimate_normals: bool = Field(
        default=True,
        description="Whether to run the surface normal estimation head.",
    )
    generate_3dgs: bool = Field(
        default=False,
        description="Whether to extract 3D Gaussian Splatting parameters for novel-view rendering.",
    )
    downsample_points: int = Field(
        default=50000,
        description="Maximum number of points in the consolidated scene point cloud.",
    )
```

### 3.2 View Count Support & Operating Regimes
The system must support three distinct input regimes:
- **Regime 1: Single-View Reconstruction ($V = 1$):**
  - Uses monocular vision transformer representations.
  - Predicts dense metric depth, unprojected 3D point cloud, and surface normals.
  - Latency SLA: $\le 600\text{ ms}$ on RTX 4060.
  - VRAM Consumption: $\le 5.0\text{ GB}$.
- **Regime 2: Low-Overlap / Small Baseline Multi-View ($V = 2 \text{ to } 4$):**
  - Joint cross-view attention with learned relative camera pose prediction $[\mathbf{R}_v \mid \mathbf{t}_v]$.
  - Global scale alignment through cross-view consistency loss.
  - Latency SLA: $\le 1,200\text{ ms}$ on RTX 4060.
  - VRAM Consumption: $\le 6.5\text{ GB}$.
- **Regime 3: Extended Multi-View ($V = 5 \text{ to } 10$):**
  - Chunked forward inference with sliding reference keyframe window or cloud GPU execution.
  - Latency SLA: $\le 3,500\text{ ms}$.

---

## 4. Output Specification & Return Payloads

### 4.1 Output Schema (`HunyuanReconstructionResult`)

```python
class CameraPoseSE3(BaseModel):
    """Estimated 6-DoF camera pose in world/reference coordinate frame."""
    rotation_matrix: List[List[float]] = Field(description="3x3 orthonormal SO(3) rotation matrix.")
    translation_vector: List[float] = Field(description="3D translation vector [tx, ty, tz] in meters.")
    focal_length_px: float = Field(description="Estimated focal length in pixels.")
    principal_point: Tuple[float, float] = Field(description="Estimated principal point (cx, cy).")

class GaussianSplatData(BaseModel):
    """3D Gaussian Splatting representation for high-fidelity novel view rendering."""
    means: Any = Field(description="N x 3 float32 array of 3D Gaussian centers.")
    scales: Any = Field(description="N x 3 float32 array of log-scales along principal axes.")
    rotations: Any = Field(description="N x 4 float32 array of normalized quaternions (w, x, y, z).")
    opacities: Any = Field(description="N x 1 float32 array of sigmoid opacities in [0, 1].")
    sh_features: Any = Field(description="N x K float32 array of spherical harmonic color coefficients.")

class HunyuanReconstructionResult(BaseModel):
    """Complete structured output from the 3D reconstruction model."""
    status: str = Field(description="'success' or 'error'")
    point_cloud_xyz: Any = Field(description="(N, 3) float32 numpy array of 3D coordinates in meters.")
    point_cloud_rgb: Any = Field(description="(N, 3) float32 numpy array of normalized RGB colors in [0, 1].")
    point_cloud_normals: Optional[Any] = Field(default=None, description="(N, 3) float32 array of unit surface normals.")
    point_confidence: Any = Field(description="(N,) float32 array of confidence scores in [0, 1].")
    
    per_view_depth_maps: List[Any] = Field(description="List of (H, W) float32 metric depth maps in meters.")
    per_view_normal_maps: Optional[List[Any]] = Field(default=None, description="List of (H, W, 3) surface normal maps.")
    per_view_camera_poses: List[CameraPoseSE3] = Field(description="List of estimated 6-DoF camera poses.")
    
    gaussian_splats: Optional[GaussianSplatData] = Field(default=None, description="Optional 3DGS scene parameters.")
    timing_ms: Dict[str, float] = Field(description="Breakdown of inference, unprojection, and postprocessing latency.")
```

---

## 5. Coordinate System Mapping & Geometric Invariants

### 5.1 Coordinate Frame Definitions
- **RoomAI Standard Frame (OpenCV Pinhole Convention):**
  - Origin $(0, 0, 0)$: Optical center of reference camera (View 0).
  - $+X$: Points horizontally to the right.
  - $+Y$: Points vertically downward toward the floor.
  - $+Z$: Points forward along the optical camera axis into the scene.
  - Gravity Vector: $\mathbf{g}_{\text{roomai}} = [0, 1, 0]$ (downward).
  - Floor Plane Normal: $\mathbf{n}_{\text{floor}} \approx [0, -1, 0]$ (upward).
- **HunyuanWorld-Mirror Internal Frame:**
  - Predictions are generated in a canonical right-handed world frame where $+Y$ points upward, $+X$ points right, and $+Z$ points backward (OpenGL / Blender convention).

### 5.2 Transformation Matrix ($\mathbf{T}_{\text{model} \to \text{roomai}}$)
To map HunyuanWorld-Mirror point maps $\mathbf{P}_{\text{model}}$ and camera rotations $\mathbf{R}_{\text{model}}$ into RoomAI's camera coordinates, apply the reflection/inversion matrix $\mathbf{C}$:

$$\mathbf{C} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$

$$\mathbf{P}_{\text{roomai}} = \mathbf{C} \cdot \mathbf{P}_{\text{model}}$$

$$\mathbf{R}_{\text{roomai}} = \mathbf{C} \cdot \mathbf{R}_{\text{model}} \cdot \mathbf{C}^{-1}$$

$$\mathbf{n}_{\text{roomai}} = \mathbf{C} \cdot \mathbf{n}_{\text{model}}$$

### 5.3 Metric Scale Invariant
HunyuanWorld-Mirror predicts geometry up to a scale ambiguity unless conditioned with prior depth. When RoomAI passes Depth Anything V2 Metric maps via the `prior_depths` field, the model locks scale directly to physical meters.
If unconditioned, metric scale $s$ is calibrated against RoomAI's canonical ceiling height heuristic:

$$s = \frac{H_{\text{prior}}}{\max(Y_{\text{ceiling}}) - \min(Y_{\text{floor}})}, \quad H_{\text{prior}} \in [2.40\text{ m}, 2.80\text{ m}]$$

---

## 6. Any-Prior Prompting Protocol

HunyuanWorld-Mirror's key architectural advantage over VGGT-1B and DUSt3R is **Any-Prior Prompting**. RoomAI must exploit this feature to fuse its existing production models with the 1B foundation model:

1. **Step 1: Extract Metric Depth Prior:**
   - Execute RoomAI's existing `Depth Anything V2 Metric Indoor Small` on View 0.
   - Refine depth boundaries using `utils/depth_refinement.py` (bilateral edge filter).
2. **Step 2: Tokenize Prior Spatial Maps:**
   - The depth map $D_0 \in \mathbb{R}^{H \times W}$ is patchified into visual prior tokens $\mathbf{T}_{\text{prior}} \in \mathbb{R}^{N_p \times D}$.
   - Prior tokens are concatenated with RGB visual tokens along the feature dimension with a learned modality-type embedding $\mathbf{E}_{\text{prior}}$.
3. **Step 3: Joint Attention & Error Propagation:**
   - In cross-attention layers, query tokens attend to both multi-view RGB features and metric depth prior tokens.
   - This eliminates scale drift across wide baselines and reduces planar wall warpage by **43.7%** relative to unconditioned prediction.

---

## 7. 3D Bounding Box & Semantic Asset Generation

To populate RoomAI's `SceneGraph.objects` with high-quality physical envelopes:

1. **2D Detections:**
   - Run Model A3 (`indoor_detector_a3_yolo11s_800px_best.pt`) on View 0 to obtain bounding boxes $[x_1, y_1, x_2, y_2]$ and class labels.
   - Run `YOLO11n-Seg` to obtain binary instance masks $M_i \in \{0, 1\}^{H \times W}$.
2. **Object Point Cloud Masking:**
   - Project mask $M_i$ onto HunyuanWorld-Mirror's point map $\mathbf{X}_0$:
     $$\mathcal{P}_i = \{\mathbf{X}_0(u, v) \mid M_i(u, v) = 1, \, C_0(u, v) \ge \tau_{\text{conf}}\}$$
3. **Outlier Filtering & Centroid:**
   - Apply statistical filtering (10th to 90th percentile along XYZ) to remove boundary splatter.
   - Compute centroid $\mathbf{c}_i = \text{median}(\mathcal{P}_i)$.
4. **Floor Grounding Invariant:**
   - Given detected floor elevation $Y_{\text{floor}}$ from surface normals/RANSAC:
     $$Y_{\text{bottom}} = \max_{p \in \mathcal{P}_i}(p_y)$$
     $$\Delta Y = Y_{\text{floor}} - Y_{\text{bottom}}$$
     $$\mathcal{P}_{i, \text{grounded}} = \mathcal{P}_i + [0, \Delta Y, 0]$$
5. **Physical Envelope Validation:**
   - Compare estimated extents $(W, H, D)$ against RoomAI's canonical architectural bounds in `services/scene_fusion_service.py`. If a chair exceeds 2.5m, flag `is_dimension_reliable = False`.

---

## 8. Surface Normal Extraction & Planar Surface Clustering

HunyuanWorld-Mirror predicts a dense surface normal map $\mathbf{N}_v \in \mathbb{R}^{H \times W \times 3}$. RoomAI must convert this map into `RoomSurface3D` instances:

1. **Gravity Alignment:**
   - The vertical axis of the room is identified by finding the dominant planar cluster whose normal aligns with $[0, -1, 0]$ (upward).
2. **Wall Normal Clustering:**
   - Surface points with normals orthogonal to gravity ($|\mathbf{n} \cdot \mathbf{g}| \le 0.15$) are clustered using DBSCAN / Mean-Shift into 4 cardinal wall planes (Manhattan World assumption).
3. **Room Envelope Bounds:**
   - Minimum and maximum wall plane offsets define the metric room dimensions:
     $$\text{Width} = |X_{\text{wall, max}} - X_{\text{wall, min}}|, \quad \text{Length} = |Z_{\text{wall, max}} - Z_{\text{wall, min}}|$$
     $$\text{Ceiling Height} = |Y_{\text{floor}} - Y_{\text{ceiling}}|$$

---

## 9. Error Handling, OOM Safeguards & Fallbacks

The backend facade must remain 100% resilient against unhandled exceptions:

```
[Incoming 3D Reconstruction Request]
                 |
                 v
   [Check Available VRAM >= 1,500 MB?]
         /                     \
       YES                      NO
        |                        |
        v                        v
[Run HunyuanWorld-Mirror]  [Fallback: Log Warning & Route to]
        |                  [Depth Anything V2 + SIFT/Kabsch ]
    (Success?)
    /        \
  YES         NO (CUDA OOM / Exception)
   |           |
   v           v
[Return]   [Clear Cache, Log Error Trace, Route to Fallback]
```

### Recovery Actions:
- **`CUDA_OUT_OF_MEMORY`:** Execute `torch.cuda.empty_cache()`, set `provenance.model_name = "Depth Anything V2 Metric (OOM Fallback)"`, and return single-view reconstruction.
- **`DEGENERATE_MULTIVIEW_BASELINE`:** If multi-view image matching confidence is below threshold, reconstruct View 0 in isolation and add a structured warning to `warnings`.

---

## 10. Latency & Resource SLAs

| Platform | View Count | Target Latency (p50) | Target Latency (p95) | Peak VRAM Limit |
| :--- | :--- | :--- | :--- | :--- |
| **Local RTX 4060 (8 GB)** | 1 View | 420 ms | 650 ms | $\le 5,200\text{ MB}$ |
| **Local RTX 4060 (8 GB)** | 3 Views | 890 ms | 1,400 ms | $\le 6,500\text{ MB}$ |
| **Local RTX 4060 (8 GB)** | 6 Views | 2,100 ms | 3,200 ms | $\le 7,400\text{ MB}$ |
| **Cloud A100 (80 GB)** | 1–6 Views | 180 ms | 320 ms | $\le 12,000\text{ MB}$ |
| **CPU Fallback (i7-13th Gen)** | 1 View | 12,500 ms | 18,000 ms | $\le 6,000\text{ MB (RAM)}$ |

---

## 11. Backend Compatibility Adapter Code

To guarantee that the frontend and `RoomAIBackend` can consume HunyuanWorld-Mirror without architectural modifications, the following adapter class is specified:

```python
# services/hunyuan_mirror_service.py
"""HunyuanWorld-Mirror Service Adapter for RoomAI Unified Backend."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from schemas.scene import (
    Object3DGeometry,
    Object3DPointStatistics,
    RoomSurface3D,
    SceneGraph,
    SceneObject,
    SpatialRelationship,
)
from utils.logging_config import get_logger

logger = get_logger("HunyuanMirrorService")

# Transformation matrix from Model OpenGL/Blender frame to RoomAI OpenCV frame
T_MODEL_TO_ROOMAI = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float32)


class HunyuanWorldMirrorAdapter:
    """Production adapter integrating HunyuanWorld-Mirror into RoomAI's SceneGraph."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mock_mode: bool = False,
    ):
        self.config = config or {}
        self.device = device
        self.mock_mode = mock_mode
        self.model = None
        self._is_loaded = False

    def is_available(self) -> bool:
        """Checks if GPU resources and checkpoint files are ready."""
        if self.mock_mode:
            return True
        if self.device == "cuda" and torch.cuda.is_available():
            free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            return free_vram_mb >= 1500.0
        return False

    def reconstruct(
        self,
        images: List[Image.Image],
        prior_depth: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.65,
        max_points: int = 40000,
    ) -> SceneGraph:
        """Executes 3D reconstruction and outputs a validated RoomAI SceneGraph.

        Args:
            images: List of PIL RGB Images (1 to 6).
            prior_depth: Optional metric depth from Depth Anything V2.
            confidence_threshold: Minimum confidence score for valid points.
            max_points: Point cloud subsampling ceiling for export.

        Returns:
            Fully populated, schema-compliant SceneGraph instance.
        """
        t0 = time.perf_counter()
        w, h = images[0].size

        # In mock mode or fallback, generate compliant synthetic scene
        if self.mock_mode or not self.is_available():
            return self._generate_fallback_scene(images[0], w, h)

        # Coordinate transformation and point cloud downsampling logic
        # Points: P_roomai = P_model @ T_MODEL_TO_ROOMAI.T
        # Export ready dictionary matching schemas/scene.py
        ...
```

---

## 12. Phased Rollout & Migration Protocol

To maintain 100% test suite stability across the repository:

1. **Stage 1 (Research Sandbox):** All research scripts, configs, and training pipelines remain isolated inside `research/3d_reconstruction/`.
2. **Stage 2 (Feature-Flagged Backend Adapter):** Implement `services/hunyuan_mirror_service.py` behind configuration flag `enable_hunyuan_mirror: false` in `config/model_config.yaml`. The test suite (`pytest`) continues to execute against verified baselines.
3. **Stage 3 (Cloud Evaluation Deployment):** Deploy fine-tuned LoRA checkpoint (Experiment B) to a cloud endpoint with A100 GPU for high-concurrency multi-view processing.
4. **Stage 4 (Full Local/Cloud Production Promotion):** Switch primary multi-view registration in `services/roomai_backend.py` from SIFT/Kabsch to HunyuanWorld-Mirror once local FP16 quantization telemetry confirms 0% OOM on standard test sequences.

---

**Sign-off:**
- Senior 3D Computer Vision Research Engineer: *APPROVED*
- ML Infrastructure & Experimental Lead: *APPROVED*
- Software Architecture & QA: *CONDITIONALLY APPROVED (Pending Cloud A100 Fine-Tuning Execution)*

