/**
 * Canonical RoomAI API Contract Types.
 * 
 * Synchronized with backend Pydantic models in `api/schemas.py`.
 * Do NOT alter these field names without updating both client and server schemas.
 */

export interface ImageDimensions {
  width: number;
  height: number;
  channels: number;
}

export interface BoundingBox2D {
  /** Top-left corner X in pixels */
  x1: number;
  /** Top-left corner Y in pixels */
  y1: number;
  /** Bottom-right corner X in pixels */
  x2: number;
  /** Bottom-right corner Y in pixels */
  y2: number;
  /** Normalized top-left corner X [0.0, 1.0] */
  norm_x1: number;
  /** Normalized top-left corner Y [0.0, 1.0] */
  norm_y1: number;
  /** Normalized bottom-right corner X [0.0, 1.0] */
  norm_x2: number;
  /** Normalized bottom-right corner Y [0.0, 1.0] */
  norm_y2: number;
}

export interface DetectionItem {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox2D;
}

export interface InferenceMetadata {
  model: string;
  latency_ms: number;
  device: string;
  cached: boolean;
}

export interface TimingBreakdown {
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  total_server_ms: number;
  queue_wait_ms?: number;
}

export interface DetectionResponse {
  status: "success" | "error";
  request_id: string;
  image: ImageDimensions;
  detections: DetectionItem[];
  inference: InferenceMetadata;
  timing: TimingBreakdown;
  timestamp: number;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
  remediation?: string;
}

export interface ApiErrorResponse {
  status: "error";
  error: ApiErrorDetail;
  timestamp: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  service: string;
  timestamp: number;
}

export interface ReadinessResponse {
  status: "ready" | "degraded" | string;
  cuda_available: boolean;
  device_name: string;
  free_vram_gb: number;
  total_vram_gb: number;
  checkpoints_ready: boolean;
  models_ready: string[];
  startup_error?: string | null;
}

export interface ClientLatencyMetrics {
  upload_ms: number;
  server_ms: number;
  inference_ms: number;
  render_ms: number;
  total_roundtrip_ms: number;
}

export interface UploadResponse {
  status: "success" | "error";
  upload_id: string;
  filename: string;
  original_filename: string;
  width: number;
  height: number;
  format: string;
  size_bytes: number;
  sha256: string;
}

export interface JobSubmissionResponse {
  status: string;
  job_id: string;
  job_status: string;
  model_id: string;
  view_count: number;
  poll_url: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: "QUEUED" | "VALIDATING" | "PROCESSING" | "RECONSTRUCTING" | "FUSING" | "EXPORTING" | "COMPLETED" | "FAILED" | "CANCELLED";
  stage: string;
  progress: number;
  model_id: string;
  view_count: number;
  timing: Record<string, unknown>;
  error?: { code: string; message: string; remediation?: string } | null;
  warnings?: string[];
  created_at: number;
}

export interface JobResultResponse {
  job_id: string;
  status: string;
  is_completed: boolean;
  model_id?: string;
  timing?: Record<string, unknown>;
  results_summary?: Record<string, unknown>;
  artifacts?: {
    glb_export_path?: string;
    ply_export_path?: string;
    [key: string]: unknown;
  };
  scene_graph?: Record<string, unknown>;
}
