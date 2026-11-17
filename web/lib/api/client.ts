import {
  ApiErrorDetail,
  ApiErrorResponse,
  ClientLatencyMetrics,
  DetectionResponse,
  HealthResponse,
  JobResultResponse,
  JobStatusResponse,
  JobSubmissionResponse,
  ReadinessResponse,
  UploadResponse,
} from "./types";
import { zeroGpuClient, ZeroGpuDetectOptions } from "../detectionClient";

export class ApiError extends Error {
  public readonly code: string;
  public readonly status: number;
  public readonly remediation?: string;

  constructor(message: string, status: number, code: string, remediation?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.remediation = remediation;
  }
}

export interface DetectOptions {
  confidenceThreshold?: number;
  modelId?: string;
  signal?: AbortSignal;
}

export interface DetectResult {
  data: DetectionResponse;
  clientLatency: ClientLatencyMetrics;
}

class RoomApiClient {
  private readonly baseUrl: string;
  private readonly defaultTimeoutMs: number;

  constructor() {
    if (typeof window !== "undefined") {
      // In the browser, use relative same-origin path by default so Next.js rewrites
      // proxy to the backend without CORS or network interface mismatches.
      this.baseUrl = "";
    } else {
      const envUrl = process.env.NEXT_PUBLIC_API_URL;
      this.baseUrl = (envUrl && envUrl.trim().length > 0)
        ? envUrl.replace(/\/+$/, "").replace("localhost", "127.0.0.1")
        : "http://127.0.0.1:8000";
    }
    const envTimeout = process.env.NEXT_PUBLIC_API_TIMEOUT_MS;
    this.defaultTimeoutMs = envTimeout ? parseInt(envTimeout, 10) : 15000;
  }

  /**
   * Health probe to verify backend liveness.
   */
  async checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
    const url = `${this.baseUrl}/health`;
    try {
      const res = await fetch(url, {
        method: "GET",
        signal,
        headers: { Accept: "application/json" },
      });
      if (!res.ok) {
        throw new ApiError(`Health check failed with status ${res.status}`, res.status, "HEALTH_CHECK_FAILED");
      }
      return (await res.json()) as HealthResponse;
    } catch (err: unknown) {
      if (err instanceof ApiError) throw err;
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new ApiError("Health check aborted", 499, "REQUEST_ABORTED");
      }

      // Attempt alternate endpoint (relative proxied or direct 127.0.0.1:8000)
      const altUrl = url.startsWith("http") ? "/health" : "http://127.0.0.1:8000/health";
      try {
        const directRes = await fetch(altUrl, {
          method: "GET",
          signal,
          headers: { Accept: "application/json" },
        });
        if (directRes.ok) {
          return (await directRes.json()) as HealthResponse;
        }
      } catch {
        // fall through
      }

      throw new ApiError(
        `Unable to reach backend at ${this.baseUrl || "http://127.0.0.1:8000"}: ${err instanceof Error ? err.message : String(err)}`,
        503,
        "BACKEND_UNREACHABLE",
        "Verify that the RoomAI FastAPI backend is running on port 8000."
      );
    }
  }

  /**
   * Readiness probe to verify GPU status, VRAM, and warmed models.
   */
  async checkReadiness(signal?: AbortSignal): Promise<ReadinessResponse> {
    const url = `${this.baseUrl}/ready`;
    try {
      const res = await fetch(url, {
        method: "GET",
        signal,
        headers: { Accept: "application/json" },
      });
      if (!res.ok) {
        throw new ApiError(`Readiness check failed with status ${res.status}`, res.status, "READINESS_CHECK_FAILED");
      }
      return (await res.json()) as ReadinessResponse;
    } catch (err: unknown) {
      if (err instanceof ApiError) throw err;
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new ApiError("Readiness check aborted", 499, "REQUEST_ABORTED");
      }

      // Attempt alternate endpoint (relative proxied or direct 127.0.0.1:8000)
      const altUrl = url.startsWith("http") ? "/ready" : "http://127.0.0.1:8000/ready";
      try {
        const directRes = await fetch(altUrl, {
          method: "GET",
          signal,
          headers: { Accept: "application/json" },
        });
        if (directRes.ok) {
          return (await directRes.json()) as ReadinessResponse;
        }
      } catch {
        // fall through
      }

      throw new ApiError(
        `Unable to reach backend readiness probe: ${err instanceof Error ? err.message : String(err)}`,
        503,
        "BACKEND_UNREACHABLE",
        "Verify that RoomAI FastAPI backend is running on port 8000."
      );
    }
  }

  /**
   * Sends an image to Model A3 / specialist models for object detection.
   * By default, Model A3 routes to Hugging Face ZeroGPU for GPU-accelerated inference.
   * Specialist models (B2 Exterior, C2 Parking) or local mode route to the local backend.
   */
  async detectImage(file: File | Blob, options: ZeroGpuDetectOptions = {}): Promise<DetectResult> {
    const isA3Model = !options.modelId ||
      options.modelId === "indoor_detector_a3_yolo11s_800px" ||
      options.modelId === "roomai-a3";

    const useZeroGpu = isA3Model && process.env.NEXT_PUBLIC_ROOMAI_INFERENCE_URL !== "local";

    if (useZeroGpu) {
      try {
        return await zeroGpuClient.detect(file, options);
      } catch (zeroGpuErr: unknown) {
        if (options.signal?.aborted) {
          throw zeroGpuErr;
        }

        // If local fallback is explicitly disabled, propagate the ZeroGPU error
        if (options.fallbackToLocal === false) {
          throw zeroGpuErr;
        }

        console.warn("ZeroGPU inference unavailable or quota-limited, attempting local fallback:", zeroGpuErr);

        // Fallback to local FastAPI backend if reachable
        try {
          return await this.detectImageLocal(file, options);
        } catch {
          // If local backend also fails, raise the original ZeroGPU error with its actionable context
          throw zeroGpuErr;
        }
      }
    }

    return await this.detectImageLocal(file, options);
  }

  /**
   * Local backend detection implementation via FastAPI (/api/v1/detect).
   */
  async detectImageLocal(file: File | Blob, options: DetectOptions = {}): Promise<DetectResult> {
    let url = `${this.baseUrl}/api/v1/detect`;
    const t0 = performance.now();

    // Prepare multipart payload
    const formData = new FormData();
    const filename = file instanceof File ? file.name : "upload.jpg";
    formData.append("file", file, filename);
    if (options.confidenceThreshold !== undefined) {
      formData.append("confidence_threshold", options.confidenceThreshold.toString());
    }
    if (options.modelId !== undefined) {
      formData.append("model_id", options.modelId);
    }

    // Set up timeout and abort handling
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => {
      timeoutController.abort(new Error(`Request timed out after ${this.defaultTimeoutMs}ms`));
    }, this.defaultTimeoutMs);

    // Merge external signal with timeout signal
    let combinedSignal: AbortSignal;
    if (options.signal) {
      combinedSignal = AbortSignal.any
        ? AbortSignal.any([options.signal, timeoutController.signal])
        : options.signal;
    } else {
      combinedSignal = timeoutController.signal;
    }

    let response: Response;
    let t1: number;

    try {
      try {
        response = await fetch(url, {
          method: "POST",
          body: formData,
          signal: combinedSignal,
          headers: {
            Accept: "application/json",
          },
        });
      } catch (firstErr) {
        // Attempt fallback to alternate endpoint (relative proxied or direct 127.0.0.1:8000)
        const fallbackUrl = url.startsWith("http") ? "/api/v1/detect" : "http://127.0.0.1:8000/api/v1/detect";
        if (!options.signal?.aborted && !timeoutController.signal.aborted) {
          try {
            response = await fetch(fallbackUrl, {
              method: "POST",
              body: formData,
              signal: combinedSignal,
              headers: {
                Accept: "application/json",
              },
            });
          } catch {
            throw firstErr;
          }
        } else {
          throw firstErr;
        }
      }
      t1 = performance.now();
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      if (options.signal?.aborted) {
        throw new ApiError("Inference request cancelled by client", 499, "REQUEST_CANCELLED");
      }
      if (timeoutController.signal.aborted) {
        throw new ApiError(
          `Detection timed out after ${this.defaultTimeoutMs}ms`,
          504,
          "TIMEOUT",
          "Try a smaller image or verify server GPU availability."
        );
      }
      throw new ApiError(
        `Network error communicating with ${this.baseUrl}: ${err instanceof Error ? err.message : String(err)}`,
        503,
        "NETWORK_ERROR",
        "Check network connection and backend status."
      );
    } finally {
      clearTimeout(timeoutId);
    }

    // Process Response
    const uploadAndServerMs = Math.round(t1 - t0);

    if (!response.ok) {
      let errorData: ApiErrorResponse | null = null;
      try {
        errorData = (await response.json()) as ApiErrorResponse;
      } catch {
        // Fallback for non-JSON errors
      }

      const detail: ApiErrorDetail = errorData?.error || {
        code: `HTTP_${response.status}`,
        message: response.statusText || `Request failed with HTTP status ${response.status}`,
        remediation: "Check server logs.",
      };

      throw new ApiError(detail.message, response.status, detail.code, detail.remediation);
    }

    let data: DetectionResponse;
    try {
      data = (await response.json()) as DetectionResponse;
    } catch (err) {
      throw new ApiError(
        `Failed to parse detection JSON response: ${err instanceof Error ? err.message : String(err)}`,
        502,
        "INVALID_RESPONSE_SCHEMA"
      );
    }

    const t2 = performance.now();
    const totalRoundtripMs = Math.round(t2 - t0);
    const serverReportedMs = data.timing?.total_server_ms || data.inference?.latency_ms || 0;
    const estimatedUploadMs = Math.max(0, Math.round(uploadAndServerMs - serverReportedMs));

    const clientLatency: ClientLatencyMetrics = {
      upload_ms: estimatedUploadMs,
      server_ms: serverReportedMs,
      inference_ms: data.inference?.latency_ms || 0,
      render_ms: 0, // Filled after DOM/Canvas paint
      total_roundtrip_ms: totalRoundtripMs,
    };

    return { data, clientLatency };
  }

  /**
   * Uploads an image to the backend for 3D room reconstruction.
   */
  async uploadImage(file: File | Blob, filename?: string): Promise<UploadResponse> {
    const formData = new FormData();
    const actualName = filename || (file instanceof File ? file.name : "scene.jpg");
    formData.append("file", file, actualName);

    const url = `${this.baseUrl}/api/v1/uploads`;
    let response: Response;

    try {
      response = await fetch(url, {
        method: "POST",
        body: formData,
        headers: { Accept: "application/json" },
      });
    } catch (firstErr) {
      const fallbackUrl = url.startsWith("http") ? "/api/v1/uploads" : "http://127.0.0.1:8000/api/v1/uploads";
      try {
        response = await fetch(fallbackUrl, {
          method: "POST",
          body: formData,
          headers: { Accept: "application/json" },
        });
      } catch {
        throw new ApiError(
          `Failed to upload image: ${firstErr instanceof Error ? firstErr.message : String(firstErr)}`,
          500,
          "UPLOAD_FAILED"
        );
      }
    }

    if (!response.ok) {
      throw new ApiError(`Image upload failed with status ${response.status}`, response.status, "UPLOAD_FAILED");
    }

    return (await response.json()) as UploadResponse;
  }

  /**
   * Submits a 3D reconstruction job for one or more upload IDs.
   */
  async submitReconstructionJob(
    uploadIds: string[],
    options: { modelId?: string; refineDepth?: boolean } = {}
  ): Promise<JobSubmissionResponse> {
    const url = `${this.baseUrl}/api/v1/jobs/reconstruct`;
    const payload = {
      upload_ids: uploadIds,
      model_id: options.modelId || "roomai_composite_v1",
      refine_depth: options.refineDepth !== undefined ? options.refineDepth : true,
    };

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (firstErr) {
      const fallbackUrl = url.startsWith("http") ? "/api/v1/jobs/reconstruct" : "http://127.0.0.1:8000/api/v1/jobs/reconstruct";
      try {
        response = await fetch(fallbackUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(payload),
        });
      } catch {
        throw new ApiError(
          `Failed to submit reconstruction job: ${firstErr instanceof Error ? firstErr.message : String(firstErr)}`,
          500,
          "JOB_SUBMISSION_FAILED"
        );
      }
    }

    if (!response.ok) {
      throw new ApiError(`Reconstruction job submission failed with HTTP ${response.status}`, response.status, "JOB_SUBMISSION_FAILED");
    }

    return (await response.json()) as JobSubmissionResponse;
  }

  /**
   * Polls status of an existing reconstruction job.
   */
  async pollJobStatus(jobId: string): Promise<JobStatusResponse> {
    const url = `${this.baseUrl}/api/v1/jobs/${jobId}`;
    let response: Response;

    try {
      response = await fetch(url, { headers: { Accept: "application/json" } });
    } catch (firstErr) {
      const fallbackUrl = url.startsWith("http") ? `/api/v1/jobs/${jobId}` : `http://127.0.0.1:8000/api/v1/jobs/${jobId}`;
      try {
        response = await fetch(fallbackUrl, { headers: { Accept: "application/json" } });
      } catch {
        throw new ApiError(`Failed to poll job status: ${firstErr instanceof Error ? firstErr.message : String(firstErr)}`, 500, "JOB_POLL_FAILED");
      }
    }

    if (!response.ok) {
      throw new ApiError(`Job status poll failed with HTTP ${response.status}`, response.status, "JOB_POLL_FAILED");
    }

    return (await response.json()) as JobStatusResponse;
  }

  /**
   * Retrieves complete 3D scene data for a completed reconstruction job.
   */
  async getJobResults(jobId: string): Promise<JobResultResponse> {
    const url = `${this.baseUrl}/api/v1/jobs/${jobId}/results`;
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new ApiError(`Failed to fetch job results (HTTP ${response.status})`, response.status, "JOB_RESULTS_FAILED");
    }
    return (await response.json()) as JobResultResponse;
  }

  /**
   * Returns direct URL to stream/download the GLB 3D asset for a job.
   */
  getExportGlbUrl(jobId: string): string {
    return `${this.baseUrl || ""}/api/v1/jobs/${jobId}/exports/glb`;
  }

  /**
   * Returns direct URL to stream/download the PLY point cloud for a job.
   */
  getExportPlyUrl(jobId: string): string {
    return `${this.baseUrl || ""}/api/v1/jobs/${jobId}/exports/ply`;
  }
}

// Singleton export
export const apiClient = new RoomApiClient();
