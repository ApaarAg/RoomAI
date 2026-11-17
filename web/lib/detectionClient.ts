/**
 * RoomAI ZeroGPU Detection Client & Normalizer.
 *
 * Connects the RoomAI frontend to the Hugging Face ZeroGPU Model A3 Inference Space:
 * https://huggingface.co/spaces/Apaarag/roomai-a3-inference
 * Direct URL: https://apaarag-roomai-a3-inference.hf.space
 *
 * Implements:
 * 1. Multipart binary upload via Gradio API (/gradio_api/upload)
 * 2. Event job submission (/gradio_api/call/detect)
 * 3. SSE streaming with status callbacks and timeout protection
 * 4. Coordinate-accurate normalization from ZeroGPU pixel coordinates to canonical RoomAI schema
 * 5. Robust error handling for cold starts, quota limits, and network errors
 */

import {
  BoundingBox2D,
  ClientLatencyMetrics,
  DetectionItem,
  DetectionResponse,
} from "./api/types";
import { ApiError, DetectOptions, DetectResult } from "./api/client";

/**
 * Raw ZeroGPU response structure from Hugging Face Space.
 */
export interface ZeroGpuBBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ZeroGpuDetectionItem {
  class: string;
  confidence: number;
  bbox: ZeroGpuBBox;
}

export interface ZeroGpuRawResponse {
  model?: string;
  model_version?: string;
  image?: {
    width: number;
    height: number;
  };
  detections: ZeroGpuDetectionItem[];
  inference?: {
    latency_ms: number;
    confidence_threshold: number;
    image_size: number;
  };
}

export interface ZeroGpuDetectOptions extends DetectOptions {
  /** Optional callback for progressive status updates during inference */
  onStatusUpdate?: (status: string) => void;
  /** Natural dimensions of the image if known in advance by client */
  imageDimensions?: { width: number; height: number };
  /** Whether to fallback to local backend if ZeroGPU is unavailable/quota-limited */
  fallbackToLocal?: boolean;
}

export const DEFAULT_INFERENCE_URL = "https://Apaarag-roomai-a3-inference.hf.space";

/**
 * Deterministic mapping of Model A3 indoor furniture classes to canonical class IDs.
 */
export const A3_INDOOR_CLASS_MAP: Record<string, number> = {
  "chair#stool": 0,
  "table": 1,
  "sofa": 2,
  "bed": 3,
  "cabinet#shelf": 4,
  "wardrobe": 5,
  "door": 6,
  "window": 7,
  "mirror": 8,
  "tv#monitor": 9,
  "lamp": 10,
  "home-appliance": 11,
};

/**
 * Generates a stable numeric class ID for known or custom classes.
 */
export function resolveClassId(className: string): number {
  const normalized = className.trim().toLowerCase();
  if (normalized in A3_INDOOR_CLASS_MAP) {
    return A3_INDOOR_CLASS_MAP[normalized];
  }
  let hash = 0;
  for (let i = 0; i < normalized.length; i++) {
    hash = (hash << 5) - hash + normalized.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) % 1000 + 100;
}

/**
 * Normalizes raw ZeroGPU response into canonical RoomAI DetectionResponse.
 *
 * Guarantees:
 * 1. Preserves original image pixel coordinates (x1, y1, x2, y2).
 * 2. Computes normalized bounding coordinates (norm_x1..norm_y2) clamped to [0, 1].
 * 3. Applies confidence threshold filtering if specified.
 * 4. Assigns consistent class IDs and request tracking identifier.
 */
export function normalizeZeroGpuResponse(
  raw: ZeroGpuRawResponse,
  options: {
    confidenceThreshold?: number;
    imageDimensions?: { width: number; height: number };
  } = {}
): DetectionResponse {
  const width = Math.max(1, raw.image?.width || options.imageDimensions?.width || 1280);
  const height = Math.max(1, raw.image?.height || options.imageDimensions?.height || 960);
  const minConf = options.confidenceThreshold ?? 0.15;

  const rawDetections = Array.isArray(raw.detections) ? raw.detections : [];
  const filtered = rawDetections.filter((d) => (d.confidence ?? 0) >= minConf);

  const detections: DetectionItem[] = filtered.map((d) => {
    const rawBbox = d.bbox || { x1: 0, y1: 0, x2: 0, y2: 0 };
    const x1 = Math.max(0, Math.min(width, Number(rawBbox.x1) || 0));
    const y1 = Math.max(0, Math.min(height, Number(rawBbox.y1) || 0));
    const x2 = Math.max(x1, Math.min(width, Number(rawBbox.x2) || 0));
    const y2 = Math.max(y1, Math.min(height, Number(rawBbox.y2) || 0));

    const norm_x1 = Math.max(0, Math.min(1, x1 / width));
    const norm_y1 = Math.max(0, Math.min(1, y1 / height));
    const norm_x2 = Math.max(norm_x1, Math.min(1, x2 / width));
    const norm_y2 = Math.max(norm_y1, Math.min(1, y2 / height));

    const bbox: BoundingBox2D = {
      x1: Math.round(x1 * 10) / 10,
      y1: Math.round(y1 * 10) / 10,
      x2: Math.round(x2 * 10) / 10,
      y2: Math.round(y2 * 10) / 10,
      norm_x1: Math.round(norm_x1 * 10000) / 10000,
      norm_y1: Math.round(norm_y1 * 10000) / 10000,
      norm_x2: Math.round(norm_x2 * 10000) / 10000,
      norm_y2: Math.round(norm_y2 * 10000) / 10000,
    };

    return {
      class_id: resolveClassId(d.class),
      class_name: d.class,
      confidence: Math.round(d.confidence * 10000) / 10000,
      bbox,
    };
  });

  const latencyMs = raw.inference?.latency_ms || 0;

  return {
    status: "success",
    request_id: `det_zerogpu_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`,
    image: {
      width,
      height,
      channels: 3,
    },
    detections,
    inference: {
      model: raw.model || "roomai-a3",
      latency_ms: latencyMs,
      device: "Hugging Face ZeroGPU / CUDA",
      cached: false,
    },
    timing: {
      preprocess_ms: 0,
      inference_ms: latencyMs,
      postprocess_ms: 0,
      total_server_ms: latencyMs,
    },
    timestamp: Date.now() / 1000,
  };
}

/**
 * Client for performing object detection directly against Hugging Face ZeroGPU Space.
 */
export class ZeroGpuDetectionClient {
  private readonly baseUrl: string;
  private readonly defaultTimeoutMs: number;

  constructor() {
    const envUrl = process.env.NEXT_PUBLIC_ROOMAI_INFERENCE_URL;
    this.baseUrl = (envUrl && envUrl.trim().length > 0)
      ? envUrl.trim().replace(/\/+$/, "")
      : DEFAULT_INFERENCE_URL;

    // ZeroGPU cold starts can take 30-45s, allow up to 60s
    const envTimeout = process.env.NEXT_PUBLIC_API_TIMEOUT_MS;
    this.defaultTimeoutMs = envTimeout ? Math.max(60000, parseInt(envTimeout, 10)) : 60000;
  }

  public getInferenceUrl(): string {
    return this.baseUrl;
  }

  /**
   * Checks whether the Hugging Face Space is reachable and responsive.
   */
  async checkReadiness(signal?: AbortSignal): Promise<{ isReady: boolean; status: string; detail?: string }> {
    try {
      const res = await fetch(`${this.baseUrl}/config`, {
        method: "GET",
        signal,
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        return { isReady: true, status: "ready" };
      }
      return { isReady: false, status: "degraded", detail: `HTTP ${res.status}` };
    } catch (err: unknown) {
      return {
        isReady: false,
        status: "unreachable",
        detail: err instanceof Error ? err.message : String(err),
      };
    }
  }

  /**
   * Executes object detection on an image using the ZeroGPU Space.
   */
  async detect(
    file: File | Blob,
    options: ZeroGpuDetectOptions = {}
  ): Promise<DetectResult> {
    const t0 = performance.now();
    options.onStatusUpdate?.("Initializing AI model...");

    // Setup timeout and abort controllers
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => {
      timeoutController.abort(new Error(`ZeroGPU inference timed out after ${this.defaultTimeoutMs}ms`));
    }, this.defaultTimeoutMs);

    let combinedSignal: AbortSignal;
    if (options.signal) {
      combinedSignal = AbortSignal.any
        ? AbortSignal.any([options.signal, timeoutController.signal])
        : options.signal;
    } else {
      combinedSignal = timeoutController.signal;
    }

    try {
      // 1. Upload image to Space
      const formData = new FormData();
      const filename = file instanceof File ? file.name : "scene.jpg";
      formData.append("files", file, filename);

      let uploadRes: Response;
      try {
        uploadRes = await fetch(`${this.baseUrl}/gradio_api/upload`, {
          method: "POST",
          body: formData,
          signal: combinedSignal,
        });
      } catch (uploadErr: unknown) {
        if (options.signal?.aborted) {
          throw new ApiError("Inference cancelled by client", 499, "REQUEST_CANCELLED");
        }
        if (timeoutController.signal.aborted) {
          throw new ApiError(
            `Connection to Hugging Face ZeroGPU timed out after ${this.defaultTimeoutMs}ms`,
            504,
            "TIMEOUT",
            "The Hugging Face Space may be waking up from sleep. Please retry in a moment."
          );
        }
        throw new ApiError(
          `Unable to reach Hugging Face Space at ${this.baseUrl}: ${uploadErr instanceof Error ? uploadErr.message : String(uploadErr)}`,
          503,
          "ZEROGPU_UNREACHABLE",
          "Verify your internet connection or check https://huggingface.co/spaces/Apaarag/roomai-a3-inference status."
        );
      }

      if (!uploadRes.ok) {
        throw new ApiError(
          `Image upload to Hugging Face Space failed with HTTP ${uploadRes.status}`,
          uploadRes.status,
          "UPLOAD_FAILED",
          "Ensure the image format is JPEG, PNG, or WebP."
        );
      }

      const uploadJson = await uploadRes.json();
      const uploadedPath = Array.isArray(uploadJson) ? uploadJson[0] : null;
      if (!uploadedPath) {
        throw new ApiError("Invalid upload response from Hugging Face Space", 502, "INVALID_UPLOAD_RESPONSE");
      }

      // 2. Submit detection job
      options.onStatusUpdate?.("Running detection...");
      const callPayload = {
        data: [
          {
            path: uploadedPath,
            meta: { _type: "gradio.FileData" },
          },
        ],
      };

      const callRes = await fetch(`${this.baseUrl}/gradio_api/call/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(callPayload),
        signal: combinedSignal,
      });

      if (!callRes.ok) {
        const errorText = await callRes.text();
        throw new ApiError(
          `Failed to initiate detection on Hugging Face Space (HTTP ${callRes.status}): ${errorText}`,
          callRes.status,
          "JOB_SUBMISSION_FAILED",
          "The Space might be reloading. Please retry shortly."
        );
      }

      const { event_id } = (await callRes.json()) as { event_id: string };
      if (!event_id) {
        throw new ApiError("No event_id returned by Hugging Face Space", 502, "INVALID_EVENT_ID");
      }

      // 3. Stream SSE response until completion
      options.onStatusUpdate?.("Analyzing room...");
      const sseRes = await fetch(`${this.baseUrl}/gradio_api/call/detect/${event_id}`, {
        method: "GET",
        signal: combinedSignal,
      });

      if (!sseRes.ok) {
        throw new ApiError(
          `Failed to open SSE stream from Hugging Face Space (HTTP ${sseRes.status})`,
          sseRes.status,
          "SSE_STREAM_FAILED"
        );
      }

      const rawResult = await this.readSseCompletion(sseRes, combinedSignal, options.onStatusUpdate);
      const t1 = performance.now();
      const totalRoundtripMs = Math.round(t1 - t0);

      // 4. Normalize response
      const normalizedData = normalizeZeroGpuResponse(rawResult, {
        confidenceThreshold: options.confidenceThreshold,
        imageDimensions: options.imageDimensions,
      });

      const serverReportedMs = normalizedData.inference.latency_ms || 0;
      const clientLatency: ClientLatencyMetrics = {
        upload_ms: Math.max(0, Math.round(totalRoundtripMs - serverReportedMs)),
        server_ms: serverReportedMs,
        inference_ms: serverReportedMs,
        render_ms: 0,
        total_roundtrip_ms: totalRoundtripMs,
      };

      return { data: normalizedData, clientLatency };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Consumes SSE event stream and returns the raw detection output.
   */
  private async readSseCompletion(
    response: Response,
    signal: AbortSignal,
    onStatusUpdate?: (status: string) => void
  ): Promise<ZeroGpuRawResponse> {
    if (!response.body) {
      throw new ApiError("Response body is null on SSE stream", 502, "EMPTY_STREAM");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "";

    try {
      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim();
          } else if (trimmed.startsWith("data:")) {
            const dataStr = trimmed.slice(5).trim();

            if (currentEvent === "progress") {
              onStatusUpdate?.("Running detection...");
            } else if (currentEvent === "generating") {
              onStatusUpdate?.("Analyzing room...");
            } else if (currentEvent === "complete") {
              try {
                const parsed = JSON.parse(dataStr);
                // ZeroGPU Space returns [ImageResult, JsonResult]
                if (Array.isArray(parsed) && parsed.length >= 2) {
                  return parsed[1] as ZeroGpuRawResponse;
                }
                if (parsed && typeof parsed === "object" && "detections" in parsed) {
                  return parsed as ZeroGpuRawResponse;
                }
                throw new Error("Unexpected payload structure in complete event");
              } catch (parseErr) {
                throw new ApiError(
                  `Failed to parse ZeroGPU complete payload: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`,
                  502,
                  "INVALID_PAYLOAD_JSON"
                );
              }
            } else if (currentEvent === "error") {
              let errorDetail = dataStr;
              try {
                const errObj = JSON.parse(dataStr);
                if (errObj && errObj.error) {
                  errorDetail = errObj.error;
                }
              } catch {
                // fall through
              }

              if (errorDetail.includes("ZeroGPU runs limit") || errorDetail.includes("quota exceeded")) {
                throw new ApiError(
                  "ZeroGPU inference quota exceeded. Please wait a moment for the hourly limit to refresh or use local inference.",
                  429,
                  "ZEROGPU_QUOTA_EXCEEDED",
                  "Hugging Face ZeroGPU limits anonymous runs per IP. Retry in a few minutes or switch to local engine."
                );
              }

              throw new ApiError(
                `ZeroGPU inference error: ${errorDetail}`,
                500,
                "ZEROGPU_INFERENCE_ERROR",
                "Verify input image validity or retry in a few moments."
              );
            }
          }
        }
      }

      if (signal.aborted) {
        throw new ApiError("Inference stream aborted by client", 499, "REQUEST_ABORTED");
      }

      throw new ApiError("Stream ended without completion event", 502, "STREAM_CLOSED_PREMATURELY");
    } finally {
      reader.releaseLock();
    }
  }
}

export const zeroGpuClient = new ZeroGpuDetectionClient();
