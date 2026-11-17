import assert from "node:assert";
import { test, describe } from "node:test";
import {
  normalizeZeroGpuResponse,
  resolveClassId,
  A3_INDOOR_CLASS_MAP,
  ZeroGpuRawResponse,
} from "../lib/detectionClient";

describe("ZeroGPU Detection Client & Normalizer", () => {
  test("resolveClassId correctly maps known 12 indoor classes to IDs 0-11", () => {
    assert.strictEqual(resolveClassId("chair#stool"), 0);
    assert.strictEqual(resolveClassId("table"), 1);
    assert.strictEqual(resolveClassId("sofa"), 2);
    assert.strictEqual(resolveClassId("bed"), 3);
    assert.strictEqual(resolveClassId("cabinet#shelf"), 4);
    assert.strictEqual(resolveClassId("wardrobe"), 5);
    assert.strictEqual(resolveClassId("door"), 6);
    assert.strictEqual(resolveClassId("window"), 7);
    assert.strictEqual(resolveClassId("mirror"), 8);
    assert.strictEqual(resolveClassId("tv#monitor"), 9);
    assert.strictEqual(resolveClassId("lamp"), 10);
    assert.strictEqual(resolveClassId("home-appliance"), 11);
  });

  test("resolveClassId assigns stable deterministic ID for custom or unmapped classes", () => {
    const id1 = resolveClassId("bookshelf");
    const id2 = resolveClassId("bookshelf");
    assert.strictEqual(id1, id2);
    assert.ok(id1 >= 100);
  });

  test("normalizes 1920x1080 image response with exact pixel & normalized coordinates", () => {
    const raw: ZeroGpuRawResponse = {
      model: "roomai-a3",
      model_version: "a3",
      image: { width: 1920, height: 1080 },
      detections: [
        {
          class: "bed",
          confidence: 0.744,
          bbox: {
            x1: 567.59,
            y1: 394.98,
            x2: 1596.96,
            y2: 898.57,
          },
        },
      ],
      inference: {
        latency_ms: 123.45,
        confidence_threshold: 0.25,
        image_size: 800,
      },
    };

    const res = normalizeZeroGpuResponse(raw);

    assert.strictEqual(res.status, "success");
    assert.ok(res.request_id.startsWith("det_zerogpu_"));
    assert.strictEqual(res.image.width, 1920);
    assert.strictEqual(res.image.height, 1080);
    assert.strictEqual(res.detections.length, 1);

    const det = res.detections[0];
    assert.strictEqual(det.class_name, "bed");
    assert.strictEqual(det.class_id, 3);
    assert.strictEqual(det.confidence, 0.744);

    // Pixel coordinates preserved
    assert.strictEqual(det.bbox.x1, 567.6);
    assert.strictEqual(det.bbox.y1, 395.0);
    assert.strictEqual(det.bbox.x2, 1597.0);
    assert.strictEqual(det.bbox.y2, 898.6);

    // Normalized coordinates accurately calculated
    assert.strictEqual(det.bbox.norm_x1, Math.round((567.6 / 1920) * 10000) / 10000);
    assert.strictEqual(det.bbox.norm_y1, Math.round((395.0 / 1080) * 10000) / 10000);
    assert.strictEqual(det.bbox.norm_x2, Math.round((1597.0 / 1920) * 10000) / 10000);
    assert.strictEqual(det.bbox.norm_y2, Math.round((898.6 / 1080) * 10000) / 10000);

    // Provenance and timing preserved
    assert.strictEqual(res.inference.model, "roomai-a3");
    assert.strictEqual(res.inference.latency_ms, 123.45);
    assert.strictEqual(res.inference.device, "Hugging Face ZeroGPU / CUDA");
  });

  test("normalizes 1280x720 image response with correct aspect ratio coordinates", () => {
    const raw: ZeroGpuRawResponse = {
      model: "roomai-a3",
      image: { width: 1280, height: 720 },
      detections: [
        {
          class: "chair#stool",
          confidence: 0.88,
          bbox: { x1: 100, y1: 200, x2: 300, y2: 500 },
        },
      ],
      inference: {
        latency_ms: 85.0,
        confidence_threshold: 0.25,
        image_size: 800,
      },
    };

    const res = normalizeZeroGpuResponse(raw);
    assert.strictEqual(res.image.width, 1280);
    assert.strictEqual(res.image.height, 720);

    const b = res.detections[0].bbox;
    assert.strictEqual(b.norm_x1, Math.round((100 / 1280) * 10000) / 10000);
    assert.strictEqual(b.norm_y1, Math.round((200 / 720) * 10000) / 10000);
    assert.strictEqual(b.norm_x2, Math.round((300 / 1280) * 10000) / 10000);
    assert.strictEqual(b.norm_y2, Math.round((500 / 720) * 10000) / 10000);
  });

  test("normalizes square 1000x1000 image response with symmetric coordinates", () => {
    const raw: ZeroGpuRawResponse = {
      model: "roomai-a3",
      image: { width: 1000, height: 1000 },
      detections: [
        {
          class: "table",
          confidence: 0.92,
          bbox: { x1: 250, y1: 250, x2: 750, y2: 750 },
        },
      ],
    };

    const res = normalizeZeroGpuResponse(raw);
    assert.strictEqual(res.image.width, 1000);
    assert.strictEqual(res.image.height, 1000);

    const b = res.detections[0].bbox;
    assert.strictEqual(b.x1, 250);
    assert.strictEqual(b.y1, 250);
    assert.strictEqual(b.norm_x1, 0.25);
    assert.strictEqual(b.norm_y1, 0.25);
    assert.strictEqual(b.norm_x2, 0.75);
    assert.strictEqual(b.norm_y2, 0.75);
  });

  test("filters detections by confidenceThreshold", () => {
    const raw: ZeroGpuRawResponse = {
      detections: [
        { class: "chair#stool", confidence: 0.25, bbox: { x1: 10, y1: 10, x2: 20, y2: 20 } },
        { class: "table", confidence: 0.55, bbox: { x1: 30, y1: 30, x2: 40, y2: 40 } },
        { class: "bed", confidence: 0.85, bbox: { x1: 50, y1: 50, x2: 60, y2: 60 } },
      ],
    };

    const resLow = normalizeZeroGpuResponse(raw, { confidenceThreshold: 0.20 });
    assert.strictEqual(resLow.detections.length, 3);

    const resMed = normalizeZeroGpuResponse(raw, { confidenceThreshold: 0.50 });
    assert.strictEqual(resMed.detections.length, 2);

    const resHigh = normalizeZeroGpuResponse(raw, { confidenceThreshold: 0.80 });
    assert.strictEqual(resHigh.detections.length, 1);
    assert.strictEqual(resHigh.detections[0].class_name, "bed");
  });

  test("clamps coordinates that extend beyond image boundaries safely", () => {
    const raw: ZeroGpuRawResponse = {
      image: { width: 500, height: 500 },
      detections: [
        {
          class: "sofa",
          confidence: 0.7,
          bbox: { x1: -50, y1: -20, x2: 600, y2: 550 },
        },
      ],
    };

    const res = normalizeZeroGpuResponse(raw);
    const b = res.detections[0].bbox;
    assert.strictEqual(b.x1, 0);
    assert.strictEqual(b.y1, 0);
    assert.strictEqual(b.x2, 500);
    assert.strictEqual(b.y2, 500);
    assert.strictEqual(b.norm_x1, 0);
    assert.strictEqual(b.norm_y1, 0);
    assert.strictEqual(b.norm_x2, 1);
    assert.strictEqual(b.norm_y2, 1);
  });

  test("handles empty detections array gracefully", () => {
    const raw: ZeroGpuRawResponse = {
      detections: [],
    };

    const res = normalizeZeroGpuResponse(raw);
    assert.strictEqual(res.detections.length, 0);
    assert.strictEqual(res.status, "success");
  });
});
