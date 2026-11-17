/**
 * RoomAI 2D-to-3D Provisional Coordinate Unprojection Pipeline.
 * 
 * IMPORTANT ARCHITECTURAL AUDIT NOTE:
 * Model A3 is an indoor 2D object detector (YOLO11s at 800px).
 * It produces 2D bounding boxes in normalized image coordinates [0, 1].
 * 
 * This module unprojects the 2D bounding box center along the 3D optical ray
 * defined by the pinhole camera geometry (FOV = 34°).
 * 
 * Because monocular 2D detection does not measure depth, this pipeline uses an
 * ASSUMED/PROVISIONAL metric scene depth (default Z = 3.0m).
 * This represents a PROVISIONAL 3D RAY PROJECTION, NOT measured metric 3D reconstruction.
 */

import { BoundingBox2D, DetectionItem } from "../api/types";

export interface CameraParameters {
  /** Vertical field of view in degrees (default 34°) */
  fovDegrees: number;
  /** Image aspect ratio width / height (default 16 / 9) */
  aspectRatio: number;
  /** Assumed provisional metric depth Z in meters (default 3.0m) */
  defaultDepthMeters: number;
}

export const DEFAULT_CAMERA_PARAMS: CameraParameters = {
  fovDegrees: 34,
  aspectRatio: 16 / 9,
  defaultDepthMeters: 3.0,
};

export interface ProvisionalSpatialObject3D {
  label: string;
  confidence: number;
  /** Provisional 3D position [x, y, z] in camera coordinate space (meters) at assumed depth */
  position: [number, number, number];
  /** Estimated 3D bounding box dimensions [width, height, depth] in meters */
  dimensions: [number, number, number];
  /** 2D Normalized Center [u, v] in [0, 1] */
  center2D: [number, number];
  /** 2D Normalized Device Coordinates [x_ndc, y_ndc] in [-1, 1] */
  ndc: [number, number];
  /** Explicit indicator that depth is an assumed constant rather than sensor measurement */
  depthIsAssumed: true;
}

// Backwards-compatible alias for existing consumers
export type SpatialObject3D = ProvisionalSpatialObject3D;

/**
 * Converts a normalized 2D bounding box to Normalized Device Coordinates (NDC).
 * 
 * Coordinate System Transformations:
 * - Image Space: (0, 0) top-left, X right [0, 1], Y down [0, 1].
 * - NDC Space: (0, 0) center, X right [-1, +1], Y UP [-1, +1].
 * 
 * Formula:
 * x_ndc = (norm_x - 0.5) * 2
 * y_ndc = -(norm_y - 0.5) * 2  <-- Inverted because Three.js +Y points up
 */
export function bboxToNDC(bbox: BoundingBox2D): {
  centerNDC: [number, number];
  sizeNDC: [number, number];
  centerNorm: [number, number];
} {
  const norm_u = (bbox.norm_x1 + bbox.norm_x2) / 2;
  const norm_v = (bbox.norm_y1 + bbox.norm_y2) / 2;
  const norm_w = Math.max(0, bbox.norm_x2 - bbox.norm_x1);
  const norm_h = Math.max(0, bbox.norm_y2 - bbox.norm_y1);

  const x_ndc = (norm_u - 0.5) * 2;
  const y_ndc = -(norm_v - 0.5) * 2; // Flip Y axis for OpenGL / Three.js

  const size_x_ndc = norm_w * 2;
  const size_y_ndc = norm_h * 2;

  return {
    centerNDC: [x_ndc, y_ndc],
    sizeNDC: [size_x_ndc, size_y_ndc],
    centerNorm: [norm_u, norm_v],
  };
}

/**
 * Unprojects a 2D bounding box to 3D metric camera coordinates using pinhole optics.
 * 
 * Optics:
 * half_fov_rad = (fov / 2) * (PI / 180)
 * tan_half_fov = tan(half_fov_rad)
 * 
 * At distance Z (depth):
 * visible_height_at_Z = 2 * Z * tan_half_fov
 * visible_width_at_Z  = visible_height_at_Z * aspect_ratio
 * 
 * X_cam = x_ndc * (visible_width_at_Z / 2)  = x_ndc * Z * tan_half_fov * aspect_ratio
 * Y_cam = y_ndc * (visible_height_at_Z / 2) = y_ndc * Z * tan_half_fov
 * Z_cam = -Z (looking down negative Z in right-handed Three.js)
 */
export function unprojectDetectionTo3D(
  detection: DetectionItem,
  cameraParams: Partial<CameraParameters> = {},
  customDepthMeters?: number
): SpatialObject3D {
  const params: CameraParameters = { ...DEFAULT_CAMERA_PARAMS, ...cameraParams };
  const z_depth = customDepthMeters ?? params.defaultDepthMeters;

  const { centerNDC, sizeNDC, centerNorm } = bboxToNDC(detection.bbox);
  const [x_ndc, y_ndc] = centerNDC;
  const [w_ndc, h_ndc] = sizeNDC;

  const halfFovRad = (params.fovDegrees * 0.5 * Math.PI) / 180;
  const tanHalfFov = Math.tan(halfFovRad);

  const x_cam = x_ndc * z_depth * tanHalfFov * params.aspectRatio;
  const y_cam = y_ndc * z_depth * tanHalfFov;
  const z_cam = -z_depth;

  const width_3d = Math.abs(w_ndc * z_depth * tanHalfFov * params.aspectRatio);
  const height_3d = Math.abs(h_ndc * z_depth * tanHalfFov);
  // Estimate depth proportionally to 2D footprint (or aspect heuristic)
  const depth_3d = Math.min(width_3d, height_3d) * 0.8;

  return {
    label: detection.class_name,
    confidence: detection.confidence,
    position: [
      parseFloat(x_cam.toFixed(4)),
      parseFloat(y_cam.toFixed(4)),
      parseFloat(z_cam.toFixed(4)),
    ],
    dimensions: [
      parseFloat(width_3d.toFixed(4)),
      parseFloat(height_3d.toFixed(4)),
      parseFloat(depth_3d.toFixed(4)),
    ],
    center2D: [parseFloat(centerNorm[0].toFixed(4)), parseFloat(centerNorm[1].toFixed(4))],
    ndc: [parseFloat(x_ndc.toFixed(4)), parseFloat(y_ndc.toFixed(4))],
    depthIsAssumed: true,
  };
}

/**
 * Batch unprojection for all detections in a response.
 */
export function unprojectAllDetections(
  detections: DetectionItem[],
  cameraParams?: Partial<CameraParameters>,
  depthMap?: number[]
): SpatialObject3D[] {
  return detections.map((det, index) => {
    const depth = depthMap && depthMap[index] ? depthMap[index] : undefined;
    return unprojectDetectionTo3D(det, cameraParams, depth);
  });
}
