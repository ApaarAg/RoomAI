"use client";

/**
 * SpatialStructureObject — INTERFACE ONLY.
 *
 * STRUCTURAL SOURCE DATA NOT VERIFIED.
 *
 * The only ROOMAI reconstruction asset available is scene_reconstruction.ply:
 * 61 colored points spanning ~0.038 x 0.103 x 0.077 units, with no faces,
 * no planes, no depth map and no room-scale extent. That is not enough to
 * legitimately derive walls, floor, ceiling or any room structure, so this
 * component deliberately renders NOTHING rather than fabricating geometry.
 *
 * When real structural data exists (e.g. a reconstructed mesh, plane
 * segmentation, or a scene-scale point cloud), implement the rendering inside
 * this component. The props below are the stable contract for the parent.
 */

export type SpatialStructureObjectProps = {
  /** 0 = hidden, 1 = fully revealed. Controlled by the parent. */
  progress?: number;
  /** 0..1 master opacity, multiplied with progress. */
  opacity?: number;
  /** Hard on/off switch. */
  visible?: boolean;
  /** Uniform scale. 1 = source units untouched. */
  scale?: number;
  /** Called once real structural data is loaded and renderable. Never fires while unverified. */
  onReady?: () => void;
  /** Called if structural data fails to load or validate. */
  onError?: (error: unknown) => void;
};

/** False until a real, verified structural source is wired in. */
export const SPATIAL_STRUCTURE_SOURCE_VERIFIED = false;

export function SpatialStructureObject(_props: SpatialStructureObjectProps) {
  // Intentionally renders nothing: no real structural data to draw.
  return null;
}

export default SpatialStructureObject;
