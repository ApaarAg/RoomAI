"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { Html } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

export type Vec3 = [number, number, number];

/** ROOMAI oxide accent. */
export const ROOMAI_OXIDE = "#8B4937";

export interface SpatialAnnotationObjectProps {
  /** Anchor position in the parent's coordinate space. Supplied by the parent. */
  position: Vec3;
  /** Small text label next to the marker. Text is supplied by the parent; nothing is computed here. */
  label?: string;
  /**
   * Optional second point in the same space as `position`. When given, a thin
   * line is drawn from `position` to `target` (a distance / measurement line).
   * The component never computes or displays a distance; put any value in `label`.
   */
  target?: Vec3;
  /** Parent-controlled visibility. Default true. */
  visible?: boolean;
  /**
   * 0 → 1, parent-controlled. Staged reveal: marker (0–0.3), line (0.2–0.8),
   * label (0.7–1). Default 1.
   */
  progress?: number;
  /** Accent colour. Default ROOMAI oxide `#8B4937`. */
  color?: string;
  /** Marker radius in scene units. Default 0.02. */
  size?: number;
  /** Show three tiny axis ticks at the anchor (a coordinate indicator). Default false. */
  axes?: boolean;
  /** If true, scene geometry can hide the annotation. Default false (always readable). */
  occlude?: boolean;
}

/* -------------------------------------------------------------------------- */
/* Constants & helpers                                                        */
/* -------------------------------------------------------------------------- */

const DOT_OPACITY = 0.85;
const LINE_OPACITY = 0.55;
const AXES_OPACITY = 0.4;
const AXES_LENGTH_IN_SIZES = 3;

const clamp01 = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

/** Smoothstep ramp of p over [a, b]. */
function ramp(p: number, a: number, b: number): number {
  const t = clamp01((p - a) / (b - a));
  return t * t * (3 - 2 * t);
}

/* -------------------------------------------------------------------------- */
/* Component                                                                  */
/* -------------------------------------------------------------------------- */

export function SpatialAnnotationObject({
  position,
  label,
  target,
  visible = true,
  progress = 1,
  color = ROOMAI_OXIDE,
  size = 0.02,
  axes = false,
  occlude = false,
}: SpatialAnnotationObjectProps) {
  const invalidate = useThree((s) => s.invalidate);

  const [px, py, pz] = position;
  const hasTarget = !!target;
  const tx = target?.[0] ?? 0;
  const ty = target?.[1] ?? 0;
  const tz = target?.[2] ?? 0;

  const groupRef = useRef<THREE.Group>(null);
  const dotRef = useRef<THREE.Mesh>(null);
  const endRef = useRef<THREE.Mesh>(null);
  const leaderRef = useRef<THREE.LineSegments>(null);
  const axesRef = useRef<THREE.LineSegments>(null);
  const labelElRef = useRef<HTMLDivElement | null>(null);

  /* ---- Resources created once, disposed on unmount ---------------------- */

  const dotGeometry = useMemo(() => new THREE.SphereGeometry(1, 16, 12), []);

  const axesGeometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1], 3),
    );
    return g;
  }, []);

  // Leader line runs from the anchor (local origin) to the target (local offset).
  const leaderGeometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 0, 0, tx - px, ty - py, tz - pz], 3),
    );
    return g;
  }, [px, py, pz, tx, ty, tz]);

  const dotMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false }),
    [],
  );
  const leaderMaterial = useMemo(
    () => new THREE.LineBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false }),
    [],
  );
  const axesMaterial = useMemo(
    () => new THREE.LineBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false }),
    [],
  );

  useEffect(() => () => dotGeometry.dispose(), [dotGeometry]);
  useEffect(() => () => axesGeometry.dispose(), [axesGeometry]);
  useEffect(() => () => leaderGeometry.dispose(), [leaderGeometry]); // also on target change
  useEffect(() => () => dotMaterial.dispose(), [dotMaterial]);
  useEffect(() => () => leaderMaterial.dispose(), [leaderMaterial]);
  useEffect(() => () => axesMaterial.dispose(), [axesMaterial]);

  /* ---- Imperative updates (no React state, no per-frame work) ----------- */

  // Latest props for the label element, which mounts asynchronously inside <Html>.
  const latest = useRef({ progress, visible, color });
  latest.current = { progress, visible, color };

  const applyLabel = useCallback(() => {
    const el = labelElRef.current;
    if (!el) return;
    const { progress: p, visible: v, color: c } = latest.current;
    const a = ramp(clamp01(p), 0.7, 1);
    el.style.color = c;
    el.style.opacity = String(a);
    el.style.display = v && a > 0 ? "block" : "none";
  }, []);

  const setLabelEl = useCallback(
    (el: HTMLDivElement | null) => {
      labelElRef.current = el;
      applyLabel();
    },
    [applyLabel],
  );

  useLayoutEffect(() => {
    const group = groupRef.current;
    if (!group) return;

    const p = clamp01(progress);
    const markerT = ramp(p, 0, 0.3);
    const lineT = ramp(p, 0.2, 0.8);
    const endT = ramp(p, 0.75, 1);

    group.visible = visible && p > 0;

    dotMaterial.color.set(color);
    leaderMaterial.color.set(color);
    axesMaterial.color.set(color);

    const depthTest = occlude;
    dotMaterial.depthTest = depthTest;
    leaderMaterial.depthTest = depthTest;
    axesMaterial.depthTest = depthTest;

    dotMaterial.opacity = DOT_OPACITY * markerT;
    leaderMaterial.opacity = LINE_OPACITY * lineT;
    axesMaterial.opacity = AXES_OPACITY * markerT;

    const dot = dotRef.current;
    if (dot) {
      dot.visible = markerT > 0;
      dot.scale.setScalar(Math.max(size * markerT, 1e-6));
    }

    const leader = leaderRef.current;
    if (leader) {
      leader.visible = hasTarget && lineT > 0;
      leader.scale.setScalar(Math.max(lineT, 1e-6)); // grows from anchor toward target
    }

    const end = endRef.current;
    if (end) {
      end.visible = hasTarget && endT > 0;
      end.position.set(tx - px, ty - py, tz - pz);
      end.scale.setScalar(Math.max(size * 0.6 * endT, 1e-6));
    }

    const axesObj = axesRef.current;
    if (axesObj) {
      axesObj.visible = axes && markerT > 0;
      axesObj.scale.setScalar(Math.max(size * AXES_LENGTH_IN_SIZES * markerT, 1e-6));
    }

    applyLabel();
    invalidate(); // parent may use frameloop="demand"
  }, [
    progress,
    visible,
    color,
    size,
    axes,
    occlude,
    hasTarget,
    px,
    py,
    pz,
    tx,
    ty,
    tz,
    dotMaterial,
    leaderMaterial,
    axesMaterial,
    applyLabel,
    invalidate,
  ]);

  const renderOrder = occlude ? 0 : 999;

  // Note: dispose={null} prevents R3F from auto-disposing shared/memoized resources on unmount;
  // manual lifecycle disposal is explicitly managed above in useEffect hooks (lines 128-133).
  return (
    <group ref={groupRef} position={position} visible={false}>
      <mesh
        ref={dotRef}
        geometry={dotGeometry}
        material={dotMaterial}
        renderOrder={renderOrder}
        raycast={() => null}
        dispose={null}
      />

      <lineSegments
        ref={axesRef}
        geometry={axesGeometry}
        material={axesMaterial}
        renderOrder={renderOrder}
        raycast={() => null}
        dispose={null}
      />

      {hasTarget && (
        <>
          <lineSegments
            ref={leaderRef}
            geometry={leaderGeometry}
            material={leaderMaterial}
            renderOrder={renderOrder}
            raycast={() => null}
            dispose={null}
          />
          <mesh
            ref={endRef}
            geometry={dotGeometry}
            material={dotMaterial}
            renderOrder={renderOrder}
            raycast={() => null}
            dispose={null}
          />
        </>
      )}

      {label ? (
        <Html center={false} zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
          <div
            ref={setLabelEl}
            style={{
              display: "none",
              opacity: 0,
              transform: "translate(10px, -50%)",
              whiteSpace: "nowrap",
              pointerEvents: "none",
              userSelect: "none",
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
              fontSize: "10px",
              lineHeight: 1,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
            }}
          >
            {label}
          </div>
        </Html>
      ) : null}
    </group>
  );
}
