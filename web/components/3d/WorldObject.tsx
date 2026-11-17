"use client";

import {
  Component,
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
} from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { clone as cloneScene } from "three/examples/jsm/utils/SkeletonUtils.js";

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

export interface WorldObjectProps {
  /** URL of the verified ROOMAI GLB. Required: if missing, the component fails with an error. */
  url?: string | null;
  /**
   * 0 → 1, parent-controlled. Drives a restrained architectural "section cut":
   * the real geometry is revealed bottom-to-top along its own vertical extent.
   * At 1 the model is fully shown and nothing is clipped. Default 1.
   */
  progress?: number;
  /** 0 → 1, parent-controlled. Multiplies each material's own opacity. Default 1. */
  opacity?: number;
  /** Parent-controlled visibility. Default true. */
  visible?: boolean;
  /**
   * Longest bounding-box dimension of the model in scene units. The model is
   * centred on the origin and uniformly scaled to this size. Default 4.
   */
  size?: number;
  /** Called once the GLB has loaded and the model is ready to render. */
  onReady?: () => void;
  /**
   * Called when the URL is missing or the GLB fails to load. If omitted, the
   * error is re-thrown so the parent's error boundary receives it. Either way,
   * nothing is rendered in place of the failed world.
   */
  onError?: (error: Error) => void;
}

/** Start fetching/parsing a GLB ahead of mounting <WorldObject />. */
export function preloadWorldAsset(url: string): void {
  useGLTF.preload(url);
}

/** Explicitly release the cached GLTF for a URL (call when it will not be shown again). */
export function releaseWorldAsset(url: string): void {
  useGLTF.clear(url);
}

/* -------------------------------------------------------------------------- */
/* Error handling                                                             */
/* -------------------------------------------------------------------------- */

const MISSING_URL_MESSAGE =
  "WorldObject: no `url` was provided. WORLD ASSET NOT VERIFIED — supply the URL of a real ROOMAI GLB. No substitute geometry is rendered.";

class WorldErrorBoundary extends Component<
  { onError?: (error: Error) => void; children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error) {
    console.error("[WorldObject] failed:", error);
    this.props.onError?.(error);
  }

  render() {
    const { error } = this.state;
    if (error) {
      // No handler: propagate so the parent can decide. Never draw stand-in geometry.
      if (!this.props.onError) throw error;
      return null;
    }
    return this.props.children;
  }
}

function MissingUrl(): never {
  throw new Error(MISSING_URL_MESSAGE);
}

/* -------------------------------------------------------------------------- */
/* Internals                                                                  */
/* -------------------------------------------------------------------------- */

interface BaseMaterialState {
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
}

const BASE_KEY = "__worldBase";
const clamp01 = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

function cloneMaterial(material: THREE.Material): THREE.Material {
  // Clone so opacity/clipping changes never leak into the cached GLTF.
  // Textures are shared by reference with the cache and are NOT cloned.
  const m = material.clone();
  const base: BaseMaterialState = {
    opacity: material.opacity,
    transparent: material.transparent,
    depthWrite: material.depthWrite,
  };
  m.userData[BASE_KEY] = base;
  return m;
}

interface PreparedModel {
  root: THREE.Object3D;
  materials: THREE.Material[];
  scale: number;
  offset: THREE.Vector3;
  /** Vertical extent of the fitted model in the wrapper group's local space. */
  minY: number;
  height: number;
}

function prepareModel(source: THREE.Object3D, size: number, url: string): PreparedModel {
  const root = cloneScene(source);
  const materials = new Set<THREE.Material>();

  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.material = Array.isArray(mesh.material)
      ? mesh.material.map((m) => {
          const c = cloneMaterial(m);
          materials.add(c);
          return c;
        })
      : (() => {
          const c = cloneMaterial(mesh.material);
          materials.add(c);
          return c;
        })();
  });

  const box = new THREE.Box3().setFromObject(root);
  const dims = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(dims.x, dims.y, dims.z);

  if (box.isEmpty() || !Number.isFinite(maxDim) || maxDim <= 0) {
    throw new Error(`WorldObject: the GLB at "${url}" contains no renderable geometry.`);
  }

  const scale = size / maxDim;
  const height = dims.y * scale;
  return {
    root,
    materials: [...materials],
    scale,
    offset: center.multiplyScalar(-scale),
    minY: -height / 2,
    height,
  };
}

interface LoadedWorldProps {
  url: string;
  progress: number;
  opacity: number;
  visible: boolean;
  size: number;
  onReady?: () => void;
}

function LoadedWorld({ url, progress, opacity, visible, size, onReady }: LoadedWorldProps) {
  // Suspends while loading; throws to the error boundary on failure.
  // The GLTF is cached by URL, so it is not re-fetched or re-parsed on re-render.
  const gltf = useGLTF(url);
  const invalidate = useThree((s) => s.invalidate);
  const gl = useThree((s) => s.gl);

  const groupRef = useRef<THREE.Group>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  // Built once per (gltf, size): cloned scene + cloned materials.
  const model = useMemo(() => prepareModel(gltf.scene, size, url), [gltf, size, url]);

  // Local-space and world-space clip planes (stable objects shared by all materials).
  const localPlane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, -1, 0), 0), []);
  const worldPlane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, -1, 0), 0), []);

  // Attach the clipping plane to every cloned material (one-time shader setup).
  useLayoutEffect(() => {
    for (const m of model.materials) {
      m.clippingPlanes = [worldPlane];
      m.needsUpdate = true;
    }
  }, [model, worldPlane]);

  // Local clipping must be enabled on the renderer; restore the previous value on unmount.
  useEffect(() => {
    const previous = gl.localClippingEnabled;
    gl.localClippingEnabled = true;
    return () => {
      gl.localClippingEnabled = previous;
    };
  }, [gl]);

  // Apply parent-controlled values imperatively (no React state, no per-frame renders).
  useLayoutEffect(() => {
    const group = groupRef.current;
    if (!group) return;

    const p = clamp01(progress);
    const o = clamp01(opacity);

    group.visible = visible && o > 0 && p > 0;

    // Section cut: keep everything below cutY. At p = 1 the plane sits above the model.
    const cutY = p >= 1 ? model.minY + model.height * 2 : model.minY + model.height * p;
    localPlane.constant = cutY;

    for (const m of model.materials) {
      const base = m.userData[BASE_KEY] as BaseMaterialState;
      const transparent = base.transparent || o < 1;
      if (m.transparent !== transparent) {
        m.transparent = transparent;
        m.needsUpdate = true;
      }
      m.opacity = base.opacity * o;
      m.depthWrite = o < 1 ? false : base.depthWrite;
    }

    invalidate(); // parent may use frameloop="demand"
  }, [progress, opacity, visible, model, localPlane, invalidate]);

  // Keep the world-space clip plane aligned with whatever transform the parent applies.
  useFrame(() => {
    const group = groupRef.current;
    if (!group || !group.visible) return;
    group.updateWorldMatrix(true, false);
    worldPlane.copy(localPlane).applyMatrix4(group.matrixWorld);
  });

  useEffect(() => {
    onReadyRef.current?.();
    invalidate();
  }, [model, invalidate]);

  // Release only what this component created. Geometries and textures belong to the
  // cached GLTF and are intentionally left alone (see releaseWorldAsset).
  useEffect(() => {
    return () => {
      for (const m of model.materials) m.dispose();
    };
  }, [model]);

  return (
    <group ref={groupRef} visible={false} dispose={null}>
      <primitive object={model.root} scale={model.scale} position={model.offset} dispose={null} />
    </group>
  );
}

/* -------------------------------------------------------------------------- */
/* Component                                                                  */
/* -------------------------------------------------------------------------- */

export function WorldObject({
  url,
  progress = 1,
  opacity = 1,
  visible = true,
  size = 4,
  onReady,
  onError,
}: WorldObjectProps) {
  return (
    // key resets the boundary when the URL changes. Suspense is provided by the parent.
    <WorldErrorBoundary key={url ?? "__missing__"} onError={onError}>
      {url ? (
        <LoadedWorld
          url={url}
          progress={progress}
          opacity={opacity}
          visible={visible}
          size={size}
          onReady={onReady}
        />
      ) : (
        <MissingUrl />
      )}
    </WorldErrorBoundary>
  );
}
