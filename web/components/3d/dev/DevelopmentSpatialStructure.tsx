"use client";

import { useEffect, useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { developmentRoom } from "./fixtureConfig";

type Props = { visible?: boolean; progress: number; opacity: number };

/** Architectural planes fixture in the shared room coordinate system. */
export function DevelopmentSpatialStructure({ visible = true, progress, opacity }: Props) {
  const invalidate = useThree((state) => state.invalidate);
  const group = useRef<THREE.Group>(null);
  const { width, height, depth, floorY, backZ } = developmentRoom;
  const geometries = useMemo(() => [
    new THREE.PlaneGeometry(width, depth),
    new THREE.PlaneGeometry(width, height),
    new THREE.PlaneGeometry(depth, height),
  ], [width, height, depth]);
  const outlineGeometry = useMemo(() => {
    const bounds = new THREE.BoxGeometry(width, height, depth);
    const edges = new THREE.EdgesGeometry(bounds);
    bounds.dispose();
    return edges;
  }, [width, height, depth]);
  const materials = useMemo(() => [0, 1, 2].map(() => new THREE.MeshBasicMaterial({
    color: "#b3aa9b", transparent: true, opacity: 0, side: THREE.DoubleSide,
    depthWrite: false, toneMapped: false,
  })), []);
  const outlineMaterial = useMemo(() => new THREE.LineBasicMaterial({
    color: "#c2b7a4", transparent: true, opacity: 0, depthWrite: false, toneMapped: false,
  }), []);
  useEffect(() => () => geometries.forEach((g) => g.dispose()), [geometries]);
  useEffect(() => () => outlineGeometry.dispose(), [outlineGeometry]);
  useEffect(() => () => materials.forEach((m) => m.dispose()), [materials]);
  useEffect(() => () => outlineMaterial.dispose(), [outlineMaterial]);
  useEffect(() => {
    const t = THREE.MathUtils.clamp(progress, 0, 1);
    for (const material of materials) material.opacity = THREE.MathUtils.clamp(opacity, 0, 1) * 0.2 * t;
    outlineMaterial.opacity = THREE.MathUtils.clamp(opacity, 0, 1) * 0.7;
    const edgeVertexCount = outlineGeometry.getAttribute("position").count;
    outlineGeometry.setDrawRange(0, Math.floor(edgeVertexCount * t / 2) * 2);
    if (group.current) {
      group.current.scale.setScalar(THREE.MathUtils.lerp(0.96, 1, t));
      group.current.visible = visible && opacity > 0;
    }
    invalidate();
  }, [progress, opacity, visible, geometries, outlineGeometry, materials, outlineMaterial, invalidate]);

  return <group ref={group} visible={visible && opacity > 0}>
    <mesh geometry={geometries[0]} material={materials[0]} position={[0, floorY, backZ / 2]} rotation={[-Math.PI / 2, 0, 0]} />
    <mesh geometry={geometries[1]} material={materials[1]} position={[0, 0, backZ]} />
    <mesh geometry={geometries[2]} material={materials[2]} position={[-width / 2, 0, backZ / 2]} rotation={[0, Math.PI / 2, 0]} />
    <mesh geometry={geometries[2]} material={materials[2]} position={[width / 2, 0, backZ / 2]} rotation={[0, Math.PI / 2, 0]} />
    <mesh geometry={geometries[0]} material={materials[0]} position={[0, floorY + height, backZ / 2]} rotation={[-Math.PI / 2, 0, 0]} />
    <lineSegments geometry={outlineGeometry} material={outlineMaterial} position={[0, 0, backZ / 2]} />
  </group>;
}
