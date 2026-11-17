"use client";

import { useEffect, useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { developmentRoom } from "./fixtureConfig";

type Props = { visible?: boolean; progress: number; opacity: number };

/** Simple blockout continuation of the development room, not a reconstruction. */
export function DevelopmentWorld({ visible = true, progress, opacity }: Props) {
  const invalidate = useThree((state) => state.invalidate);
  const group = useRef<THREE.Group>(null);
  const { width, height, depth, floorY, backZ } = developmentRoom;
  const geometry = useMemo(() => [
    new THREE.PlaneGeometry(width, depth),
    new THREE.PlaneGeometry(width, height),
    new THREE.PlaneGeometry(width, depth),
    new THREE.BoxGeometry(1.5, 0.18, 1.8),
    new THREE.BoxGeometry(0.72, 1.8, 0.5),
  ], [width, height, depth]);
  const materials = useMemo(() => [
    new THREE.MeshBasicMaterial({ color: "#82796b", side: THREE.DoubleSide, transparent: true, opacity: 0, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: "#504a42", side: THREE.DoubleSide, transparent: true, opacity: 0, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: "#a09687", side: THREE.DoubleSide, transparent: true, opacity: 0, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: "#9b8d77", transparent: true, opacity: 0, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: "#817666", transparent: true, opacity: 0, depthWrite: false }),
  ], []);
  useEffect(() => () => geometry.forEach((g) => g.dispose()), [geometry]);
  useEffect(() => () => materials.forEach((m) => m.dispose()), [materials]);
  useEffect(() => {
    const p = THREE.MathUtils.clamp(progress, 0, 1);
    const alpha = THREE.MathUtils.clamp(opacity, 0, 1);
    for (const material of materials) {
      material.opacity = alpha;
      const opaque = alpha >= 0.999;
      if (material.transparent === opaque) {
        material.transparent = !opaque;
        material.depthWrite = opaque;
        material.needsUpdate = true;
      }
    }
    if (group.current) {
      group.current.scale.setScalar(THREE.MathUtils.lerp(0.98, 1, p));
      group.current.visible = visible && opacity > 0;
    }
    invalidate();
  }, [progress, opacity, visible, materials, invalidate]);

  return <group ref={group} visible={visible && opacity > 0}>
    <mesh geometry={geometry[0]} material={materials[0]} position={[0, floorY, backZ / 2]} rotation={[-Math.PI / 2, 0, 0]} />
    <mesh geometry={geometry[1]} material={materials[1]} position={[0, 0, backZ]} />
    <mesh geometry={geometry[1]} material={materials[1]} position={[-width / 2, 0, backZ / 2]} rotation={[0, Math.PI / 2, 0]} />
    <mesh geometry={geometry[1]} material={materials[1]} position={[width / 2, 0, backZ / 2]} rotation={[0, Math.PI / 2, 0]} />
    <mesh geometry={geometry[2]} material={materials[2]} position={[0, floorY + height, backZ / 2]} rotation={[-Math.PI / 2, 0, 0]} />
    <mesh geometry={geometry[3]} material={materials[3]} position={[0.25, floorY + 0.09, -2.35]} />
    <mesh geometry={geometry[4]} material={materials[4]} position={[-0.98, floorY + 0.9, -3.05]} />
    <mesh geometry={geometry[4]} material={materials[4]} position={[0.98, floorY + 0.9, -3.05]} />
  </group>;
}
