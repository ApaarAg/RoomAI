"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { pointCloudConfig } from "./pointCloudConfig";

export const POINT_CLOUD_URL = pointCloudConfig.assetUrl;

export type PointCloudObjectProps = {
  progress: number;
  opacity?: number;
  visible?: boolean;
  onReady: () => void;
  onError: () => void;
};

function parseVerifiedPly(payload: string) {
  const lines = payload.trim().split(/\r?\n/);
  const headerEnd = lines.indexOf("end_header");
  if (headerEnd < 0 || lines[0] !== "ply" || lines[1] !== "format ascii 1.0") {
    throw new Error("Unsupported point-cloud PLY format");
  }

  const countLine = lines.find((line) => line.startsWith("element vertex "));
  const count = countLine ? Number(countLine.split(" ")[2]) : NaN;
  if (count !== pointCloudConfig.vertexCount || lines.length - headerEnd - 1 !== count) {
    throw new Error("Verified point-cloud vertex count does not match the job artifact");
  }

  const positions = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const values = lines[headerEnd + 1 + index].trim().split(/\s+/).map(Number);
    if (values.length !== 9 || !values.every(Number.isFinite)) throw new Error("Invalid point-cloud vertex data");
    positions.set(values.slice(0, 3), index * 3);
    colors.set(values.slice(3, 6), index * 3);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
  geometry.computeBoundingSphere();
  return geometry;
}

export function PointCloudObject({ progress, opacity = 1, visible = true, onReady, onError }: PointCloudObjectProps) {
  const ownedGeometry = useRef<THREE.BufferGeometry | null>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const invalidate = useThree((state) => state.invalidate);
  const displayProgress = THREE.MathUtils.clamp(progress, 0, 1);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    fetch(pointCloudConfig.assetUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load point cloud: ${response.status}`);
        return response.text();
      })
      .then((payload) => {
        const loaded = parseVerifiedPly(payload);
        if (!active) {
          loaded.dispose();
          return;
        }
        ownedGeometry.current = loaded;
        setGeometry(loaded);
        onReady();
        invalidate();
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) onError();
      });

    return () => {
      active = false;
      controller.abort();
      ownedGeometry.current?.dispose();
      ownedGeometry.current = null;
    };
  }, [invalidate, onError, onReady]);

  const material = useMemo(() => new THREE.PointsMaterial({
    vertexColors: true,
    size: pointCloudConfig.pointSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0,
    depthWrite: false,
  }), []);

  useEffect(() => () => material.dispose(), [material]);

  useEffect(() => {
    material.opacity = displayProgress * opacity * pointCloudConfig.opacity;
    invalidate();
  }, [displayProgress, opacity, invalidate, material]);

  if (!geometry) return null;
  return <points geometry={geometry} material={material} visible={visible && opacity > 0} />;
}
