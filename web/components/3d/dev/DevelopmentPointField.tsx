"use client";

import { useEffect, useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { developmentRoom } from "./fixtureConfig";
import { photographConfig } from "../photographConfig";

type Props = {
  visible?: boolean;
  progress: number;
  opacity: number;
  rgbTexture: THREE.Texture | null;
};

/** Deterministic architectural sampling fixture. This is not ROOMAI output. */
export function DevelopmentPointField({ visible = true, progress, opacity, rgbTexture }: Props) {
  const invalidate = useThree((state) => state.invalidate);
  const points = useRef<THREE.Points>(null);
  const geometry = useMemo(() => {
    const columns = 44;
    const rows = 30;
    const starts: number[] = [];
    const targets: number[] = [];
    const colors: number[] = [];
    const { width, height, floorY, backZ } = developmentRoom;
    const image = rgbTexture?.image as CanvasImageSource | undefined;
    let pixels: Uint8ClampedArray | null = null;
    if (image && typeof document !== "undefined") {
      try {
        const sampler = document.createElement("canvas");
        sampler.width = columns;
        sampler.height = rows;
        const context = sampler.getContext("2d", { willReadFrequently: true });
        if (context) {
          context.drawImage(image, 0, 0, columns, rows);
          pixels = context.getImageData(0, 0, columns, rows).data;
        }
      } catch {
        pixels = null;
      }
    }
    const color = new THREE.Color();

    for (let iy = 0; iy < rows; iy += 1) for (let ix = 0; ix < columns; ix += 1) {
      const u = ix / (columns - 1);
      const v = iy / (rows - 1);
      starts.push((u - 0.5) * photographConfig.width, (0.5 - v) * photographConfig.height, 0.018);

      // Preserve the image grid while deterministic depth layers fan it into the room volume.
      const depthLayer = ((ix * 17 + iy * 31) % 19) / 18;
      targets.push(
        -width / 2 + u * width,
        floorY + (1 - v) * height,
        -0.08 - depthLayer * (Math.abs(backZ) - 0.08),
      );

      const offset = (iy * columns + ix) * 4;
      const red = pixels ? pixels[offset] / 255 : 0.72;
      const green = pixels ? pixels[offset + 1] / 255 : 0.68;
      const blue = pixels ? pixels[offset + 2] / 255 : 0.61;
      color.setRGB(red, green, blue, THREE.SRGBColorSpace);
      colors.push(color.r, color.g, color.b);
    }

    const result = new THREE.BufferGeometry();
    result.setAttribute("position", new THREE.Float32BufferAttribute(starts, 3));
    result.setAttribute("startPosition", new THREE.Float32BufferAttribute(starts, 3));
    result.setAttribute("targetPosition", new THREE.Float32BufferAttribute(targets, 3));
    result.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    result.computeBoundingSphere();
    return result;
  }, [rgbTexture]);
  const material = useMemo(() => new THREE.PointsMaterial({
    size: 0.018, sizeAttenuation: true, vertexColors: true,
    transparent: true, depthWrite: false, opacity: 0, toneMapped: false,
  }), []);

  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    material.opacity = THREE.MathUtils.clamp(opacity, 0, 1);
    if (points.current) {
      const t = THREE.MathUtils.clamp(progress, 0, 1);
      const position = geometry.getAttribute("position") as THREE.BufferAttribute;
      const start = geometry.getAttribute("startPosition") as THREE.BufferAttribute;
      const target = geometry.getAttribute("targetPosition") as THREE.BufferAttribute;
      for (let i = 0; i < position.count; i += 1) {
        position.setXYZ(
          i,
          THREE.MathUtils.lerp(start.getX(i), target.getX(i), t),
          THREE.MathUtils.lerp(start.getY(i), target.getY(i), t),
          THREE.MathUtils.lerp(start.getZ(i), target.getZ(i), t),
        );
      }
      position.needsUpdate = true;
      geometry.computeBoundingSphere();
      points.current.visible = visible && opacity > 0;
    }
    invalidate();
  }, [progress, opacity, visible, geometry, material, invalidate]);

  return <points ref={points} geometry={geometry} material={material} visible={visible && opacity > 0} />;
}
