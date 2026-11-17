"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useThree, useFrame, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { depthFieldConfig, photographConfig } from "./photographConfig";

type DepthFieldObjectProps = {
  interactive: boolean;
  progress: number;
  rgbTexture: THREE.Texture | null;
  opacity?: number;
  flatOpacity?: number;
  gridOpacity?: number;
  onError: () => void;
};

type LoadedTextures = {
  rgb: THREE.Texture;
  depth: THREE.DataTexture;
};

async function loadDepthTexture(signal: AbortSignal) {
  const response = await fetch(depthFieldConfig.browserAssetUrl, { signal });
  if (!response.ok) throw new Error(`Unable to load raw depth texture: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const expectedByteLength = depthFieldConfig.dimensions[0] * depthFieldConfig.dimensions[1] * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedByteLength) throw new Error(`Unexpected raw depth payload size: ${bytes.byteLength}`);

  const values = new Float32Array(bytes);
  if (!values.every(Number.isFinite)) throw new Error("Raw depth texture contains non-finite values");

  const texture = new THREE.DataTexture(
    values,
    depthFieldConfig.dimensions[0],
    depthFieldConfig.dimensions[1],
    THREE.RedFormat,
    THREE.FloatType,
  );
  texture.colorSpace = THREE.NoColorSpace;
  texture.flipY = true;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

export function DepthFieldObject({ interactive, progress, rgbTexture, opacity = 1, flatOpacity = 0, gridOpacity = 0, onError }: DepthFieldObjectProps) {
  const group = useRef<THREE.Group>(null);
  const ownedTextures = useRef<LoadedTextures | null>(null);
  const [textures, setTextures] = useState<LoadedTextures | null>(null);
  const invalidate = useThree((state) => state.invalidate);
  const displayProgress = THREE.MathUtils.clamp(progress, 0, 1);
  const displayOpacity = THREE.MathUtils.clamp(opacity, 0, 1);
  const displayFlatOpacity = THREE.MathUtils.clamp(flatOpacity, 0, 1);
  const displayGridOpacity = THREE.MathUtils.clamp(gridOpacity, 0, 1);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    if (!rgbTexture) return () => controller.abort();
    loadDepthTexture(controller.signal)
      .then((depth) => {
        if (!active) {
          depth.dispose();
          return;
        }
        const loaded = { rgb: rgbTexture, depth };
        ownedTextures.current = loaded;
        setTextures(loaded);
        invalidate();
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) onError();
      });

    return () => {
      active = false;
      controller.abort();
      ownedTextures.current?.depth.dispose();
      ownedTextures.current = null;
    };
  }, [invalidate, onError, rgbTexture]);

  const material = useMemo(() => {
    if (!textures) return null;
    const [cropLeft, cropRight] = depthFieldConfig.sourceCropX;
    return new THREE.ShaderMaterial({
      uniforms: {
        uRgbTexture: { value: textures.rgb },
        uDepthTexture: { value: textures.depth },
        uDepthMin: { value: depthFieldConfig.observedRange[0] },
        uDepthMax: { value: depthFieldConfig.observedRange[1] },
        uDepthScale: { value: depthFieldConfig.displacementScale },
        uDirection: { value: depthFieldConfig.renderDirection },
        uProgress: { value: 0 },
        uOpacity: { value: 1 },
        uGridOpacity: { value: 0 },
        uCropLeft: { value: cropLeft },
        uCropRight: { value: cropRight },
      },
      vertexShader: `
        varying vec2 vUv;
        uniform sampler2D uDepthTexture;
        uniform float uDepthMin;
        uniform float uDepthMax;
        uniform float uDepthScale;
        uniform float uDirection;
        uniform float uProgress;
        uniform float uOpacity;
        uniform float uCropLeft;
        uniform float uCropRight;

        void main() {
          vUv = uv;
          float cropWidth = uCropRight - uCropLeft;
          vec2 depthUv = vec2((uv.x - uCropLeft) / cropWidth, uv.y);
          float insideCrop = smoothstep(uCropLeft, uCropLeft + 0.02, uv.x)
            * (1.0 - smoothstep(uCropRight - 0.02, uCropRight, uv.x));
          float rawDepth = texture2D(uDepthTexture, clamp(depthUv, 0.0, 1.0)).r;
          float normalizedDepth = clamp((rawDepth - uDepthMin) / (uDepthMax - uDepthMin), 0.0, 1.0);
          vec3 transformed = position;
          transformed.z += (normalizedDepth - 0.5) * uDepthScale * uDirection * uProgress * insideCrop;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D uRgbTexture;
        uniform float uOpacity;
        uniform float uGridOpacity;

        void main() {
          vec3 color = texture2D(uRgbTexture, vUv).rgb;
          vec2 gridUv = vUv * vec2(72.0, 54.0);
          vec2 cellEdge = min(fract(gridUv), 1.0 - fract(gridUv));
          float gridLine = 1.0 - step(0.018, min(cellEdge.x, cellEdge.y));
          color = mix(color, vec3(0.72, 0.43, 0.31), gridLine * uGridOpacity);
          gl_FragColor = vec4(color, uOpacity);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }
      `,
      toneMapped: false,
      transparent: true,
      depthWrite: false,
    });
  }, [textures]);

  useEffect(() => () => material?.dispose(), [material]);

  useEffect(() => {
    if (!material) return;
    material.uniforms.uProgress.value = displayProgress;
    material.uniforms.uOpacity.value = displayOpacity;
    material.uniforms.uGridOpacity.value = displayGridOpacity;
    invalidate();
  }, [displayGridOpacity, displayOpacity, displayProgress, invalidate, material]);

  const targetRotation = useRef({ x: 0, y: 0 });

  useFrame((_, delta) => {
    if (!group.current) return;
    const factor = Math.min(1, delta * 9);
    const curX = group.current.rotation.x;
    const curY = group.current.rotation.y;
    const diffX = targetRotation.current.x - curX;
    const diffY = targetRotation.current.y - curY;

    if (Math.abs(diffX) > 0.0002 || Math.abs(diffY) > 0.0002) {
      group.current.rotation.x += diffX * factor;
      group.current.rotation.y += diffY * factor;
      if (Math.abs(targetRotation.current.x - group.current.rotation.x) < 0.0002) {
        group.current.rotation.x = targetRotation.current.x;
      }
      if (Math.abs(targetRotation.current.y - group.current.rotation.y) < 0.0002) {
        group.current.rotation.y = targetRotation.current.y;
      }
      invalidate();
    }
  });

  const tilt = (event: ThreeEvent<PointerEvent>) => {
    if (!interactive) return;
    targetRotation.current.y = THREE.MathUtils.clamp(event.pointer.x, -1, 1) * 0.04;
    targetRotation.current.x = -THREE.MathUtils.clamp(event.pointer.y, -1, 1) * 0.028;
    invalidate();
  };

  const reset = () => {
    targetRotation.current.x = 0;
    targetRotation.current.y = 0;
    invalidate();
  };

  if (!material || !textures) return null;

  return (
    <group ref={group} onPointerMove={tilt} onPointerLeave={reset}>
      <mesh position={[0, 0, -(depthFieldConfig.displacementScale / 2 + 0.025)]} visible={displayOpacity > 0}>
        <planeGeometry args={[photographConfig.width + photographConfig.mountBorder, photographConfig.height + photographConfig.mountBorder]} />
        <meshBasicMaterial color="#151412" transparent opacity={displayOpacity} depthWrite={false} />
      </mesh>
      <mesh>
        <planeGeometry args={[photographConfig.width, photographConfig.height, depthFieldConfig.segments[0], depthFieldConfig.segments[1]]} />
        <primitive object={material} attach="material" />
      </mesh>
      {displayFlatOpacity > 0 && <mesh position={[0, 0, 0.012]}>
        <planeGeometry args={[photographConfig.width, photographConfig.height]} />
        <meshBasicMaterial map={textures.rgb} transparent opacity={displayFlatOpacity} depthWrite={false} toneMapped={false} />
      </mesh>}
    </group>
  );
}
