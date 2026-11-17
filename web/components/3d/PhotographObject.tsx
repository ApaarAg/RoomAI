"use client";

import { useEffect, useRef, useState } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { photographConfig } from "./photographConfig";

type PhotographObjectProps = {
  interactive: boolean;
  opacity?: number;
  visible?: boolean;
  onReady: () => void;
  onError: () => void;
  onTextureLoaded?: (texture: THREE.Texture) => void;
};

export function PhotographObject({ interactive, opacity = 1, visible = true, onReady, onError, onTextureLoaded }: PhotographObjectProps) {
  const group = useRef<THREE.Group>(null);
  const ownedTexture = useRef<THREE.Texture | null>(null);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const invalidate = useThree((state) => state.invalidate);

  const onReadyRef = useRef(onReady);
  const onErrorRef = useRef(onError);
  const onTextureLoadedRef = useRef(onTextureLoaded);

  useEffect(() => {
    onReadyRef.current = onReady;
    onErrorRef.current = onError;
    onTextureLoadedRef.current = onTextureLoaded;
  });

  useEffect(() => {
    let active = true;
    const loader = new THREE.TextureLoader();
    loader.load(photographConfig.imageUrl, (loaded) => {
      if (!active) {
        loaded.dispose();
        return;
      }
      loaded.colorSpace = THREE.SRGBColorSpace;
      loaded.anisotropy = 4;
      ownedTexture.current = loaded;
      setTexture(loaded);
      onTextureLoadedRef.current?.(loaded);
      onReadyRef.current();
      invalidate();
    }, undefined, () => {
      if (active) onErrorRef.current();
    });
    return () => {
      active = false;
      ownedTexture.current?.dispose();
      ownedTexture.current = null;
    };
  }, [invalidate]);

  useEffect(() => {
    invalidate();
  }, [opacity, visible, invalidate]);

  const tilt = (event: ThreeEvent<PointerEvent>) => {
    if (!interactive || !group.current) return;
    const x = THREE.MathUtils.clamp(event.pointer.x, -1, 1);
    const y = THREE.MathUtils.clamp(event.pointer.y, -1, 1);
    group.current.rotation.y = x * 0.035;
    group.current.rotation.x = -y * 0.025;
    invalidate();
  };

  const reset = () => {
    if (!interactive || !group.current) return;
    group.current.rotation.set(0, 0, 0);
    invalidate();
  };

  if (!texture) return null;

  return (
    <group ref={group} position={[0, 0, 0]} rotation={[0, 0, 0]} scale={1} visible={visible && opacity > 0} onPointerMove={tilt} onPointerLeave={reset}>
      <mesh position={[0, 0, -0.025]}>
        <planeGeometry args={[photographConfig.width + photographConfig.mountBorder, photographConfig.height + photographConfig.mountBorder]} />
        <meshBasicMaterial color="#151412" />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[photographConfig.width, photographConfig.height]} />
        <meshBasicMaterial map={texture} transparent={opacity < 1} opacity={opacity} depthWrite={opacity >= 1} toneMapped={false} />
      </mesh>
    </group>
  );
}
