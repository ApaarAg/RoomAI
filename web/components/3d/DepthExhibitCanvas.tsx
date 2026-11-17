"use client";

import { Canvas } from "@react-three/fiber";
import { useCallback, useEffect, useState } from "react";
import type { Texture } from "three";
import { CameraController } from "./CameraController";
import { DepthFieldObject } from "./DepthFieldObject";
import { PhotographObject } from "./PhotographObject";
import { photographConfig } from "./photographConfig";

type DepthExhibitCanvasProps = {
  className?: string;
  progress: number;
  gridOpacity?: number;
};

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

export function DepthExhibitCanvas({
  className = "",
  progress,
  gridOpacity = 0,
}: DepthExhibitCanvasProps) {
  const [webglAvailable, setWebglAvailable] = useState(false);
  const [photoReady, setPhotoReady] = useState(false);
  const [rgbTexture, setRgbTexture] = useState<Texture | null>(null);
  const [displayProgress, setDisplayProgress] = useState(clamp01(progress));
  const [interactive, setInteractive] = useState(false);

  useEffect(() => {
    const probe = document.createElement("canvas");
    const context = probe.getContext("webgl2") ?? probe.getContext("webgl");
    if (context) {
      context.getExtension("WEBGL_lose_context")?.loseContext();
      setWebglAvailable(true);
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
    const updateMotionPreference = () => {
      setInteractive(!reducedMotion.matches && finePointer.matches);
    };
    updateMotionPreference();
    reducedMotion.addEventListener("change", updateMotionPreference);
    finePointer.addEventListener("change", updateMotionPreference);
    return () => {
      reducedMotion.removeEventListener("change", updateMotionPreference);
      finePointer.removeEventListener("change", updateMotionPreference);
    };
  }, []);

  useEffect(() => {
    setDisplayProgress(clamp01(progress));
  }, [progress]);

  const onPhotoReady = useCallback(() => setPhotoReady(true), []);
  const onLoadError = useCallback(() => setPhotoReady(false), []);
  const onTextureLoaded = useCallback((texture: Texture) => setRgbTexture(texture), []);

  return (
    <div className={`depth-exhibit ${className}`}>
      <img
        className={`depth-exhibit-fallback${photoReady && webglAvailable ? " is-hidden" : ""}`}
        src={photographConfig.imageUrl}
        alt={photographConfig.imageAlt}
      />
      {webglAvailable && (
        <Canvas
          className={`depth-exhibit-canvas${photoReady ? " is-ready" : ""}`}
          aria-hidden="true"
          dpr={[1, 1.25]}
          frameloop="demand"
          camera={{ fov: 34, position: [0, 0, 5] }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <CameraController progress={Math.min(displayProgress, 0.58)} />
          <group rotation={[-0.035 - displayProgress * 0.06, displayProgress * 0.08, 0]}>
            <PhotographObject
              interactive={interactive}
              opacity={1 - displayProgress}
              onReady={onPhotoReady}
              onError={onLoadError}
              onTextureLoaded={onTextureLoaded}
            />
            <DepthFieldObject
              interactive={interactive}
              rgbTexture={rgbTexture}
              progress={displayProgress}
              opacity={displayProgress}
              gridOpacity={gridOpacity}
              onError={onLoadError}
            />
          </group>
        </Canvas>
      )}
    </div>
  );
}
