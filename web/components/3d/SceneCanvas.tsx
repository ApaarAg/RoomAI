"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import type { Texture } from "three";
import { CameraController } from "./CameraController";
import { DepthFieldObject } from "./DepthFieldObject";
import { useNarrative, pointCloudAvailable, spatialStructureAvailable, worldAvailable } from "./NarrativeController";
import { PointCloudObject } from "./PointCloudObject";
import { pointCloudConfig } from "./pointCloudConfig";
import { PhotographObject } from "./PhotographObject";
import SpatialStructureObject, { SPATIAL_STRUCTURE_SOURCE_VERIFIED } from "./SpatialStructureObject";
import { WorldObject } from "./WorldObject";
import { SpatialAnnotationObject } from "./SpatialAnnotationObject";
import { DevelopmentPointField } from "./dev/DevelopmentPointField";
import { DevelopmentSpatialStructure } from "./dev/DevelopmentSpatialStructure";
import { DevelopmentWorld } from "./dev/DevelopmentWorld";
import { USE_DEVELOPMENT_FIXTURES } from "./dev/fixtureConfig";
import { photographConfig } from "./photographConfig";
import { RepresentationController } from "./RepresentationController";

const smoothstep = (value: number, start: number, end: number) => {
  const t = Math.min(1, Math.max(0, (value - start) / (end - start)));
  return t * t * (3 - 2 * t);
};

export function SceneCanvas() {
  const narrative = useNarrative();
  const [webglAvailable, setWebglAvailable] = useState(false);
  const [photoReady, setPhotoReady] = useState(false);
  const [rgbTexture, setRgbTexture] = useState<Texture | null>(null);
  const [interactive, setInteractive] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    if (context) {
      const loseContext = context.getExtension("WEBGL_lose_context");
      loseContext?.loseContext();
      setWebglAvailable(true);
    }
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
    const update = () => {
      setReducedMotion(reducedMotion.matches);
      setInteractive(!reducedMotion.matches && finePointer.matches);
    };
    update();
    reducedMotion.addEventListener("change", update);
    finePointer.addEventListener("change", update);
    return () => {
      reducedMotion.removeEventListener("change", update);
      finePointer.removeEventListener("change", update);
    };
  }, []);

  const ready = useCallback(() => setPhotoReady(true), []);
  const failed = useCallback(() => setPhotoReady(false), []);
  const setSharedRgb = useCallback((texture: Texture) => setRgbTexture(texture), []);

  const visualProgress = reducedMotion ? 0 : narrative.masterProgress;

  return (
    <div className="photograph-stage">
      <img className={`photograph-fallback${photoReady && webglAvailable ? " is-hidden" : ""}`} src={photographConfig.imageUrl} alt={photographConfig.imageAlt} />
      {webglAvailable && (
        <Canvas
          className={`photograph-canvas${photoReady ? " is-ready" : ""}`}
          aria-hidden="true"
          dpr={[1, 1.5]}
          frameloop="demand"
          camera={{ fov: 34, position: [0, 0, 5] }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <RepresentationController representation={narrative.representation} progress={narrative.representationProgress}>
            <CameraController progress={USE_DEVELOPMENT_FIXTURES ? visualProgress : Math.min(visualProgress, 0.58)} />
            <Suspense fallback={null}>
              <ambientLight intensity={0.75} />
              <directionalLight position={[-2, 4, 3]} intensity={1.1} color="#eee7dc" />
              <PhotographObject
                interactive={interactive}
                opacity={1 - smoothstep(visualProgress, 0.14, 0.36)}
                visible
                onReady={ready}
                onError={failed}
                onTextureLoaded={setSharedRgb}
              />
              <DepthFieldObject
                interactive={interactive}
                rgbTexture={rgbTexture}
                progress={reducedMotion ? 0 : narrative.depthProgress}
                opacity={smoothstep(visualProgress, 0.16, 0.34) * (USE_DEVELOPMENT_FIXTURES ? 1 - smoothstep(visualProgress, 0.38, 0.52) : 1)}
                flatOpacity={0}
                onError={failed}
              />
              {USE_DEVELOPMENT_FIXTURES ? <>
                <DevelopmentPointField
                  visible
                  progress={reducedMotion ? 0 : narrative.pointsProgress}
                  rgbTexture={rgbTexture}
                  opacity={smoothstep(visualProgress, 0.34, 0.48) * (1 - smoothstep(visualProgress, 0.58, 0.7))}
                />
                <DevelopmentSpatialStructure
                  visible
                  progress={reducedMotion ? 0 : narrative.structureProgress}
                  opacity={smoothstep(visualProgress, 0.55, 0.68) * (1 - smoothstep(visualProgress, 0.76, 0.86))}
                />
                <DevelopmentWorld
                  visible
                  progress={reducedMotion ? 0 : narrative.worldProgress}
                  opacity={smoothstep(visualProgress, 0.74, 0.94)}
                />
              </> : <>
                {pointCloudAvailable && pointCloudConfig.availableForExhibit && <PointCloudObject progress={narrative.masterProgress} visible opacity={1} onReady={ready} onError={failed} />}
                <SpatialStructureObject visible={spatialStructureAvailable && SPATIAL_STRUCTURE_SOURCE_VERIFIED} progress={narrative.masterProgress} opacity={1} />
                {worldAvailable && <WorldObject url={null} visible={false} progress={0} opacity={0} onError={() => undefined} />}
              </>}
              <SpatialAnnotationObject
                position={[-1.42, 1.03, 0.12]}
                progress={USE_DEVELOPMENT_FIXTURES ? 1 : narrative.depthProgress}
                visible={!USE_DEVELOPMENT_FIXTURES || visualProgress >= 0.34}
                size={0.014}
              />
            </Suspense>
          </RepresentationController>
        </Canvas>
      )}
    </div>
  );
}

