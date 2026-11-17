"use client";

import { useEffect } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { photographConfig } from "./photographConfig";

type CameraControllerProps = {
  progress?: number;
};

export function CameraController({ progress: narrativeProgress = 0 }: CameraControllerProps) {
  const { camera, size, invalidate } = useThree();

  useEffect(() => {
    if (!(camera instanceof THREE.PerspectiveCamera)) return;
    const aspect = size.width / Math.max(size.height, 1);
    const fitHeight = Math.max(photographConfig.height, photographConfig.width / aspect) * 1.18;
    const distance = fitHeight / (2 * Math.tan((camera.fov * Math.PI) / 360));
    const progress = THREE.MathUtils.clamp(narrativeProgress, 0, 1);
    const crossing = THREE.MathUtils.smoothstep(progress, 0.8, 1);
    const inspection = THREE.MathUtils.smoothstep(progress, 0.18, 0.58);
    const roomReveal = THREE.MathUtils.smoothstep(progress, 0.42, 0.78);
    camera.position.set(
      THREE.MathUtils.lerp(0.42, 0, crossing),
      THREE.MathUtils.lerp(0, 0.18, crossing),
      THREE.MathUtils.lerp(distance, -0.16, crossing) - inspection * 0.06,
    );
    const roomTargetZ = THREE.MathUtils.lerp(0, -1.35, roomReveal);
    camera.lookAt(
      0,
      THREE.MathUtils.lerp(0, -1, roomReveal) + THREE.MathUtils.lerp(0, 0.15, crossing),
      THREE.MathUtils.lerp(roomTargetZ, -2.15, crossing),
    );
    camera.aspect = aspect;
    camera.near = 0.1;
    camera.far = Math.max(distance * 10, 100);
    camera.updateProjectionMatrix();
    invalidate();
  }, [camera, narrativeProgress, size.width, size.height, invalidate]);

  return null;
}
