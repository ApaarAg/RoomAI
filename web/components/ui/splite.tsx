"use client";

import React, { Suspense, lazy, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

const Spline = lazy(() => import("@splinetool/react-spline"));

interface SplineSceneProps {
  scene: string;
  className?: string;
}

interface RobotNodes {
  rightArm: THREE.Object3D;
  rightElbow: THREE.Object3D;
  rightForearm: THREE.Object3D;
  rightHand: THREE.Object3D;
  leftArm: THREE.Object3D | null;
  leftElbow: THREE.Object3D | null;
  head: THREE.Object3D | null;
  camera: THREE.Camera;
  scene: THREE.Scene;
}

interface RestRotations {
  arm: { x: number; y: number; z: number };
  elbow: { x: number; y: number; z: number };
  forearm: { x: number; y: number; z: number };
  head: { x: number; y: number; z: number };
  leftArm: { x: number; y: number; z: number };
  leftElbow: { x: number; y: number; z: number };
}

export function SplineScene({ scene, className }: SplineSceneProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const badgeTextRef = useRef<HTMLSpanElement>(null);
  const reticleRef = useRef<HTMLDivElement>(null);

  // Mutable refs for 60 FPS animation loop without React re-render thrashing
  const splineAppRef = useRef<any>(null);
  const nodesRef = useRef<RobotNodes | null>(null);
  const restRef = useRef<RestRotations | null>(null);
  const isDraggingRef = useRef<boolean>(false);
  const isHoveredRef = useRef<boolean>(false);
  const planeRef = useRef<THREE.Plane>(new THREE.Plane());
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());

  const mouseNdcRef = useRef<THREE.Vector2>(new THREE.Vector2(-999, -999));
  const mouseCanvasPxRef = useRef<{ x: number; y: number }>({ x: -999, y: -999 });
  const lastInteractionTimeRef = useRef<number>(Date.now());

  // Current interpolated rotations and targets
  const currentAnglesRef = useRef({
    armZ: 0,
    armX: 0,
    elbowZ: 0,
    leftArmZ: 0,
    leftElbowZ: 0,
    headY: 0,
    headX: 0,
  });

  const targetAnglesRef = useRef({
    armZ: 0,
    armX: 0,
    elbowZ: 0,
    leftArmZ: 0,
    leftElbowZ: 0,
    headY: 0,
    headX: 0,
  });

  // Analytical 2-bone IK solver with mechanical constraints
  const solveIK = useCallback((targetX: number, targetY: number, targetZ: number) => {
    const rest = restRef.current;
    if (!rest) return;

    // Bone lengths discovered from actual robot hierarchy:
    // arm (pivot) -> elbow: L1 = 88.10
    // elbow -> hand: L2 = 111.10
    const L1 = 88.1;
    const L2 = 111.1;
    const maxReach = L1 + L2 - 5.0; // 194.20
    const minReach = 65.0;

    const dist = Math.sqrt(targetX * targetX + targetY * targetY + targetZ * targetZ);
    if (dist < 0.001) return;

    const d = Math.max(minReach, Math.min(maxReach, dist));

    // Law of Cosines for elbow flexion: cos(beta) = (L1^2 + L2^2 - d^2) / (2 * L1 * L2)
    const cosBeta = Math.max(-1, Math.min(1, (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2)));
    const beta = Math.acos(cosBeta);
    const rawElbowAngle = Math.PI - beta;

    // Constrain elbow flexion between 0 (straight) and 135 deg (2.35 rad)
    const elbowAngle = Math.max(0, Math.min(2.35, rawElbowAngle));

    // Law of Cosines for shoulder angle relative to chord: cos(alpha) = (L1^2 + d^2 - L2^2) / (2 * L1 * d)
    const cosAlpha = Math.max(-1, Math.min(1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)));
    const alpha = Math.acos(cosAlpha);

    // Target spherical angles relative to shoulder pivot (downward -Y axis)
    const azimuth = Math.atan2(targetX, -targetY);
    const elevation = Math.asin(Math.max(-1, Math.min(1, targetZ / d)));

    // Constrain shoulder rotation ranges
    const clampedArmZ = Math.max(-0.87, Math.min(1.57, azimuth * 0.8));
    const clampedArmX = Math.max(-1.3, Math.min(1.4, elevation + alpha * 0.5));

    targetAnglesRef.current.armZ = rest.arm.z + clampedArmZ;
    targetAnglesRef.current.armX = rest.arm.x - clampedArmX;
    targetAnglesRef.current.elbowZ = rest.elbow.z + elbowAngle;

    // Subtle counter-motion on opposite (left) arm for physical balance
    targetAnglesRef.current.leftArmZ = -azimuth * 0.15;
    targetAnglesRef.current.leftElbowZ = elbowAngle * 0.18;
  }, []);

  // Frame update loop
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();

    const loop = (time: number) => {
      animId = requestAnimationFrame(loop);

      const dt = Math.min(0.1, (time - lastTime) / 1000);
      lastTime = time;

      const nodes = nodesRef.current;
      const rest = restRef.current;
      const app = splineAppRef.current;
      if (!nodes || !rest || !app) return;

      const { rightArm, rightElbow, rightHand, leftArm, leftElbow, head, camera, scene } = nodes;
      const isDragging = isDraggingRef.current;
      const isHovered = isHoveredRef.current;
      const wrapper = wrapperRef.current;

      // 1. Calculate Hand 3D World Position and Project to Canvas Pixels
      const handWorld = new THREE.Vector3();
      rightHand.getWorldPosition(handWorld);

      let handScreenPx = { x: -999, y: -999 };
      if (wrapper) {
        const rect = wrapper.getBoundingClientRect();
        const projected = handWorld.clone().project(camera);
        handScreenPx.x = (projected.x * 0.5 + 0.5) * rect.width;
        handScreenPx.y = (-(projected.y * 0.5) + 0.5) * rect.height;

        // Position reticle over hand
        if (reticleRef.current) {
          reticleRef.current.style.transform = `translate3d(${handScreenPx.x}px, ${handScreenPx.y}px, 0) translate(-50%, -50%)`;
        }

        // Test hover distance if not currently dragging
        if (!isDragging) {
          const mX = mouseCanvasPxRef.current.x;
          const mY = mouseCanvasPxRef.current.y;
          const dx = mX - handScreenPx.x;
          const dy = mY - handScreenPx.y;
          const distPx = Math.sqrt(dx * dx + dy * dy);

          const hoveredNow = distPx < 48;
          if (hoveredNow !== isHoveredRef.current) {
            isHoveredRef.current = hoveredNow;
            if (hoveredNow) {
              lastInteractionTimeRef.current = Date.now();
              if (wrapper) wrapper.style.cursor = "grab";
              if (badgeRef.current) {
                badgeRef.current.classList.add("visible");
                badgeRef.current.classList.remove("active");
              }
              if (badgeTextRef.current) {
                badgeTextRef.current.textContent = "◈ DRAG HAND TO MANIPULATE";
              }
              if (reticleRef.current) {
                reticleRef.current.classList.add("visible");
                reticleRef.current.classList.remove("active");
              }
            } else {
              if (wrapper) wrapper.style.cursor = "default";
              if (badgeRef.current) badgeRef.current.classList.remove("visible", "active");
              if (reticleRef.current) reticleRef.current.classList.remove("visible", "active");
            }
          }
        }
      }

      // 2. Process Interaction Target
      if (isDragging) {
        lastInteractionTimeRef.current = Date.now();
        // Raycast from cursor through camera onto locked interaction plane
        raycasterRef.current.setFromCamera(mouseNdcRef.current, camera);
        const targetWorld = new THREE.Vector3();
        if (raycasterRef.current.ray.intersectPlane(planeRef.current, targetWorld)) {
          // Convert world target into shoulder local coordinate space
          const targetLocal = targetWorld.clone();
          rightArm.parent!.worldToLocal(targetLocal);
          targetLocal.sub(rightArm.position);

          solveIK(targetLocal.x, targetLocal.y, targetLocal.z);
        }
      } else if (!isHovered) {
        // Return smoothly toward baseline rest pose
        targetAnglesRef.current.armZ = rest.arm.z;
        targetAnglesRef.current.armX = rest.arm.x;
        targetAnglesRef.current.elbowZ = rest.elbow.z;
        targetAnglesRef.current.leftArmZ = 0;
        targetAnglesRef.current.leftElbowZ = 0;

        // Micro harmonic breathing when idle (> 1.2s since last interaction)
        const idleDuration = (Date.now() - lastInteractionTimeRef.current) / 1000;
        if (idleDuration > 1.2) {
          const breath = Math.sin(time * 0.0018) * 0.012;
          const breathPitch = Math.cos(time * 0.0018) * 0.008;
          targetAnglesRef.current.armZ += breath;
          targetAnglesRef.current.armX += breathPitch;
          targetAnglesRef.current.leftArmZ -= breath * 0.5;
        }
      } else {
        // Hover anticipation (subtle upward lift)
        targetAnglesRef.current.armX = rest.arm.x - 0.05;
        targetAnglesRef.current.elbowZ = rest.elbow.z + 0.08;
      }

      // 3. Head Tracking (Subtle orientation toward cursor)
      if (head) {
        if (mouseNdcRef.current.x > -10) {
          const targetHeadY = rest.head.y + Math.max(-0.32, Math.min(0.32, mouseNdcRef.current.x * 0.28));
          const targetHeadX = rest.head.x - Math.max(-0.22, Math.min(0.22, mouseNdcRef.current.y * 0.18));
          targetAnglesRef.current.headY = targetHeadY;
          targetAnglesRef.current.headX = targetHeadX;
        } else {
          targetAnglesRef.current.headY = rest.head.y;
          targetAnglesRef.current.headX = rest.head.x;
        }
      }

      // 4. Smooth Damped Interpolation (Critical Damping without Overshoot)
      const lerpSpeed = isDragging ? 18.0 : 9.0;
      const factor = 1 - Math.exp(-lerpSpeed * dt);

      currentAnglesRef.current.armZ += (targetAnglesRef.current.armZ - currentAnglesRef.current.armZ) * factor;
      currentAnglesRef.current.armX += (targetAnglesRef.current.armX - currentAnglesRef.current.armX) * factor;
      currentAnglesRef.current.elbowZ += (targetAnglesRef.current.elbowZ - currentAnglesRef.current.elbowZ) * factor;
      currentAnglesRef.current.leftArmZ += (targetAnglesRef.current.leftArmZ - currentAnglesRef.current.leftArmZ) * factor;
      currentAnglesRef.current.leftElbowZ += (targetAnglesRef.current.leftElbowZ - currentAnglesRef.current.leftElbowZ) * factor;
      currentAnglesRef.current.headY += (targetAnglesRef.current.headY - currentAnglesRef.current.headY) * factor;
      currentAnglesRef.current.headX += (targetAnglesRef.current.headX - currentAnglesRef.current.headX) * factor;

      // 5. Apply Rotations to Three.js Robot Nodes
      rightArm.rotation.z = currentAnglesRef.current.armZ;
      rightArm.rotation.x = currentAnglesRef.current.armX;
      rightArm.updateMatrix();

      rightElbow.rotation.z = currentAnglesRef.current.elbowZ;
      rightElbow.updateMatrix();

      if (leftArm && leftElbow) {
        leftArm.rotation.z = rest.leftArm.z + currentAnglesRef.current.leftArmZ;
        leftArm.updateMatrix();
        leftElbow.rotation.z = rest.leftElbow.z + currentAnglesRef.current.leftElbowZ;
        leftElbow.updateMatrix();
      }

      if (head) {
        head.rotation.y = currentAnglesRef.current.headY;
        head.rotation.x = currentAnglesRef.current.headX;
        head.updateMatrix();
      }

      // 6. Request Scene Render
      scene.updateMatrixWorld(true);
      if (app.requestRender) {
        app.requestRender();
      }
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [solveIK]);

  // Pointer event handlers on stage container
  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const app = splineAppRef.current;
    const nodes = nodesRef.current;
    if (!nodes || !app) return;

    if (isHoveredRef.current) {
      isDraggingRef.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);

      // Disable default OrbitControls so hand dragging takes precedence
      if (app._controls) {
        app._controls.enabled = false;
      }

      // Lock interaction plane through current hand world position facing the camera
      const handWorld = new THREE.Vector3();
      nodes.rightHand.getWorldPosition(handWorld);

      const normal = nodes.camera.position.clone().sub(handWorld).normalize();
      planeRef.current.setFromNormalAndCoplanarPoint(normal, handWorld);

      // Update UI affordance to active manipulating state
      if (wrapperRef.current) wrapperRef.current.style.cursor = "grabbing";
      if (badgeRef.current) {
        badgeRef.current.classList.add("visible", "active");
      }
      if (badgeTextRef.current) {
        badgeTextRef.current.textContent = "● MANIPULATING RIGHT ARM // IK SOLVER ACTIVE";
      }
      if (reticleRef.current) {
        reticleRef.current.classList.add("visible", "active");
      }
    } else {
      // Empty canvas drag: OrbitControls remain active for scene rotation
      if (app._controls) {
        app._controls.enabled = true;
      }
    }
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pxX = e.clientX - rect.left;
    const pxY = e.clientY - rect.top;

    mouseCanvasPxRef.current.x = pxX;
    mouseCanvasPxRef.current.y = pxY;

    // Three.js NDC (-1 to +1)
    mouseNdcRef.current.x = (pxX / rect.width) * 2 - 1;
    mouseNdcRef.current.y = -(pxY / rect.height) * 2 + 1;
  }, []);

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (isDraggingRef.current) {
      isDraggingRef.current = false;
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Safe fallback
      }

      const app = splineAppRef.current;
      // Re-enable camera orbit controls
      if (app?._controls) {
        app._controls.enabled = true;
      }

      // Revert UI affordances
      if (wrapperRef.current) {
        wrapperRef.current.style.cursor = isHoveredRef.current ? "grab" : "default";
      }
      if (badgeRef.current) {
        if (isHoveredRef.current) {
          badgeRef.current.classList.remove("active");
          if (badgeTextRef.current) {
            badgeTextRef.current.textContent = "◈ DRAG HAND TO MANIPULATE";
          }
        } else {
          badgeRef.current.classList.remove("visible", "active");
        }
      }
      if (reticleRef.current) {
        if (isHoveredRef.current) {
          reticleRef.current.classList.remove("active");
        } else {
          reticleRef.current.classList.remove("visible", "active");
        }
      }
    }
  }, []);

  const handlePointerLeave = useCallback(() => {
    mouseNdcRef.current.set(-999, -999);
    mouseCanvasPxRef.current.x = -999;
    mouseCanvasPxRef.current.y = -999;

    if (!isDraggingRef.current) {
      isHoveredRef.current = false;
      if (wrapperRef.current) wrapperRef.current.style.cursor = "default";
      if (badgeRef.current) badgeRef.current.classList.remove("visible", "active");
      if (reticleRef.current) reticleRef.current.classList.remove("visible", "active");
    }
  }, []);

  const handleSplineLoad = useCallback((spline: any) => {
    splineAppRef.current = spline;
    if (typeof window !== "undefined") {
      (window as any).splineApp = spline;
    }

    const scene: THREE.Scene = spline._scene;
    const camera: THREE.Camera = spline._camera;

    let rightArm: THREE.Object3D | null = null;
    let rightElbow: THREE.Object3D | null = null;
    let rightForearm: THREE.Object3D | null = null;
    let rightHand: THREE.Object3D | null = null;
    let leftArm: THREE.Object3D | null = null;
    let leftElbow: THREE.Object3D | null = null;
    let head: THREE.Object3D | null = null;

    scene.traverse((node: any) => {
      if (node.name === "Head" && node.type === "Group") {
        head = node;
      }
      if (node.name === "Hand" && node.type === "Group" && node.parent?.name === "Top part") {
        node.traverse((child: any) => {
          if (child.name === "arm") rightArm = child;
          if (child.name === "elbow") rightElbow = child;
          if (child.name === "forearm") rightForearm = child;
          if (child.name === "Hand" && child.type === "Mesh") rightHand = child;
        });
      }
      if (node.name === "Hand Instance") {
        node.traverse((child: any) => {
          if (child.name === "arm") leftArm = child;
          if (child.name === "elbow") leftElbow = child;
        });
      }
    });

    const rArm = rightArm as THREE.Object3D | null;
    const rElbow = rightElbow as THREE.Object3D | null;
    const rHand = rightHand as THREE.Object3D | null;
    const rForearm = rightForearm as THREE.Object3D | null;
    const lArm = leftArm as THREE.Object3D | null;
    const lElbow = leftElbow as THREE.Object3D | null;
    const h = head as THREE.Object3D | null;

    if (rArm && rElbow && rHand) {
      rArm.matrixAutoUpdate = true;
      rElbow.matrixAutoUpdate = true;
      if (rForearm) rForearm.matrixAutoUpdate = true;
      if (lArm) lArm.matrixAutoUpdate = true;
      if (lElbow) lElbow.matrixAutoUpdate = true;
      if (h) h.matrixAutoUpdate = true;

      nodesRef.current = {
        rightArm: rArm,
        rightElbow: rElbow,
        rightForearm: rForearm || rElbow,
        rightHand: rHand,
        leftArm: lArm,
        leftElbow: lElbow,
        head: h,
        camera,
        scene,
      };

      const rest: RestRotations = {
        arm: { x: rArm.rotation.x, y: rArm.rotation.y, z: rArm.rotation.z },
        elbow: { x: rElbow.rotation.x, y: rElbow.rotation.y, z: rElbow.rotation.z },
        forearm: rForearm
          ? { x: rForearm.rotation.x, y: rForearm.rotation.y, z: rForearm.rotation.z }
          : { x: 0, y: 0, z: 0 },
        head: h
          ? { x: h.rotation.x, y: h.rotation.y, z: h.rotation.z }
          : { x: 0, y: 0, z: 0 },
        leftArm: lArm
          ? { x: lArm.rotation.x, y: lArm.rotation.y, z: lArm.rotation.z }
          : { x: 0, y: 0, z: 0 },
        leftElbow: lElbow
          ? { x: lElbow.rotation.x, y: lElbow.rotation.y, z: lElbow.rotation.z }
          : { x: 0, y: 0, z: 0 },
      };

      restRef.current = rest;
      currentAnglesRef.current = {
        armZ: rest.arm.z,
        armX: rest.arm.x,
        elbowZ: rest.elbow.z,
        leftArmZ: rest.leftArm.z,
        leftElbowZ: rest.leftElbow.z,
        headY: rest.head.y,
        headX: rest.head.x,
      };
      targetAnglesRef.current = { ...currentAnglesRef.current };
    }
  }, []);

  return (
    <div
      ref={wrapperRef}
      className={`relative w-full h-full select-none ${className || ""}`}
      style={{ touchAction: "none" }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onPointerLeave={handlePointerLeave}
    >
      <Suspense
        fallback={
          <div className="w-full h-full flex items-center justify-center min-h-[300px]" style={{ minHeight: "300px" }}>
            <div className="spline-loading-indicator">
              <span className="loader" />
              <span className="spline-loading-label">INITIALIZING 3D ENVIRONMENT...</span>
            </div>
          </div>
        }
      >
        <Spline scene={scene} className="w-full h-full" onLoad={handleSplineLoad} />
      </Suspense>

      {/* Interactive HUD Overlay for Articulated Character Manipulation */}
      <div className="robot-manipulation-overlay" aria-hidden="true">
        {/* Subtle monospaced affordance badge */}
        <div ref={badgeRef} className="robot-affordance-badge">
          <span className="robot-affordance-dot" />
          <span ref={badgeTextRef}>◈ DRAG HAND TO MANIPULATE</span>
        </div>

        {/* Ambient reticle centered over robot hand */}
        <div ref={reticleRef} className="robot-hand-reticle">
          <div className="robot-hand-reticle-ring" />
          <div className="robot-hand-reticle-crosshair" />
        </div>
      </div>
    </div>
  );
}