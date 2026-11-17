"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface SpatialObjectInfo {
  id: string;
  name: string;
  category: string;
  dimensions: { width: number; height: number; depth: number };
  position: { x: number; y: number; z: number };
  confidence: number;
  elevation: string;
  description: string;
}

const OBJECT_METADATA_MAP: Record<string, SpatialObjectInfo> = {
  "Node_chair#stool_001": {
    id: "chair-001",
    name: "TASK CHAIR // STOOL",
    category: "Furniture / Seating",
    dimensions: { width: 0.46, height: 0.62, depth: 0.63 },
    position: { x: 0.6, y: 0.35, z: 4.77 },
    confidence: 0.924,
    elevation: "Floor contact (+0.66m plane)",
    description: "Multi-legged indoor seating oriented toward primary workspace.",
  },
  "Node_table_002": {
    id: "table-002",
    name: "SIDE TABLE // PEDESTAL",
    category: "Furniture / Surface",
    dimensions: { width: 0.39, height: 0.36, depth: 0.44 },
    position: { x: 0.22, y: 0.48, z: 5.3 },
    confidence: 0.887,
    elevation: "Floor mounted (+0.66m plane)",
    description: "Compact secondary architectural horizontal surface.",
  },
  "Node_home-appliance_003": {
    id: "appliance-003",
    name: "CEILING FIXTURE // VENTILATION UNIT",
    category: "Appliance / Spatial Fixture",
    dimensions: { width: 2.57, height: 1.03, depth: 1.8 },
    position: { x: 0.35, y: -1.11, z: 6.3 },
    confidence: 0.841,
    elevation: "Overhead ceiling span (1.77m floor clearance)",
    description: "Wide architectural ceiling/wall ventilation appliance envelope.",
  },
  "Node_table_004": {
    id: "table-004",
    name: "PRIMARY DINING / WORK TABLE",
    category: "Furniture / Primary Surface",
    dimensions: { width: 1.54, height: 0.59, depth: 0.74 },
    position: { x: -0.05, y: 0.37, z: 3.17 },
    confidence: 0.956,
    elevation: "Forefront central surface (+0.66m floor contact)",
    description: "High-confidence dominant horizontal plane in immediate foreground.",
  },
  "Node_decoration_005": {
    id: "decoration-005",
    name: "WALL FIXTURE // PANEL DECORATION",
    category: "Decoration / Wall Mount",
    dimensions: { width: 0.15, height: 0.71, depth: 0.42 },
    position: { x: -0.9, y: 0.03, z: 5.08 },
    confidence: 0.785,
    elevation: "Left wall mount (0.63m floor clearance)",
    description: "Planar acoustic or decorative panel mounted to lateral wall boundary.",
  },
};

interface Room3DCanvasProps {
  stageProgress: number; // 0.0 to 1.0 across stages 4-6
  activeStageIndex: number; // 0 to 5
  isInteractiveOrbit: boolean;
  onObjectHover: (obj: SpatialObjectInfo | null) => void;
  onResetViewRef?: (fn: () => void) => void;
  onSetPresetRef?: (fn: (preset: "default" | "top" | "iso") => void) => void;
}

export function Room3DCanvas({
  stageProgress,
  activeStageIndex,
  isInteractiveOrbit,
  onObjectHover,
  onResetViewRef,
  onSetPresetRef,
}: Room3DCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const originalPointPositions = useRef<Float32Array | null>(null);
  const objectMeshesRef = useRef<THREE.Mesh[]>([]);
  const lineSegmentsRef = useRef<THREE.LineSegments[]>([]);
  const floorGridRef = useRef<THREE.GridHelper | null>(null);
  const roomGroupRef = useRef<THREE.Group | null>(null);
  const hoveredMeshRef = useRef<THREE.Mesh | null>(null);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2(-999, -999));
  const animationFrameId = useRef<number | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Camera preset handler
  const setCameraPreset = useCallback((preset: "default" | "top" | "iso") => {
    if (!cameraRef.current || !controlsRef.current) return;
    const controls = controlsRef.current;
    const camera = cameraRef.current;

    if (preset === "default") {
      controls.target.set(0.1, 0, -4.5);
      camera.position.set(0, 0.4, 0.2);
    } else if (preset === "top") {
      controls.target.set(0.1, 0, -4.5);
      camera.position.set(0.1, 5.5, -4.5);
    } else if (preset === "iso") {
      controls.target.set(0.1, 0, -4.5);
      camera.position.set(3.2, 2.8, -1.5);
    }
    controls.update();
  }, []);

  const resetView = useCallback(() => {
    setCameraPreset("default");
  }, [setCameraPreset]);

  // Expose reset and preset handlers to parent
  useEffect(() => {
    if (onResetViewRef) onResetViewRef(resetView);
    if (onSetPresetRef) onSetPresetRef(setCameraPreset);
  }, [onResetViewRef, onSetPresetRef, resetView, setCameraPreset]);

  // Orbit controls toggle
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.enabled = isInteractiveOrbit || activeStageIndex >= 5;
    }
  }, [isInteractiveOrbit, activeStageIndex]);

  // Initialize Three.js scene
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 560;

    // 1. Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    rendererRef.current = renderer;

    // 2. Scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0c0d10, 0.035);
    sceneRef.current = scene;

    // 3. Camera
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.05, 50);
    camera.position.set(0, 0.4, 0.2);
    cameraRef.current = camera;

    // 4. OrbitControls
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.target.set(0.1, 0, -4.5);
    controls.minDistance = 0.3;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI / 2 + 0.12; // Do not go under floor
    controls.enabled = false;
    controlsRef.current = controls;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffecd6, 1.2);
    dirLight.position.set(2, 4, 1);
    scene.add(dirLight);

    const accentLight = new THREE.PointLight(0xb47762, 2.0, 10);
    accentLight.position.set(0, 0.5, -4);
    scene.add(accentLight);

    // 6. Room Wrapper Group (Inverts Camera Y coordinate to align floor to ground)
    const roomGroup = new THREE.Group();
    roomGroup.rotation.x = Math.PI; // Rotates camera-space +Y down to -Y (ground)
    scene.add(roomGroup);
    roomGroupRef.current = roomGroup;

    // 7. Architectural Floor Grid
    const floorGrid = new THREE.GridHelper(8, 24, 0xb47762, 0x222730);
    floorGrid.position.set(0.2, -0.66, -4.5);
    floorGrid.material.transparent = true;
    floorGrid.material.opacity = 0.45;
    scene.add(floorGrid);
    floorGridRef.current = floorGrid;

    // 8. GLTF Loader
    const loader = new GLTFLoader();
    const modelUrl = "/assets/roomai/3d/roomai_composite_test.glb";

    loader.load(
      modelUrl,
      (gltf) => {
        const loadedGroup = gltf.scene;
        const objectMeshes: THREE.Mesh[] = [];
        const lineSegments: THREE.LineSegments[] = [];

        loadedGroup.traverse((child) => {
          if (child.name === "PointCloudNode" && (child as THREE.Points).isPoints) {
            const points = child as THREE.Points;
            const geom = points.geometry;
            const posAttr = geom.attributes.position;
            const count = posAttr.count;

            // Preserve original positions for unprojection interpolation
            const orig = new Float32Array(posAttr.array);
            originalPointPositions.current = orig;

            // Generate vertex colors based on depth (Z axis)
            const colors = new Float32Array(count * 3);
            const colorNear = new THREE.Color(0xf6f2ec); // Warm near white
            const colorMid = new THREE.Color(0xb47762); // Architectural amber
            const colorFar = new THREE.Color(0x3a424e); // Deep steel graphite
            const tempColor = new THREE.Color();

            for (let i = 0; i < count; i++) {
              const z = orig[i * 3 + 2];
              // Normalized depth between 2.5m and 7.2m
              const t = THREE.MathUtils.clamp((z - 2.5) / (7.2 - 2.5), 0, 1);
              if (t < 0.45) {
                tempColor.lerpColors(colorNear, colorMid, t / 0.45);
              } else {
                tempColor.lerpColors(colorMid, colorFar, (t - 0.45) / 0.55);
              }
              colors[i * 3] = tempColor.r;
              colors[i * 3 + 1] = tempColor.g;
              colors[i * 3 + 2] = tempColor.b;
            }

            geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

            const pointsMaterial = new THREE.PointsMaterial({
              size: 0.026,
              vertexColors: true,
              transparent: true,
              opacity: 0.9,
              sizeAttenuation: true,
            });

            points.material = pointsMaterial;
            pointsRef.current = points;
          } else if (child.name.startsWith("Node_") && (child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            const meta = OBJECT_METADATA_MAP[mesh.name];

            // Setup transparent architectural volume
            mesh.material = new THREE.MeshBasicMaterial({
              color: 0x181a20,
              transparent: true,
              opacity: 0.15,
              depthWrite: false,
            });

            mesh.userData = {
              nodeName: mesh.name,
              meta: meta || {
                id: mesh.name,
                name: mesh.name.replace("Node_", "").toUpperCase(),
                category: "Detected Object",
                dimensions: { width: 0.5, height: 0.5, depth: 0.5 },
                position: { x: 0, y: 0, z: 0 },
                confidence: 0.85,
                elevation: "Floor contact",
                description: "Segmented spatial geometric volume",
              },
            };

            // Architectural Wireframe with Corner Bracket Accents
            const edges = new THREE.EdgesGeometry(mesh.geometry);
            const lineMat = new THREE.LineBasicMaterial({
              color: 0x687282,
              transparent: true,
              opacity: 0.6,
              linewidth: 1,
            });
            const wireframe = new THREE.LineSegments(edges, lineMat);
            mesh.add(wireframe);

            objectMeshes.push(mesh);
            lineSegments.push(wireframe);
          }
        });

        objectMeshesRef.current = objectMeshes;
        lineSegmentsRef.current = lineSegments;

        roomGroup.add(loadedGroup);
        setIsLoading(false);
      },
      undefined,
      (error) => {
        console.error("Failed to load roomai_composite_test.glb:", error);
        setLoadError("3D Reconstruction GLB asset could not be loaded.");
        setIsLoading(false);
      }
    );

    // 9. Raycasting on pointer move
    const handlePointerMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    };

    const handlePointerLeave = () => {
      mouseRef.current.x = -999;
      mouseRef.current.y = -999;
      if (hoveredMeshRef.current) {
        // Reset hovered mesh
        const wire = hoveredMeshRef.current.children[0] as THREE.LineSegments;
        if (wire && wire.material) {
          (wire.material as THREE.LineBasicMaterial).color.setHex(0x687282);
          (wire.material as THREE.LineBasicMaterial).opacity = 0.6;
        }
        (hoveredMeshRef.current.material as THREE.MeshBasicMaterial).opacity = 0.15;
        (hoveredMeshRef.current.material as THREE.MeshBasicMaterial).color.setHex(0x181a20);
        hoveredMeshRef.current = null;
        onObjectHover(null);
      }
    };

    canvas.addEventListener("mousemove", handlePointerMove);
    canvas.addEventListener("mouseleave", handlePointerLeave);

    // 10. Resize Observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newW, height: newH } = entry.contentRect;
        if (newW > 0 && newH > 0) {
          camera.aspect = newW / newH;
          camera.updateProjectionMatrix();
          renderer.setSize(newW, newH);
        }
      }
    });
    resizeObserver.observe(container);

    // 11. Render Loop
    let lastTime = 0;
    const animate = (time: number) => {
      animationFrameId.current = requestAnimationFrame(animate);
      const delta = (time - lastTime) / 1000;
      lastTime = time;

      controls.update();

      // Raycast objects when in Stage 6 or interactive mode
      const meshes = objectMeshesRef.current;
      if (meshes.length > 0 && mouseRef.current.x !== -999) {
        raycasterRef.current.setFromCamera(mouseRef.current, camera);
        const intersects = raycasterRef.current.intersectObjects(meshes, false);

        if (intersects.length > 0) {
          const hitMesh = intersects[0].object as THREE.Mesh;
          if (hoveredMeshRef.current !== hitMesh) {
            // Unhover previous
            if (hoveredMeshRef.current) {
              const prevWire = hoveredMeshRef.current.children[0] as THREE.LineSegments;
              if (prevWire) {
                (prevWire.material as THREE.LineBasicMaterial).color.setHex(0x687282);
                (prevWire.material as THREE.LineBasicMaterial).opacity = 0.6;
              }
              (hoveredMeshRef.current.material as THREE.MeshBasicMaterial).opacity = 0.15;
              (hoveredMeshRef.current.material as THREE.MeshBasicMaterial).color.setHex(0x181a20);
            }

            // Highlight new
            hoveredMeshRef.current = hitMesh;
            const newWire = hitMesh.children[0] as THREE.LineSegments;
            if (newWire) {
              (newWire.material as THREE.LineBasicMaterial).color.setHex(0xb47762);
              (newWire.material as THREE.LineBasicMaterial).opacity = 1.0;
            }
            (hitMesh.material as THREE.MeshBasicMaterial).opacity = 0.35;
            (hitMesh.material as THREE.MeshBasicMaterial).color.setHex(0xb47762);

            const meta = hitMesh.userData.meta as SpatialObjectInfo;
            onObjectHover(meta);
          }
        } else if (hoveredMeshRef.current) {
          const prevWire = hoveredMeshRef.current.children[0] as THREE.LineSegments;
          if (prevWire) {
            (prevWire.material as THREE.LineBasicMaterial).color.setHex(0x687282);
            (prevWire.material as THREE.LineBasicMaterial).opacity = 0.6;
          }
          (hoveredMeshRef.current.material as THREE.MeshBasicMaterial).opacity = 0.15;
          (hoveredMeshRef.current.material as THREE.MeshBasicMaterial).color.setHex(0x181a20);
          hoveredMeshRef.current = null;
          onObjectHover(null);
        }
      }

      renderer.render(scene, camera);
    };

    animationFrameId.current = requestAnimationFrame(animate);

    // Cleanup on unmount
    return () => {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
      resizeObserver.disconnect();
      canvas.removeEventListener("mousemove", handlePointerMove);
      canvas.removeEventListener("mouseleave", handlePointerLeave);

      controls.dispose();
      renderer.dispose();

      // Dispose scene objects
      scene.traverse((obj) => {
        if ((obj as THREE.Mesh).isMesh || (obj as THREE.Points).isPoints) {
          const mesh = obj as THREE.Mesh;
          if (mesh.geometry) mesh.geometry.dispose();
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => m.dispose());
          } else if (mesh.material) {
            mesh.material.dispose();
          }
        }
      });
    };
  }, [onObjectHover]);

  // Sync scroll stage progress with 3D elements
  useEffect(() => {
    // stageProgress is normalized progress across the 3D portion:
    // If activeStageIndex < 3: 3D canvas is mostly hidden/flat
    // Stage 3 (Unprojection): Points expand from 2D plane (Z=0) into full 3D depth
    // Stage 4 (Point Cloud): Points full opacity, object wireframes opacity 0
    // Stage 5 (3D Room): Object wireframes and floor grid full opacity
    const points = pointsRef.current;
    const orig = originalPointPositions.current;
    const floorGrid = floorGridRef.current;
    const lineSegments = lineSegmentsRef.current;
    const objectMeshes = objectMeshesRef.current;

    if (!points || !orig) return;

    const geom = points.geometry;
    const posAttr = geom.attributes.position;
    const count = posAttr.count;

    if (activeStageIndex <= 2) {
      // Very beginning: flatten points toward image plane
      for (let i = 0; i < count; i++) {
        posAttr.setXYZ(i, orig[i * 3], orig[i * 3 + 1], 0.1);
      }
      posAttr.needsUpdate = true;
      if (points.material) (points.material as THREE.PointsMaterial).opacity = 0;
      if (floorGrid) floorGrid.material.opacity = 0;
      lineSegments.forEach((line) => {
        (line.material as THREE.LineBasicMaterial).opacity = 0;
      });
      objectMeshes.forEach((m) => {
        (m.material as THREE.MeshBasicMaterial).opacity = 0;
      });
    } else if (activeStageIndex === 3) {
      // Stage 04: Depth Unprojection - interpolate Z from 0 to full depth
      const unprojectFactor = THREE.MathUtils.clamp(stageProgress * 2.0, 0.05, 1.0);
      for (let i = 0; i < count; i++) {
        const targetZ = orig[i * 3 + 2];
        const curZ = 0.2 + (targetZ - 0.2) * unprojectFactor;
        // Pinhole ray divergence: X and Y scale with depth
        const curX = orig[i * 3] * (curZ / targetZ);
        const curY = orig[i * 3 + 1] * (curZ / targetZ);
        posAttr.setXYZ(i, curX, curY, curZ);
      }
      posAttr.needsUpdate = true;
      if (points.material) {
        (points.material as THREE.PointsMaterial).opacity = unprojectFactor * 0.85;
      }
      if (floorGrid) floorGrid.material.opacity = unprojectFactor * 0.2;
      lineSegments.forEach((line) => {
        (line.material as THREE.LineBasicMaterial).opacity = 0;
      });
      objectMeshes.forEach((m) => {
        (m.material as THREE.MeshBasicMaterial).opacity = 0;
      });
    } else if (activeStageIndex === 4) {
      // Stage 05: Point Cloud - full positions, high visibility, subtle object wireframe hint
      for (let i = 0; i < count; i++) {
        posAttr.setXYZ(i, orig[i * 3], orig[i * 3 + 1], orig[i * 3 + 2]);
      }
      posAttr.needsUpdate = true;
      if (points.material) (points.material as THREE.PointsMaterial).opacity = 0.92;
      if (floorGrid) floorGrid.material.opacity = 0.35;
      lineSegments.forEach((line) => {
        (line.material as THREE.LineBasicMaterial).opacity = 0.2;
      });
      objectMeshes.forEach((m) => {
        (m.material as THREE.MeshBasicMaterial).opacity = 0.05;
      });
    } else if (activeStageIndex >= 5) {
      // Stage 06: Spatial Geometry & Interactive Room - full geometry, full wireframes
      for (let i = 0; i < count; i++) {
        posAttr.setXYZ(i, orig[i * 3], orig[i * 3 + 1], orig[i * 3 + 2]);
      }
      posAttr.needsUpdate = true;
      if (points.material) (points.material as THREE.PointsMaterial).opacity = 0.75;
      if (floorGrid) floorGrid.material.opacity = 0.5;
      lineSegments.forEach((line) => {
        (line.material as THREE.LineBasicMaterial).opacity = 0.7;
      });
      objectMeshes.forEach((m) => {
        (m.material as THREE.MeshBasicMaterial).opacity = 0.18;
      });
    }
  }, [activeStageIndex, stageProgress]);

  return (
    <div className="room-3d-canvas-wrapper" ref={containerRef}>
      <canvas ref={canvasRef} className="room-3d-webgl-canvas" />

      {isLoading && (
        <div className="room-3d-loading-overlay">
          <div className="loading-spinner" />
          <span>SYNTHESIZING 3D SPATIAL ASSET // GLTF 2.0 (484KB)</span>
        </div>
      )}

      {loadError && (
        <div className="room-3d-error-banner">
          <span>NOTICE: {loadError}</span>
        </div>
      )}
    </div>
  );
}
