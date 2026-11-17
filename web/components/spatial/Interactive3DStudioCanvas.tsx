"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { SpatialObject3D } from "@/lib/spatial/coordinates";

interface Interactive3DStudioCanvasProps {
  imageSrc: string;
  spatialObjects?: SpatialObject3D[];
  hoveredIdx?: number | null;
  onHoverObject?: (idx: number | null) => void;
  glbUrl?: string | null;
  isReconstructing?: boolean;
  reconstructionProgress?: number;
  reconstructionStage?: string;
  colorMode?: "rgb" | "depth";
  showBoxes?: boolean;
}

export function Interactive3DStudioCanvas({
  imageSrc,
  spatialObjects = [],
  hoveredIdx = null,
  onHoverObject,
  glbUrl,
  isReconstructing = false,
  reconstructionProgress = 0,
  reconstructionStage = "",
  colorMode = "rgb",
  showBoxes = false,
}: Interactive3DStudioCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  const pointCloudRef = useRef<THREE.Points | null>(null);
  const glbGroupRef = useRef<THREE.Group | null>(null);
  const objectMeshesRef = useRef<THREE.Mesh[]>([]);
  const objectBoxesGroupRef = useRef<THREE.Group | null>(null);

  const animationFrameId = useRef<number | null>(null);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2(-999, -999));

  const [activePreset, setActivePreset] = useState<"default" | "top" | "iso">("default");
  const [activeColorMode, setActiveColorMode] = useState<"rgb" | "depth">(colorMode);
  const [pointCount, setPointCount] = useState<number>(0);

  // Set camera view presets
  const setCameraPreset = useCallback((preset: "default" | "top" | "iso") => {
    setActivePreset(preset);
    if (!cameraRef.current || !controlsRef.current) return;
    const controls = controlsRef.current;
    const camera = cameraRef.current;

    if (preset === "default") {
      controls.target.set(0, 0, -3.2);
      camera.position.set(0, 0.4, 0.4);
    } else if (preset === "top") {
      controls.target.set(0, 0, -3.2);
      camera.position.set(0, 5.0, -3.2);
    } else if (preset === "iso") {
      controls.target.set(0, 0, -3.2);
      camera.position.set(2.8, 2.2, -1.0);
    }
    controls.update();
  }, []);

  const resetView = useCallback(() => {
    setCameraPreset("default");
  }, [setCameraPreset]);

  // 1. Initialize Three.js WebGL Scene
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 520;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    rendererRef.current = renderer;

    // Scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0c0d10, 0.035);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(46, width / height, 0.05, 50);
    camera.position.set(0, 0.4, 0.4);
    cameraRef.current = camera;

    // OrbitControls
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, -3.2);
    controls.minDistance = 0.2;
    controls.maxDistance = 16;
    controls.maxPolarAngle = Math.PI / 2 + 0.12; // Prevent camera dipping below floor
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xfff1e0, 1.4);
    dirLight.position.set(3, 5, 2);
    scene.add(dirLight);

    const accentLight = new THREE.PointLight(0xe08365, 2.2, 12);
    accentLight.position.set(0, 0.8, -3.2);
    scene.add(accentLight);

    // Architectural Ground Plane Grid (in meters)
    const floorGrid = new THREE.GridHelper(8, 20, 0xb47762, 0x222730);
    floorGrid.position.set(0, -0.65, -3.2);
    (floorGrid.material as THREE.Material).transparent = true;
    (floorGrid.material as THREE.Material).opacity = 0.55;
    scene.add(floorGrid);

    // Group for 3D Bounding Boxes
    const objectBoxesGroup = new THREE.Group();
    scene.add(objectBoxesGroup);
    objectBoxesGroupRef.current = objectBoxesGroup;

    // Animation Render Loop
    let isSubscribed = true;
    const animate = () => {
      if (!isSubscribed) return;
      controls.update();

      // Raycasting for object hover detection
      if (cameraRef.current && objectMeshesRef.current.length > 0) {
        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
        const intersects = raycasterRef.current.intersectObjects(objectMeshesRef.current, false);
        if (intersects.length > 0) {
          const hitMesh = intersects[0].object as THREE.Mesh;
          const idx = hitMesh.userData.objectIndex;
          if (idx !== undefined && idx !== hoveredIdx) {
            onHoverObject?.(idx);
          }
        }
      }

      renderer.render(scene, camera);
      animationFrameId.current = requestAnimationFrame(animate);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      isSubscribed = false;
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  // 2. Generate Unprojected 3D Point Cloud from 2D Image
  useEffect(() => {
    if (!sceneRef.current || !imageSrc) return;
    const scene = sceneRef.current;

    // If a backend GLB is provided, don't generate synthetic point cloud
    if (glbUrl) {
      if (pointCloudRef.current) {
        scene.remove(pointCloudRef.current);
        pointCloudRef.current = null;
      }
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageSrc;

    img.onload = () => {
      const sampleW = 160;
      const sampleH = 120;
      const offscreen = document.createElement("canvas");
      offscreen.width = sampleW;
      offscreen.height = sampleH;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(img, 0, 0, sampleW, sampleH);
      const imgData = ctx.getImageData(0, 0, sampleW, sampleH).data;

      const numPoints = sampleW * sampleH;
      const positions = new Float32Array(numPoints * 3);
      const colors = new Float32Array(numPoints * 3);

      const fovRad = (38 * Math.PI) / 180;
      const aspect = sampleW / sampleH;
      const tanHalfFov = Math.tan(fovRad / 2);

      const colorNear = new THREE.Color(0xf6f2ec);
      const colorMid = new THREE.Color(0xb47762);
      const colorFar = new THREE.Color(0x3a424e);
      const tempColor = new THREE.Color();

      let ptr = 0;
      for (let y = 0; y < sampleH; y++) {
        for (let x = 0; x < sampleW; x++) {
          const pixelIdx = (y * sampleW + x) * 4;
          const r = imgData[pixelIdx] / 255;
          const g = imgData[pixelIdx + 1] / 255;
          const b = imgData[pixelIdx + 2] / 255;

          // Normalized Device Coordinates [-1, 1]
          const xNdc = (x / sampleW - 0.5) * 2;
          const yNdc = -(y / sampleH - 0.5) * 2;

          // Depth synthesis: pseudo-depth based on perspective geometry + luminance
          // Foreground floor & furniture is closer (Z ~ 2.2m), ceiling/far wall is deeper (Z ~ 5.2m)
          const normY = y / sampleH;
          const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
          const baseDepth = 2.0 + normY * 1.6 + (1 - brightness) * 1.2;
          const zDepth = THREE.MathUtils.clamp(baseDepth, 1.8, 6.0);

          // Pinhole camera ray unprojection
          const xCam = xNdc * zDepth * tanHalfFov * aspect;
          const yCam = yNdc * zDepth * tanHalfFov;
          const zCam = -zDepth;

          positions[ptr * 3] = xCam;
          positions[ptr * 3 + 1] = yCam;
          positions[ptr * 3 + 2] = zCam;

          if (activeColorMode === "rgb") {
            colors[ptr * 3] = r;
            colors[ptr * 3 + 1] = g;
            colors[ptr * 3 + 2] = b;
          } else {
            const t = THREE.MathUtils.clamp((zDepth - 1.8) / (6.0 - 1.8), 0, 1);
            if (t < 0.45) {
              tempColor.lerpColors(colorNear, colorMid, t / 0.45);
            } else {
              tempColor.lerpColors(colorMid, colorFar, (t - 0.45) / 0.55);
            }
            colors[ptr * 3] = tempColor.r;
            colors[ptr * 3 + 1] = tempColor.g;
            colors[ptr * 3 + 2] = tempColor.b;
          }

          ptr++;
        }
      }

      // Remove existing point cloud if present
      if (pointCloudRef.current) {
        scene.remove(pointCloudRef.current);
        pointCloudRef.current.geometry.dispose();
        pointCloudRef.current = null;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: 0.024,
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        sizeAttenuation: true,
      });

      const pointCloud = new THREE.Points(geometry, material);
      scene.add(pointCloud);
      pointCloudRef.current = pointCloud;
      setPointCount(numPoints);
    };
  }, [imageSrc, activeColorMode, glbUrl]);

  // 3. Load GLB when provided
  useEffect(() => {
    if (!sceneRef.current || !glbUrl) return;
    const scene = sceneRef.current;

    const loader = new GLTFLoader();
    loader.load(glbUrl, (gltf) => {
      if (glbGroupRef.current) {
        scene.remove(glbGroupRef.current);
      }
      const model = gltf.scene;
      glbGroupRef.current = model;
      scene.add(model);
    });

    return () => {
      if (glbGroupRef.current && sceneRef.current) {
        sceneRef.current.remove(glbGroupRef.current);
        glbGroupRef.current = null;
      }
    };
  }, [glbUrl]);

  // 4. Render 3D Bounding Envelopes only if explicitly requested (disabled in pure 3D room construction)
  useEffect(() => {
    if (!objectBoxesGroupRef.current) return;
    const group = objectBoxesGroupRef.current;

    // Clear existing boxes
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
      }
    }
    objectMeshesRef.current = [];

    if (!showBoxes || !spatialObjects || spatialObjects.length === 0) {
      return;
    }

    spatialObjects.forEach((obj, idx) => {
      const [w, h, d] = obj.dimensions;
      const [posX, posY, posZ] = obj.position;

      // 3D Box Volume Mesh (for raycasting and tint)
      const boxGeom = new THREE.BoxGeometry(w, h, d);
      const isHovered = hoveredIdx === idx;

      const fillMat = new THREE.MeshBasicMaterial({
        color: isHovered ? 0xe08365 : 0xb47762,
        transparent: true,
        opacity: isHovered ? 0.32 : 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const boxMesh = new THREE.Mesh(boxGeom, fillMat);
      boxMesh.position.set(posX, posY, posZ);
      boxMesh.userData = { objectIndex: idx, label: obj.label };
      group.add(boxMesh);
      objectMeshesRef.current.push(boxMesh);

      // Wireframe Outline
      const edges = new THREE.EdgesGeometry(boxGeom);
      const lineMat = new THREE.LineBasicMaterial({
        color: isHovered ? 0xe08365 : 0xb47762,
        linewidth: isHovered ? 2 : 1,
        transparent: true,
        opacity: isHovered ? 1.0 : 0.7,
      });
      const wireframe = new THREE.LineSegments(edges, lineMat);
      wireframe.position.copy(boxMesh.position);
      group.add(wireframe);

      // Floating Label Billboard Sprite
      const canvasLabel = document.createElement("canvas");
      canvasLabel.width = 256;
      canvasLabel.height = 64;
      const ctx = canvasLabel.getContext("2d");
      if (ctx) {
        ctx.fillStyle = isHovered ? "rgba(224, 131, 101, 0.95)" : "rgba(10, 9, 8, 0.88)";
        ctx.strokeStyle = isHovered ? "#FFFFFF" : "#B47762";
        ctx.lineWidth = 2;
        ctx.roundRect(4, 4, 248, 56, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#F2F0EA";
        ctx.font = "bold 20px monospace";
        ctx.fillText(obj.label.toUpperCase(), 16, 28);
        ctx.font = "14px monospace";
        ctx.fillStyle = isHovered ? "#0A0908" : "#E08365";
        ctx.fillText(`${w}m × ${h}m (${Math.round(obj.confidence * 100)}%)`, 16, 48);

        const texture = new THREE.CanvasTexture(canvasLabel);
        const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.set(posX, posY + h / 2 + 0.18, posZ);
        sprite.scale.set(0.65, 0.16, 1);
        group.add(sprite);
      }
    });
  }, [spatialObjects, hoveredIdx, showBoxes]);

  // Pointer move for Raycaster
  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  };

  const handlePointerLeave = () => {
    mouseRef.current.x = -999;
    mouseRef.current.y = -999;
    onHoverObject?.(null);
  };

  return (
    <div className="interactive-3d-studio-container" ref={containerRef}>
      {/* 1. Camera & View Controls Toolbar */}
      <div className="studio-3d-toolbar">
        <div className="toolbar-group">
          <span className="toolbar-tag">3D CAMERA:</span>
          <button
            type="button"
            className={`toolbar-btn ${activePreset === "default" ? "is-active" : ""}`}
            onClick={() => setCameraPreset("default")}
          >
            PERSPECTIVE
          </button>
          <button
            type="button"
            className={`toolbar-btn ${activePreset === "top" ? "is-active" : ""}`}
            onClick={() => setCameraPreset("top")}
          >
            PLAN (TOP)
          </button>
          <button
            type="button"
            className={`toolbar-btn ${activePreset === "iso" ? "is-active" : ""}`}
            onClick={() => setCameraPreset("iso")}
          >
            ISO 45°
          </button>
          <button
            type="button"
            className="toolbar-btn btn-reset"
            onClick={resetView}
            title="Reset camera orientation"
          >
            RESET ⟲
          </button>
        </div>

        <div className="toolbar-group">
          <span className="toolbar-tag">STYLE:</span>
          <button
            type="button"
            className={`toolbar-btn ${activeColorMode === "rgb" ? "is-active" : ""}`}
            onClick={() => setActiveColorMode("rgb")}
          >
            RGB PHOTOREAL
          </button>
          <button
            type="button"
            className={`toolbar-btn ${activeColorMode === "depth" ? "is-active" : ""}`}
            onClick={() => setActiveColorMode("depth")}
          >
            DEPTH GRADIENT
          </button>
        </div>

        <div className="toolbar-meta">
          <span>{pointCount > 0 ? `${pointCount.toLocaleString()} 3D VERTICES` : "3D SCENE READY"}</span>
          <span>·</span>
          <span>{spatialObjects.length} 3D BOUNDS</span>
        </div>
      </div>

      {/* 2. Interactive WebGL Canvas */}
      <canvas
        ref={canvasRef}
        className="studio-3d-canvas"
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
      />

      {/* 3. Real-Time HUD Overlay for Hovered Object */}
      {hoveredIdx !== null && spatialObjects[hoveredIdx] && (
        <div className="studio-3d-hover-hud">
          <div className="hud-badge-title">
            <span className="hud-dot" />
            <strong>{spatialObjects[hoveredIdx].label.toUpperCase()}</strong>
            <span className="hud-conf">
              {Math.round(spatialObjects[hoveredIdx].confidence * 100)}% CONF
            </span>
          </div>
          <div className="hud-dims-row">
            <span>POS: [ {spatialObjects[hoveredIdx].position.map((v) => v.toFixed(2)).join(", ")} ] m</span>
            <span>·</span>
            <span>DIMS: {spatialObjects[hoveredIdx].dimensions.map((d) => d.toFixed(2)).join(" × ")} m</span>
          </div>
        </div>
      )}

      {/* 4. Reconstruction In-Progress Overlay */}
      {isReconstructing && (
        <div className="studio-3d-reconstructing-overlay">
          <div className="recon-spinner" />
          <div className="recon-text">
            <strong>SYNTHESIZING 3D METRIC ROOM RECONSTRUCTION</strong>
            <p>{reconstructionStage || "Running Depth Anything V2 & RANSAC ground recovery..."}</p>
            <div className="recon-progress-bar">
              <div
                className="recon-progress-fill"
                style={{ width: `${Math.round(reconstructionProgress * 100)}%` }}
              />
            </div>
            <span>{Math.round(reconstructionProgress * 100)}% COMPLETE</span>
          </div>
        </div>
      )}

      {/* 5. Navigation Hint Pill */}
      <div className="studio-3d-nav-hint">
        <span>LEFT DRAG TO ORBIT · RIGHT DRAG TO PAN · WHEEL TO ZOOM</span>
      </div>
    </div>
  );
}
