# RoomAI Immersive Experience — Technical Implementation Plan

**Status:** planning document; no application code is specified here.  
**Inputs:** Phase 1 Creative Direction and Phase 2 Visual Blueprint / Section-by-Section Storyboard supplied with the request.  
**Existing project context:** RoomAI is currently a Python/FastAPI reconstruction prototype. The repo exposes `/api/v1/uploads`, `/api/v1/jobs`, `/api/v1/models`, and `/api/v1/exports`, with GLB and PLY exports. Keep the new web experience as a separate frontend application in this repository and integrate the real API.

## 1. Executive architecture

Build one immersive editorial website around one verified indoor reconstruction. The first five storyboard states (Entry, Photograph, Depth, Point Field, Crossing, World) form a controlled cinematic sequence; Engine, Explore, Evidence, Limitations, Future, and Project shift progressively toward explanation and proof. Narrative continuity comes from visual and camera language, not from forcing every section into one enormous Three.js scene. Use separate scene modules and explicit handoffs to reduce GPU residency and state coupling.

The core runtime has four layers:

1. **Next.js document and interface layer:** semantic content, navigation, section labels, controls, benchmark/limitation copy, keyboard behavior, and non-WebGL fallback.
2. **R3F scene layer:** photographic plane, depth proxy, point cloud, room reconstruction, representation modes, spatial selection, and measurement line.
3. **Motion coordinator:** GSAP + ScrollTrigger maps scroll to section state and camera/representation parameters. Lenis is optional polish, configured to preserve native scrolling and reduced-motion behavior.
4. **FastAPI integration:** explicit `PRECOMPUTED` exhibit data for the initial story and separately labeled `LIVE` upload/inference flow where deployed and available. A precomputed sequence must never be presented as an active inference job.

### Rendering and scene strategy

- Use **R3F v9 on its stable WebGL renderer path** for the first release. The point-cloud look may use custom GLSL/BufferGeometry, and production postprocessing/Drei compatibility is more valuable here than adopting an alpha renderer. Keep a renderer adapter boundary so a later WebGPU evaluation does not infect content and scene state.
- A single shared `<Canvas>` can host the cinematic states and the small persistent world viewport while preserving one context. Compose scene-specific groups and switch visibility/asset residency at intentional section boundaries. If the project later needs distinct render qualities or scene-specific context teardown, use a second canvas only after profiling; multiple contexts compete for GPU resources.
- Use a deliberately authored `ExhibitScene` asset bundle: source RGB, metric depth map, colored point samples (or positions + colors), final GLB, and a small manifest describing dimensions, units, bounds, camera framing, provenance, and supported representation modes. Derive states from that same reconstruction. Do not synthesize unsupported geometry merely to make every selector look complete.
- Keep typography, controls, narrative, technical diagrams, metrics, and accessibility text in HTML/CSS. Use 3D only when position, depth, camera, or spatial selection matters.
- The cinematic camera is a scripted, deterministic spline/keyframe path. The Explore camera is a constrained user camera with damping, near/far bounds, target bounds, and a reset pose. They are separate control modes; scroll must not fight a live OrbitControls gesture.

### Data truth and API contract

Use API routes already present in the repo: upload image(s), submit reconstruction, poll `/api/v1/jobs/{job_id}`, read `/results`, cancel if needed, and fetch the export (GLB/PLY). Add a typed frontend API adapter and runtime validation for responses. Confirm the backend's actual status vocabulary, progress granularity, CORS, upload limits, and artifact URL behavior before wiring UI. Downloaded assets need authenticated/signed or same-origin URLs if deployment requires them. The current endpoint for a scene graph is a summary placeholder, so the frontend must not depend on semantic object labels being available. Coordinates and wall/object labels in storyboard examples are illustrative unless supplied by real artifact metadata.

`LIVE` means the actual request was accepted and completed by the deployed backend. `PRECOMPUTED` means a stored, named reconstruction is being viewed. If live inference is unavailable, show the precomputed experience with an explicit label and a clear route to the project/API status. Do not imply live GPU inference with canned progress or a pre-rendered scroll animation.

## 2. Section-by-section implementation map

Dependencies notation: **Core** = React/Next/TypeScript; **3D** = Three.js, R3F, Drei; **Motion** = GSAP, ScrollTrigger; **Scroll** = Lenis (optional); **UI** = Tailwind CSS. Shared packages are listed once in §4; section entries name only the dependencies actively used.

### 0. Global Experience Architecture — the exhibition system

- **Components:** `ExperienceShell`, `ExperienceCanvas`, `SectionIndex`, `ProgressMarker`, `SkipIntroLink`, `SceneStatus`, `FallbackExperience`.
- **3D assets:** shared exhibit manifest; room assets referenced by sections below; procedural dark volume/floor plane and architectural guide lines.
- **Rendering:** one client-only canvas behind semantic DOM sections; separate scene groups for cinematic and informational modes. Use dynamic import with SSR disabled for the canvas, and reserve a stable canvas aspect/viewport to avoid layout shifts.
- **Animation:** one scroll-state coordinator with named labels; scene state transitions are reversible when scrolling backward. Hard cuts at story-to-evidence and documentation transitions; morphs only for meaningful representation changes.
- **Interaction:** persistent index jumps directly to section anchors, updates active index, can skip the intro, and returns to index/top. Section navigation must not require replaying the film.
- **Dependencies:** Core, 3D, Motion, UI; Scroll only if it passes compatibility/performance checks.
- **Performance:** cap device pixel ratio; pause/limit frame loop offscreen or while static; asset prefetch only for the hero bundle; provide a no-WebGL path and reduced-motion state.

### 1. Entry — The Dark Room

- **Components:** `EntrySection`, `RoomaiWordmark`, `TechnicalMeta`, `ScrollCue`.
- **3D assets:** source interior photograph as a plane; subtle floor plane; soft key light; abstract surrounding volume. Generate floor/volume/guide lines procedurally. Source image must be an owned/licensed room image and be the same source used by the reconstruction.
- **Rendering:** WebGL scene with a perspective camera and photographic plane. Keep the 35–50 mm equivalent framing configurable for desktop/mobile; image occupies roughly 55–65% width at desktop as blueprint directs.
- **Animation:** darkness → silhouette → illuminated still, then hold image while the camera/world begins to move on scroll. Small entrance delay only; no cascading copy reveal.
- **Interaction:** pointer parallax capped to a few pixels; scroll cue and keyboard-accessible “Skip intro.” Disable parallax for touch and reduced motion.
- **Dependencies:** Core, 3D, Motion; UI.
- **Performance:** low-cost plane/light setup; source texture with mipmaps and responsive resolution variants; no bloom or atmospheric particle layer.

### 2. Transform — The Photograph (`01 / TRANSFORM`)

- **Components:** `TransformSection`, `HeroStatement`, `ImageMetadata`, `StageLabel`.
- **3D assets:** same RGB image and scene framing; technical metadata (e.g. image dimensions/view count) must come from the real exhibit manifest.
- **Rendering:** keep the photograph as a flat plane initially, with display statement and metadata in DOM overlays. Do not turn ordinary text into 3D text.
- **Animation:** image scales/frames into prominence; restrained split between static image and beginning spatial movement. Copy enters once and remains readable.
- **Interaction:** scroll advances into depth; section index can jump away. No gameplay controls during Watch.
- **Dependencies:** Core, 3D, Motion, UI.
- **Performance:** same texture as Entry; avoid duplicating large textures or reloading the asset on section changes.

### 3. Depth — The Image Acquires Space (`01 / DEPTH`)

- **Components:** `DepthSection`, `DepthLegend`, `DepthReadout`, optional `DepthContourOverlay`.
- **3D assets:** metric depth map aligned to source RGB; image-space camera calibration/intrinsics if available; otherwise mark the effect as a depth-based visualization, not calibrated world geometry.
- **Rendering:** subdivided plane or vertex-displaced image mesh (grid density selected by target device). A custom shader or CPU-generated positions map each pixel/sample to z. RGB remains dominant; optional contours are a restrained overlay, not a full false-color heatmap unless the visitor selects that mode.
- **Animation:** scroll drives depth displacement from 0 to a bounded scale and increases foreground/background separation. Do not use arbitrary per-object offsets. Camera dolly and rise remain subtle.
- **Interaction:** pointer may reveal a local contour/measurement only where depth data exists. Labels show actual depth/unit confidence or use neutral “DEPTH / ESTIMATED” copy where uncertainty is unavailable.
- **Dependencies:** Core, 3D, Motion; custom shader only if shader path selected.
- **Performance:** avoid CPU per-frame vertex updates; use GPU displacement or precomputed geometry; lower grid/sample resolution on mobile; no shadow map for this stage.

### 4. Point Field — Pixels Become Space (`01 / POINTS`)

- **Components:** `PointFieldSection`, `PointCloud`, `PointLegend`, `PointCountLabel`.
- **3D assets:** colored points unprojected from the real depth/RGB pair; point size/visibility bounds; optional low-density and high-density buffers. These can be derived offline from API exports or the approved source reconstruction. No random noise particles.
- **Rendering:** `BufferGeometry`/R3F `Points` with packed position/color attributes and a restrained point material. Preserve object/floor/wall structure through real spatial positions. Use a deterministic subset/LOD rather than randomly changing point locations.
- **Animation:** a point-cloud draw/opacity and surface-to-points morph driven by scroll. To create a genuine morph, store correspondence between image samples and 3D samples; if correspondence does not exist, use an honest staged crossfade/translation with stable sample positions rather than claiming a physically exact conversion.
- **Interaction:** minimal cursor parallax; no particle explosion. Optional point density can step down based on measured frame rate/device tier.
- **Dependencies:** Core, 3D, Motion; Drei optional for bounds helpers.
- **Performance:** point count and overdraw are the main risks. Establish initial tiers (e.g. low tens of thousands for mobile, higher desktop, benchmark actual device matrix); cull by frustum, cap DPR, avoid per-point React objects, and avoid transparent points where possible. Build buffers once and mutate uniforms/attributes only when needed.

### 5. Crossing — Enter the Image (`01 / CROSSING`)

- **Components:** `CrossingBeat`, `CameraRail`, `StageLabel`, optional `SoundToggle`.
- **3D assets:** same point cloud; reconstructed room GLB or a low-detail reveal proxy; warm light target ahead. Optional short, licensed/synthesized tonal sound asset with captions/controls.
- **Rendering:** keep the camera within known exhibit bounds and direct it through a safe corridor in the points. Blend point representation into the room asset only if geometry and coordinate systems align.
- **Animation:** scroll-linked dolly with controlled acceleration/deceleration over roughly 3–8 seconds of perceived motion; intensity follows scroll progress, not a free-running timer. Reveal light/material progression from RGB points to desaturated points to surfaces/hybrid. The camera never shakes or spins.
- **Interaction:** cinematic mode owns camera; user may skip via index/keyboard. Sound is opt-in and never required to understand the transition.
- **Dependencies:** Core, 3D, Motion; UI.
- **Performance:** pre-load the low-detail room before crossing; do not parse a large GLB during the pinned sequence; have a cut/fade fallback for missing room asset, reduced motion, and low capability.

### 6. The World — The Reconstructed Room (`01 / WORLD`)

- **Components:** `WorldReveal`, `ReconstructionScene`, `ExplorePrompt`, `RepresentationBar` (introduced after reveal).
- **3D assets:** selected verified reconstruction GLB, point cloud and RGB/depth references; optimized low and high quality variants if useful. Reality, Depth, Points, Geometry, Hybrid modes all refer to the same scene bundle and disclose when a mode is unavailable.
- **Rendering:** GLB via Drei `useGLTF`/Three loader; camera, color management, physically restrained materials and one broad soft light. Do not claim a full digital twin if the artifact is only depth-unprojected points.
- **Animation:** complete one composed entry → central perspective → settle camera move; then relinquish camera to Explore. The `EXPLORE` prompt arrives after the camera rests; UI is absent during the visual payoff.
- **Interaction:** enter Explore mode to enable orbit/pan/zoom, reset pose, representation switching, and object inspection where metadata supports it.
- **Dependencies:** Core, 3D, Motion, UI.
- **Performance:** prewarm only exhibit assets; use Draco/Meshopt only if measured; compressed textures; limit dynamic shadows; release heavy stage-only buffers after crossing if needed.

### 7. Engine — How the World Was Built (`02 / ENGINE`)

- **Components:** `EngineSection`, `PipelineRail`, `PipelineStage`, `StageDescription`, `EngineSceneDock`.
- **3D assets:** same reconstruction and point/depth buffers; simple process nodes/icons and connectors are DOM/SVG or procedural lines, not an unrelated scene.
- **Rendering:** room remains as a small scene dock or side stage; a vertical semantic HTML pipeline explains RGB → Depth → Spatial Representation → 3D Points → Interactive World. Selecting a stage updates the same exhibit scene representation.
- **Animation:** stage activation follows scroll or explicit selection; small camera shifts only. Pipeline transitions share state with the viewer without creating a second copy of the scene.
- **Interaction:** keyboard-accessible stage buttons and linkable stage state; labels and explanations stay in DOM.
- **Dependencies:** Core, 3D, Motion, UI.
- **Performance:** keep the scene modest while the informational section is active; mount/suspend unnecessary cinematic effects; avoid a large diagram library.

### 8. Explore — The Spatial Instrument (`03 / EXPLORE`)

- **Components:** `ExploreViewer`, `CameraControls`, `RepresentationSelector`, `ResetViewButton`, `ObjectInspector`, `MeasurementTool`, `ModeStatus`.
- **3D assets:** exhibit scene and supported representations; actual object IDs/semantics only if delivered by a real scene graph; otherwise allow surface/point selection and describe it generically. Measurement is based on metric coordinates only when the artifact is metric.
- **Rendering:** canvas dominates desktop (~75–85%); OrbitControls for bounded orbit/pan/zoom with damping; raycaster for hover/click; overlay annotations via DOM or Drei `<Html>` only when screen-space alignment is useful. Use HTML for labels and metadata.
- **Animation:** representation switch morphs a shared buffer/material when correspondence is possible; otherwise transition visibly and honestly (e.g. brief dissolve). Selection outline uses oxide accent; measurement line draws between two valid 3D hits and reports distance in meters only with metric scale.
- **Interaction:** orbit, pan, zoom, reset, representations, selection, two-point measurement. Escape clears selection; keyboard alternatives and touch gestures; no scroll conflict while pointer is captured by the viewer.
- **Dependencies:** Core, 3D, Motion for entry/exit only, UI. Drei controls/loaders; no additional motion library.
- **Performance:** render on demand or with a capped loop while idle; dispose temporary raycast/measurement objects; throttle hover; use a low-cost outline/selection treatment rather than full-scene postprocessing.

### 9. Evidence — Prove It (`04 / EVIDENCE`)

- **Components:** `EvidenceSection`, `MetricDefinition`, `SceneComparison`, `EvaluationProtocol`, `SystemDiagram`, `ArtifactLink`.
- **3D assets:** up to six approved scene thumbnails and optional small reconstruction swatches, generated from evaluation assets with provenance. Avoid requiring a large live canvas here.
- **Rendering:** warm white/stone HTML section. Metrics and labels are accessible text; charts/comparisons can be SVG/HTML. System diagram uses semantic DOM/SVG. Include dataset, split, protocol, model version/date and limitations beside any benchmark values.
- **Animation:** hard cut/clear visual change from cinematic world, then restrained reveal. No count-up animation that changes the meaning or hides values.
- **Interaction:** metric definitions expandable or linked; scene comparison controls can select a scene and representation; export links point to actual files.
- **Dependencies:** Core, UI; Motion for modest entrances only. 3D optional and static/low quality only.
- **Performance:** static images and SVG are cheaper and more accessible than a persistent WebGL scene. Keep benchmark data in a typed content file with an explicit verification source.

### 10. Limitations — Intellectual Honesty

- **Components:** `LimitationsSection`, `ResearchNote`, `CapabilityStatus`.
- **3D assets:** none required; optional low-key image crop only.
- **Rendering:** editorial HTML on quiet stone/light or dark field. Explain current single-view scope, scene-to-scene variability, lack of Stage 2 Gaussian Splatting in the prototype UI, deployment/containerization status, and evaluation breadth only after reconfirming the current audit.
- **Animation:** near-static; simple hard cut from Evidence and deliberate change in visual density.
- **Interaction:** links to methodology, issue log, or model details; no interactive claims that imply unsupported capability.
- **Dependencies:** Core, UI; no 3D required.
- **Performance:** static content; ensure disclosure remains prominent and readable on narrow screens.

### 11. Future — The Space Expands (`05 / FUTURE`)

- **Components:** `FutureSection`, `FutureDirection`, `CapabilityTag`.
- **3D assets:** abstract procedural camera-frustum diagrams and cloned/branched representations for multi-view, scene graph, mesh, CAD/design, Gaussian representation. These are conceptual diagrams, not fabricated product screenshots.
- **Rendering:** SVG/DOM diagrams first; optional low-complexity Three.js points/lines to pull the room out into a wider abstract space. Label each as research/future direction.
- **Animation:** pull camera outward; one room branches into conceptual views/stages. Use controlled transitions and avoid showing an unimplemented result as already achieved.
- **Interaction:** selecting a direction reveals a concise explanation and status label; all details accessible without WebGL.
- **Dependencies:** Core, UI, Motion; 3D only for spatial diagram.
- **Performance:** use lightweight instancing/lines; do not load future pipeline model assets.

### 12. Project — Return to Reality (`06 / PROJECT`)

- **Components:** `ProjectSection`, `ProjectLinks`, `FinalStatement`, `SmallReconstructionStill`, footer metadata.
- **3D assets:** one low-resolution still or pre-rendered thumbnail from the verified reconstruction; no full viewer required.
- **Rendering:** warm white, minimal HTML; project links for GitHub, research, documentation, datasets, model information, limitations, and contact as available. Final copy returns to “I didn’t look at a 3D model. I entered a photograph.”
- **Animation:** quiet final statement with a deliberate pause; simple CTA arrow movement, then stop. Respect reduced motion.
- **Interaction:** ordinary accessible links; no SaaS conversion funnel. CTA text “Explore the project →”.
- **Dependencies:** Core, UI; Motion optional for the final pause/microinteraction.
- **Performance:** static image and no active canvas required; final content remains usable if WebGL has failed.

## 3. Camera, state, and scroll design

### Camera modes

| Mode | Owner | Behavior | Exit condition |
|---|---|---|---|
| `CINEMATIC` | Scroll coordinator | Scripted keyframes/spline, bounded dolly, no user input | World reveal settles or intro is skipped |
| `INSPECT` | Viewer controls | Orbit/pan/zoom with limits and known reset pose | User exits viewer / selects another section |
| `STATIC` | Scene controller | Camera fixed, render-on-demand where safe | Section needs motion or resize |
| `FALLBACK` | DOM/video controller | Pre-rendered sequence or static layered image | WebGL is unavailable/over budget |

Model scroll progress as named semantic states (`entry`, `photo`, `depth`, `points`, `crossing`, `world`) and expose normalized progress to the scene controller. Avoid direct DOM scroll reads in `useFrame`; avoid React state updates per frame. ScrollTrigger writes lightweight refs/uniforms, with a single animation loop consuming them. When a visitor jumps via navigation, use ScrollTo/anchor relocation and land at a meaningful state without forcing every preceding cinematic beat.

Lenis should not be a prerequisite. If enabled, initialize/dispose it in one client-side provider and use its documented GSAP ScrollTrigger synchronization; don't run multiple independent RAF loops. Use native scroll when reduced motion is requested, on low-power/mobile devices where smoothing harms responsiveness, and for accessibility. Pin only the hero film; long pinned regions complicate mobile browser chrome, focus, and ScrollTrigger refresh.

### Representation state model

`REALITY`, `DEPTH`, `POINTS`, `GEOMETRY`, `HYBRID` are a view over one exhibit asset, not five unrelated scenes. Each mode needs explicit availability metadata. Reality is the RGB texture; Depth uses the real aligned depth asset; Points uses unprojected samples; Geometry uses only valid surfaced geometry available from the reconstruction; Hybrid composes compatible representations. If a mode cannot be produced by the actual artifact, disable it with a short explanation instead of faking it. Keep selected mode and selected point/object in a small typed context; keep high-frequency camera/scroll values in refs.

## 4. Proposed project structure

Add the frontend under `web/` to avoid disturbing the existing Python app. This is the target structure, not an instruction to generate files now.

```text
RoomAI/
├── web/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── globals.css
│   │   ├── loading.tsx
│   │   ├── error.tsx
│   │   └── api/                  # optional same-origin proxy/BFF routes only
│   ├── components/
│   │   ├── experience/           # ExperienceShell, CanvasBoundary, fallback
│   │   ├── sections/             # one DOM section per storyboard beat
│   │   ├── navigation/           # SectionIndex, skip, focus management
│   │   ├── viewer/               # controls, representations, inspector, measure
│   │   ├── evidence/             # metrics, comparisons, protocol, diagram
│   │   └── ui/                   # buttons, labels, status, loading/error states
│   ├── scenes/
│   │   ├── shared/               # lights, camera rails, frame/bounds helpers
│   │   ├── entry/                # dark room and photographic plane
│   │   ├── transform/            # depth/point transition
│   │   ├── reconstruction/       # GLB, points, depth, representation adapters
│   │   └── future/               # conceptual diagrams only
│   ├── animation/
│   │   ├── gsap-client.ts        # client-only registration/setup
│   │   ├── scroll-coordinator.ts # named timeline/state mapping
│   │   └── motion-tokens.ts
│   ├── hooks/                    # reduced motion, capability, viewport, section
│   ├── state/                    # typed exhibit/viewer state (low-frequency only)
│   ├── lib/
│   │   ├── api/                  # typed FastAPI client, schemas, errors
│   │   ├── assets/               # exhibit manifest and URL resolution
│   │   ├── performance/          # device tier and quality policy
│   │   └── validation/           # runtime API/manifest validation
│   ├── content/                  # section copy, metrics, limitations, links
│   ├── public/
│   │   ├── assets/exhibit/        # optimized public showcase bundle
│   │   ├── images/                # posters, scene thumbnails, depth previews
│   │   ├── fonts/                 # licensed/self-hosted fonts
│   │   └── sequences/             # optional reduced-motion/mobile film
│   ├── types/                    # ExhibitManifest, ApiJob, Representation types
│   ├── next.config.ts
│   ├── package.json
│   └── tsconfig.json
├── app/                           # existing Python application; retain
├── artifacts/                     # existing generated outputs; curate, do not ship all
├── docs/ROOMAI_TECHNICAL_IMPLEMENTATION_PLAN.md
└── ...existing backend and research files
```

The production showcase assets should be curated and copied/exported into `web/public/assets/exhibit/`; do not expose or bundle the full `artifacts/jobs` directory. Keep private uploads and generated per-user results out of the static public folder.

## 5. Dependency and version recommendations

Use the latest patch within the recommended compatible stable lines when scaffolding, then commit the lockfile and upgrade intentionally. Do not use floating `latest` at deployment. The 3D ecosystem is currently in a renderer transition; stable compatibility is more important than adopting an alpha for novelty.

| Package | Recommendation | Reason / note |
|---|---|---|
| Node.js | 22 LTS (current supported LTS at project bootstrap) | Match Next.js supported runtime; pin via `.nvmrc`/Volta and CI. Recheck at implementation start. |
| Next.js | 16.x Active LTS, latest patched release | App Router, metadata, server rendering for text, route-level lazy loading. Avoid canary. |
| React / React DOM | 19.2.x compatible with chosen Next 16 release | Use same exact minor/patch pair. |
| TypeScript | 5.9.x or the version supported by chosen Next toolchain | Strict mode; align with Next generated config. |
| Three.js | Current stable r18x release compatible with R3F 9 | Pin exact package and `@types/three` in lock; use standard WebGLRenderer path. |
| `@react-three/fiber` | 9.x stable | Pairs with React 19. R3F 10 remains alpha in consulted docs and changes the default renderer to WebGPU; that is a poor baseline for GLSL/custom shader and postprocessing compatibility. |
| `@react-three/drei` | Latest stable release with peer support for R3F 9 | Use loaders, OrbitControls, bounds, Html selectively; verify peer ranges at install. |
| `gsap` | 3.13+ stable (latest patched 3.x) | ScrollTrigger is included as a plugin; register explicitly and scope/clean up in React. |
| `lenis` | 1.x stable, optional | Only if the team wants smoothing after native-scroll prototype; documented GSAP integration required. |
| Tailwind CSS | 4.x stable | Current Next integration uses `@tailwindcss/postcss`; keep custom visual tokens in CSS variables. |
| `@gltf-transform/cli` | build-time tool, pinned | Optimize GLB (prune/texture/mesh compression); keep source and optimized outputs. |
| `draco3d` / Meshopt | choose one tested mesh compression route | Adds decoder cost and complexity; compare loading benefit against a small showcase model. |
| `zod` | current stable 4.x only if adopted by backend contract | Runtime-validate API/manifests; otherwise a lightweight schema validator can be omitted. Avoid duplicating schemas without a contract source. |

No Framer Motion, Redux, physics engine, chart framework, or postprocessing package is required initially. GSAP owns scroll choreography; R3F owns spatial interaction; CSS owns interface transitions. Add a rendering/effects dependency only when a specific authored shot needs it and the budget permits it.

## 6. Asset production checklist

### Must-have source and showcase assets

- [ ] One high-quality, rights-cleared indoor RGB source image and its exact image dimensions/view metadata.
- [ ] Matching metric depth map, coordinate convention, camera intrinsics if available, invalid-depth mask, units, scale confidence, and alignment notes.
- [ ] Dense colored PLY or equivalent point data with known axis convention, bounds, sample count, and provenance.
- [ ] Verified GLB output for the selected exhibit; validate whether it contains points, mesh, normals, textures, hierarchy, and meaningful metric scale before choosing which representation modes to expose.
- [ ] A stable exhibit manifest linking source image, depth, point data, GLB, poster, bounds, camera framing, supported modes, model/version, generation date, and `PRECOMPUTED` status.
- [ ] A low-end/mobile point buffer and optional low-poly GLB generated from the same source/correspondence. Preserve stable sampled positions so transition states remain coherent.
- [ ] Evidence thumbnails/side-by-side outputs for the benchmark scenes, with consent/license and exact relationship to the published evaluation.
- [ ] Fallback sequence: compressed pre-rendered hero transformation or image layers, plus poster and static final room image. Label this fallback as precomputed.
- [ ] Licensed display/UI/mono fonts; self-host only the weights used, with system-font fallback.
- [ ] Optional short tonal sound cue, disabled by default, with mute control and no essential information encoded only in sound.

### Blender / asset preparation

Use Blender for cleanup, framing, coordinate/origin normalization, optional proxy surfaces, and curated camera stills. Do not “beautify” the benchmark reconstruction into a result that implies unsupported model quality. Preserve a pristine source export and log all transforms. Export GLB with compressed textures where appropriate; validate material color space, scale, axes, and transparency in the browser. Do not use generated stock interiors in place of the real selected product reconstruction. Procedural assets should be limited to architectural planes, lines, camera rails, and conceptual future diagrams.

## 7. Asset loading and GPU budget

- **Progressive first impression:** render the dark environment, HTML wordmark and poster immediately. Load the source image first, then depth/point buffers, then the world GLB before the crossing trigger. Never block the opening on all assets.
- **Code splitting:** keep Canvas/R3F/GSAP in client-only chunks; lazy-import the Explore viewer and evidence scene utilities. Avoid loading UI packages into the initial canvas bundle.
- **GLB:** prune unused nodes/materials; resize textures to the exhibit's projected need; evaluate KTX2/Basis texture compression and Meshopt/Draco on real mobile networks. Avoid both mesh compression formats simultaneously without measured gain. Include decoder assets and fallbacks.
- **PLY:** convert showcase PLY to packed binary/typed buffers at build time or backend export time; avoid parsing giant ASCII PLY on the main thread. Consider a small custom manifest + binary position/color files if PLY loader cost is excessive.
- **Textures:** JPEG/AVIF for photo posters where supported; GPU compressed format for runtime textures if the pipeline supports it; mipmaps for oblique photographic planes. Never ship full sensor-sized images if downsampled versions serve the shot.
- **GPU:** cap DPR (initial target max 1.5 desktop / 1.25 mobile, tune on devices); adapt points and viewport resolution; disable expensive shadows/postprocessing in reduced tiers; use one broad shadow or baked shading; avoid multiple full-screen effects and large transparent overdraw.
- **Lifecycle:** dispose geometry/materials/textures on true unmount; `useGLTF` cache intentionally; avoid creating objects in React render/useFrame; pause render loop when canvas is offscreen or scene state is static. Handle WebGL context loss with a recoverable message/fallback.
- **Network:** serve immutable versioned assets with long-lived cache headers; keep first viewport transfer intentionally small and report measurements; show loading status with a useful poster while GLB/points decode.

Starting performance targets (validate, then adjust): hero poster visible in <2 s on a representative mid-tier mobile connection; no blocking transfer of full GLB before user reaches the sequence; sustain near 60 fps on target desktop and at least 30 fps on supported mobile for the main transform; no more than one active high-density scene; avoid long tasks during scroll. These are acceptance targets, not promises independent of hardware/network.

## 8. Responsive and accessibility strategy

| Tier | Experience |
|---|---|
| Desktop | Full point density, full camera crossing, interactive room, spatial index, measurement tool. Adaptive DPR and quality still apply. |
| Tablet | Lower point samples, shorter crossing, one broad light, reduced shadow/effect cost, persistent accessible controls. Preserve the image → depth → points → world story. |
| Mobile | Portrait crop/framing, much lower point density, shorter camera rail, touch OrbitControls, bottom-sheet inspector, compact index. Turn off smoothing where it harms touch; no hover-only discovery. |
| Low capability / no WebGL | Pre-rendered labeled transformation sequence or layered image sequence; interactive final room may be a lightweight static/orbit clip only if proven stable; otherwise offer GLB/PLY links and full explanatory content. Never leave a blank canvas. |
| Reduced motion | Native scroll; no parallax or pinned long camera move; direct, fast state transitions; retain a static image/depth/point representation explanation and working section navigation. |

Use semantic headings/landmarks, skip link, focus-visible states, keyboard-operated index and viewer controls, accessible names/state for mode toggles, Escape to leave/clear inspect mode, sufficient contrast, text descriptions for each visual representation, no essential information conveyed by color alone, and `prefers-reduced-motion`. Canvas must have an accessible label and a DOM description. Test zoom/reflow; avoid trapping focus inside the pinned cinematic region.

## 9. Development milestones and acceptance criteria

Implement in the sequence prescribed in the blueprint. Each milestone should leave a reviewable, functioning slice. The plan itself is not an instruction to start implementation.

1. **Technical audit and asset contract** — inspect actual API schemas/statuses and verified reconstruction assets; choose one exhibit; document coordinate system, dimensions, supported view modes and limitations. **Accept when:** a manifest can describe the exhibit without guessed semantics and API live/precomputed distinctions are settled.
2. **Static visual composition** — build semantic page sections, grid, typography, color/material progression, spatial index, fallback and responsive content using poster/stills. **Accept when:** every storyboard section exists, is navigable and readable without WebGL.
3. **Hero camera choreography** — Entry → Transform → Depth with measured pinned scroll; named timeline labels; skip/jump behavior. **Accept when:** forward/backward scroll and direct section jumps are deterministic, resize-safe, and usable with reduced motion.
4. **Integrate actual reconstruction** — load the selected verified GLB and associated RGB/depth; establish camera framing and fallback. **Accept when:** visible result matches the approved artifact and metadata; no fabricated geometry or scale.
5. **Point-cloud transformation and crossing** — build true data-derived points and mapping; stage LODs; choreograph crossing into world. **Accept when:** recognizable structure is data-derived, scroll reverses cleanly, low-tier fallback works, and no heavy parsing occurs in the pinned sequence.
6. **Interactive room** — modes, bounded orbit/pan/zoom, reset, selection, optional metric measurement. **Accept when:** every visible control works by mouse, touch and keyboard; unsupported representations are explained/disabled; units are trustworthy.
7. **HTML technical layer and API statuses** — Engine pipeline, status/error/loading UI, Evidence and Limitations content, optional real backend submission. **Accept when:** PRECOMPUTED/LIVE are unambiguous and evidence copy matches a verified evaluation source.
8. **Remaining section transitions and Future/Project** — hard cuts and quiet end sequence. **Accept when:** no transition creates forced playback or blocks anchor navigation.
9. **Responsive implementation and optimization** — quality tiers, compression, code split, adaptive points, reduced-motion/mobile sequence. **Accept when:** supported-device matrix meets agreed frame/network targets and context-loss/error fallback works.
10. **Creative polish and release review** — only after the core data and camera work: precise labels, restrained interactions, sound opt-in, microinteraction timing. **Accept when:** visual review confirms restraint, all source/metric claims trace to evidence, and no fabricated capability remains.

### Verification matrix

- **Story/state:** every state can be entered by scroll, reversed, skipped, and deep-linked/jumped to; browser refresh and resize do not strand a pinned section.
- **Representations/data:** compare source RGB, depth, PLY and GLB coordinate framing; check axis, scale, bounds, point colors, and all advertised modes. Metric measurement tested against known distances.
- **Rendering/performance:** Windows/macOS desktop integrated GPU and discrete GPU; iOS Safari and Android Chrome representative mid-tier devices; record FPS/frame time, transfer and decode time, peak memory where available, first poster time; inspect thermal throttling.
- **Input/accessibility:** keyboard-only, screen reader labels, touch gestures, high zoom/reflow, reduced motion, no hover, no sound, WebGL disabled/context loss.
- **API:** queued/running/completed/failed/cancelled/timeout, bad upload, network drop, CORS, unavailable model/GPU, malformed response, expired export, retry and cancel behavior. No fake completion state.
- **Truth:** verify metrics against dated report and protocol; verify capability status labels; future work visually marked as future; all claims/links reviewed by project owner.

Per repo instruction, implementation tests/commands are not run as part of this planning deliverable.

## 10. Technical risks and mitigations

| Risk | Severity | Why it is hard | Mitigation / decision gate |
|---|---|---|---|
| Hero point-cloud morph has no sample correspondence | High | A visually convincing morph can be false or chaotic if RGB/depth/point coordinates don't line up. | Audit asset transforms first; preserve index correspondence or present a truthful dissolve/translation. Make a small hero prototype before story polish. |
| Current GLB may not include real surfaces/semantic objects | High | Representation switch, inspection labels, and measurements rely on data the backend may not export. | Inspect actual artifacts; expose only supported modes; generic surface inspection if semantic metadata absent; extend backend contract only when needed. |
| WebGL point count and fill rate on mobile | High | Points can overdraw, consume upload bandwidth and fragment budget, and heat phones. | Profile representative devices; deterministic LOD tiers, cap DPR, short crossing, mobile cinematic fallback. |
| Scroll, Lenis, ScrollTrigger, and R3F clocks drift | High | Multiple RAF loops and pinned layouts create lag, jumps, and camera/scroll disagreement. | Begin native scroll; if Lenis is retained, follow official synchronization, one coordinated RAF, cleanup/context ownership, test resize and iOS. |
| React/Three compatibility churn | Medium-high | R3F 10 is in alpha with WebGPU default and GLSL compatibility differences in consulted docs. | Baseline on stable R3F 9 + WebGL; lock exact versions; isolate renderer-specific code. Re-evaluate WebGPU after ecosystem release/compatibility. |
| Next.js SSR/client boundary | Medium | Three, GSAP, WebGL, window access and loaders are client-only. | Keep page and content server-renderable; dynamically import only canvas/animation providers with SSR disabled; avoid browser globals in module scope. |
| Source/backend schema differs from assumed frontend states | High | Existing job systems may expose coarse states, not live stage progress; artifact URLs may be filesystem-local. | Contract audit in milestone 1; use truthful coarse statuses; implement BFF/proxy/signed asset delivery where deployment requires. |
| Metrics and benchmark claims stale or underspecified | High | Report values need dataset/split/protocol/model/date and limitations. | Treat pasted metric examples as draft until reverified against locked report; content source with citation/provenance and review date. |
| Scroll pinning harms small screens and accessibility | Medium-high | Mobile browser chrome, keyboard focus and zoom make long pinned scenes fragile. | Pin only hero; short mobile version; reduced-motion no-pin path; nav skip always available. |
| Large GLB/PLY decode stalls interaction | Medium-high | Network transfer and main-thread parsing can block scroll and controls. | Preconvert binary, compress selectively, load before trigger, test decode time, retain poster fallback. |
| Browser context loss / GPU exhaustion | Medium | Long one-page experiences can retain buffers across sections. | Explicit disposal/quality downgrade/context-restoration UX; do not mount multiple dense canvases. |
| “Geometry” / “Hybrid” overstates current output | High | A depth-derived point cloud is not automatically a true surface mesh or hybrid digital twin. | Verify export contents, use capability status metadata, label visualization method and limitations accurately. |

## 11. Key implementation decisions

- Prefer **stable R3F 9 + WebGL** for initial production; revisit WebGPU only after R3F/Drei/custom shader dependencies have stable support.
- Prefer **native scrolling first**. Lenis is a progressive enhancement, not a foundation.
- Use **one selected real room** and derive all demonstration states from it. Other scenes appear only as labeled benchmark evidence.
- Build the cinematic narrative from **actual RGB/depth/point/GLB data**. Use pre-rendering as an explicit fallback, never as fake live behavior.
- Keep evidence and limitations as calm, accessible HTML. The technical credibility should come from precise provenance, not extra effects.
- Make the frontend an additive `web/` app beside FastAPI. Do not replace the existing prototype stack or expose the full local artifacts directory.

## 12. Sources for volatile compatibility guidance

Version guidance should be rechecked when scaffolding, because these packages change frequently. This plan was checked against primary documentation on 2026-11-11:

- [Next.js support policy](https://nextjs.org/support-policy) — Next.js 16 Active LTS and 15 Maintenance LTS at check time.
- [React Three Fiber installation / compatibility](https://r3f.docs.pmnd.rs/getting-started/installation) — v9 pairs with React 19; v8 with React 18.
- [React Three Fiber v10 migration guide](https://r3f.docs.pmnd.rs/next/migration/v10) — v10 alpha, WebGPU default, WebGL-only shader/postprocessing caveats, and Drei peer caveat at check time.
- [GSAP installation](https://gsap.com/docs/v3/Installation/) and [ScrollTrigger](https://gsap.com/docs/v3/Plugins/ScrollTrigger/) — package installation, plugin registration and scroll integration.
- [Tailwind CSS Next.js guide](https://tailwindcss.com/docs/installation/framework-guides/nextjs) — current PostCSS integration.
- [Lenis repository documentation](https://github.com/darkroomengineering/lenis) — reduced-motion behavior and GSAP ScrollTrigger integration guidance.
- [Three.js WebGLRenderer docs](https://threejs.org/docs/pages/WebGLRenderer) — pixel ratio and renderer configuration considerations.
