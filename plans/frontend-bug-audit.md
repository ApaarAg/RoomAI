# ROOMAI Frontend Bug Audit — Independent QA Report

**Date**: 2026-11-16  
**Scope**: `web/` — Next.js 15, React 19, R3F 9.8, Three.js 0.187, GSAP 3.15  
**Methodology**: Static code analysis (read-only). No browser automation available. No `execute_command` access — `git status`, `tsc`, and `next dev` could not be run.

---

## CONFIRMED BUGS

### CRITICAL

| # | Bug | File(s) | Evidence | Recommended Fix |
|---|-----|---------|----------|-----------------|
| C1 | **Nested `NarrativeController` — double scroll listeners & RAF loops** | [`web/app/page.tsx`](web/app/page.tsx:12) + [`web/components/3d/SceneCanvas.tsx`](web/components/3d/SceneCanvas.tsx:64) | `page.tsx` wraps the entire page in `<NarrativeController>`. `SceneCanvas.tsx` creates a **second** `<NarrativeController>` inside the Canvas. This produces two independent `window.addEventListener("scroll", ...)` listeners, two `requestAnimationFrame` lerp loops, and two React context providers. The inner context shadows the outer one, so `useNarrative()` inside the Canvas returns the inner state while the Hero uses the outer state. | Remove the inner `<NarrativeController>` wrapper from `SceneCanvas.tsx`. The Canvas already receives narrative state via the outer context. Use `useNarrative()` inside `SceneCanvas` instead of creating a new provider. |
| C2 | **Point cloud asset URL 404** | [`web/PointCloudObject.tsx`](web/PointCloudObject.tsx:13) | `POINT_CLOUD_URL = "/assets/scene_reconstruction.ply"` resolves to `public/assets/scene_reconstruction.ply` which **does not exist**. The actual file is at `public/assets/roomai/exhibit-01/point-cloud/scene_reconstruction.ply`. | Change URL to `"/assets/roomai/exhibit-01/point-cloud/scene_reconstruction.ply"`. |

### HIGH

| # | Bug | File(s) | Evidence | Recommended Fix |
|---|-----|---------|----------|-----------------|
| H1 | **Root-level orphan `.tsx` files create ambiguous import surface** | [`web/RepresentationController.tsx`](web/RepresentationController.tsx), [`web/WorldObject.tsx`](web/WorldObject.tsx), [`web/SpatialStructureObject.tsx`](web/SpatialStructureObject.tsx), [`web/SpatialAnnotationObject.tsx`](web/SpatialAnnotationObject.tsx), [`web/PointCloudObject.tsx`](web/PointCloudObject.tsx) | Five `.tsx` files live at the `web/` root. The `@/*` path alias maps to `./*` (web root), so `@/RepresentationController` resolves to the root file. Meanwhile, `web/components/3d/PointCloudObject.tsx` is a **separate** file imported via relative path `"./PointCloudObject"` from `SceneCanvas.tsx`. This creates two `PointCloudObject` components with different implementations. The root-level files are imported by `SceneCanvas.tsx` via `@/` aliases, but their placement at root is unconventional and risks accidental imports. | Move root-level `.tsx` files into `web/components/3d/` and update all `@/` imports to `@/components/3d/`. Delete the duplicate `web/components/3d/PointCloudObject.tsx` if the root version is canonical, or vice versa. |
| H2 | **`PhotographObject` effect re-runs on callback identity changes** | [`web/components/3d/PhotographObject.tsx`](web/components/3d/PhotographObject.tsx:46) | The texture-loading `useEffect` includes `onReady`, `onError`, `onTextureLoaded` in its dependency array. If a parent passes inline arrow functions, the texture reloads on every render. Currently safe because `SceneCanvas` and `DepthExhibitCanvas` use `useCallback` with `[]` deps, but this is fragile. | Use refs for callbacks (`onReadyRef.current = onReady`) and exclude them from the dependency array, matching the pattern used in `PointCloudObject.tsx` (line 100-101 comment). |
| H3 | **`ResultViewer` — unused `mobileQuery` variable** | [`web/components/sections/ResultViewer.tsx`](web/components/sections/ResultViewer.tsx:71) | `const mobileQuery = window.matchMedia("(max-width: 767px)")` is created but never used. The resize handler on line 225 checks `window.innerWidth < 768` directly instead. Dead code. | Remove the unused `mobileQuery` variable. |

### MEDIUM

| # | Bug | File(s) | Evidence | Recommended Fix |
|---|-----|---------|----------|-----------------|
| M1 | **`scroll-padding-top` doesn't adapt to mobile nav height** | [`web/app/globals.css`](web/app/globals.css:19) | `html { scroll-padding-top: 90px }` is fixed. On mobile (≤767px), the nav height increases to 104px (line 223: `min-height: 104px`). Section anchors will be partially hidden behind the taller nav on mobile. | Add a media query: `@media (max-width: 767px) { html { scroll-padding-top: 112px; } }` |
| M2 | **No app-level error boundary** | [`web/app/layout.tsx`](web/app/layout.tsx) | If a Three.js component throws (e.g., WebGL context loss, shader compilation failure), the entire page crashes with no fallback UI. Only `WorldObject` has an internal error boundary. | Add an `error.tsx` file at `web/app/error.tsx` per Next.js conventions, or wrap the body in a custom error boundary. |
| M3 | **`DepthFieldObject` `useFrame` with `frameloop="demand"` — potential infinite micro-loop** | [`web/components/3d/DepthFieldObject.tsx`](web/components/3d/DepthFieldObject.tsx:165) | The `useFrame` tilt animation calls `invalidate()` when rotation diff > 0.0001. With `frameloop="demand"`, this creates a self-sustaining loop until the rotation settles. The threshold should prevent infinite loops, but floating-point precision on very small deltas could cause indefinite micro-animation. | Add a maximum frame count or use `THREE.MathUtils.damp` with a slightly larger epsilon (e.g., 0.001). |
| M4 | **`gsap.registerPlugin(ScrollTrigger)` at module level** | [`web/components/sections/PipelineExperience.tsx`](web/components/sections/PipelineExperience.tsx:9), [`web/components/sections/ResultViewer.tsx`](web/components/sections/ResultViewer.tsx:13) | `ScrollTrigger` requires `window`/`document`. While these are `"use client"` components, if they're ever imported in a server context (e.g., by a parent server component), it will throw. Currently safe because all parents are client components. | Move `gsap.registerPlugin(ScrollTrigger)` inside a `useEffect` or a lazy initialization guard: `if (typeof window !== "undefined") gsap.registerPlugin(ScrollTrigger)`. |
| M5 | **`ResultViewer` — `ScrollTrigger` missing `scrub` property** | [`web/components/sections/ResultViewer.tsx`](web/components/sections/ResultViewer.tsx:174) | The `ScrollTrigger.create()` on line 174 does not include a `scrub` property. The `onUpdate` callback manually implements scrubbing via `gsap.quickTo`, which works but is more complex than using the built-in `scrub: 0.34`. The `PipelineExperience` uses `scrub: 0.35` directly. Inconsistency between the two. | Consider using `scrub: 0.34` directly for consistency with `PipelineExperience`, or document why the manual approach is needed. |

### LOW

| # | Bug | File(s) | Evidence | Recommended Fix |
|---|-----|---------|----------|-----------------|
| L1 | **24 images preloaded on page mount** | [`web/components/hero/NarrativeFrameSequence.tsx`](web/components/hero/NarrativeFrameSequence.tsx:57), [`web/components/sections/PipelineExperience.tsx`](web/components/sections/PipelineExperience.tsx:36), [`web/components/sections/ResultViewer.tsx`](web/components/sections/ResultViewer.tsx:48) | 12 sequence frames + 6 pipeline images + 6 result images = 24 images preloaded via `new Image()` on mount. This causes bandwidth contention on load. | Stagger preloading with `requestIdleCallback` or prioritize only above-the-fold images. |
| L2 | **`will-change` on 12 simultaneous elements** | [`web/app/globals.css`](web/app/globals.css:299) | `.sequence-frame { will-change: opacity, transform }` applies to all 12 frames simultaneously. `will-change` creates a GPU compositor layer per element, consuming VRAM. On low-end devices, this can cause performance degradation. | Apply `will-change` only to the 2-3 currently visible frames via inline styles, not via CSS class. |
| L3 | **Google Fonts loaded via `<link>` instead of `next/font`** | [`web/app/layout.tsx`](web/app/layout.tsx:14-16) | Fonts are loaded via `<link>` tags in `<head>`. This bypasses Next.js font optimization (automatic `font-display: swap`, subsetting, no FOUT). | Migrate to `next/font/google` for DM Sans, DM Mono, and Instrument Serif. |
| L4 | **`SpatialAnnotationObject` — `dispose={null}` is non-standard** | [`web/components/3d/SpatialAnnotationObject.tsx`](web/SpatialAnnotationObject.tsx:241) | `<mesh dispose={null}>` and `<lineSegments dispose={null}>` tell R3F not to auto-dispose. This is correct since the component manages disposal manually, but it's an unusual pattern that could confuse maintainers. | Add a comment explaining why `dispose={null}` is used, or switch to R3F's `attach` pattern. |
| L5 | **`WorldObject` — `useFrame` runs even when `frameloop="demand"`** | [`web/WorldObject.tsx`](web/WorldObject.tsx:245) | The `useFrame` callback updates the world-space clip plane. With `frameloop="demand"`, this only runs when something invalidates. If the parent group moves without calling `invalidate()`, the clip plane will be stale. Currently safe because `CameraController` calls `invalidate()` on progress changes. | Add an early return if `!group.visible` to skip matrix updates when hidden. |

---

## POTENTIAL BUGS (Unconfirmed — requires runtime verification)

| # | Concern | File(s) | What to Check |
|---|---------|---------|---------------|
| P1 | **Hydration mismatch from `useState(false)` for WebGL** | [`web/components/3d/SceneCanvas.tsx`](web/components/3d/SceneCanvas.tsx:28), [`web/components/3d/DepthExhibitCanvas.tsx`](web/components/3d/DepthExhibitCanvas.tsx:26) | `webglAvailable` starts as `false` and is set to `true` in a `useEffect`. The server renders the `<img>` fallback. If the client also has WebGL, it renders the `<Canvas>`. This should not cause a hydration mismatch because the `<img>` is always rendered (just hidden via CSS class). Verify in browser console. |
| P2 | **`ResultViewer` — `ScrollTrigger.refresh()` timing** | [`web/components/sections/ResultViewer.tsx`](web/components/sections/ResultViewer.tsx:240) | `ScrollTrigger.refresh()` is called inside `requestAnimationFrame` in the cleanup. If the component unmounts before the RAF fires, `ScrollTrigger.refresh()` runs on a destroyed component. Verify no console errors on fast navigation. |
| P3 | **`PipelineExperience` — `ScrollTrigger.refresh()` same pattern** | [`web/components/sections/PipelineExperience.tsx`](web/components/sections/PipelineExperience.tsx:240) | Same RAF + `ScrollTrigger.refresh()` pattern as `ResultViewer`. |
| P4 | **`DevelopmentPointField` — canvas 2D context for color sampling** | [`web/components/3d/dev/DevelopmentPointField.tsx`](web/components/3d/dev/DevelopmentPointField.tsx:31) | Creates a `<canvas>` element and draws the RGB texture to sample pixel colors. The canvas is never removed from memory explicitly (though it should be GC'd). Verify no memory leak on repeated mounts. |
| P5 | **`DepthFieldObject` — `DataTexture` disposal on abort** | [`web/components/3d/DepthFieldObject.tsx`](web/components/3d/DepthFieldObject.tsx:83) | On cleanup, `ownedTextures.current?.depth.dispose()` is called. But if `loadDepthTexture` succeeds and `active` is already `false`, the depth texture is disposed in the `.then()` callback (line 68). This double-dispose could throw. Verify no "already disposed" warnings. |

---

## PERFORMANCE OPPORTUNITIES

| # | Opportunity | Impact | Recommendation |
|---|------------|--------|----------------|
| PF1 | **`next/font` for font loading** | Eliminates FOUT, reduces layout shift | Migrate Google Fonts `<link>` tags to `next/font/google` |
| PF2 | **Staggered image preloading** | Reduces bandwidth contention on load | Use `requestIdleCallback` or priority hints for below-fold images |
| PF3 | **Conditional `will-change`** | Reduces GPU memory on low-end devices | Apply `will-change` only to active frames via JS, not CSS |
| PF4 | **`useMemo` for `stageAnnotations` array** | Avoids recreation on every render | `NarrativeFrameSequence.tsx` line 16: wrap `stageAnnotations` in `useMemo` or move outside component |
| PF5 | **`dpr={[1, 1.5]}` on Canvas** | Good — caps pixel ratio at 1.5x | Already well-configured. Consider `[1, 1]` on mobile for battery savings |

---

## THREE.JS / R3F DISPOSAL AUDIT

| Component | Geometries Disposed | Materials Disposed | Textures Disposed | Status |
|-----------|--------------------|--------------------|--------------------|--------|
| [`SpatialAnnotationObject`](web/SpatialAnnotationObject.tsx) | ✅ Lines 128-130 | ✅ Lines 131-133 | N/A | **PASS** |
| [`DevelopmentPointField`](web/components/3d/dev/DevelopmentPointField.tsx) | ✅ Line 79 | ✅ Line 80 | N/A | **PASS** |
| [`DevelopmentSpatialStructure`](web/components/3d/dev/DevelopmentSpatialStructure.tsx) | ✅ Lines 33-34 | ✅ Lines 35-36 | N/A | **PASS** |
| [`DevelopmentWorld`](web/components/3d/dev/DevelopmentWorld.tsx) | ✅ Line 29 | ✅ Line 30 | N/A | **PASS** |
| [`DepthFieldObject`](web/components/3d/DepthFieldObject.tsx) | N/A (JSX) | ✅ Line 153 | ✅ Lines 68, 83 | **PASS** |
| [`PhotographObject`](web/components/3d/PhotographObject.tsx) | N/A (JSX) | N/A (JSX) | ✅ Lines 28, 43 | **PASS** |
| [`PointCloudObject`](web/PointCloudObject.tsx) | ✅ Lines 63, 73, 97 | ✅ Lines 98 | N/A | **PASS** |
| [`WorldObject`](web/WorldObject.tsx) | N/A (useGLTF cached) | Cloned per instance | N/A | **PASS** (clones managed by R3F) |

**Verdict**: Three.js resource disposal is **well-handled** across all components. No geometry, material, or texture leaks detected.

---

## GSAP / SCROLLTRIGGER CLEANUP AUDIT

| Component | ScrollTrigger Killed | Timeline Killed | Event Listeners Removed | Status |
|-----------|---------------------|-----------------|------------------------|--------|
| [`PipelineExperience`](web/components/sections/PipelineExperience.tsx) | ✅ Line 245 | ✅ Line 246 | ✅ Lines 243-244 | **PASS** |
| [`ResultViewer`](web/components/sections/ResultViewer.tsx) | ✅ Line 159 (rebuild), line 240 (cleanup) | ✅ Lines 160, 185 | ✅ Lines 225-228 | **PASS** |
| [`NarrativeController`](web/components/3d/NarrativeController.tsx) | N/A (no GSAP) | N/A | ✅ Lines 140-145 | **PASS** |

**Verdict**: GSAP/ScrollTrigger cleanup is **correct** in all components. No leaked listeners or timelines.

---

## LIMITATIONS OF THIS AUDIT

1. **No browser access** — Cannot verify runtime behavior, console errors, network requests, or visual rendering.
2. **No `execute_command`** — Cannot run `git status`, `tsc --noEmit`, `next build`, or `next lint`.
3. **No WebGL context inspection** — Cannot verify GPU memory usage, shader compilation, or texture upload performance.
4. **No network waterfall analysis** — Cannot verify asset loading order, caching headers, or bundle sizes.
5. **Static analysis only** — Some bugs (hydration, race conditions, memory leaks) require runtime observation.

---

## SUMMARY

| Category | Count |
|----------|-------|
| Critical | 2 |
| High | 3 |
| Medium | 5 |
| Low | 5 |
| Potential (unconfirmed) | 5 |
| Performance opportunities | 5 |

**Overall assessment**: The codebase is well-structured with strong disposal discipline and correct GSAP cleanup. The two critical bugs (nested `NarrativeController` and broken point cloud URL) should be addressed immediately. The root-level orphan files are a code organization concern that will cause confusion as the project grows. Three.js resource management is exemplary — all geometries, materials, and textures are properly disposed.