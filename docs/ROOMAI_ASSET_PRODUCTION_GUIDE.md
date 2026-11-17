# ROOMAI Asset Production Guide

**Phase:** 4 — 3D Asset Production  
**State:** production is planned, not started. This guide creates no final or placeholder model, image, texture, sound, or effect. Asset creation requires approval of the checklist at the end of this phase.  
**Project constraints:** no website implementation; no FastAPI changes; production must keep the product's actual reconstruction distinct from future concepts and precomputed media.

**Current candidate status from a read-only scan:** the repo has one verification GLB and PLY plus three uploaded images, but their pairing, matching metric-depth source, and image rights have not been confirmed. The initial production task after approval is therefore exhibit provenance/coordinate audit; it is not yet safe to label a particular file the hero scene. Do not copy the large scene-graph JSON into runtime assets.

## 1. Goal and order of work

Build an asset pipeline around one verified indoor scene. The milestone is a first playable reconstruction sequence: photograph → depth → data-derived points → camera crossing → actual reconstructed room. Do not spend time producing multiple rooms, decorative particle systems, speculative Gaussian splats, or finished evidence art before this scene works.

Recommended production order:

1. **Select and audit the exhibit.** Choose one source image and its exact matching model outputs. Confirm RGB dimensions, depth alignment, invalid pixels, depth unit/scale, coordinate axes/handedness, origin, bounds, GLB primitives, PLY count/colors, and any semantic metadata. Confirm rights to the source photo.
2. **Freeze a manifest.** Record the exhibit ID, source/model version, generation date, hashes, axis conversion, units, camera framing, export availability, and modes supported by the artifact. The manifest is the authority for all later processing.
3. **Create source-preserving review copies.** Inspect GLB/PLY in Blender; do not alter the canonical files. Record screenshots/observations in `02-working/qa/` after approval to begin production.
4. **Generate web buffers and LODs.** Convert the selected output to compact data-derived point buffers. Make deterministic mobile and desktop samples. Produce explicit correspondence only if sample mapping can be proven.
5. **Validate image/depth/point/GLB registration.** Render overlays and known-point checks in a standalone review scene; use Blender for manual inspection and a small Three.js sandbox only if coordinate behavior cannot be inspected adequately in Blender.
6. **Tune presentation materials and camera.** Set framing, light, and color with the actual reconstruction. The source photograph's lighting remains dominant in early states; the room gets only restrained architectural fill.
7. **Build the runtime effect kit.** Depth displacement, point appearance, camera rail, selection/measurement marks, and Future diagrams are code-authored. Keep them deterministic and separately reviewable.
8. **Optimize delivery copies.** Optimize GLB and image derivatives only after an unoptimized baseline renders correctly. Keep original and optimized files side-by-side and compare quality.
9. **Capture fallback and review renders.** Optional fallback sequence is captured from the real exhibit and marked `PRECOMPUTED`; it cannot impersonate active inference.
10. **Package accepted runtime assets.** Copy only reviewed runtime deliverables to the future frontend public asset directory. Keep private sources and intermediate exports out of the public bundle.

## 2. Recommended tools by asset class

| Asset / job | Primary tool | Secondary / alternative | Why / guardrail |
|---|---|---|---|
| Inspect RoomAI GLB, frame camera, check axes, remove accidental export clutter | Blender | Three.js viewer for browser-specific verification | Blender is the review/cleanup tool, not a license to replace an imperfect reconstruction with a fabricated room. Preserve canonical input and log every transform. |
| Prepare a valid surface proxy from a valid mesh GLB | Blender | glTF-Transform simplify/meshoptimizer | Only if the source genuinely contains surfaces and decimation preserves meaningful shape. Never convert an unstructured point cloud into fake watertight geometry. |
| Canonical/runtime GLB optimization | `@gltf-transform/cli` (reproducible CLI pipeline) | Blender export for authoring changes | Prune/deduplicate/resize and test Meshopt or Draco as one final geometry route. Test decoder, quality, file size, and load time in the target browser. |
| PLY inspection | Blender | CloudCompare / Open3D viewer, if already available | Visual validation only; do not edit point positions in an untracked GUI pass. Log any downsampling/transforms. |
| Point cloud runtime conversion and deterministic LOD | Small standalone Python or Node build script, outside the backend | Blender Python only for a one-off audited conversion | Preserve canonical PLY. Output binary position/color buffers plus manifest; do not make per-point scene objects. No FastAPI changes. |
| Source photo/depth visualization review | Krita, GIMP, Affinity Photo, or existing image tool | Python Pillow/OpenCV for reproducible crops/resizes | Preserve raw photo and metric depth. Image tools may create presentation derivatives only; never paint depth to improve the story. |
| Runtime depth mesh/displacement | Three.js / R3F BufferGeometry or shader | Precomputed mesh buffer for low-end fallback | GPU path avoids CPU updates per frame. Use real aligned depth and clear invalid-pixel behavior. |
| Photo plane, lights, fog, camera rails, measurements, markers, point shaders | Three.js/R3F + GSAP | Blender previs for camera composition | Production state remains controllable by scroll, reversible, and accessible. Use Three.js for runtime spatial behavior. |
| Engine pipeline and Future concepts | SVG/HTML/CSS first; Three.js lines/instancing only where spatial behavior helps | Figma/Illustrator for reference layout | Precise, lightweight, editable, and clearly labeled conceptual states. These are diagrams, not fake model outputs. |
| Optional hero fallback sequence | Blender render or deterministic Three.js capture from the approved exhibit | DaVinci Resolve for edit/encode | Same selected reconstruction; no AI-generated replacement imagery; include poster and `PRECOMPUTED` provenance. Do after live scene visual is approved. |
| Optional crossing tone | Web Audio synthesis or commissioned file mastered in Reaper/Audacity | Licensed downloaded sound with documented redistribution terms | Short, quiet, opt-in and nonessential. Browser autoplay is not assumed. |
| Fonts | Licensed self-hostable WOFF2 from chosen foundry/open font source | System font stack | Limit to three families and needed weights. Record license and subset. |
| Spline | **Not recommended for final hero data path** | Use for quick storyboard/blockout only if convenient | Spline is not the source of truth for depth correspondences, high-count points, model provenance, or deterministic camera/scroll state. Export compatibility and runtime constraints must be proven before any production use. |
| AI image generator | **No shipping use planned** | Private non-shipping composition study only, if separately requested | Product evidence, hero photo, room, and benchmarks must be actual RoomAI data. No speculative visual should resemble a finished feature. |

## 3. Visual style and asset rules

### Palette / signal semantics

| Token | Starting value | Use |
|---|---|---|
| Obsidian | `#0A0908` | Entry and opening void |
| Charcoal | `#151412` | Secondary dark field / technical transition |
| Stone | `#B9B4AA` | Neutral architectural surfaces and evidence background |
| Warm white | `#F2F0EA` | Typography and later illuminated sections |
| Deep oxide | `#8B4937` | Semantic active/selected/depth/measurement signal only |

Treat these as approved starting values from the storyboard, then calibrate against source photograph color management and real screen capture. Oxide is not a general button fill or orange glow. Keep contrast quiet but readable.

### Material language

- **Early:** photographic RGB remains authoritative; use a matte plane and minimal dark staging.
- **Depth:** preserve RGB and the photograph's own light; show depth through parallax/displacement, not generic new lighting.
- **Points:** sample colors from actual RGB; reveal sparse desaturation only through a controlled shader/material parameter.
- **World:** preserve the limits and visual character of the real reconstruction. One broad warm fill, realistic restrained shadow, no AAA material replacement.
- **Evidence / limitations:** primarily flat paper/stone colors and HTML/SVG, not a 3D material showcase.
- **Future:** abstract linework and spatial symbols; always label FUTURE / RESEARCH.

### Geometry and visual restraint

- Use one room. Do not add modeled props to hide missing reconstruction data.
- Architectural staging geometry is simple: floor plane, possible framing plane, finite camera path, sparse guide lines.
- No generic random particles, star fields, neon edges, glowing AI motifs, rainbow point palette, excessive bloom, fake volumetrics, or floating glass UI.
- Do not bake labels, metrics, typography, navigation, or accessibility instructions into textures.
- Keep object labels and sample metrics outside the model unless they come from verified model metadata.

## 4. Technical specifications and budgets

All figures below are starting budgets for review, not claims that every existing artifact meets them. Preserve canonical source files and profile target devices before locking final delivery thresholds.

| Type | Working/master format | Runtime format | Starting target |
|---|---|---|---|
| Source room image | Original as received; preserve color profile | `.avif` or `.webp`; `.jpg` fallback | 1,536–2,048 px long edge for canvas texture. Make smaller poster as needed. Preserve original outside public runtime. |
| Depth | Raw backend output plus lossless 16-bit PNG/TIFF or numeric `.npy` in private working data | 16-bit PNG or packed float/half-float binary + manifest | Match RGB pixel registration and resolution; no color LUT in master; explicit invalid value/mask and unit/scale metadata. |
| Points | Canonical `.ply` from RoomAI | Packed binary `.bin`/typed buffers; optional `.ply` download remains separate | Float32 XYZ + uint8 RGB baseline; 40k–120k desktop samples, 15k–35k mobile, stable deterministic subset. Quantize only after measured precision/perceptual tests. |
| Reconstruction | Canonical `.glb` and original `.ply` | Optimized `.glb` with required textures/decoders | Prefer ≤15 MB for initial exhibit delivery, stretch ceiling 25 MB only if measured and delayed until needed. If model is point-only, use the point buffers rather than mislabeling it as mesh. |
| Valid surface/proxy mesh | `.blend` source only if artist edits are needed | `.glb` | Aim ≤100k visible triangles desktop, ≤50k mobile proxy; 1–2 materials; no unnecessary rig or hidden objects. These limits do not apply to point-only data. |
| Textures in a genuine mesh GLB | `.png`/`.tif` source | KTX2/Basis or WebP only after cross-device verification | 1K–2K for hero scene; 512–1K for small surfaces. Avoid 4K maps unless a close inspect view justifies them. |
| Depth grid | Source depth map | Runtime-generated buffer | Start 256² desktop, 128² tablet, 64–96² mobile grid; compare edge quality and GPU cost. This is geometry sampling, not a change to depth measurement values. |
| Presentation stills | Lossless review image | `.avif`/`.webp` | 1,000–1,400 px long edge for section/poster use. |
| Optional video fallback | High-quality master encode | H.264 MP4 baseline; WebM/AV1 only if supported and useful | 8–15 s, 24/30 fps, 1080p desktop and portrait/mobile crop as needed. Keep poster-first loading and explicit PRECOMPUTED status. |
| Optional sound | 48 kHz WAV master | OGG/MP3/AAC browser fallback | 1–2 s, restrained peak, mono or narrow stereo; do not spatialize without a meaningful directional source. |
| Materials | Blender node setup only for authentic asset material work | glTF metallic-roughness standard material | 1–2 materials for a valid proxy; avoid shader complexity and texture maps where values suffice. |

**Compression rule:** establish an uncompressed baseline; compare at least one candidate geometry compression (Meshopt or Draco) and one texture route; inspect decoded output in target browser and on the target hardware. Compression is accepted only when visual fidelity, decode time, loading time, and bundle support improve as a whole. Keep the compression/transcode pipeline reproducible and keep source data untouched.

## 5. Naming convention

Use lowercase ASCII, hyphen-separated tokens. Include exhibit and role, representation/quality where relevant, version, and extension. Do not embed dates in the asset filename unless the date is part of evidence identity; record generation date in the manifest.

```text
roomai-{exhibit}-{asset}-{representation}-{tier}-v{major}.{minor}.{ext}
```

Examples (examples only; they are **not files created by this phase**):

```text
roomai-exhibit-01-source-rgb-master-v1.0.tif
roomai-exhibit-01-depth-metric-master-v1.0.png
roomai-exhibit-01-points-rgb-desktop-v1.0.bin
roomai-exhibit-01-points-rgb-mobile-v1.0.bin
roomai-exhibit-01-scene-source-v1.0.glb
roomai-exhibit-01-scene-web-desktop-v1.0.glb
roomai-exhibit-01-manifest-v1.0.json
roomai-exhibit-01-poster-precomputed-v1.0.avif
roomai-transition-crossing-tone-v1.0.ogg
```

Version increments: patch = metadata/copy/encoding fix without perceptible geometry change; minor = new quality tier or compatible representation addition; major = coordinate, unit, source, topology, or schema change. The manifest references exact file hashes and a manifest schema version.

## 6. Storage structure

The empty folder tree is created at `asset-production/roomai/`. It separates non-public originals and working files from reviewed runtime exports. The intended future frontend delivery folder is shown but is not populated in this phase.

```text
asset-production/
└── roomai/
    ├── 00-intake-private/          # originals; never serve from web
    │   ├── rgb/
    │   ├── depth/
    │   ├── exports-glb/
    │   ├── exports-ply/
    │   ├── source-manifests/
    │   └── licenses-and-consent/
    ├── 01-review/                  # approved exhibit selection and audit records
    │   ├── exhibit-01/
    │   ├── coordinate-audits/
    │   └── provenance/
    ├── 02-working/                 # non-destructive working artifacts
    │   ├── blender/
    │   ├── image-derivatives/
    │   ├── point-conversion/
    │   ├── depth-validation/
    │   ├── camera-previs/
    │   └── qa/
    ├── 03-masters/                 # approved high-quality derivative masters
    │   ├── exhibit-01/
    │   │   ├── rgb/
    │   │   ├── depth/
    │   │   ├── points/
    │   │   ├── glb/
    │   │   ├── correspondence/
    │   │   └── manifest/
    │   ├── evidence/
    │   ├── fallback-sequence/
    │   └── audio/
    ├── 04-runtime/                 # only optimized, accepted outputs
    │   ├── exhibit-01/
    │   │   ├── textures/
    │   │   ├── depth/
    │   │   ├── points/
    │   │   ├── glb/
    │   │   ├── posters/
    │   │   └── manifest/
    │   ├── evidence/
    │   ├── fallback/
    │   ├── audio/
    │   └── fonts/
    ├── 05-license-ledger/
    └── 06-release-records/

web/public/assets/roomai/           # future frontend copy destination; not populated now
├── exhibit-01/
├── evidence/
├── fallback/
├── audio/
├── fonts/
└── manifests/
```

`00-intake-private` can contain sensitive or licensed originals; keep it out of static serving. The repo's existing `artifacts/jobs/` remains untouched. Only copy curated, approved, optimized assets into the future public folder. If the selected source must not be published, use approved artifact delivery rather than public/static storage.

## 7. Production acceptance gates

### Gate A — Exhibit selected

- Source image and all output artifacts are paired to the same run/model.
- Source rights are known and public display is approved.
- Axis direction, scale, units, coordinate origin, bounds, depth validity, RGB-depth registration, and GLB/PLY contents are understood.
- Every planned representation is marked supported or unavailable.

### Gate B — Transformation fidelity

- Depth displacement corresponds to the selected metric/estimated depth and preserves the image.
- Point samples visually recover recognizable furniture, wall, and floor structure; positions are product-derived, not random.
- A morph is enabled only if sample correspondence is valid. Otherwise transition strategy is a transparent crossfade/translation.
- Crossing camera remains within a known safe path and arrives at the same coordinate-aligned reconstruction.

### Gate C — Visual system

- Obsidian → charcoal → stone → warm white palette progression is consistent.
- Oxide is reserved for meaningful active/selected/measurement states.
- Architectural light, controlled shadow and matte/material restraint are preserved.
- No unsupported visual claim, generic AI motif, or decorative particle layer is added.

### Gate D — Delivery and reliability

- Asset hash, manifest, license, file size, decode time and supported tier are recorded.
- Desktop, tablet, mobile and reduced-motion variants use matching coordinate systems and source identity.
- Optimized output has been compared with canonical output in-browser.
- Context loss, missing asset, unsupported mode, and poster fallback are understood.

## 8. Expensive/risky items and faithful alternatives

| Risky item | Cost/risk | Preferred alternative that keeps the concept intact |
|---|---|---|
| Per-pixel photograph → point-cloud physical morph | Needs stable pixel/sample correspondence and can create visual noise or incorrect implied geometry | Derive deterministic correspondence from aligned RGB/depth. If impossible, stage a precise image-plane separation then dissolve/crossfade into spatially faithful points; label the point state as a representation, not a literal pixel trajectory. |
| Dense 40k+ point cloud on mobile | Fill-rate, memory, thermal limits, high DPR multiplies fragment work | Deterministic 15k–35k mobile sample, lower DPR, shorter crossing; retain recognizable shape and use pre-rendered sequence only as fallback. |
| Full mesh/digital twin from sparse/point output | Often impossible without inventing surfaces; can misrepresent model quality | Use points as the actual spatial representation. Only show `GEOMETRY` if the real export contains valid surface geometry; otherwise disable with clear status. |
| Gaussian splat animation/asset | Stage 2 3DGS is not integrated in current prototype UI | Future diagram only, labelled research/future. Do not fabricate splat output. |
| Real volumetric fog/postprocessing | Screen-space cost, mobile heat, stylistic drift toward game/cyberpunk | Use background gradient, restrained scene fog or a single light falloff approximation. Let actual point geometry carry the spectacle. |
| Long pinned desktop hero on mobile | Browser chrome, touch scrolling, accessibility focus, dropped frame risk | Shorten the camera crossing; native scroll; portrait crop; reduced-motion/fallback path. |
| Multiple canvases/full scene across every section | GPU/context memory and scene-management overhead | One canvas, separate scene groups, load/unload at meaningful boundaries; static HTML/SVG for Evidence/Future/Project. |
| Soundscape/spatial audio | Autoplay and user-control/accessibility burden | One opt-in 1–2 s tonal cue at crossing or no sound; visual transition remains complete without it. |
| Blender-authored room to clean up noisy output | May improve aesthetics by hiding the real limitation | Preserve a source GLB, disclose any derived cleanup, use direct output as product proof. Do not add guessed walls/furniture. |

## 9. Approval checklist before asset creation

Approval is requested for the **production scope and first asset batch**, not for website implementation. Once approved, Phase 4 asset creation can start with the exhibit audit and P0 bundle. This plan does not presume that generated assets should be used if source provenance, rights, or supported representations fail review.

- [ ] Approve one real RoomAI exhibit as the hero source (RGB + matching depth + PLY/point data + GLB).
- [ ] Approve the P0 first playable scope: image/depth/points/crossing/real room, desktop first and mobile quality tier.
- [ ] Approve style/material rules and the no-fake-geometry/no-fake-live/no-AI-hero constraints.
- [ ] Approve target output formats and starting budgets; final file sizes/point budgets will be tuned by device profiling.
- [ ] Approve optional work order after P0: evidence imagery, future diagrams, fallback sequence, fonts, then optional sound/contours.

**No models, textures, images, audio, or runtime effects have been authored by creating this guide or the inventory.**

## 10. Technical references

- [Khronos glTF 2.0 registry](https://registry.khronos.org/glTF/) — glTF is the runtime delivery format; GLB packages scene JSON and binary buffers in one container.
- [Khronos glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) — format, PBR materials, extensions, and GLB details.
- [Blender glTF 2.0 exporter manual](https://docs.blender.org/manual/en/3.0/addons/import_export/scene_gltf2.html) — Blender supports `.glb` output; consult the manual matching the installed Blender release before production.
- [glTF Transform](https://github.com/donmccurdy/glTF-Transform) — reproducible GLB optimization, geometry compression, and texture processing options.

Tool versions and browser compression support should be checked and locked when production starts. This guide avoids declaring a version as production-approved before the actual toolchain and target browsers are selected.
