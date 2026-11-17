# ROOMAI static exhibition

`web/` is the Phase 1 static website. It is a component-based Next.js and TypeScript presentation of the ROOMAI system, its evaluation, current limits, and future work. The site uses no runtime reconstruction API, Three.js, React Three Fiber, WebGL, or scene animation.

## Run locally

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`.

## Structure

- `app/`: document metadata and global design tokens / responsive styles.
- `components/navigation/`: anchor navigation and current section indicator.
- `components/hero/`: opening exhibition composition.
- `components/sections/`: transform, engine, explore, evidence, future, and project sections.
- `components/ui/`: shared technical labels, metrics, pipeline, and architectural elevation plate.
- `data/siteContent.ts`: navigation, pipeline, evaluation values, and future directions.

Static architectural drawings are clearly labeled placeholders. The view-mode labels are non-functional by design; 3D interaction and reconstruction assets belong to Phase 2.
