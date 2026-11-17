// Pinned release gate using the official Khronos glTF Validator npm package.
// Install reproducibly with npm ci --prefix validation/gltf-validator.
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const file = process.argv[2];
if (!file) {
  console.error('Usage: node scripts/validate_glb_khronos.mjs <asset.glb>');
  process.exit(2);
}

let validator;
try {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const require = createRequire(resolve(scriptDir, '../validation/gltf-validator/package.json'));
  validator = require('gltf-validator');
} catch (error) {
  console.error('Pinned Khronos validator dependency is unavailable. Run npm ci --prefix validation/gltf-validator.');
  process.exit(2);
}
const scriptDir = dirname(fileURLToPath(import.meta.url));
const validatorPackage = JSON.parse(await readFile(
  resolve(scriptDir, '../validation/gltf-validator/node_modules/gltf-validator/package.json'), 'utf8',
));

const bytes = await readFile(file);
const report = await validator.validateBytes(new Uint8Array(bytes), {
  uri: file,
  format: 'glb',
  externalResourceFunction: async () => { throw new Error('Unexpected external resource in standalone GLB'); },
});
console.log(JSON.stringify({
  uri: file,
  validatorVersion: validatorPackage.version,
  assetInfo: report.info ?? null,
  errors: report.issues.numErrors,
  warnings: report.issues.numWarnings,
  infos: report.issues.numInfos,
  hints: report.issues.numHints,
  issues: report.issues.messages,
}, null, 2));
if (report.issues.numErrors > 0) process.exitCode = 1;
