# RoomAI Frontend Handoff — Backend API Integration Guide

## Overview

This document provides everything a frontend developer needs to integrate with the RoomAI Backend API. The backend is production-ready and serves as the complete 3D reconstruction backend.

## Quick Start

### Backend Startup
```bash
cd F:\RoomAI
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/api/v1/openapi.json

### Health Check
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## Integration Flow

```
1. Upload Images     → POST /api/v1/uploads (×N)
2. Submit Job        → POST /api/v1/jobs/reconstruct
3. Poll Status       → GET /api/v1/jobs/{job_id} (every 1-2s)
4. Get Results       → GET /api/v1/jobs/{job_id}/results
5. Download Assets   → GET /api/v1/jobs/{job_id}/exports/glb|ply
```

---

## Step-by-Step Examples

### 1. Upload Images

```javascript
async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/v1/uploads', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }
  
  return response.json(); // { upload_id, width, height, ... }
}

// Usage
const uploads = await Promise.all(
  fileList.map(file => uploadImage(file))
);
// uploads = [{ upload_id: "upl_abc123", ... }, ...]
```

### 2. Submit Reconstruction Job

```javascript
async function submitReconstruction(uploadIds, options = {}) {
  const response = await fetch('http://localhost:8000/api/v1/jobs/reconstruct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      upload_ids: uploadIds,
      model_id: options.model_id || 'roomai_composite_v1',
      refine_depth: options.refine_depth !== false,
      downsample_points: options.downsample_points || 40000,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }
  
  return response.json(); // { job_id, poll_url, ... }
}

// Usage
const job = await submitReconstruction(
  uploads.map(u => u.upload_id),
  { model_id: 'worldmirror1_lora_r8' }
);
// job = { job_id: "job_1731234567_abc123", poll_url: "/api/v1/jobs/...", ... }
```

### 3. Poll Job Status

```javascript
async function pollJobStatus(jobId, onProgress) {
  const response = await fetch(`http://localhost:8000/api/v1/jobs/${jobId}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }
  
  const status = await response.json();
  
  // status = {
  //   job_id, status, stage, progress (0-1),
  //   timing, error, warnings, created_at, updated_at
  // }
  
  if (onProgress) onProgress(status);
  
  if (status.status === 'COMPLETED') return status;
  if (status.status === 'FAILED') throw new Error(status.error?.message || 'Job failed');
  if (status.status === 'CANCELLED') throw new Error('Job cancelled');
  
  // Continue polling
  await new Promise(r => setTimeout(r, 1000));
  return pollJobStatus(jobId, onProgress);
}

// Usage
const completedJob = await pollJobStatus(job.job_id, (status) => {
  console.log(`${status.stage} - ${Math.round(status.progress * 100)}%`);
  updateProgressBar(status.progress);
});
```

### 4. Get Results & Download Assets

```javascript
async function getJobResults(jobId) {
  const response = await fetch(`http://localhost:8000/api/v1/jobs/${jobId}/results`);
  return response.json();
}

async function downloadExport(jobId, format) {
  // format: 'glb' or 'ply'
  const response = await fetch(`http://localhost:8000/api/v1/jobs/${jobId}/exports/${format}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }
  
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  
  // Trigger download
  const a = document.createElement('a');
  a.href = url;
  a.download = `roomai_${jobId}.${format}`;
  a.click();
  URL.revokeObjectURL(url);
}

// Usage
const results = await getJobResults(job.job_id);
// results.scene_graph has full 3D data
// results.artifacts has export info

await downloadExport(job.job_id, 'glb');
await downloadExport(job.job_id, 'ply');
```

---

## Complete Integration Example (React)

```jsx
import { useState, useCallback } from 'react';

function RoomAIReconstruction() {
  const [uploads, setUploads] = useState([]);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const handleUpload = async (files) => {
    try {
      const uploaded = await Promise.all(
        Array.from(files).map(uploadImage)
      );
      setUploads(uploaded);
      setError(null);
    } catch (e) { setError(e.message); }
  };
  
  const handleReconstruct = async () => {
    if (uploads.length === 0) return;
    try {
      const newJob = await submitReconstruction(
        uploads.map(u => u.upload_id),
        { model_id: 'worldmirror1_lora_r8' }
      );
      setJob(newJob);
      setError(null);
      await pollJobStatus(newJob.job_id, setStatus);
      const res = await getJobResults(newJob.job_id);
      setResults(res);
    } catch (e) { setError(e.message); }
  };
  
  return (
    <div>
      <input type="file" multiple accept="image/*" onChange={e => handleUpload(e.target.files)} />
      {uploads.map(u => <img key={u.upload_id} src={URL.createObjectURL(u.file)} width={100} />)}
      
      <button onClick={handleReconstruct} disabled={!uploads.length}>
        Reconstruct 3D
      </button>
      
      {status && <div>Stage: {status.stage} - {Math.round(status.progress * 100)}%</div>}
      {error && <div style={{color: 'red'}}>Error: {error}</div>}
      
      {results && (
        <div>
          <h3>Complete! Points: {results.results_summary?.valid_points}</h3>
          <button onClick={() => downloadExport(job.job_id, 'glb')}>Download GLB</button>
          <button onClick={() => downloadExport(job.job_id, 'ply')}>Download PLY</button>
        </div>
      )}
    </div>
  );
}
```

---

## Model Selection Guide

| Model ID | Use Case | Views | Features |
|----------|----------|-------|----------|
| `roomai_composite_v1` | Production default | 1 | YOLO + Depth + Fusion |
| `worldmirror1_baseline` | Reference/rollback | 1-6 | Pretrained Hunyuan |
| `worldmirror1_lora_r8` | Best geometry (candidate) | 1-6 | Fine-tuned heads |
| `worldmirror1_stage2` | Experimental 3DGS | 2-6 | Gaussian Splatting |

**Recommendation**: Use `roomai_composite_v1` for production. Use `worldmirror1_lora_r8` for best geometric quality (validation candidate).

---

## Error Handling

All errors follow this format:
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human readable message",
    "remediation": "Suggested fix"
  },
  "timestamp": 1234567890.123
}
```

**Common Codes**:
- `VALIDATION_ERROR` (400): Bad request
- `NOT_FOUND` (404): Resource missing
- `SERVICE_UNAVAILABLE` (503): Model not ready
- `GPU_ERROR` (500): OOM - reduce resolution/views
- `JOB_CANCELLED` (499): User cancelled

---

## Rate Limits & Performance

- **Concurrent jobs**: 1 (RTX 4060 8GB constraint)
- **Typical latency**: 15-30s per job (2 views)
- **Max upload**: 20MB, 8192px, 100MP
- **Polling interval**: 1-2 seconds recommended

---

## WebSocket / Real-time (Future)

Currently: **Polling only**. WebSocket support planned for v2.

---

## Testing Endpoints

```bash
# Health
curl http://localhost:8000/health

# Models
curl http://localhost:8000/api/v1/models

# Upload test
curl -F "file=@test.jpg" http://localhost:8000/api/v1/uploads

# Full flow (see test_e2e_api_workflow.py)
```

---

## OpenAPI / TypeScript Types

Generate types from OpenAPI spec:
```bash
npx openapi-typescript http://localhost:8000/api/v1/openapi.json -o types/api.ts
```

Or use the provided `API_CONTRACT.md` for manual typing.

---

## Support

- Backend logs: Structured JSON with `trace_id`, `job_id`, `model_id`
- Metrics: `GET /metrics` (Prometheus)
- Issues: Check job `error` field and server logs