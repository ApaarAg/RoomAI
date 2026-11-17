"""Integration test for Hugging Face ZeroGPU A3 Space across multiple image resolutions."""

import io
import json
import requests
from PIL import Image

SPACE_URL = "https://apaarag-roomai-a3-inference.hf.space"


def test_zerogpu_resolutions():
    test_sizes = [
        ("1920x1080", (1920, 1080)),
        ("1280x720", (1280, 720)),
        ("1000x1000", (1000, 1000)),
    ]

    for label, (w, h) in test_sizes:
        img = Image.new("RGB", (w, h), color=(180, 150, 130))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # 1. Upload
        up_res = requests.post(
            f"{SPACE_URL}/gradio_api/upload",
            files={"files": (f"{label}.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        assert up_res.status_code == 200, f"Upload failed: {up_res.text}"
        uploaded_path = up_res.json()[0]

        # 2. Call detect
        call_res = requests.post(
            f"{SPACE_URL}/gradio_api/call/detect",
            json={"data": [{"path": uploaded_path, "meta": {"_type": "gradio.FileData"}}]},
            timeout=30,
        )
        assert call_res.status_code == 200, f"Call failed: {call_res.text}"
        event_id = call_res.json()["event_id"]

        # 3. Stream result and parse complete event
        stream_res = requests.get(
            f"{SPACE_URL}/gradio_api/call/detect/{event_id}",
            stream=True,
            timeout=60,
        )
        final_payload = None
        current_event = None

        for line in stream_res.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("event:"):
                current_event = decoded[6:].strip()
            elif decoded.startswith("data:"):
                data_str = decoded[5:].strip()
                if current_event == "complete":
                    final_payload = json.loads(data_str)
                    break
                elif current_event == "error":
                    err_info = json.loads(data_str) if data_str.startswith("{") else data_str
                    raise RuntimeError(f"ZeroGPU error: {err_info}")

        assert final_payload is not None, f"Did not receive final payload for {label}"
        assert len(final_payload) >= 2, f"Payload structure unexpected: {final_payload}"
        det_output = final_payload[1]
        print(f"SUCCESS: {label} -> reported dims: {det_output['image']}, latency: {det_output['inference']['latency_ms']}ms")
        assert det_output["image"]["width"] == w
        assert det_output["image"]["height"] == h


if __name__ == "__main__":
    test_zerogpu_resolutions()
