import urllib.request
from pathlib import Path

img_path = Path("web/public/assets/roomai/result-viewer/01-rgb.png")
boundary = "----WebKitFormBoundaryXYZ"
header = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="01-rgb.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode("utf-8")
footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
data = header + img_path.read_bytes() + footer

for base in ["http://127.0.0.1:8000", "http://localhost:3000"]:
    try:
        req = urllib.request.Request(
            f"{base}/api/v1/detect",
            data=data,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req) as resp:
            print(base, "SUCCESS:", resp.status, resp.read()[:100].decode("utf-8"))
    except Exception as e:
        print(base, "FAILED:", e)
