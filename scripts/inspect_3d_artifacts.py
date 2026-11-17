"""Detailed GLB and PLY Artifact Inspection Script."""

import json
from pathlib import Path
import struct
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GLB_PATH = REPO_ROOT / "artifacts" / "final_backend_verification" / "reconstructed_scene.glb"
PLY_PATH = REPO_ROOT / "artifacts" / "final_backend_verification" / "reconstructed_pointcloud.ply"
JSON_PATH = REPO_ROOT / "artifacts" / "final_backend_verification" / "reconstructed_scene_graph.json"


def inspect():
    print("=" * 70)
    print("1. INSPECTING GLB CONTAINER")
    print("=" * 70)
    with open(GLB_PATH, "rb") as f:
        glb_data = f.read()

    magic, version, length = struct.unpack("<4sII", glb_data[:12])
    print(f"GLB Magic: {magic.decode('ascii')}, Version: {version}, Declared Length: {length:,} bytes (Actual: {len(glb_data):,} bytes)")

    c0_len, c0_type = struct.unpack("<I4s", glb_data[12:20])
    print(f"Chunk 0 Length: {c0_len:,} bytes, Type: {c0_type.decode('ascii')}")
    json_bytes = glb_data[20 : 20 + c0_len]
    gltf = json.loads(json_bytes.decode("utf-8"))

    print("\nglTF JSON Manifest:")
    print("  Asset:", gltf.get("asset"))
    print("  Scene Default:", gltf.get("scene"))
    print("  Scenes:", gltf.get("scenes"))
    nodes = gltf.get("nodes", [])
    print(f"  Nodes ({len(nodes)}):")
    for i, n in enumerate(nodes):
        print(f"    Node {i}: name='{n.get('name')}', mesh_index={n.get('mesh')}, translation={n.get('translation')}, rotation={n.get('rotation')}, matrix={n.get('matrix')}")

    meshes = gltf.get("meshes", [])
    print(f"\n  Meshes ({len(meshes)}):")
    for i, m in enumerate(meshes):
        print(f"    Mesh {i}: name='{m.get('name')}'")
        for p_idx, p in enumerate(m.get("primitives", [])):
            mode = p.get("mode", 4)
            mode_map = {0: "POINTS", 1: "LINES", 2: "LINE_LOOP", 3: "LINE_STRIP", 4: "TRIANGLES"}
            print(f"      Primitive {p_idx}: mode={mode} ({mode_map.get(mode, 'UNKNOWN')}), attributes={p.get('attributes')}, indices={p.get('indices')}")

    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    print(f"\n  Accessors ({len(accessors)}) & BufferViews ({len(buffer_views)}):")

    c1_offset = 20 + c0_len
    c1_len, c1_type = struct.unpack("<I4s", glb_data[c1_offset : c1_offset + 8])
    print(f"  Chunk 1 (BIN) Length: {c1_len:,} bytes, Type: {c1_type.decode('ascii')}")
    bin_data = glb_data[c1_offset + 8 : c1_offset + 8 + c1_len]

    for i, acc in enumerate(accessors):
        bv = buffer_views[acc["bufferView"]]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        acc_type = acc["type"]
        comp_type = acc["componentType"]
        print(f"    Accessor {i}: type={acc_type}, compType={comp_type}, count={count:,}, bufferView={acc['bufferView']}, min={acc.get('min')}, max={acc.get('max')}")

        if acc_type == "VEC3" and comp_type == 5126:
            raw_floats = np.frombuffer(bin_data[offset : offset + count * 12], dtype=np.float32).reshape(-1, 3)
            all_finite = bool(np.all(np.isfinite(raw_floats)))
            print(f"      Data Check: shape={raw_floats.shape}, all_finite={all_finite}, sample_min={raw_floats.min(axis=0)}, sample_max={raw_floats.max(axis=0)}")

    print("\n" + "=" * 70)
    print("2. INSPECTING PLY POINT CLOUD ARTIFACT")
    print("=" * 70)
    with open(PLY_PATH, "r", encoding="utf-8", errors="replace") as f:
        ply_lines = [f.readline().strip() for _ in range(25)]

    print("PLY Header:")
    for line in ply_lines[:15]:
        print("  ", line)

    with open(PLY_PATH, "rb") as f:
        ply_content = f.read().decode("utf-8", errors="replace").splitlines()

    end_header_idx = [idx for idx, line in enumerate(ply_content) if line.strip() == "end_header"][0]
    vertex_rows = ply_content[end_header_idx + 1 :]
    print(f"\nHeader ends at line {end_header_idx}. Declared data rows: {len(vertex_rows):,}")
    print("  First vertex:", vertex_rows[0])
    print("  Sample vertex (mid):", vertex_rows[len(vertex_rows) // 2])
    print("  Last vertex:", vertex_rows[-1])

    print("\n" + "=" * 70)
    print("3. INSPECTING SCENE GRAPH JSON VS EXPORTED GLB/PLY")
    print("=" * 70)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        sg = json.load(f)

    xyz = sg.get("point_cloud", {}).get("xyz", [])
    rgb = sg.get("point_cloud", {}).get("rgb", [])
    objs = sg.get("objects", [])

    print(f"SceneGraph JSON point cloud xyz total points: {len(xyz):,}")
    print(f"SceneGraph JSON point cloud rgb total colors: {len(rgb):,}")
    print(f"SceneGraph JSON objects recognized: {len(objs)}")
    for obj in objs:
        geom = obj.get("geometry", {})
        print(f"  Object: id='{obj.get('object_id')}', class='{obj.get('semantic_class')}', centroid={geom.get('centroid')}, extent={geom.get('extent')}")

    print("\n" + "=" * 70)
    print("4. COMPARISON CONCLUSION: DOES GLB CONTAIN POINT CLOUD OR ONLY BOUNDING BOXES?")
    print("=" * 70)
    # Check meshes in GLB
    has_pc_mesh = any(m.get("name") == "Room_PointCloud" for m in meshes)
    box_meshes = [m.get("name") for m in meshes if m.get("name", "").startswith("Box_")]
    pc_accessor = None
    for m in meshes:
        if m.get("name") == "Room_PointCloud":
            pos_acc_idx = m["primitives"][0]["attributes"]["POSITION"]
            pc_accessor = accessors[pos_acc_idx]

    if has_pc_mesh and pc_accessor:
        print(f"RESULT: GLB CONTAINS POINT CLOUD with {pc_accessor['count']:,} points (mode=POINTS).")
        print(f"RESULT: GLB ALSO CONTAINS {len(box_meshes)} BOUNDING BOX MESHES: {box_meshes}.")
        print(f"NOTE: The original SceneGraph had {len(xyz):,} points. The exported GLB contains {pc_accessor['count']:,} points due to the 40,000 max_points viewer cap.")
        print(f"NOTE: The exported PLY contains {len(vertex_rows):,} points.")
    else:
        print("RESULT: GLB CONTAINS ONLY BOUNDING BOXES (NO POINT CLOUD MESH FOUND).")


if __name__ == "__main__":
    inspect()

