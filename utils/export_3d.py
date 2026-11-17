"""3D Scene and Point Cloud Exporter for RoomAI.

Generates:
1. Standard ASCII PLY files for dense point clouds and vertex-colored meshes.
2. Binary glTF 2.0 (GLB) files containing 3D point clouds and furniture 3D bounding boxes.
Completely standalone with zero external dependencies (pure Python standard library + numpy).
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def export_point_cloud_ply(
    xyz: List[List[float]],
    rgb: Optional[List[List[float]]] = None,
    output_path: Optional[Path] = None,
    units: str = "m",
    coordinate_frame: str = "opencv_camera_x_right_y_down_z_forward",
) -> bytes:
    """Exports XYZ coordinates and RGB colors to standard PLY ASCII format.

    Args:
        xyz: List of [x, y, z] coordinates in meters.
        rgb: Optional list of [r, g, b] colors in [0, 1] range.
        output_path: Optional path to write file directly to disk.

    Returns:
        PLY file content as bytes.
    """
    # Allow passing SceneGraph instance directly as first parameter
    if hasattr(xyz, "point_cloud"):
        pc = getattr(xyz, "point_cloud", None)
        if isinstance(pc, dict):
            if rgb is None:
                rgb = pc.get("rgb")
            xyz = pc.get("xyz", [])

    if xyz is None or len(xyz) == 0:
        xyz_arr = np.zeros((0, 3), dtype=np.float32)
        valid_mask = np.zeros(0, dtype=bool)
    else:
        xyz_arr = np.array(xyz, dtype=np.float32)
        if xyz_arr.ndim == 1 or (xyz_arr.ndim == 2 and xyz_arr.shape[1] != 3):
            xyz_arr = np.zeros((0, 3), dtype=np.float32)
            valid_mask = np.zeros(0, dtype=bool)
        else:
            valid_mask = np.isfinite(xyz_arr).all(axis=1)
            xyz_arr = xyz_arr[valid_mask]

    n_vertices = len(xyz_arr)
    has_rgb = rgb is not None and len(rgb) == len(xyz) and n_vertices > 0

    if has_rgb:
        rgb_arr = np.array(rgb, dtype=np.float32)[valid_mask]
        rgb_bytes = np.clip(rgb_arr * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_bytes = np.full((n_vertices, 3), 180, dtype=np.uint8)

    # Construct PLY ASCII Header
    header = (
        "ply\n"
        "format ascii 1.0\n"
        "comment RoomAI 3D Scene Reconstruction Export\n"
        f"comment units {units}\n"
        f"comment coordinate_frame {coordinate_frame}\n"
        "comment representation colored_point_cloud_not_mesh\n"
        f"element vertex {n_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    lines = [header]
    for i in range(n_vertices):
        x, y, z = xyz_arr[i]
        r, g, b = rgb_bytes[i]
        lines.append(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")

    content = "".join(lines).encode("utf-8")
    if output_path:
        Path(output_path).write_bytes(content)
    return content


def create_box_mesh(
    centroid: Tuple[float, float, float],
    extent: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 8 vertices and 12 triangular faces for a 3D bounding box.

    Args:
        centroid: (cx, cy, cz) in meters.
        extent: (w, h, d) in meters.

    Returns:
        Tuple of (vertices: [8, 3] float32, indices: [36] uint16).
    """
    cx, cy, cz = centroid
    w, h, d = extent
    hw, hh, hd = w / 2.0, h / 2.0, d / 2.0

    # 8 corner vertices
    v = np.array([
        [cx - hw, cy - hh, cz - hd],  # 0
        [cx + hw, cy - hh, cz - hd],  # 1
        [cx + hw, cy + hh, cz - hd],  # 2
        [cx - hw, cy + hh, cz - hd],  # 3
        [cx - hw, cy - hh, cz + hd],  # 4
        [cx + hw, cy - hh, cz + hd],  # 5
        [cx + hw, cy + hh, cz + hd],  # 6
        [cx - hw, cy + hh, cz + hd],  # 7
    ], dtype=np.float32)

    # 12 triangles (2 per cube face)
    triangles = np.array([
        0, 1, 2,  0, 2, 3,  # Front
        4, 6, 5,  4, 7, 6,  # Back
        0, 4, 5,  0, 5, 1,  # Bottom
        3, 2, 6,  3, 6, 7,  # Top
        0, 3, 7,  0, 7, 4,  # Left
        1, 5, 6,  1, 6, 2,  # Right
    ], dtype=np.uint16)

    return v, triangles


def export_scene_glb(
    point_cloud: Optional[Dict[str, Any]] = None,
    objects: Optional[List[Any]] = None,
    output_path: Optional[Path] = None,
) -> bytes:
    """Exports 3D reconstructed scene with point cloud and furniture 3D bounding boxes to binary GLB (glTF 2.0).

    Args:
        point_cloud: Dict with 'xyz' and optional 'rgb'.
        objects: List of SceneObject with geometry and semantic labels.
        output_path: Optional path to write file directly to disk.

    Returns:
        Binary GLB bytes.
    """
    binary_buffers = []
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []

    current_byte_offset = 0

    def add_buffer_data(data_bytes: bytes, target_type: Optional[int] = None) -> int:
        nonlocal current_byte_offset
        # glTF requires 4-byte buffer alignment
        pad = (4 - (len(data_bytes) % 4)) % 4
        padded = data_bytes + b"\x00" * pad

        bv_idx = len(buffer_views)
        bv_dict = {
            "buffer": 0,
            "byteOffset": current_byte_offset,
            "byteLength": len(data_bytes),
        }
        if target_type:
            bv_dict["target"] = target_type
        buffer_views.append(bv_dict)

        binary_buffers.append(padded)
        current_byte_offset += len(padded)
        return bv_idx

    # Allow passing SceneGraph instance directly as first parameter
    if hasattr(point_cloud, "point_cloud"):
        if objects is None:
            objects = getattr(point_cloud, "objects", None)
        point_cloud = getattr(point_cloud, "point_cloud", None)

    # 1. Process point cloud points if present
    if point_cloud and isinstance(point_cloud, dict) and point_cloud.get("xyz"):
        xyz_raw = np.array(point_cloud["xyz"], dtype=np.float32)
        if xyz_raw.ndim == 2 and xyz_raw.shape[1] == 3 and len(xyz_raw) > 0:
            valid = np.isfinite(xyz_raw).all(axis=1)
            xyz_arr = xyz_raw[valid]
        else:
            xyz_arr = np.zeros((0, 3), dtype=np.float32)

        # Subsample point cloud for GLB viewer efficiency (up to 40k points)
        if len(xyz_arr) > 40000:
            sub_idx = np.random.choice(len(xyz_arr), 40000, replace=False)
            xyz_arr = xyz_arr[sub_idx]

        if len(xyz_arr) > 0:
            pos_bytes = xyz_arr.tobytes()
            pos_bv = add_buffer_data(pos_bytes, target_type=34962)  # ARRAY_BUFFER

            pos_acc = len(accessors)
            accessors.append({
                "bufferView": pos_bv,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(xyz_arr),
                "type": "VEC3",
                "max": xyz_arr.max(axis=0).tolist(),
                "min": xyz_arr.min(axis=0).tolist(),
            })

            # Point cloud primitive (mode 0 = POINTS)
            mesh_idx = len(meshes)
            meshes.append({
                "name": "Room_PointCloud",
                "primitives": [{
                    "attributes": {"POSITION": pos_acc},
                    "mode": 0,  # POINTS
                }],
            })
            nodes.append({"name": "PointCloudNode", "mesh": mesh_idx})

    # 2. Process 3D furniture bounding boxes
    if objects:
        for obj in objects:
            geom = getattr(obj, "geometry", None)
            if not geom or not getattr(geom, "centroid", None) or not getattr(geom, "extent", None):
                continue
            c = geom.centroid
            e = geom.extent
            if not (len(c) == 3 and len(e) == 3 and all(np.isfinite(c)) and all(np.isfinite(e)) and all(v > 0 for v in e)):
                continue

            v_arr, idx_arr = create_box_mesh(geom.centroid, geom.extent)
            v_bytes = v_arr.tobytes()
            i_bytes = idx_arr.tobytes()

            v_bv = add_buffer_data(v_bytes, target_type=34962)
            i_bv = add_buffer_data(i_bytes, target_type=34963)  # ELEMENT_ARRAY_BUFFER

            v_acc = len(accessors)
            accessors.append({
                "bufferView": v_bv,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": 8,
                "type": "VEC3",
                "max": v_arr.max(axis=0).tolist(),
                "min": v_arr.min(axis=0).tolist(),
            })

            i_acc = len(accessors)
            accessors.append({
                "bufferView": i_bv,
                "byteOffset": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(idx_arr),
                "type": "SCALAR",
                "max": [int(idx_arr.max())],
                "min": [int(idx_arr.min())],
            })

            mesh_idx = len(meshes)
            sem_name = getattr(obj, "semantic_class", "furniture")
            obj_id = getattr(obj, "object_id", f"obj_{mesh_idx}")
            meshes.append({
                "name": f"Box_{sem_name}_{obj_id}",
                "primitives": [{
                    "attributes": {"POSITION": v_acc},
                    "indices": i_acc,
                    "mode": 4,  # TRIANGLES
                }],
            })
            nodes.append({"name": f"Node_{obj_id}", "mesh": mesh_idx})

    # Assemble complete glTF JSON structure
    total_bin_bytes = b"".join(binary_buffers)
    gltf_dict = {
        "asset": {"version": "2.0", "generator": "RoomAI Phase 7 3D Reconstruction Exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(total_bin_bytes)}],
    }

    json_str = json.dumps(gltf_dict, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON chunk to 4-byte boundary with spaces
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad

    # Build binary GLB container:
    # 12-byte header: magic (0x46546C67), version (2), total_length
    # Chunk 0 (JSON): length, type (0x4E4F534A 'JSON'), data
    # Chunk 1 (BIN):  length, type (0x004E4942 'BIN\x00'), data
    total_glb_len = 12 + (8 + len(json_bytes)) + (8 + len(total_bin_bytes))

    header_bytes = struct.pack("<4sII", b"glTF", 2, total_glb_len)
    chunk0_header = struct.pack("<I4s", len(json_bytes), b"JSON")
    chunk1_header = struct.pack("<I4s", len(total_bin_bytes), b"BIN\x00")

    glb_content = header_bytes + chunk0_header + json_bytes + chunk1_header + total_bin_bytes

    if output_path:
        Path(output_path).write_bytes(glb_content)
    return glb_content


def export_colored_point_cloud_glb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_path: Optional[Path] = None,
    max_points: Optional[int] = 40000,
    sampling_seed: int = 0,
    units: str = "m",
    coordinate_frame: str = "opencv_camera_x_right_y_down_z_forward",
) -> bytes:
    """Export only the canonical colored point set (no semantic box geometry)."""
    xyz_arr = np.asarray(xyz, dtype=np.float32)
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3 or rgb_arr.shape != xyz_arr.shape:
        raise ValueError("xyz and rgb must be matching Nx3 arrays")
    finite = np.isfinite(xyz_arr).all(axis=1) & np.isfinite(rgb_arr).all(axis=1)
    xyz_arr = xyz_arr[finite]
    rgb_arr = np.clip(rgb_arr[finite], 0.0, 1.0)
    if not len(xyz_arr):
        raise ValueError("Cannot export an empty point cloud")
    if max_points is not None and max_points > 0 and len(xyz_arr) > max_points:
        rng = np.random.default_rng(sampling_seed)
        selection = np.sort(rng.choice(len(xyz_arr), size=max_points, replace=False))
        xyz_arr, rgb_arr = xyz_arr[selection], rgb_arr[selection]

    positions = np.ascontiguousarray(xyz_arr, dtype="<f4").tobytes()
    # Vertex attribute starts must remain 4-byte aligned. Store RGB in a
    # 4-byte-stride element (unused pad byte) while exposing a VEC3 accessor.
    colors_rgb = np.rint(rgb_arr * 255.0).astype(np.uint8)
    colors_rgba_stride = np.zeros((len(colors_rgb), 4), dtype=np.uint8)
    colors_rgba_stride[:, :3] = colors_rgb
    colors = colors_rgba_stride.tobytes()
    color_offset = len(positions) + ((4 - len(positions) % 4) % 4)
    binary_content = positions + b"\x00" * (color_offset - len(positions)) + colors
    binary = binary_content + b"\x00" * ((4 - len(binary_content) % 4) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "RoomAI Canonical Point Cloud Exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "RoomAI_PointCloud", "mesh": 0}],
        "meshes": [{"name": "RoomAI_Colored_Points", "primitives": [{
            "attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0,
        }]}],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": len(xyz_arr), "type": "VEC3", "min": xyz_arr.min(axis=0).tolist(), "max": xyz_arr.max(axis=0).tolist()},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5121, "normalized": True, "count": len(rgb_arr), "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": len(colors), "byteStride": 4, "target": 34962},
        ],
        "buffers": [{"byteLength": len(binary_content)}],
        "extras": {"roomai": {"representation": "point_cloud", "units": units,
                                "coordinate_frame": coordinate_frame, "surface_mesh": False}},
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    content = (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    )
    if output_path:
        Path(output_path).write_bytes(content)
    return content


def validate_ply_bytes(ply_bytes: bytes) -> Dict[str, Any]:
    """Validates and parses ASCII PLY file bytes.

    Args:
        ply_bytes: Raw bytes of PLY file.

    Returns:
        Dict with 'is_valid': bool, 'vertex_count': int, 'error': Optional[str].
    """
    if not isinstance(ply_bytes, (bytes, bytearray)):
        return {"is_valid": False, "vertex_count": 0, "error": "Input is not bytes"}
    try:
        text = ply_bytes.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or lines[0] != "ply":
            return {"is_valid": False, "vertex_count": 0, "error": "Missing 'ply' header magic"}

        end_header_idx = -1
        vertex_count = 0
        for idx, line in enumerate(lines):
            if line.startswith("element vertex"):
                parts = line.split()
                if len(parts) >= 3 and parts[2].isdigit():
                    vertex_count = int(parts[2])
            elif line == "end_header":
                end_header_idx = idx
                break

        if end_header_idx == -1:
            return {"is_valid": False, "vertex_count": 0, "error": "Missing 'end_header' token"}

        data_lines = lines[end_header_idx + 1 :]
        if len(data_lines) != vertex_count:
            return {
                "is_valid": False,
                "vertex_count": vertex_count,
                "error": f"Header declared {vertex_count} vertices, but found {len(data_lines)} data lines",
            }
        fields = []
        for row in data_lines:
            try:
                values = row.split()
                fields.append((float(values[0]), float(values[1]), float(values[2]), int(values[3]), int(values[4]), int(values[5])))
            except (ValueError, IndexError):
                return {"is_valid": False, "vertex_count": vertex_count, "error": "Malformed XYZ/RGB vertex row"}
        if fields and (not np.isfinite(np.asarray(fields, dtype=np.float64)).all()):
            return {"is_valid": False, "vertex_count": vertex_count, "error": "Non-finite PLY values"}
        if fields and any(any(color < 0 or color > 255 for color in row[3:]) for row in fields):
            return {"is_valid": False, "vertex_count": vertex_count, "error": "PLY color outside [0,255]"}
        return {"is_valid": True, "vertex_count": vertex_count, "error": None, "has_rgb": True}
    except Exception as exc:
        return {"is_valid": False, "vertex_count": 0, "error": str(exc)}


def extract_point_cloud_ply(ply_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Parse XYZ and uchar RGB from RoomAI's ASCII PLY point export."""
    validation = validate_ply_bytes(ply_bytes)
    if not validation.get("is_valid"):
        raise ValueError(f"Invalid PLY: {validation.get('error')}")
    text = ply_bytes.decode("utf-8")
    rows = text.split("end_header\n", 1)[1].splitlines()
    values = np.asarray([[float(part) for part in row.split()] for row in rows if row.strip()], dtype=np.float64)
    return values[:, :3].astype(np.float32), values[:, 3:6].astype(np.float32) / 255.0


def validate_glb_bytes(glb_bytes: bytes) -> Dict[str, Any]:
    """Validates and parses binary glTF 2.0 (GLB) container bytes.

    Args:
        glb_bytes: Raw bytes of GLB file.

    Returns:
        Dict with 'is_valid': bool, 'mesh_count': int, 'json_data': dict, 'error': Optional[str].
    """
    if not isinstance(glb_bytes, (bytes, bytearray)):
        return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": "Input is not bytes"}
    if len(glb_bytes) < 20:
        return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": "File too small (< 20 bytes)"}
    try:
        magic, version, length = struct.unpack("<4sII", glb_bytes[:12])
        if magic != b"glTF":
            return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": f"Invalid magic '{magic}', expected 'glTF'"}
        if version != 2:
            return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": f"Unsupported glTF version {version}, expected 2"}
        if length != len(glb_bytes):
            return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": f"Header length {length} does not match payload length {len(glb_bytes)}"}

        c0_len, c0_type = struct.unpack("<I4s", glb_bytes[12:20])
        if c0_type != b"JSON":
            return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": f"First chunk is '{c0_type}', expected 'JSON'"}
        json_end = 20 + c0_len
        if c0_len % 4 or json_end + 8 > len(glb_bytes):
            return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": "Invalid JSON chunk length or missing BIN chunk"}
        json_bytes = glb_bytes[20:json_end]
        json_dict = json.loads(json_bytes.decode("utf-8").strip())
        bin_len, bin_type = struct.unpack("<I4s", glb_bytes[json_end:json_end + 8])
        bin_start = json_end + 8
        bin_end = bin_start + bin_len
        if bin_type != b"BIN\x00" or bin_end != len(glb_bytes) or bin_len % 4:
            return {"is_valid": False, "mesh_count": 0, "json_data": json_dict, "error": "Invalid BIN chunk type, length, or alignment"}
        binary = glb_bytes[bin_start:bin_end]

        buffer_length = json_dict.get("buffers", [{}])[0].get("byteLength", -1)
        if buffer_length < 0 or buffer_length > len(binary):
            return {"is_valid": False, "mesh_count": 0, "json_data": json_dict, "error": "Buffer byteLength exceeds BIN chunk"}
        for view in json_dict.get("bufferViews", []):
            offset = int(view.get("byteOffset", 0))
            length_view = int(view.get("byteLength", -1))
            if offset < 0 or length_view < 0 or offset + length_view > buffer_length or offset % 4:
                return {"is_valid": False, "mesh_count": 0, "json_data": json_dict, "error": "Invalid bufferView range or alignment"}
        accessor_map = json_dict.get("accessors", [])
        for accessor in accessor_map:
            view_index = accessor.get("bufferView")
            if view_index is None or not (0 <= int(view_index) < len(json_dict.get("bufferViews", []))):
                return {"is_valid": False, "mesh_count": 0, "json_data": json_dict, "error": "Accessor references missing bufferView"}
            view = json_dict["bufferViews"][int(view_index)]
            type_components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
            component_bytes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
            count = int(accessor.get("count", -1))
            component_size = component_bytes.get(int(accessor.get("componentType", -1)))
            components = type_components.get(accessor.get("type"))
            offset = int(accessor.get("byteOffset", 0))
            element_size = component_size * components if component_size and components else -1
            stride = int(view.get("byteStride", element_size))
            required = ((count - 1) * stride + element_size) if element_size > 0 and count > 0 else -1
            if required < 0 or offset < 0 or offset + required > int(view["byteLength"]):
                return {"is_valid": False, "mesh_count": 0, "json_data": json_dict, "error": "Invalid accessor type, count, or buffer range"}
        for mesh in json_dict.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                for attribute, accessor_index in primitive.get("attributes", {}).items():
                    if not (0 <= int(accessor_index) < len(accessor_map)):
                        return {"is_valid": False, "mesh_count": 0, "json_data": json_dict, "error": f"Primitive references missing {attribute} accessor"}

        mesh_count = len(json_dict.get("meshes", []))
        return {
            "is_valid": True,
            "mesh_count": mesh_count,
            "binary": binary,
            "json_data": json_dict,
            "error": None,
        }
    except Exception as exc:
        return {"is_valid": False, "mesh_count": 0, "json_data": None, "error": str(exc)}


def extract_glb_point_cloud(glb_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Decode POSITION and normalized COLOR_0 arrays from a RoomAI point GLB."""
    validation = validate_glb_bytes(glb_bytes)
    if not validation.get("is_valid"):
        raise ValueError(f"Invalid GLB: {validation.get('error')}")
    gltf = validation["json_data"]
    primitive = gltf["meshes"][0]["primitives"][0]
    attrs = primitive.get("attributes", {})
    if "POSITION" not in attrs or "COLOR_0" not in attrs:
        raise ValueError("GLB point primitive must contain POSITION and COLOR_0")
    binary = validation["binary"]
    arrays = []
    for semantic in ("POSITION", "COLOR_0"):
        accessor = gltf["accessors"][attrs[semantic]]
        view = gltf["bufferViews"][accessor["bufferView"]]
        offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        count = int(accessor["count"])
        stride = int(view.get("byteStride", 12 if semantic == "POSITION" else 3))
        if semantic == "POSITION":
            array = np.frombuffer(binary, dtype="<f4", count=count * 3, offset=offset).reshape(count, 3).copy()
        else:
            if stride == 4:
                strided = np.ndarray((count, 4), dtype=np.uint8, buffer=binary, offset=offset, strides=(stride, 1))
                array = strided[:, :3].astype(np.float32) / 255.0
            else:
                array = np.frombuffer(binary, dtype=np.uint8, count=count * 3, offset=offset).reshape(count, 3).astype(np.float32) / 255.0
        arrays.append(array)
    return arrays[0], arrays[1]


