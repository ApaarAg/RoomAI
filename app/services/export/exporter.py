"""RoomAI Export Service.

Validated 3D asset export (GLB, PLY) with format verification.
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


class ExportService:
    """Exports validated 3D reconstruction assets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_points = self.config.get("max_points", 100000)

    def export_point_cloud_ply(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
    ) -> bytes:
        """Export point cloud as ASCII PLY with validation."""
        if len(points) == 0:
            # Empty PLY header
            header = (
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 0\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            ply_bytes = header.encode("ascii")
            if output_path:
                output_path.write_bytes(ply_bytes)
            return ply_bytes

        n = len(points)
        has_colors = colors is not None and len(colors) == n
        has_normals = normals is not None and len(normals) == n

        # Build header
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_colors:
            lines.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])

        if has_normals:
            lines.extend([
                "property float nx",
                "property float ny",
                "property float nz",
            ])

        lines.append("end_header")

        # Build vertex data
        vertex_lines = []
        for i in range(n):
            parts = [f"{points[i,0]:.6f}", f"{points[i,1]:.6f}", f"{points[i,2]:.6f}"]
            if has_colors:
                c = colors[i]
                # Colors expected in [0,1], convert to [0,255]
                parts.extend([
                    str(int(np.clip(c[0] * 255, 0, 255))),
                    str(int(np.clip(c[1] * 255, 0, 255))),
                    str(int(np.clip(c[2] * 255, 0, 255))),
                ])
            if has_normals:
                n_ = normals[i]
                parts.extend([f"{n_[0]:.6f}", f"{n_[1]:.6f}", f"{n_[2]:.6f}"])
            vertex_lines.append(" ".join(parts))

        ply_content = "\n".join(lines) + "\n" + "\n".join(vertex_lines) + "\n"
        ply_bytes = ply_content.encode("ascii")

        if output_path:
            output_path.write_bytes(ply_bytes)

        return ply_bytes

    def export_scene_glb(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
    ) -> bytes:
        """Export scene as binary GLB (glTF 2.0) with point cloud and bounding boxes."""
        # Minimal GLB structure for point cloud + boxes
        # This is a simplified but valid GLB

        accessors = []
        buffer_views = []
        meshes = []
        nodes = []
        materials = []

        # Main buffer data
        buffer_data = bytearray()
        buffer_offset = 0

        # Point cloud mesh
        if len(points) > 0:
            # Positions
            pos_data = points.astype(np.float32).tobytes()
            pos_offset = buffer_offset
            buffer_data.extend(pos_data)
            buffer_offset += len(pos_data)

            pos_accessor = {
                "bufferView": len(buffer_views),
                "componentType": 5126,  # FLOAT
                "count": len(points),
                "type": "VEC3",
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
            }
            accessors.append(pos_accessor)

            pos_buffer_view = {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": len(pos_data),
                "target": 34962,  # ARRAY_BUFFER
            }
            buffer_views.append(pos_buffer_view)

            # Colors
            col_accessor_idx = None
            if colors is not None and len(colors) == len(points):
                col_data = (np.clip(colors, 0, 1) * 255).astype(np.uint8).tobytes()
                col_offset = buffer_offset
                buffer_data.extend(col_data)
                buffer_offset += len(col_data)

                col_accessor = {
                    "bufferView": len(buffer_views),
                    "componentType": 5121,  # UNSIGNED_BYTE
                    "count": len(colors),
                    "type": "VEC3",
                    "normalized": True,
                }
                col_accessor_idx = len(accessors)
                accessors.append(col_accessor)

                col_buffer_view = {
                    "buffer": 0,
                    "byteOffset": col_offset,
                    "byteLength": len(col_data),
                    "target": 34962,
                }
                buffer_views.append(col_buffer_view)

            # Primitive
            primitive = {"attributes": {"POSITION": len(accessors) - 1 - (1 if col_accessor_idx is not None else 0)}}
            if col_accessor_idx is not None:
                primitive["attributes"]["COLOR_0"] = col_accessor_idx

            meshes.append({"primitives": [primitive]})
            nodes.append({"mesh": len(meshes) - 1, "name": "point_cloud"})

        # Bounding box objects
        if objects:
            for obj in objects:
                geom = obj.get("geometry")
                if not geom:
                    continue
                centroid = np.array(geom.get("centroid", [0,0,0]), dtype=np.float32)
                extent = np.array(geom.get("extent", [1,1,1]), dtype=np.float32)

                # Create box corners (8 vertices)
                half = extent / 2
                corners = np.array([
                    [-half[0], -half[1], -half[2]],
                    [ half[0], -half[1], -half[2]],
                    [ half[0],  half[1], -half[2]],
                    [-half[0],  half[1], -half[2]],
                    [-half[0], -half[1],  half[2]],
                    [ half[0], -half[1],  half[2]],
                    [ half[0],  half[1],  half[2]],
                    [-half[0],  half[1],  half[2]],
                ], dtype=np.float32) + centroid

                # Box edges (12 edges = 24 indices for LINES)
                edges = np.array([
                    0,1, 1,2, 2,3, 3,0,
                    4,5, 5,6, 6,7, 7,4,
                    0,4, 1,5, 2,6, 3,7,
                ], dtype=np.uint16)

                box_pos_data = corners.tobytes()
                pos_offset = buffer_offset
                buffer_data.extend(box_pos_data)
                buffer_offset += len(box_pos_data)

                box_accessor = {
                    "bufferView": len(buffer_views),
                    "componentType": 5126,
                    "count": 8,
                    "type": "VEC3",
                    "min": corners.min(axis=0).tolist(),
                    "max": corners.max(axis=0).tolist(),
                }
                accessors.append(box_accessor)

                box_bv = {
                    "buffer": 0,
                    "byteOffset": pos_offset,
                    "byteLength": len(box_pos_data),
                    "target": 34962,
                }
                buffer_views.append(box_bv)

                # Index buffer for edges
                idx_data = edges.tobytes()
                idx_offset = buffer_offset
                buffer_data.extend(idx_data)
                buffer_offset += len(idx_data)

                idx_accessor = {
                    "bufferView": len(buffer_views),
                    "componentType": 5123,  # UNSIGNED_SHORT
                    "count": len(edges),
                    "type": "SCALAR",
                }
                accessors.append(idx_accessor)

                idx_bv = {
                    "buffer": 0,
                    "byteOffset": idx_offset,
                    "byteLength": len(idx_data),
                    "target": 34963,  # ELEMENT_ARRAY_BUFFER
                }
                buffer_views.append(idx_bv)

                # Wireframe material
                mat_idx = len(materials)
                materials.append({
                    "name": f"box_{obj.get('object_id', 'unknown')}",
                    "emissiveFactor": [1.0, 0.0, 0.0],
                    "alphaMode": "OPAQUE",
                })

                primitive = {
                    "attributes": {"POSITION": len(accessors) - 2},
                    "indices": len(accessors) - 1,
                    "material": mat_idx,
                    "mode": 1,  # LINES
                }
                meshes.append({"primitives": [primitive]})
                nodes.append({
                    "mesh": len(meshes) - 1,
                    "name": f"box_{obj.get('object_id', 'unknown')}",
                })

        # Build GLTF JSON
        gltf = {
            "asset": {"version": "2.0", "generator": "RoomAI ExportService"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}],
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buffer_data)}],
        }

        # Encode GLB
        json_str = json.dumps(gltf, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")

        # Pad JSON to 4-byte alignment
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes += b" " * json_padding

        # Pad buffer to 4-byte alignment
        buffer_padding = (4 - len(buffer_data) % 4) % 4
        buffer_data += b"\x00" * buffer_padding

        # GLB header (12 bytes)
        # Magic: "gltF" (0x46546C67), Version: 2, Length: total
        total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)
        glb_header = struct.pack("<I II", 0x46546C67, 2, total_length)

        # JSON chunk (8 bytes header + data)
        json_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # "JSON"

        # Binary chunk (8 bytes header + data)
        bin_header = struct.pack("<II", len(buffer_data), 0x004E4942)  # "BIN\0"

        glb_bytes = glb_header + json_header + json_bytes + bin_header + buffer_data

        if output_path:
            output_path.write_bytes(glb_bytes)

        return bytes(glb_bytes)

    def validate_ply(self, ply_bytes: bytes) -> Dict[str, Any]:
        """Validate PLY file structure."""
        try:
            text = ply_bytes.decode("ascii")
            lines = text.strip().split("\n")

            if not lines[0].startswith("ply"):
                return {"is_valid": False, "error": "Missing PLY magic"}

            if "format ascii" not in lines[1]:
                return {"is_valid": False, "error": "Only ASCII PLY supported"}

            vertex_count = 0
            for line in lines:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[2])
                    break

            # Count actual vertices after header
            header_end = lines.index("end_header") + 1
            actual_vertices = len(lines) - header_end

            return {
                "is_valid": True,
                "vertex_count": vertex_count,
                "actual_vertices": actual_vertices,
                "format": "ascii",
            }
        except Exception as exc:
            return {"is_valid": False, "error": str(exc)}

    def validate_glb(self, glb_bytes: bytes) -> Dict[str, Any]:
        """Validate GLB file structure."""
        try:
            if len(glb_bytes) < 12:
                return {"is_valid": False, "error": "File too small for GLB header"}

            magic, version, length = struct.unpack("<III", glb_bytes[:12])

            if magic != 0x46546C67:  # "gltF"
                return {"is_valid": False, "error": f"Invalid magic: {magic:#x}"}

            if version != 2:
                return {"is_valid": False, "error": f"Unsupported version: {version}"}

            if length != len(glb_bytes):
                return {"is_valid": False, "error": f"Length mismatch: header={length}, actual={len(glb_bytes)}"}

            # Parse chunks
            offset = 12
            mesh_count = 0

            while offset < len(glb_bytes):
                if offset + 8 > len(glb_bytes):
                    break
                chunk_length, chunk_type = struct.unpack("<II", glb_bytes[offset:offset+8])
                offset += 8

                if chunk_type == 0x4E4F534A:  # JSON
                    json_bytes = glb_bytes[offset:offset+chunk_length]
                    import json
                    gltf = json.loads(json_bytes.decode("utf-8"))
                    mesh_count = len(gltf.get("meshes", []))

                offset += chunk_length

            return {
                "is_valid": True,
                "mesh_count": mesh_count,
                "version": version,
            }
        except Exception as exc:
            return {"is_valid": False, "error": str(exc)}

    def export_scene(
        self,
        scene_data: Dict[str, Any],
        output_dir: Path,
        formats: List[str] = None,
        max_points: int = 40000,
    ) -> Dict[str, Any]:
        """Export scene in multiple formats."""
        formats = formats or ["glb", "ply"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract point cloud
        pc = scene_data.get("point_cloud", {})
        points = np.array(pc.get("xyz", []), dtype=np.float32)
        colors = np.array(pc.get("rgb", []), dtype=np.float32) if pc.get("rgb") else None
        normals = np.array(pc.get("normals", []), dtype=np.float32) if pc.get("normals") else None

        # Downsample if needed
        if len(points) > max_points:
            step = len(points) // max_points
            idx = np.arange(0, len(points), step)[:max_points]
            points = points[idx]
            if colors is not None:
                colors = colors[idx]
            if normals is not None:
                normals = normals[idx]

        results = {}

        if "ply" in formats:
            ply_path = output_dir / "scene_reconstruction.ply"
            ply_bytes = self.export_point_cloud_ply(points, colors, normals, ply_path)
            ply_val = self.validate_ply(ply_bytes)
            results["ply"] = {
                "path": str(ply_path),
                "size_bytes": len(ply_bytes),
                "valid": ply_val["is_valid"],
                "vertex_count": ply_val.get("vertex_count", 0),
            }

        if "glb" in formats:
            glb_path = output_dir / "scene_reconstruction.glb"
            objects = scene_data.get("objects", [])
            glb_bytes = self.export_scene_glb(points, colors, objects, glb_path)
            glb_val = self.validate_glb(glb_bytes)
            results["glb"] = {
                "path": str(glb_path),
                "size_bytes": len(glb_bytes),
                "valid": glb_val["is_valid"],
                "mesh_count": glb_val.get("mesh_count", 0),
            }

        return results


# Singleton
_export_service: Optional[ExportService] = None


def get_export_service() -> ExportService:
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
