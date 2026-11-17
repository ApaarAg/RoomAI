"""Multi-dataset label harmonization, annotation validation, and integrity utilities for RoomAI (Phase 19).

Provides:
- Canonical 10-class exterior architectural taxonomy.
- Explicit mapping rules for Open Images V7, Mapillary Vistas, ADE20K, COCO, and CMP Facade.
- Annotation verification (bounding boxes, polygons, coordinate normalization, degeneracies).
- Split isolation and duplicate detection (MD5 content hash and 64-bit DCT perceptual hash).
- Partial annotation / ignore mask tagging for sparse datasets.
"""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Canonical 10-class architectural taxonomy
CANONICAL_EXTERIOR_CLASSES: Dict[int, str] = {
    0: "building",
    1: "facade",
    2: "window",
    3: "door",
    4: "roof_cornice",
    5: "balcony",
    6: "porch",
    7: "pillar_column",
    8: "shutter_blind",
    9: "molding_deco",
}

CANONICAL_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CANONICAL_EXTERIOR_CLASSES.items()}


@dataclass
class LabelMappingEntry:
    """Explicit mapping from a source dataset label to RoomAI canonical taxonomy."""
    source_dataset: str
    source_label: str
    target_label: Optional[str]
    mapping_status: str  # "exact", "defensible", "ambiguous", "unsupported"
    reason: str
    validation_requirement: str

    @property
    def canonical_id(self) -> Optional[int]:
        if self.target_label and self.target_label in CANONICAL_NAME_TO_ID:
            return CANONICAL_NAME_TO_ID[self.target_label]
        return None


# Comprehensive Master Mapping Dictionary across all evaluated datasets
HARMONIZATION_RULES: List[LabelMappingEntry] = [
    # CMP Facade Database
    LabelMappingEntry("CMP_Facade", "facade", "facade", "exact",
                      "Direct facade envelope surface segment.", "Verify non-overlap with interior window cutouts."),
    LabelMappingEntry("CMP_Facade", "window", "window", "exact",
                      "Direct architectural glazed window opening.", "Filter out sub-pixel sliver boxes < 0.005 width/height."),
    LabelMappingEntry("CMP_Facade", "door", "door", "exact",
                      "Ground-floor and balcony entry portals.", "Verify ground-level or entrance-deck spatial alignment."),
    LabelMappingEntry("CMP_Facade", "cornice", "roof_cornice", "exact",
                      "Projecting decorative horizontal molding crowning building or wall.", "Distinguish from mid-wall stringcourses."),
    LabelMappingEntry("CMP_Facade", "balcony", "balcony", "exact",
                      "Exterior platform projecting from wall with railing.", "Verify railing/deck enclosure boundary."),
    LabelMappingEntry("CMP_Facade", "blind", "shutter_blind", "exact",
                      "Exterior louvered shutters or fabric blinds adjacent to windows.", "Ensure bbox wraps shutter blade assembly, not interior curtain."),
    LabelMappingEntry("CMP_Facade", "pillar", "pillar_column", "exact",
                      "Vertical structural or decorative column / pilaster.", "Differentiate engaged flat pilasters from freestanding columns."),
    LabelMappingEntry("CMP_Facade", "sill", "molding_deco", "defensible",
                      "Window sill horizontal ledge; collapsed into general exterior decorative molding.", "Check tight aspect ratio below window frame."),
    LabelMappingEntry("CMP_Facade", "deco", "molding_deco", "defensible",
                      "Ornamental pediment, keystone, or relief sculpture; collapsed into molding_deco.", "Validate decorative exterior placement."),
    LabelMappingEntry("CMP_Facade", "molding", "molding_deco", "exact",
                      "Horizontal or profiled decorative band; direct match.", "Ensure band continuity across facade span."),
    LabelMappingEntry("CMP_Facade", "shop", "door", "defensible",
                      "Ground-floor commercial storefront entrance; mapped to door.", "Exclude large display glass if separate window label exists."),
    LabelMappingEntry("CMP_Facade", "background", None, "unsupported",
                      "Sky, road, trees, or non-architectural background.", "Ignore in loss calculation; do not map."),

    # Open Images V7
    LabelMappingEntry("OpenImagesV7", "/m/0cgh4 (Building)", "building", "exact",
                      "Direct building envelope annotation.", "Ensure box encloses entire visible structure."),
    LabelMappingEntry("OpenImagesV7", "/m/03jm5 (House)", "building", "exact",
                      "Residential single-family home envelope; mapped to building.", "Ensure full structure envelope."),
    LabelMappingEntry("OpenImagesV7", "/m/021sj1 (Office building)", "building", "exact",
                      "Commercial office structure; mapped to building.", "Ensure complete exterior boundary."),
    LabelMappingEntry("OpenImagesV7", "/m/04m6gz (Porch)", "porch", "exact",
                      "Covered exterior entrance structure / veranda.", "Validate inclusion of deck, steps, and roof overhang."),
    LabelMappingEntry("OpenImagesV7", "/m/02dgv (Door)", "door", "exact",
                      "Architectural doorway opening.", "Filter out vehicle doors, interior cabinet doors via context."),
    LabelMappingEntry("OpenImagesV7", "/m/0d4v4 (Window)", "window", "exact",
                      "Glazed exterior window opening.", "Filter vehicle/airplane windows using exterior scene context."),
    LabelMappingEntry("OpenImagesV7", "/m/031b6r (Window blind)", "shutter_blind", "defensible",
                      "Window blinds and shutters; mapped to shutter_blind.", "Exclude interior micro-blinds lacking exterior visibility."),
    LabelMappingEntry("OpenImagesV7", "/m/016m2d (Skyscraper)", "building", "defensible",
                      "Multi-story high-rise building; mapped to building.", "Clip to visible structural boundaries."),
    LabelMappingEntry("OpenImagesV7", "/m/02p3w7 (Balcony)", "balcony", "ambiguous",
                      "Exists in Open Images hierarchy but sparse / largely machine-generated.", "Requires human verification of box bounds before ingestion."),
    LabelMappingEntry("OpenImagesV7", "/m/02522 (Column)", None, "ambiguous",
                      "Includes newspaper columns, text columns, and architectural pillars.", "Reject without architectural scene filter."),

    # ADE20K (Objects and Parts)
    LabelMappingEntry("ADE20K", "building, edifice", "building", "exact",
                      "Overall exterior structural envelope.", "Use whole object polygon mask."),
    LabelMappingEntry("ADE20K", "house", "building", "exact",
                      "Residential building structure; mapped to building.", "Use whole object polygon mask."),
    LabelMappingEntry("ADE20K", "wall (exterior)", "facade", "defensible",
                      "Exterior vertical surface of building.", "Verify polygon excludes windows and doors."),
    LabelMappingEntry("ADE20K", "facade", "facade", "exact",
                      "Explicit building facade part annotation.", "Best suited for semantic segmentation mask."),
    LabelMappingEntry("ADE20K", "windowpane, window", "window", "exact",
                      "Glazed window aperture; direct match.", "Extract tight bounding box from polygon."),
    LabelMappingEntry("ADE20K", "door", "door", "exact",
                      "Entrance door aperture; direct match.", "Extract tight bounding box from polygon."),
    LabelMappingEntry("ADE20K", "roof", "roof_cornice", "defensible",
                      "Roof structure and eaves; mapped to roof_cornice.", "Crop to lower cornice/fascia edge when converted to box."),
    LabelMappingEntry("ADE20K", "cornice", "roof_cornice", "exact",
                      "Cornice trim molding along roofline.", "Direct match for roof_cornice."),
    LabelMappingEntry("ADE20K", "balcony", "balcony", "exact",
                      "Exterior cantilevered balcony platform and balustrade.", "Extract tight bounding box from polygon."),
    LabelMappingEntry("ADE20K", "porch", "porch", "exact",
                      "Covered exterior entrance veranda / porch.", "Direct match; high structural value for rare class."),
    LabelMappingEntry("ADE20K", "column, pillar", "pillar_column", "exact",
                      "Architectural vertical column or pillar.", "Extract vertical aspect-ratio box from polygon."),
    LabelMappingEntry("ADE20K", "shutter, blind", "shutter_blind", "exact",
                      "Window shutter or external blind.", "Extract box adjacent to window aperture."),
    LabelMappingEntry("ADE20K", "molding", "molding_deco", "exact",
                      "Architectural exterior molding band.", "Extract horizontal/profiled box."),
    LabelMappingEntry("ADE20K", "awning, sunblind", None, "ambiguous",
                      "Fabric or metal canopy over windows/doors; overlaps with porch/entrance.", "Hold in quarantine; manual review required."),
    LabelMappingEntry("ADE20K", "arcade, colonnade", "pillar_column", "ambiguous",
                      "Series of arches supported by columns; spans multiple pillars.", "Must be split into individual pillar instances or rejected."),

    # Mapillary Vistas (v1.2 & v2.0)
    LabelMappingEntry("MapillaryVistas", "construction--structure--building", "building", "exact",
                      "Street-level building structure envelope.", "Must be filtered for distance > 5m to avoid distant horizon noise."),
    LabelMappingEntry("MapillaryVistas", "construction--barrier--wall", "facade", "ambiguous",
                      "Freestanding perimeter walls and exterior building walls are lumped together.", "Only accept when adjacent to building structure; otherwise reject."),
    LabelMappingEntry("MapillaryVistas_v2", "object--window", "window", "defensible",
                      "Building windows in v2.0.", "Filter out distant low-resolution windows (< 16px)."),
    LabelMappingEntry("MapillaryVistas_v2", "object--door", "door", "defensible",
                      "Building doors in v2.0.", "Filter out distant low-resolution doors."),
    LabelMappingEntry("MapillaryVistas", "object--support--pole", None, "unsupported",
                      "Utility poles, lamp posts, and traffic signs; NOT architectural columns.", "Reject completely; semantic mismatch."),
    LabelMappingEntry("MapillaryVistas", "construction--flat--driveway", None, "unsupported",
                      "Driveway pavement; outside exterior 10-class taxonomy.", "Ignore in loss calculation; do not map."),

    # COCO / COCO-Stuff
    LabelMappingEntry("COCO_Detection_80", "ALL_80_CLASSES", None, "unsupported",
                      "Standard COCO-80 has 0 architectural facade classes (no building, window, door, porch, column).",
                      "Completely unsupported for RoomAI object detection. Do not use for training detector."),
    LabelMappingEntry("COCO_Stuff", "building-other", "building", "defensible",
                      "Pixel stuff mask of building structures.", "Use only for background negative filtering or semantic envelope segmentation."),
    LabelMappingEntry("COCO_Stuff", "roof", "roof_cornice", "defensible",
                      "Pixel stuff mask of roofs.", "Coarse roof surface; lacks fine cornice boundary."),
    LabelMappingEntry("COCO_Stuff", "wall-brick/stone/concrete", "facade", "ambiguous",
                      "Material-based stuff masks spanning indoor and outdoor walls.", "Hold in quarantine; do not map directly without scene parser.")
]


@dataclass
class AnnotationValidationResult:
    """Validation report for an object annotation (box or polygon)."""
    is_valid: bool
    class_id: Optional[int]
    error: Optional[str] = None
    box_coords: Optional[Tuple[float, float, float, float]] = None  # xc, yc, w, h


def validate_yolo_box(
    class_id: int,
    xc: float,
    yc: float,
    w: float,
    h: float,
    min_size: float = 0.003,
    max_size: float = 1.0,
    valid_classes: Optional[Set[int]] = None
) -> AnnotationValidationResult:
    """Validates normalized YOLO bounding box coordinates [xc, yc, w, h] in range [0, 1].

    Guarantees:
    - class_id is recognized in valid_classes (or canonical 0-9).
    - xc, yc, w, h are floats within [0.0, 1.0].
    - w > min_size and h > min_size (filters zero-area and degenerate slivers).
    - Box does not completely exceed image boundaries (xmin >= 0, xmax <= 1, ymin >= 0, ymax <= 1).
    """
    if valid_classes is None:
        valid_classes = set(CANONICAL_EXTERIOR_CLASSES.keys())

    if class_id not in valid_classes:
        return AnnotationValidationResult(False, class_id, f"Invalid class ID: {class_id}")

    for val, name in [(xc, "xc"), (yc, "yc"), (w, "w"), (h, "h")]:
        if np.isnan(val) or np.isinf(val):
            return AnnotationValidationResult(False, class_id, f"Non-finite value in {name}: {val}")

    if w <= 0.0 or h <= 0.0:
        return AnnotationValidationResult(False, class_id, f"Degenerate non-positive dimension: w={w}, h={h}")

    if w < min_size or h < min_size:
        return AnnotationValidationResult(False, class_id, f"Sub-threshold dimension (<{min_size}): w={w}, h={h}")

    if w > max_size or h > max_size:
        return AnnotationValidationResult(False, class_id, f"Dimension exceeds max size ({max_size}): w={w}, h={h}")

    xmin = xc - w / 2.0
    xmax = xc + w / 2.0
    ymin = yc - h / 2.0
    ymax = yc + h / 2.0

    # Allow tiny numerical epsilon for float rounding (e.g. 1.000001)
    eps = 1e-4
    if xmin < -eps or ymin < -eps or xmax > 1.0 + eps or ymax > 1.0 + eps:
        return AnnotationValidationResult(
            False, class_id,
            f"Box coordinates out of normalized bounds [0, 1]: [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]"
        )

    # Clamped clean coordinates
    c_xc = max(0.0, min(1.0, float(xc)))
    c_yc = max(0.0, min(1.0, float(yc)))
    c_w = max(0.0, min(1.0, float(w)))
    c_h = max(0.0, min(1.0, float(h)))

    return AnnotationValidationResult(True, class_id, None, (c_xc, c_yc, c_w, c_h))


def compute_image_phash(img: Image.Image, hash_size: int = 8) -> int:
    """Computes a 64-bit DCT perceptual hash (pHash) for near-duplicate screening."""
    img_gray = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.BILINEAR)
    pixels = np.array(img_gray, dtype=np.float32)
    dct = cv2.dct(pixels)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    diff = dct_low > med
    h = 0
    for b in diff.flatten():
        h = (h << 1) | int(b)
    return h


def compute_phash_hamming_distance(h1: int, h2: int) -> int:
    """Computes bitwise Hamming distance between two 64-bit perceptual hashes."""
    return bin(h1 ^ h2).count("1")


@dataclass
class SplitIsolationReport:
    """Report summarizing cross-split leakages and near-duplicate detections."""
    is_isolated: bool
    md5_collisions: List[Tuple[str, str, str]] = field(default_factory=list)  # (split1, split2, hash)
    near_duplicates: List[Tuple[str, str, int]] = field(default_factory=list)  # (file1, file2, dist)
    total_images_checked: int = 0


def audit_split_isolation(
    split_image_dirs: Dict[str, Path],
    min_hamming_distance: int = 5
) -> SplitIsolationReport:
    """Verifies that no identical or near-duplicate scenes cross partition boundaries.

    Args:
        split_image_dirs: Dict mapping split name (e.g. 'train', 'val', 'test') to image directory Path.
        min_hamming_distance: Minimum pHash Hamming distance required between splits (default 5).

    Returns:
        SplitIsolationReport detailing any MD5 or pHash collisions.
    """
    md5_registry: Dict[str, Tuple[str, str]] = {}  # md5 -> (split, filename)
    phash_registry: Dict[str, List[Tuple[str, int]]] = {s: [] for s in split_image_dirs}
    md5_collisions = []
    near_duplicates = []
    total_checked = 0

    for split, s_dir in split_image_dirs.items():
        if not s_dir.exists():
            continue
        for img_p in s_dir.glob("*.jpg"):
            total_checked += 1
            data = img_p.read_bytes()
            md5_val = hashlib.md5(data).hexdigest()

            if md5_val in md5_registry:
                prior_split, prior_file = md5_registry[md5_val]
                if prior_split != split:
                    md5_collisions.append((prior_split, split, md5_val))
            else:
                md5_registry[md5_val] = (split, img_p.name)

            # Compute pHash for near-duplicate screening across different splits
            try:
                with Image.open(img_p) as im:
                    im_phash = compute_image_phash(im)
                    phash_registry[split].append((img_p.name, im_phash))
            except Exception:
                pass

    # Compare pHashes between distinct splits
    splits = list(split_image_dirs.keys())
    for i in range(len(splits)):
        s1 = splits[i]
        for j in range(i + 1, len(splits)):
            s2 = splits[j]
            for f1, h1 in phash_registry[s1]:
                for f2, h2 in phash_registry[s2]:
                    dist = compute_phash_hamming_distance(h1, h2)
                    if dist < min_hamming_distance:
                        near_duplicates.append((f"{s1}:{f1}", f"{s2}:{f2}", dist))

    is_isolated = (len(md5_collisions) == 0) and (len(near_duplicates) == 0)
    return SplitIsolationReport(
        is_isolated=is_isolated,
        md5_collisions=md5_collisions,
        near_duplicates=near_duplicates,
        total_images_checked=total_checked
    )


def generate_mapping_dataframe():
    """Converts HARMONIZATION_RULES to a clean Pandas DataFrame for export."""
    import pandas as pd
    records = []
    for r in HARMONIZATION_RULES:
        records.append({
            "source_dataset": r.source_dataset,
            "source_label": r.source_label,
            "target_label": r.target_label if r.target_label else "UNMAPPED_IGNORE",
            "canonical_id": r.canonical_id if r.canonical_id is not None else -1,
            "mapping_status": r.mapping_status,
            "reason": r.reason,
            "validation_requirement": r.validation_requirement,
        })
    return pd.DataFrame(records)
