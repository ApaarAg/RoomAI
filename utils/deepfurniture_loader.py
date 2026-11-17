"""Hugging Face datasets adapter for Dalby123/DeepFurniture."""

import json
from pathlib import Path
from typing import Optional
from PIL import Image

try:
    from datasets import Dataset, DatasetDict, Features, Value, Sequence
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

CATEGORY_MAP = {
    1: "cabinet#shelf",
    2: "table",
    3: "chair#stool",
    4: "lamp",
    5: "door",
    6: "bed",
    7: "sofa",
    8: "plant",
    9: "decoration",
    10: "curtain",
    11: "home-appliance",
}


def load_deepfurniture_dataset(data_dir: Optional[str] = None):
    """Loads DeepFurniture scenes as a Hugging Face DatasetDict.

    Args:
        data_dir: Path to prepared DeepFurniture directory. Defaults to RoomAI/data/deepfurniture.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits containing:
            - scene_id (str)
            - image_path (str)
            - width (int)
            - height (int)
            - categories (List[str])
            - bboxes (List[List[float]]) in normalized [x_min, y_min, x_max, y_max] format.
    """
    if not HAS_DATASETS:
        raise ImportError("Hugging Face datasets package is required. Install via pip install datasets.")

    base_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data" / "deepfurniture"
    splits = {}

    split_map = {"train": "train", "val": "validation", "test": "test"}

    for split_src, split_name in split_map.items():
        img_dir = base_dir / "images" / split_src
        lbl_dir = base_dir / "labels" / split_src

        records = []
        if img_dir.exists():
            for img_file in sorted(img_dir.glob("*.jpg")):
                stem = img_file.stem
                lbl_file = lbl_dir / f"{stem}.txt"

                bboxes = []
                category_names = []

                if lbl_file.exists():
                    with open(lbl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                xc = float(parts[1])
                                yc = float(parts[2])
                                bw = float(parts[3])
                                bh = float(parts[4])

                                x_min = round(max(0.0, xc - bw / 2.0), 4)
                                y_min = round(max(0.0, yc - bh / 2.0), 4)
                                x_max = round(min(1.0, xc + bw / 2.0), 4)
                                y_max = round(min(1.0, yc + bh / 2.0), 4)

                                bboxes.append([x_min, y_min, x_max, y_max])
                                category_names.append(CATEGORY_MAP.get(cls_id + 1, "unknown"))

                records.append({
                    "scene_id": stem,
                    "image_path": str(img_file),
                    "categories": category_names,
                    "bboxes": bboxes,
                    "total_objects": len(bboxes),
                })

        splits[split_name] = Dataset.from_list(records)

    return DatasetDict(splits)
