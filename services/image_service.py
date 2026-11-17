"""Image manipulation and visualization service for RoomAI."""

from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from schemas.responses import DetectionItem, SegmentationMask
from utils.logging_config import get_logger

logger = get_logger("ImageService")

# High-contrast, aesthetic color palette for detection boxes and segmentation masks
PALETTE = [
    (59, 130, 246),   # Royal Blue
    (16, 185, 129),   # Emerald Green
    (245, 158, 11),   # Amber
    (239, 68, 68),    # Crimson Red
    (139, 92, 246),   # Violet
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (249, 115, 22),   # Orange
]


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[DetectionItem],
    box_width: int = 3,
) -> Image.Image:
    """Draws clear bounding boxes with label pills and confidence percentages on the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = annotated.size

    # Attempt to load a default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, item in enumerate(detections):
        color = PALETTE[idx % len(PALETTE)]
        left, top, right, bottom = item.box.to_pixel_coords(img_w, img_h)

        # Draw box outline
        draw.rectangle([left, top, right, bottom], outline=color, width=box_width)

        # Format label pill
        label_text = f"{item.label} {int(item.confidence * 100)}%"

        # Estimate text bbox
        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((0, 0), label_text, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        else:
            text_w = len(label_text) * 7
            text_h = 12

        pill_top = max(0, top - text_h - 4)
        pill_bottom = pill_top + text_h + 4
        pill_right = min(img_w, left + text_w + 8)

        # Draw filled pill background
        draw.rectangle([left, pill_top, pill_right, pill_bottom], fill=color)
        # Draw white text
        draw.text((left + 4, pill_top + 2), label_text, fill=(255, 255, 255), font=font)

    return annotated


def create_segmentation_overlay(
    image: Image.Image,
    mask_array: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blends a 2D category mask with the original image using semi-transparent color overlay."""
    img_rgb = np.array(image.convert("RGB"))
    h, w = mask_array.shape[:2]

    # Resize mask if not strictly matching image
    if (img_rgb.shape[0], img_rgb.shape[1]) != (h, w):
        pil_mask = Image.fromarray(mask_array.astype(np.uint8))
        pil_mask = pil_mask.resize((img_rgb.shape[1], img_rgb.shape[0]), Image.Resampling.NEAREST)
        mask_array = np.array(pil_mask)

    color_mask = np.zeros_like(img_rgb)
    unique_ids = np.unique(mask_array)

    for uid in unique_ids:
        if uid == 0:
            continue  # Keep background clean or map subtle color
        color = PALETTE[(int(uid) - 1) % len(PALETTE)]
        color_mask[mask_array == uid] = color

    # Alpha blend where mask is non-zero
    is_masked = (mask_array > 0)[:, :, np.newaxis]
    blended = np.where(
        is_masked,
        (1 - alpha) * img_rgb + alpha * color_mask,
        img_rgb,
    ).astype(np.uint8)

    return Image.fromarray(blended)


def create_depth_heatmap(depth_array: np.ndarray) -> Image.Image:
    """Converts a 2D floating-point or uint8 depth map into a colored heatmap visualization."""
    norm = depth_array.astype(np.float32)
    min_val, max_val = norm.min(), norm.max()
    if max_val > min_val:
        norm = (norm - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(norm)

    # Use a clean 3-channel gradient from dark purple to bright yellow (Plasma-like)
    r = (np.clip(2.0 * norm, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(2.0 * (1.0 - np.abs(norm - 0.5)), 0, 1) * 220).astype(np.uint8)
    b = (np.clip(2.0 * (1.0 - norm), 0, 1) * 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb)


def create_side_by_side_comparison(
    image1: Image.Image,
    image2: Image.Image,
    label1: str = "Original Room",
    label2: str = "Redesigned Concept",
) -> Image.Image:
    """Composes two images horizontally with clear header labels for comparison."""
    # Match height
    target_h = max(image1.height, image2.height)
    w1 = int(image1.width * (target_h / image1.height))
    w2 = int(image2.width * (target_h / image2.height))

    img1_resized = image1.resize((w1, target_h), Image.Resampling.LANCZOS)
    img2_resized = image2.resize((w2, target_h), Image.Resampling.LANCZOS)

    header_h = 36
    total_w = w1 + w2 + 10  # 10px divider
    total_h = target_h + header_h

    canvas = Image.new("RGB", (total_w, total_h), color=(26, 32, 44))
    draw = ImageDraw.Draw(canvas)

    # Paste images
    canvas.paste(img1_resized, (0, header_h))
    canvas.paste(img2_resized, (w1 + 10, header_h))

    # Add header labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((12, 10), label1, fill=(226, 232, 240), font=font)
    draw.text((w1 + 22, 10), label2, fill=(226, 232, 240), font=font)

    return canvas
