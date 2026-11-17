import cv2
import numpy as np
import os

dir_p = "web/public/assets/roomai/pipeline"
out_dir = "artifacts/transition_audit/midpoints"
os.makedirs(out_dir, exist_ok=True)

stages = [
    ("01", "01_rgb_image.webp"),
    ("02", "02_hunyuan_scene_priors.webp"),
    ("03", "03_raw_depth_concept.webp"),
    ("04", "04_depth_unprojection_concept.webp"),
    ("05", "05_point_cloud_concept.webp"),
    ("06", "06_ply_glb_export_concept.webp"),
]

imgs = [cv2.imread(os.path.join(dir_p, s[1])) for s in stages]
h, w, _ = imgs[0].shape

def render_composite(img_from, img_to, raw_mix):
    # Equal-energy crossfade
    out_op = np.cos(raw_mix * (np.pi / 2))
    in_op = np.sin(raw_mix * (np.pi / 2))

    # Micro-scale
    s_out = 1.0 + 0.003 * raw_mix
    s_in = 0.997 + 0.003 * raw_mix

    # Scaled versions
    M_out = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s_out)
    scaled_out = cv2.warpAffine(img_from, M_out, (w, h), borderMode=cv2.BORDER_REFLECT)
    scaled_out = np.clip(scaled_out.astype(np.float32) * (1.0 - 0.03 * raw_mix), 0, 255)

    M_in = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s_in)
    scaled_in = cv2.warpAffine(img_to, M_in, (w, h), borderMode=cv2.BORDER_REFLECT)
    scaled_in = np.clip(scaled_in.astype(np.float32) * (0.97 + 0.03 * raw_mix), 0, 255)

    # Feathered aperture gradient mask
    # 20% width feather centered at sweepX
    sweep_x = raw_mix * w
    feather = 0.20 * w
    x_coords = np.arange(w, dtype=np.float32)
    # Mask is 1 left of (sweep_x - feather/2), 0 right of (sweep_x + feather/2)
    mask = 1.0 - np.clip((x_coords - (sweep_x - feather / 2)) / feather, 0, 1)
    mask_2d = np.tile(mask[np.newaxis, :, np.newaxis], (h, 1, 3))

    # Composite:
    # Base is scaled_out modulated by out_op
    # Incoming is scaled_in modulated by in_op * mask_2d
    # Normalizing factor so luminance never drops
    comp = scaled_out * out_op * (1.0 - mask_2d * in_op) + scaled_in * in_op * mask_2d
    return np.clip(comp, 0, 255).astype(np.uint8)

print("Rendering visual midpoint audit...")
for i in range(len(stages) - 1):
    name_from, file_from = stages[i]
    name_to, file_to = stages[i+1]
    
    start_img = imgs[i]
    mid_img = render_composite(imgs[i], imgs[i+1], 0.5)
    end_img = imgs[i+1]
    
    cv2.imwrite(f"{out_dir}/trans_{name_from}_to_{name_to}_00_start.png", start_img)
    cv2.imwrite(f"{out_dir}/trans_{name_from}_to_{name_to}_50_midpoint.png", mid_img)
    cv2.imwrite(f"{out_dir}/trans_{name_from}_to_{name_to}_100_end.png", end_img)
    
    # Save a side-by-side comparison strip
    small_start = cv2.resize(start_img, (480, 270))
    small_mid = cv2.resize(mid_img, (480, 270))
    small_end = cv2.resize(end_img, (480, 270))
    
    cv2.putText(small_start, f"{name_from} (0%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(small_mid, f"MIDPOINT (50%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(small_end, f"{name_to} (100%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    strip = np.hstack([small_start, small_mid, small_end])
    cv2.imwrite(f"{out_dir}/strip_{name_from}_to_{name_to}.png", strip)
    print(f"Transition {name_from} -> {name_to}: Start, Midpoint (50%), End rendered successfully.")
