"""
ROOMAI — PHASE 7.1: CONTROLLED FORENSIC INSPECTION OF DEPTH SEMANTICS
Compares raw and scale-aligned depth for Pretrained Baseline vs Stage 1 on Hypersim and ScanNet.
"""
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror

def load_m(p):
    d = torch.load(p, map_location='cpu')
    sd = d.get('state_dict', d)
    sd = {k.replace('model.', ''): v for k, v in sd.items()}
    m = WorldMirror(
        img_size=518, patch_size=14, embed_dim=1024, gs_dim=256,
        enable_cam=True, enable_pts=True, enable_depth=True, enable_norm=True,
        enable_gs=False, enable_cond=False
    ).to('cuda', dtype=torch.bfloat16)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def main():
    m_base = load_m(REPO_ROOT / 'research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt')
    m_s1 = load_m(REPO_ROOT / 'research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt')

    manifest_path = REPO_ROOT / 'research/3d_reconstruction/data_manifests/indoor_ground_truth_manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        m = json.load(f)

    val_frames = [f for f in m['frames'] if f['split'] == 'val']
    hyp_frames = [f for f in val_frames if f['dataset'] == 'hypersim'][:5]
    
    # 5 frames across scene0030_00, 01, 02
    scan_00 = [f for f in val_frames if f['scene_id'] == 'scene0030_00'][:2]
    scan_01 = [f for f in val_frames if f['scene_id'] == 'scene0030_01'][:2]
    scan_02 = [f for f in val_frames if f['scene_id'] == 'scene0030_02'][:1]
    scan_frames = scan_00 + scan_01 + scan_02

    print("=" * 80)
    print("PHASE 7.1: CONTROLLED FORENSIC PROBE — RAW VS SCALE-ALIGNED DEPTH")
    print("=" * 80)

    for label, flist in [('HYPERSIM (CAD Synthetic)', hyp_frames), ('SCANNET (IR Structured Light Sensor)', scan_frames)]:
        print(f"\n{'#'*40}\nDOMAIN: {label}\n{'#'*40}")
        for fr in flist:
            rgb_path = REPO_ROOT / 'research/3d_reconstruction/data/ground_truth' / fr['paths']['rgb']
            depth_path = REPO_ROOT / 'research/3d_reconstruction/data/ground_truth' / fr['paths']['depth']
            mask_path = REPO_ROOT / 'research/3d_reconstruction/data/ground_truth' / fr['paths']['mask']
            cam_path = REPO_ROOT / 'research/3d_reconstruction/data/ground_truth' / fr['paths']['cam']

            rgb_img = Image.open(rgb_path).convert('RGB')
            rgb = torch.from_numpy(np.array(rgb_img, dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).unsqueeze(0).to('cuda', dtype=torch.bfloat16)
            gt_d = np.load(depth_path)
            mask = np.array(Image.open(mask_path)) > 0
            valid = mask & (gt_d > 0.1) & np.isfinite(gt_d)

            with torch.no_grad():
                out_b = m_base(views={'img': rgb}, cond_flags=[0,0,0], is_inference=True)
                out_s1 = m_s1(views={'img': rgb}, cond_flags=[0,0,0], is_inference=True)

            pred_b = out_b['depth'].squeeze().float().cpu().numpy()
            pred_s1 = out_s1['depth'].squeeze().float().cpu().numpy()

            # Raw AbsRel and RMSE
            absrel_b_raw = float(np.mean(np.abs(pred_b[valid] - gt_d[valid]) / gt_d[valid]))
            absrel_s1_raw = float(np.mean(np.abs(pred_s1[valid] - gt_d[valid]) / gt_d[valid]))
            rmse_b_raw = float(np.sqrt(np.mean((pred_b[valid] - gt_d[valid])**2)))
            rmse_s1_raw = float(np.sqrt(np.mean((pred_s1[valid] - gt_d[valid])**2)))

            # Official WorldMirror median scaling: scale_factor = median(gt) / median(pred)
            med_gt = float(np.median(gt_d[valid]))
            med_b = float(np.median(pred_b[valid]))
            med_s1 = float(np.median(pred_s1[valid]))

            s_b = med_gt / max(med_b, 1e-6)
            s_s1 = med_gt / max(med_s1, 1e-6)

            pred_b_scaled = pred_b * s_b
            pred_s1_scaled = pred_s1 * s_s1

            absrel_b_scaled = float(np.mean(np.abs(pred_b_scaled[valid] - gt_d[valid]) / gt_d[valid]))
            absrel_s1_scaled = float(np.mean(np.abs(pred_s1_scaled[valid] - gt_d[valid]) / gt_d[valid]))
            rmse_b_scaled = float(np.sqrt(np.mean((pred_b_scaled[valid] - gt_d[valid])**2)))
            rmse_s1_scaled = float(np.sqrt(np.mean((pred_s1_scaled[valid] - gt_d[valid])**2)))

            print(f"Frame {fr['scene_id']} #{fr['frame_id']:04d}:")
            print(f"  GT depth range:    min={gt_d[valid].min():.2f}m, max={gt_d[valid].max():.2f}m, median={med_gt:.2f}m")
            print(f"  Base RAW depth:    min={pred_b[valid].min():.2f}, max={pred_b[valid].max():.2f}, median={med_b:.2f} | Scale s={s_b:.3f}")
            print(f"  Stage1 RAW depth:  min={pred_s1[valid].min():.2f}, max={pred_s1[valid].max():.2f}, median={med_s1:.2f} | Scale s={s_s1:.3f}")
            print(f"  RAW AbsRel:        Base = {absrel_b_raw:.4f} | Stage1 = {absrel_s1_raw:.4f} (Ratio S1/Base: {absrel_s1_raw/absrel_b_raw:.2f}x)")
            print(f"  RAW RMSE:          Base = {rmse_b_raw:.3f}m | Stage1 = {rmse_s1_raw:.3f}m")
            print(f"  SCALED AbsRel:     Base = {absrel_b_scaled:.4f} | Stage1 = {absrel_s1_scaled:.4f} (Ratio S1/Base: {absrel_s1_scaled/absrel_b_scaled:.2f}x)")
            print(f"  SCALED RMSE:       Base = {rmse_b_scaled:.3f}m | Stage1 = {rmse_s1_scaled:.3f}m\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()

