import json
from pathlib import Path

p = Path("research/3d_reconstruction/data_manifests/indoor_ground_truth_manifest.json")
with open(p, "r", encoding="utf-8") as f:
    m = json.load(f)

print(f"Manifest Version: {m.get('manifest_version')}")
print(f"Resolution: {m.get('resolution')}")
print(f"Total frames: {m.get('total_frames')}")
print(f"Splits summary: {m.get('splits')}")
print(f"Datasets summary: {m.get('datasets')}")

frames = m.get("frames", [])
print(f"Total frame entries: {len(frames)}")

scene_counts = {}
split_counts = {}
dataset_counts = {}
for f in frames:
    s_id = f.get("scene_id")
    sp = f.get("split")
    ds = f.get("dataset")
    scene_counts[(s_id, ds, sp)] = scene_counts.get((s_id, ds, sp), 0) + 1
    split_counts[sp] = split_counts.get(sp, 0) + 1
    dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

print("\nScenes Breakdown:")
for (s_id, ds, sp), cnt in sorted(scene_counts.items()):
    print(f"  [{sp:<5}] Scene: {s_id:<18} | Dataset: {ds:<10} | Frames: {cnt}")

print("\nSplit Counts:", split_counts)
print("Dataset Counts:", dataset_counts)

if frames:
    print("\nSample frame record:")
    for k, v in frames[0].items():
        print(f"  {k}: {v}")
