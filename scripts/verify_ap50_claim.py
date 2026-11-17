import json
from pathlib import Path

with open("reports/exterior_phase9_b4_evaluation.json") as f:
    d = json.load(f)

matrix = d["results_matrix"]
b2 = matrix["B2 (YOLO11s Baseline)"]
b4 = matrix["B4 (YOLO11s Curated)"]

print("=========================================================================================")
print("EXAMINING CLAIM: 'B4 achieves higher AP50 than B2 across all 10 classes'")
print("=========================================================================================")

for proto in ["Canonical_Original_134", "Canonical_Corrected_134", "Curated_Total_154"]:
    print(f"\n--- Protocol: {proto} ---")
    b2_proto = b2[proto]["per_class"]
    b4_proto = b4[proto]["per_class"]
    
    all_higher = True
    for cid in range(10):
        # find class name
        cname = [k for k, v in b2_proto.items() if v["class_id"] == cid][0]
        b2_ap = b2_proto[cname]["ap50"]
        b4_ap = b4_proto[cname]["ap50"]
        diff = b4_ap - b2_ap
        is_higher = diff > 0
        if not is_higher:
            all_higher = False
        status = "HIGHER (+)" if diff > 0 else ("LOWER (-)" if diff < 0 else "EQUAL (=)")
        print(f"  Class {cid} ({cname:<14}): B2={b2_ap:.4f} | B4={b4_ap:.4f} | diff={diff:+.4f} | {status}")
    print(f"-> Claim supported on {proto}? {all_higher}")
