"""Debug runner to capture exact error."""
import sys
import traceback

try:
    import train_stage1_corrected
    device = "cuda"
    train_stage1_corrected.run_training_experiment(
        "debug_exp",
        steps=5,
        lr=1e-4,
        dataset_filter="hypersim",
        device=device,
    )
except Exception as e:
    print("CAUGHT EXCEPTION IN DEBUG RUNNER:", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

