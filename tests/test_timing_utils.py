"""Tests for StageTimer and InferenceTimer timing utilities."""

import time
from utils.timing_utils import InferenceTimer, StageTimer, format_duration


def test_format_duration():
    assert format_duration(45) == "00:45"
    assert format_duration(125) == "02:05"
    assert format_duration(3665) == "01:01:05"
    assert format_duration(-5) == "00:00"


def test_stage_timer_lifecycle():
    timer = StageTimer()
    timer.start_stage("Unit Test Stage", estimated_sec=10.0, total_items=5, expected_gpu_vram_gb=1.5)
    assert timer.is_active is True
    assert timer.stage_name == "Unit Test Stage"

    for i in range(1, 6):
        timer.update(i, current_metrics={"loss": 0.5 - i * 0.05}, min_interval_sec=0.0)

    summary = timer.finish_stage(artifacts=["checkpoints/test.pt"])
    assert timer.is_active is False
    assert summary["stage_name"] == "Unit Test Stage"
    assert summary["completed_items"] == 5
    assert "checkpoints/test.pt" in summary["artifacts"]
    assert summary["actual_duration_sec"] >= 0


def test_inference_timer_breakdown_and_summary():
    timer = InferenceTimer()
    for _ in range(5):
        record = timer.record(
            load_ms=10.0,
            preprocess_ms=2.0,
            inference_ms=18.0,
            fusion_ms=1.5,
            postprocess_ms=1.5,
        )
        assert record["total_ms"] == 33.0
        assert record["throughput_fps"] > 0

    summary = timer.summarize()
    assert summary["count"] == 5
    assert summary["mean_total_ms"] == 33.0
    assert summary["mean_inference_ms"] == 18.0
    assert summary["mean_fusion_ms"] == 1.5
    assert summary["mean_fps"] > 0
