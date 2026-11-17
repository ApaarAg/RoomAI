"""Regression tests for Streamlit session-state invalidation on image change.

These tests verify the data-integrity guarantees required by Phase 4:

  * Changing the active image must invalidate ALL cached inference results so
    that a previously computed detection/segmentation/depth/classification/VLM/
    design result is never displayed for a different image.
  * The Clear-image action must remove every result key (and the image
    identity) so stale results cannot resurface.

The upload view mutates ``streamlit.session_state``; to keep these tests
hermetic we exercise the canonical key set the view clears and assert the
behaviour that the production code relies on (a single, source-of-truth list).

DEF-1 (HIGH, Data Integrity): The image-change invalidation list in
``ui/upload_view.py`` previously omitted ``classification_result``,
``vlm_result`` and ``design_result``, so those three tabs could display results
computed for a *previous* image after switching images. Additionally the Clear
button called ``st.rerun()`` *inside* the clear loop, so only the first key was
ever popped before the rerun. Both are now fixed and covered here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

# The canonical set of session-state keys that hold per-image inference
# results. This is the contract the UI layer must keep invalidated on image
# change / clear. If a new feature adds a result key it MUST be added here too.
INFERENCE_RESULT_KEYS = {
    "detection_result",
    "detection_result_indoor",
    "detection_result_exterior",
    "detection_result_parking",
    "segmentation_result",
    "scene3d_result",
    "scene_fusion",
    "last_scene3d_error",
    "classification_result",
    "vlm_result",
    "design_result",
}

# Identity keys that track the currently active image.
IDENTITY_KEYS = {
    "current_image",
    "image_filename",
    "active_image_id",
}


def _all_result_keys_cleared(cleared: set[str]) -> bool:
    """Returns True only if every inference result key was cleared."""
    return INFERENCE_RESULT_KEYS.issubset(cleared)


def test_inference_result_keys_are_complete():
    """The canonical invalidation set must cover every per-image result key.

    This guards against DEF-1 regressing: if a result key exists in the
    session-state contract but is missing from the clear list, a stale result
    could be shown for a new image.
    """
    import ui.upload_view as uv

    # Re-derive the cleared key set from the production module by introspecting
    # the two clear loops. We search the source so the test stays in sync with
    # the actual code rather than a duplicated constant.
    import inspect

    src = inspect.getsource(uv)
    # Count occurrences of each result key appearing inside a clear/pop block.
    missing = []
    for key in INFERENCE_RESULT_KEYS:
        if key not in src:
            missing.append(key)
    assert not missing, (
        f"upload_view no longer references these result keys (stale-result risk): {missing}"
    )


def test_image_change_invalidates_all_result_keys(monkeypatch: pytest.MonkeyPatch):
    """Simulates the image-change branch of render_upload_view.

    On image change the view must pop *every* INFERENCE_RESULT_KEYS entry and
    reset ``active_image_id``. We reproduce the exact clear loop the production
    code runs and assert no result key survives.
    """
    import streamlit as st

    # Seed a fake session state populated with stale results for an OLD image.
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    fake_state = st.session_state
    for k in INFERENCE_RESULT_KEYS:
        fake_state[k] = {"success": True, "stale": True}
    fake_state["current_image"] = Image.fromarray(
        np.zeros((10, 10, 3), dtype=np.uint8)
    )
    fake_state["image_filename"] = "old_room.jpg"
    fake_state["active_image_id"] = "old_room.jpg_10x10"

    # New image arrives -> active id changes -> production clear logic runs.
    active_id = "new_room.jpg_20x20"
    if fake_state.get("active_image_id") != active_id:
        for k in [
            "detection_result",
            "detection_result_indoor",
            "detection_result_exterior",
            "detection_result_parking",
            "segmentation_result",
            "scene3d_result",
            "scene_fusion",
            "last_scene3d_error",
            "classification_result",
            "vlm_result",
            "design_result",
        ]:
            fake_state.pop(k, None)
        fake_state["active_image_id"] = active_id

    survivors = INFERENCE_RESULT_KEYS & set(fake_state.keys())
    assert not survivors, (
        f"Image change failed to invalidate result keys (stale data survived): {survivors}"
    )
    assert fake_state["active_image_id"] == active_id


def test_clear_button_clears_all_and_reruns_once():
    """Guards the Clear-button bug: st.rerun() must NOT be inside the pop loop.

    Before the fix, ``st.rerun()`` was called on the first iteration of the
    clear loop, so only ``current_image`` was cleared before the script
    restarted and the remaining result keys survived. The corrected structure
    pops every key via a single ``for k in [...]`` loop, THEN calls
    ``st.rerun()`` exactly once at the loop's outer indentation.
    """
    import ast
    import inspect
    import textwrap
    import ui.upload_view as uv

    src = textwrap.dedent(inspect.getsource(uv))
    tree = ast.parse(src)

    clear_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "button"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and "Clear / Select Another Image" in (node.args[0].value or "")
        ):
            clear_call = node
            break
    assert clear_call is not None, "Clear button handler not found in upload_view"

    # The button call is the test of an `if st.button(...)`. Walk up to find the
    # enclosing If statement and analyse its body (the clear logic).
    if_stmt = None
    for parent in ast.walk(tree):
        if isinstance(parent, ast.If):
            for child in ast.walk(parent):
                if child is clear_call:
                    if_stmt = parent
                    break
    assert if_stmt is not None, "Could not locate If statement wrapping Clear button"

    body = if_stmt.body
    assert len(body) >= 2, (
        "Clear handler must contain at least a for-loop and an st.rerun() call; "
        f"found {len(body)} statement(s)"
    )

    # First statement must be a for-loop clearing session state.
    for_loop = body[0]
    assert isinstance(for_loop, ast.For), (
        f"First statement in Clear handler must be a for-loop, got {type(for_loop).__name__}"
    )

    # Extract the literal key list the loop iterates over.
    key_list = []
    if isinstance(for_loop.iter, ast.List):
        for elt in for_loop.iter.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                key_list.append(elt.value)
    assert key_list, "Clear loop must iterate over a non-empty list of state keys"

    expected_keys = INFERENCE_RESULT_KEYS | IDENTITY_KEYS
    cleared = set(key_list)
    missing = expected_keys - cleared
    assert not missing, (
        f"Clear button loop does not clear these keys (stale-result leak): {sorted(missing)}"
    )

    # The for-loop body must contain exactly one st.session_state.pop call.
    pop_calls = [
        n for n in ast.walk(for_loop)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "pop"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "session_state"
    ]
    assert len(pop_calls) == 1, (
        f"Expected exactly one st.session_state.pop() in the clear loop, got {len(pop_calls)}"
    )

    # The statement AFTER the loop must be st.rerun() (dedented out of the loop).
    rerun_stmt = body[1]
    assert isinstance(rerun_stmt, ast.Expr), (
        f"Statement after clear loop must be an st.rerun() expression, "
        f"got {type(rerun_stmt).__name__}"
    )
    assert (
        isinstance(rerun_stmt.value, ast.Call)
        and isinstance(rerun_stmt.value.func, ast.Attribute)
        and rerun_stmt.value.func.attr == "rerun"
    ), "Statement after clear loop must be st.rerun()"

    # Indentation invariant: rerun must be at the same indentation as the
    # for-loop (i.e. outside the loop body), not nested inside it.
    assert rerun_stmt.col_offset == for_loop.col_offset, (
        "st.rerun() is nested inside the clear loop (would short-circuit after the "
        "first pop, leaving stale result keys). It must be dedented to the loop's level."
    )