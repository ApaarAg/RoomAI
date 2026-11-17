"""Empirical Challenger Test Suite for Milestone 1: State Invalidation & Workspace Consolidation.

This suite executes exhaustive adversarial stress tests against:
1. ui/upload_view.py:
   - Dynamic upload of Image A with mock inference results populated for all 11 keys.
   - Upload of Image B with identical filename and dimensions but distinct content (SHA-256 difference).
   - Verification that all 11 inference keys are purged when switching from Image A to Image B.
   - Re-upload of Image A verifying SHA-256 identity match and purging of Image B's results.
   - Idempotency test: re-rendering the same image without content change preserves cached results.
   - Clearing image (uploaded_file is None) verifying all 11 inference keys and 3 identity keys are purged.
   - Corrupted/invalid image upload verifying complete state purge.
   - UploadedFile stream handling: verifying both getvalue() and read()/seek() fallbacks work.
   - Full AST verification of Clear button loop, None-file handler, invalid file handler, and rerun scoping.
2. app.py & ui/analysis_view.py:
   - Verification of workspace consolidation (exactly 3 primary workspaces: Detection, VLM, 3D Reconstruction).
   - AST verification of app.py confirming elimination of legacy outer radio selector.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import textwrap
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest

import app
import ui.analysis_view as av
import ui.upload_view as uv


# Canonical keys according to PROJECT.md and test_state_invalidation.py
CANONICAL_11_INFERENCE_KEYS = [
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
]

IDENTITY_KEYS = [
    "current_image",
    "image_filename",
    "active_image_id",
]

ALL_14_KEYS = IDENTITY_KEYS + CANONICAL_11_INFERENCE_KEYS


class DummyUploadedFile(io.BytesIO):
    """Simulates streamlit.runtime.uploaded_file_manager.UploadedFile with getvalue()."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
        self.size = len(data)


class DummyStreamWithoutGetValue:
    """Simulates a file-like object that lacks .getvalue() (only has read, seek, tell)."""

    def __init__(self, data: bytes, name: str):
        self._bio = io.BytesIO(data)
        self.name = name
        self.size = len(data)

    def read(self, *args, **kwargs):
        return self._bio.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._bio.seek(*args, **kwargs)

    def tell(self):
        return self._bio.tell()


class MockColumnContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_test_image(color: tuple[int, int, int], size: tuple[int, int] = (640, 480)) -> bytes:
    """Creates a synthetic RGB image with specified solid color and dimensions."""
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_app_config() -> Dict[str, Any]:
    return {
        "upload": {
            "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp"],
            "max_file_size_mb": 15.0,
            "max_image_dimension": 4096,
        }
    }


class TestEmpiricalStateInvalidation:
    """Empirically tests state invalidation by executing ui.upload_view.render_upload_view."""

    def test_image_a_to_b_state_purge_with_identical_filename_and_dimensions(self, mock_app_config):
        """Simulate upload of Image A, populate 11 keys, then upload Image B with identical filename/size."""
        session_state: Dict[str, Any] = {}

        # 1. Create Image A and Image B with identical dimensions and filename
        bytes_a = create_test_image(color=(255, 0, 0), size=(640, 480))
        bytes_b = create_test_image(color=(0, 255, 0), size=(640, 480))

        sha_a = hashlib.sha256(bytes_a).hexdigest()
        sha_b = hashlib.sha256(bytes_b).hexdigest()
        assert sha_a != sha_b, "Image A and Image B must have distinct SHA-256 hashes"

        file_a = DummyUploadedFile(bytes_a, name="same_name_room.jpg")
        file_b = DummyUploadedFile(bytes_b, name="same_name_room.jpg")

        # Mock streamlit components
        with patch("ui.upload_view.st") as mock_st, patch("ui.upload_view.render_image") as mock_render_img:
            mock_st.session_state = session_state
            mock_st.columns.return_value = (MockColumnContext(), MockColumnContext())
            mock_st.button.return_value = False

            # --- STEP 1: Upload Image A ---
            mock_st.file_uploader.return_value = file_a
            result_img_a = uv.render_upload_view(mock_app_config)

            assert result_img_a is not None, "Image A should validate and render successfully"
            assert session_state["active_image_id"] == sha_a, "active_image_id must match SHA-256 of Image A"
            assert session_state["image_filename"] == "same_name_room.jpg"

            # Populate mock inference results for ALL 11 keys
            for k in CANONICAL_11_INFERENCE_KEYS:
                session_state[k] = {"timestamp": 12345, "model": k, "data": [1, 2, 3]}

            for k in CANONICAL_11_INFERENCE_KEYS:
                assert k in session_state, f"Key {k} should be present in session_state"

            # --- STEP 2: Upload Image B (same filename, same dimensions, different content) ---
            file_b.seek(0)
            mock_st.file_uploader.return_value = file_b
            result_img_b = uv.render_upload_view(mock_app_config)

            assert result_img_b is not None, "Image B should validate and render successfully"
            assert session_state["active_image_id"] == sha_b, "active_image_id must update to SHA-256 of Image B"
            assert session_state["active_image_id"] != sha_a, "SHA-256 must change despite identical filename"

            # CRITICAL CHECK: Every single one of the 11 inference keys MUST be purged!
            surviving_keys = [k for k in CANONICAL_11_INFERENCE_KEYS if k in session_state]
            assert not surviving_keys, f"Inference keys survived image switch from A to B: {surviving_keys}"

            # --- STEP 3: Re-upload Image A ---
            # Populate mock inference results for Image B
            for k in CANONICAL_11_INFERENCE_KEYS:
                session_state[k] = {"image_b_result": True}

            file_a_repeat = DummyUploadedFile(bytes_a, name="same_name_room.jpg")
            mock_st.file_uploader.return_value = file_a_repeat
            result_img_a2 = uv.render_upload_view(mock_app_config)

            assert result_img_a2 is not None
            assert session_state["active_image_id"] == sha_a, "active_image_id must revert to SHA-256 of Image A"
            surviving_b_keys = [k for k in CANONICAL_11_INFERENCE_KEYS if k in session_state]
            assert not surviving_b_keys, f"Image B inference keys survived switch back to Image A: {surviving_b_keys}"

    def test_same_image_re_render_preserves_cached_results(self, mock_app_config):
        """Verify idempotency: re-rendering the EXACT SAME image preserves valid cached results."""
        session_state: Dict[str, Any] = {}
        bytes_a = create_test_image(color=(128, 128, 128), size=(640, 480))
        sha_a = hashlib.sha256(bytes_a).hexdigest()

        with patch("ui.upload_view.st") as mock_st, patch("ui.upload_view.render_image"):
            mock_st.session_state = session_state
            mock_st.columns.return_value = (MockColumnContext(), MockColumnContext())
            mock_st.button.return_value = False

            # Initial upload
            file_a = DummyUploadedFile(bytes_a, name="test.jpg")
            mock_st.file_uploader.return_value = file_a
            uv.render_upload_view(mock_app_config)

            # Store computed results
            for k in CANONICAL_11_INFERENCE_KEYS:
                session_state[k] = {"computed_for": sha_a}

            # Re-render with identical image
            file_a_same = DummyUploadedFile(bytes_a, name="test.jpg")
            mock_st.file_uploader.return_value = file_a_same
            uv.render_upload_view(mock_app_config)

            # Assert results are NOT wiped when image is identical
            for k in CANONICAL_11_INFERENCE_KEYS:
                assert k in session_state, f"Idempotent re-render should preserve key {k}"
                assert session_state[k] == {"computed_for": sha_a}

    def test_uploaded_file_is_none_cleanses_all_14_keys(self, mock_app_config):
        """When uploaded_file is None (file cleared or removed), all 11 inference keys + 3 identity keys are purged."""
        session_state: Dict[str, Any] = {}
        for k in ALL_14_KEYS:
            session_state[k] = "dummy_value"

        with patch("ui.upload_view.st") as mock_st:
            mock_st.session_state = session_state
            mock_st.file_uploader.return_value = None

            res = uv.render_upload_view(mock_app_config)

            assert res is None, "render_upload_view must return None when uploaded_file is None"
            mock_st.info.assert_called_once()

            surviving = [k for k in ALL_14_KEYS if k in session_state]
            assert not surviving, f"Keys survived uploaded_file=None: {surviving}"

    def test_corrupted_file_upload_cleanses_all_14_keys(self, mock_app_config):
        """When an invalid/corrupted file is uploaded, all 14 keys are purged and an error is shown."""
        session_state: Dict[str, Any] = {}
        for k in ALL_14_KEYS:
            session_state[k] = "old_stale_value"

        corrupted_bytes = b"NOT_A_REAL_IMAGE_CORRUPTED_DATA_12345"
        corrupt_file = DummyUploadedFile(corrupted_bytes, name="corrupted.jpg")

        with patch("ui.upload_view.st") as mock_st:
            mock_st.session_state = session_state
            mock_st.file_uploader.return_value = corrupt_file

            res = uv.render_upload_view(mock_app_config)

            assert res is None, "render_upload_view must return None for corrupted image"
            mock_st.error.assert_called_once()
            assert "Upload Error" in mock_st.error.call_args[0][0]

            surviving = [k for k in ALL_14_KEYS if k in session_state]
            assert not surviving, f"Keys survived corrupted upload: {surviving}"

    def test_clear_button_execution_and_rerun(self, mock_app_config):
        """Clicking the Clear button purges all 14 keys and immediately calls st.rerun()."""
        session_state: Dict[str, Any] = {}
        bytes_a = create_test_image(color=(100, 150, 200), size=(640, 480))
        file_a = DummyUploadedFile(bytes_a, name="room.jpg")

        with patch("ui.upload_view.st") as mock_st, patch("ui.upload_view.render_image"):
            mock_st.session_state = session_state
            mock_st.columns.return_value = (MockColumnContext(), MockColumnContext())
            mock_st.file_uploader.return_value = file_a

            # Simulate Clear button being clicked
            mock_st.button.return_value = True

            # Populate keys before click
            for k in ALL_14_KEYS:
                session_state[k] = "test_val"

            uv.render_upload_view(mock_app_config)

            mock_st.rerun.assert_called_once()
            surviving = [k for k in ALL_14_KEYS if k in session_state]
            assert not surviving, f"Keys survived Clear button click: {surviving}"

    def test_uploaded_file_without_getvalue_fallback(self, mock_app_config):
        """Verify stream fallback for file objects lacking .getvalue() (using tell, read, seek)."""
        session_state: Dict[str, Any] = {}
        bytes_a = create_test_image(color=(50, 75, 100), size=(640, 480))
        sha_a = hashlib.sha256(bytes_a).hexdigest()

        file_stream = DummyStreamWithoutGetValue(bytes_a, name="stream_room.jpg")

        with patch("ui.upload_view.st") as mock_st, patch("ui.upload_view.render_image"):
            mock_st.session_state = session_state
            mock_st.columns.return_value = (MockColumnContext(), MockColumnContext())
            mock_st.button.return_value = False
            mock_st.file_uploader.return_value = file_stream

            res = uv.render_upload_view(mock_app_config)

            assert res is not None
            assert session_state["active_image_id"] == sha_a, "active_image_id must correctly read stream via read()/seek()"


class TestEmpiricalASTIntegrity:
    """Rigorous AST structural analysis on ui/upload_view.py and app.py."""

    def test_upload_view_ast_all_pop_blocks_complete(self):
        """Ensure all 4 clear/pop sites in ui/upload_view.py contain the required keys."""
        src = inspect.getsource(uv)
        tree = ast.parse(src)

        # Find all for loops whose target is 'k' and whose body has st.session_state.pop
        pop_loops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "k":
                if isinstance(node.iter, ast.List):
                    keys = [elt.value for elt in node.iter.elts if isinstance(elt, ast.Constant)]
                    pop_loops.append(keys)

        assert len(pop_loops) >= 4, f"Expected at least 4 key-clearing loops, found {len(pop_loops)}"

        # Loop 1: uploaded_file is None (must clear all 14 keys)
        # Loop 2: validation not valid (must clear all 14 keys)
        # Loop 3: active_image_id changed (must clear all 11 inference keys)
        # Loop 4: clear button clicked (must clear all 14 keys)
        full_clear_loops = [loop for loop in pop_loops if set(CANONICAL_11_INFERENCE_KEYS).issubset(set(loop))]
        assert len(full_clear_loops) >= 4, (
            f"Not all pop loops cover all 11 inference keys. Found: {pop_loops}"
        )

        # Check that the 3 full-purge loops contain IDENTITY_KEYS as well
        fourteen_key_loops = [loop for loop in pop_loops if set(ALL_14_KEYS).issubset(set(loop))]
        assert len(fourteen_key_loops) >= 3, (
            f"Expected at least 3 loops purging all 14 keys (None, invalid, Clear button). Found: {len(fourteen_key_loops)}"
        )

    def test_upload_view_clear_button_rerun_scoping(self):
        """AST check: st.rerun() must be dedented outside the for-loop in clear button handler."""
        src = textwrap.dedent(inspect.getsource(uv))
        tree = ast.parse(src)

        clear_if = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Attribute)
                and node.test.func.attr == "button"
            ):
                if node.test.args and "Clear" in getattr(node.test.args[0], "value", ""):
                    clear_if = node
                    break

        assert clear_if is not None, "Clear button If block not found in AST"

        # Body must have For loop as body[0] and Expr(st.rerun()) as body[1]
        for_loop = clear_if.body[0]
        assert isinstance(for_loop, ast.For), f"First child must be ast.For, got {type(for_loop).__name__}"

        rerun_expr = clear_if.body[1]
        assert isinstance(rerun_expr, ast.Expr), f"Second child must be ast.Expr, got {type(rerun_expr).__name__}"
        assert rerun_expr.value.func.attr == "rerun"

        # Crucial: rerun_expr must not be inside for_loop.body
        for inner in ast.walk(for_loop):
            if inner is rerun_expr or (
                isinstance(inner, ast.Call)
                and isinstance(getattr(inner, "func", None), ast.Attribute)
                and inner.func.attr == "rerun"
            ):
                pytest.fail("st.rerun() was found nested inside the for-loop! It must be at the loop's outer indentation.")

    def test_app_consolidated_navigation_ast(self):
        """AST check: app.py must not have outer radio selector, and directly call render_analysis_view."""
        from pathlib import Path
        app_file = Path(__file__).resolve().parent.parent / "app.py"
        src = app_file.read_text(encoding="utf-8")
        tree = ast.parse(src)

        # 1. No st.radio in main()
        radio_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(getattr(n, "func", None), ast.Attribute)
            and n.func.attr == "radio"
        ]
        assert len(radio_calls) == 0, f"app.py should not contain st.radio calls, found {len(radio_calls)}"

        # 2. Must call render_analysis_view
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(getattr(n, "func", None), ast.Name)
            and n.func.id == "render_analysis_view"
        ]
        assert len(calls) >= 1, "app.py must call render_analysis_view"

    def test_analysis_view_three_primary_workspaces(self):
        """Verify ui/analysis_view.py contains exactly the three primary workspaces in st.tabs."""
        src = inspect.getsource(av.render_analysis_view)
        tree = ast.parse(src)

        tabs_args = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(getattr(node, "func", None), ast.Attribute)
                and node.func.attr == "tabs"
                and node.args
                and isinstance(node.args[0], ast.List)
            ):
                tabs_args = [elt.value for elt in node.args[0].elts if isinstance(elt, ast.Constant)]
                break

        assert tabs_args is not None, "st.tabs call not found in render_analysis_view"
        assert len(tabs_args) == 3, f"Expected exactly 3 tabs, got {tabs_args}"
        assert any("Object Detection" in t for t in tabs_args)
        assert any("Vision Q&A" in t for t in tabs_args)
        assert any("3D Scene Reconstruction" in t for t in tabs_args)


class TestEmpiricalUIComponents:
    """Tests header branding, live badges, and sidebar components."""

    def test_render_header_real_and_mock_modes(self):
        """Verify render_header executes cleanly for both real mode and mock mode."""
        import ui.components as uc

        with patch("ui.components.st") as mock_st:
            # 1. Real mode
            uc.render_header({"mock_mode": False})
            mock_st.markdown.assert_called_once()
            html_call = mock_st.markdown.call_args[0][0]
            assert "RoomAI" in html_call or "Room<span" in html_call
            assert "Understand Your Space. Visualize It in 3D." in html_call
            assert "Real Inference (GPU)" in html_call
            assert "Checkpoints: YOLO11s A3 (800px) + Metric Depth V2" in html_call

            # 2. Mock mode
            mock_st.reset_mock()
            uc.render_header({"mock_mode": True})
            mock_st.markdown.assert_called_once()
            html_mock_call = mock_st.markdown.call_args[0][0]
            assert "Mock Mode (Simulated)" in html_mock_call

    def test_sidebar_presets_and_device_status(self):
        """Verify sidebar presets and device status execute without exception."""
        import ui.components as uc

        with patch("ui.components.st") as mock_st:
            mock_st.sidebar = MagicMock()
            mock_st.sidebar.columns.return_value = (MockColumnContext(), MockColumnContext())
            uc.render_sidebar_presets()
            assert mock_st.sidebar.markdown.called

            uc.render_device_status_sidebar()
            assert mock_st.sidebar.markdown.called


if __name__ == "__main__":
    pytest.main(["-v", __file__])
