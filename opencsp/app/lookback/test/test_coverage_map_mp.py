# test_coverage_map_mp_e2e_png.py
import os
import numpy as np
import pytest
import imageio.v2 as imageio

import opencsp.app.lookback.coverage_map_mp as cm


@pytest.fixture
def threshold_fractions():
    return [0.25, 0.5]


@pytest.fixture
def stub_lbt(monkeypatch):
    """Avoid creating real checkpoint files (keep everything in-memory)."""
    state = {"checkpoint": None, "saves": 0}

    def load_checkpoint(folder, filename):
        return state["checkpoint"]

    def save_checkpoint(folder, filename, data):
        state["checkpoint"] = dict(data)
        state["saves"] += 1

    monkeypatch.setattr(
        cm, "lbt", type("LBT", (), {"load_checkpoint": load_checkpoint, "save_checkpoint": save_checkpoint})()
    )
    return state


def test_process_images_in_batches_writes_real_pngs(tmp_path, threshold_fractions, monkeypatch):
    """
    End-to-end for PNG output from process_images_in_batches:
      - write real input PNGs
      - run batch processor
      - verify output PNGs exist, are uint8, and match expected OR aggregation
    """
    # Create two small grayscale input images (2x3)
    # img1: some pixels above thresholds
    img1 = np.array([[0, 80, 0], [0, 0, 200]], dtype=np.uint8)
    img2 = np.array([[70, 0, 0], [0, 130, 0]], dtype=np.uint8)

    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    imageio.imwrite(p1, img1)
    imageio.imwrite(p2, img2)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cm.process_images_in_batches(
        image_files=[str(p1), str(p2)],
        threshold_fractions=threshold_fractions,
        max_intensity=255,
        is_raw=False,
        batch_num=1,
        output_folder=str(out_dir),
        prefix="traditional",
    )

    # Expected binary maps:
    # threshold 0.25 => > 63.75: pixels {80, 70, 130, 200} are True
    expected_025 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8) * 255
    # threshold 0.5 => > 127.5: pixels {130, 200} are True
    expected_05 = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.uint8) * 255

    for t, expected in [(0.25, expected_025), (0.5, expected_05)]:
        out_path = out_dir / f"traditional_binary_map_{int(t*100):02d}_1.png"
        assert out_path.exists()

        got = imageio.imread(out_path)
        # Some readers may return (H,W,1); squeeze to be safe
        got = np.squeeze(got)

        assert got.dtype == np.uint8
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)


def test_compile_binary_maps_real_pngs(tmp_path, threshold_fractions):
    """
    End-to-end for compile_binary_maps:
      - write two batch binary-map PNGs per threshold
      - compile
      - verify compiled output equals pixelwise OR (max) of batches
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    prefix = "traditional"

    # Batch 1 and 2 maps (2x2) expressed as 0/255 uint8
    b1 = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    b2 = np.array([[0, 0], [0, 255]], dtype=np.uint8)
    expected_compiled = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    for t in threshold_fractions:
        imageio.imwrite(out_dir / f"{prefix}_binary_map_{int(t*100):02d}_1.png", b1)
        imageio.imwrite(out_dir / f"{prefix}_binary_map_{int(t*100):02d}_2.png", b2)

    cm.compile_binary_maps(str(out_dir), threshold_fractions, prefix=prefix)

    for t in threshold_fractions:
        compiled_path = out_dir / f"{prefix}_compiled_binary_map_{int(t*100):02d}.png"
        assert compiled_path.exists()
        got = np.squeeze(imageio.imread(compiled_path))
        assert got.dtype == np.uint8
        assert np.array_equal(got, expected_compiled)


def test_construct_binary_maps_parallel_traditional_end_to_end(tmp_path, threshold_fractions, stub_lbt, monkeypatch):
    """
    End-to-end-ish for construct_binary_maps_parallel on traditional images:
      - writes real input PNGs
      - runs full pipeline (batch outputs + compiled outputs)
      - uses in-memory checkpointing (no checkpoint files)
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"

    # Create 3 images to ensure multiple batches if batch_size=2
    imgs = [
        np.array([[0, 70], [0, 0]], dtype=np.uint8),
        np.array([[0, 0], [200, 0]], dtype=np.uint8),
        np.array([[130, 0], [0, 0]], dtype=np.uint8),
    ]
    for i, arr in enumerate(imgs, start=1):
        imageio.imwrite(img_dir / f"{i:02d}.png", arr)

    # Speed up tests: make ThreadPoolExecutor single-threaded deterministically (optional)
    # monkeypatch.setattr(cm, "ThreadPoolExecutor", lambda max_workers=4: __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1))

    cm.construct_binary_maps_parallel(
        image_folder=str(img_dir),
        output_folder=str(out_dir),
        checkpoint_folder=str(ckpt_dir),
        threshold_fractions=threshold_fractions,
        batch_size=2,
        max_workers=2,
        checkpoint_file="ckpt.json",
    )

    # Verify compiled outputs exist
    for t in threshold_fractions:
        compiled_path = out_dir / f"traditional_compiled_binary_map_{int(t*100):02d}.png"
        assert compiled_path.exists()

    # Verify compiled content for one threshold (0.5) matches expectation:
    # > 127.5: pixels 200 and 130 => True at (1,0) and (0,0) in 2x2
    expected_05 = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    got_05 = np.squeeze(imageio.imread(out_dir / "traditional_compiled_binary_map_50.png"))
    assert np.array_equal(got_05, expected_05)

    # Confirm checkpointing occurred (in-memory)
    assert stub_lbt["saves"] >= 1
    assert "traditional" in (stub_lbt["checkpoint"].get("completed_thresholds") or [])


@pytest.mark.parametrize("rgb", [False, True])
def test_process_image_real_input_png(tmp_path, rgb):
    """
    End-to-end for process_image reading a real PNG from disk via imageio.
    """
    if rgb:
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[0, 0, :] = 200  # mean 200
        arr[1, 1, :] = 50  # mean 50
    else:
        arr = np.array([[200, 0], [0, 50]], dtype=np.uint8)

    p = tmp_path / ("in_rgb.png" if rgb else "in_gray.png")
    imageio.imwrite(p, arr)

    out = cm.process_image(str(p), threshold_fractions=[0.5], max_intensity=255, is_raw=False)
    assert out is not None
    got = out[0.5].astype(np.uint8)

    # > 127.5 => only the 200 pixel is True
    expected = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert np.array_equal(got, expected)
