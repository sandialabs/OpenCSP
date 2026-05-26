# test_time_history_array_mp_npz.py
import json
from pathlib import Path

import numpy as np
import pytest

import opencsp.app.lookback.time_history_array_mp_npz as time_hist


class DummyPool:
    """Deterministic stand-in for multiprocessing.Pool used by the module."""

    def __init__(self, processes=1):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, iterable):
        for item in iterable:
            yield func(item)


def _write_placeholder_files(image_dir: Path, names):
    image_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (image_dir / n).write_bytes(b"not-a-real-image")


def test_process_image_bit_depth_cv_thresholding(monkeypatch, tmp_path):
    # 8-bit image: max 255, threshold fraction 0.5 => 127.5
    img = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    img_path = tmp_path / "img001.png"
    img_path.write_bytes(b"x")

    monkeypatch.setattr(time_hist.cv2, "imread", lambda p, f: img, raising=True)

    name, binary_list = time_hist.process_image_bit_depth_cv(str(img_path), 0.5)
    assert name == "img001.png"

    binary = np.array(binary_list, dtype=bool)
    assert binary.shape == img.shape
    assert binary.tolist() == [[False, False, True, True]]  # > 127.5


def test_process_batch_writes_npz_and_json(monkeypatch, tmp_path):
    out = tmp_path / "out"
    pct_key = "50"
    (out / pct_key).mkdir(parents=True, exist_ok=True)

    # Two images with known arrays returned by cv2.imread
    arrays = {"a.png": np.array([[0, 200]], dtype=np.uint8), "b.png": np.array([[130, 120]], dtype=np.uint8)}
    _write_placeholder_files(tmp_path, ["a.png", "b.png"])

    def fake_imread(path, flags):
        return arrays[Path(path).name]

    monkeypatch.setattr(time_hist.cv2, "imread", fake_imread, raising=True)

    batch_paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]

    batch_index = time_hist.process_batch(
        batch_index=1, batch_paths=batch_paths, percentage=0.5, output_folder=str(out), percentage_key=pct_key
    )
    assert batch_index == 1

    npz_path = out / pct_key / "threshold_050_batch_0001.npz"
    assert npz_path.exists()

    data = np.load(npz_path)
    arr = data["arr_0"]  # np.savez_compressed default key
    assert arr.shape == (2, 1, 2)

    # threshold 127.5:
    # a.png: [[0, 200]] -> [[False, True]]
    # b.png: [[130,120]] -> [[True, False]]
    assert arr[0].tolist() == [[False, True]]
    assert arr[1].tolist() == [[True, False]]

    json_path = out / pct_key / "threshold_050_batch_0001.json"
    assert json_path.exists()
    names = json.loads(json_path.read_text(encoding="utf-8"))
    assert names == ["a.png", "b.png"]


def test_create_binary_pixel_array_parallel_creates_expected_batches_and_checkpoint(monkeypatch, tmp_path):
    # Avoid real multiprocessing for test determinism
    monkeypatch.setattr(time_hist, "Pool", DummyPool, raising=True)

    image_dir = tmp_path / "images"
    out = tmp_path / "out"
    ckpt = tmp_path / "ckpt"

    # 5 images, batch_size=3, overlap=1 => step = 2
    # batches:
    #   start 0..2 => batch 1 (3 imgs)
    #   start 2..4 => batch 2 (3 imgs)
    #   start 4..4 => batch 3 (1 img)
    names = [f"img{i:03d}.png" for i in range(5)]
    _write_placeholder_files(image_dir, names)

    arrays = {n: np.array([[i]], dtype=np.uint8) for i, n in enumerate(names)}

    def fake_imread(path, flags):
        return arrays[Path(path).name]

    monkeypatch.setattr(time_hist.cv2, "imread", fake_imread, raising=True)

    time_hist.create_binary_pixel_array_parallel_with_multiprocessing(
        image_folder_path=str(image_dir),
        percentages=[0.5],
        output_folder=str(out),
        checkpoint_folder=str(ckpt),
        batch_size=3,
        overlap=1,
        checkpoint_file="chk.json",
        num_workers=2,
    )

    pct_key = "50"

    npz_files = sorted((out / pct_key).glob("threshold_050_batch_*.npz"))
    assert [p.name for p in npz_files] == [
        "threshold_050_batch_0001.npz",
        "threshold_050_batch_0002.npz",
        "threshold_050_batch_0003.npz",
    ]

    # Verify checkpoint reflects all processed batches (uses the real lbt implementation)
    ckpt_data = time_hist.lbt.load_checkpoint(str(ckpt), "chk.json")
    assert "processed_batches" in ckpt_data
    assert pct_key in ckpt_data["processed_batches"]
    assert sorted(ckpt_data["processed_batches"][pct_key]) == [1, 2, 3]


def test_create_binary_pixel_array_parallel_resumes_from_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(time_hist, "Pool", DummyPool, raising=True)

    image_dir = tmp_path / "images"
    out = tmp_path / "out"
    ckpt = tmp_path / "ckpt"

    names = [f"img{i:03d}.png" for i in range(4)]
    _write_placeholder_files(image_dir, names)

    arrays = {n: np.array([[255]], dtype=np.uint8) for n in names}

    def fake_imread(path, flags):
        return arrays[Path(path).name]

    monkeypatch.setattr(time_hist.cv2, "imread", fake_imread, raising=True)

    # Seed a checkpoint that says batch 1 is already processed for 50%
    checkpoint_file = "chk.json"
    checkpoint_data = {"processed_batches": {"50": [1]}}
    time_hist.lbt.save_checkpoint(str(ckpt), checkpoint_file, checkpoint_data)

    # Spy to ensure batch 1 is skipped
    called = []
    real = time_hist.process_batch_multiprocessing

    def spy(args):
        called.append(args[0])  # batch_index
        return real(args)

    monkeypatch.setattr(time_hist, "process_batch_multiprocessing", spy, raising=True)

    time_hist.create_binary_pixel_array_parallel_with_multiprocessing(
        image_folder_path=str(image_dir),
        percentages=[0.5],
        output_folder=str(out),
        checkpoint_folder=str(ckpt),
        batch_size=3,
        overlap=1,
        checkpoint_file=checkpoint_file,
        num_workers=2,
    )

    # With 4 images, batch_size=3 overlap=1 => step=2 => batches 1 and 2
    assert called == [2]  # batch 1 skipped

    npz_files = sorted((out / "50").glob("threshold_050_batch_*.npz"))
    assert [p.name for p in npz_files] == ["threshold_050_batch_0002.npz"]


def test_create_binary_pixel_array_parallel_raises_when_no_images(monkeypatch, tmp_path):
    monkeypatch.setattr(time_hist, "Pool", DummyPool, raising=True)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out = tmp_path / "out"
    ckpt = tmp_path / "ckpt"

    with pytest.raises(ValueError, match="No valid image files found"):
        time_hist.create_binary_pixel_array_parallel_with_multiprocessing(
            image_folder_path=str(empty_dir),
            percentages=[0.5],
            output_folder=str(out),
            checkpoint_folder=str(ckpt),
            batch_size=3,
            overlap=1,
            checkpoint_file="chk.json",
            num_workers=2,
        )
