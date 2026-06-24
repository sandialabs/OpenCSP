# test_time_history_transitions_npz_mp.py
import os
import json
import gzip
import importlib
from pathlib import Path

import numpy as np
import pytest

import opencsp.app.lookback.time_history_transitions_npz_mp as th


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """
    Create an isolated working directory and chdir into it so the module-level
    logging setup writes under tmp_path (./error_logs).
    """
    monkeypatch.chdir(tmp_path)
    # Reload after chdir so logger is configured relative to tmp cwd if desired.
    importlib.reload(th)
    return tmp_path


@pytest.fixture()
def stub_lbt(monkeypatch):
    """
    Stub out lbt I/O helpers so tests do not depend on OpenCSP internals.
    Uses gzip+json for compressed json read/write and minimal checkpoint helpers.
    """

    def write_compressed_json(data, file_path, print_path=False):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

    def read_compressed_json(file_path):
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)

    def read_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_checkpoint(folder, name):
        p = Path(folder) / name
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_checkpoint(folder, name, data, print_path=False):
        Path(folder).mkdir(parents=True, exist_ok=True)
        p = Path(folder) / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def frame_number_from_img_name(item):
        # Accept full path or name; parse last run of digits.
        import re

        s = str(item)
        m = re.search(r"(\d+)", os.path.basename(s))
        if not m:
            raise ValueError(f"Could not parse frame number from {item}")
        return int(m.group(1))

    # This module expects read_batch_from_npz returns list[dict] with key "binary_array"
    def read_batch_from_npz(npz_file):
        arr = np.load(npz_file, allow_pickle=True)["binary_array"]
        return [{"binary_array": arr}]

    monkeypatch.setattr(th.lbt, "write_compressed_json", write_compressed_json, raising=True)
    monkeypatch.setattr(th.lbt, "read_compressed_json", read_compressed_json, raising=True)
    monkeypatch.setattr(th.lbt, "read_json", read_json, raising=True)
    monkeypatch.setattr(th.lbt, "load_checkpoint", load_checkpoint, raising=True)
    monkeypatch.setattr(th.lbt, "save_checkpoint", save_checkpoint, raising=True)
    monkeypatch.setattr(th.lbt, "frame_number_from_img_name", frame_number_from_img_name, raising=True)
    monkeypatch.setattr(th.lbt, "read_batch_from_npz", read_batch_from_npz, raising=True)


def _make_npz_and_json(folder: Path, idx: int, frames_binary_array, img_names):
    """
    Create a single batch pair:
      - batch_{idx:04d}.npz containing key 'binary_array'
      - batch_{idx:04d}.json containing list of image names aligned to first axis
    """
    npz_path = folder / f"batch_{idx:04d}.npz"
    json_path = folder / f"batch_{idx:04d}.json"
    np.savez_compressed(npz_path, binary_array=np.asarray(frames_binary_array, dtype=np.uint8))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(img_names), f)
    return str(npz_path), str(json_path)


def test_process_npz_file_within_batch_detects_transitions(workdir, stub_lbt):
    npz_folder = workdir / "npz"
    npz_folder.mkdir()

    # Two frames, 2x2. Pixel (0,0): 0->1 bright. Pixel (1,1): 1->0 dark.
    frames = [[[0, 0], [0, 1]], [[1, 0], [0, 0]]]
    img_names = ["frame_0001.png", "frame_0002.png"]
    npz_path, json_path = _make_npz_and_json(npz_folder, 1, frames, img_names)

    pixels = [(0, 0), (1, 1)]
    results = th.process_npz_file_within_batch(npz_path, json_path, pixels)

    assert str((0, 0)) in results
    assert str((1, 1)) in results

    assert results[str((0, 0))] == [
        {"transition": "bright", "from_frame": "frame_0001.png", "to_frame": "frame_0002.png"}
    ]
    assert results[str((1, 1))] == [
        {"transition": "dark", "from_frame": "frame_0001.png", "to_frame": "frame_0002.png"}
    ]


def test_conditional_process_skips_processed_batch(workdir, stub_lbt):
    npz_folder = workdir / "npz"
    npz_folder.mkdir()

    frames = [[[0]], [[1]]]
    img_names = ["frame_0001.png", "frame_0002.png"]
    npz_path, json_path = _make_npz_and_json(npz_folder, 1, frames, img_names)

    checkpoint_data = {"processed_batches": [0]}
    out = th.conditional_process((0, npz_path, json_path, [(0, 0)], checkpoint_data))
    assert out is None


def test_write_and_merge_intermediate_results(workdir, stub_lbt):
    outdir = workdir / "out"
    outdir.mkdir()

    r1 = {str((0, 0)): [{"transition": "bright", "from_frame": "a", "to_frame": "b"}]}
    r2 = {
        str((0, 0)): [{"transition": "dark", "from_frame": "b", "to_frame": "c"}],
        str((1, 1)): [{"transition": "bright", "from_frame": "a", "to_frame": "b"}],
    }

    th.write_intermediate_results(r1, str(outdir), 0)
    th.write_intermediate_results(r2, str(outdir), 1)

    final_name = "compiled_results.json.gz"
    th.merge_results(str(outdir), final_name)

    merged = th.lbt.read_compressed_json(outdir / final_name)

    assert merged[str((0, 0))] == r1[str((0, 0))] + r2[str((0, 0))]
    assert merged[str((1, 1))] == r2[str((1, 1))]


def test_analyze_pixel_brightness_parallel_npz_creates_final_output(workdir, stub_lbt, monkeypatch):
    # Avoid spawning processes in unit tests: replace ProcessPoolExecutor with a synchronous stub.
    class DummyExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):  # context manager
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            return [fn(x) for x in iterable]

    monkeypatch.setattr(th, "ProcessPoolExecutor", DummyExecutor, raising=True)

    npz_folder = workdir / "npz"
    out_folder = workdir / "out"
    ckpt_folder = workdir / "ckpt"
    npz_folder.mkdir()

    # Two batches. Each batch has two frames 1x1; transitions:
    # batch1: 0->1 bright
    # batch2: 1->0 dark
    _make_npz_and_json(npz_folder, 1, [[[0]], [[1]]], ["frame_0001.png", "frame_0002.png"])
    _make_npz_and_json(npz_folder, 2, [[[1]], [[0]]], ["frame_0003.png", "frame_0004.png"])

    final_output = "final_results.json.gz"
    th.analyze_pixel_brightness_parallel_npz(
        npz_folder=str(npz_folder),
        pixel_locations=[(0, 0)],
        output_folder=str(out_folder),
        final_output_file=final_output,
        checkpoint_folder=str(ckpt_folder),
        checkpoint_file_name="checkpoint.json",
    )

    final_path = out_folder / final_output
    assert final_path.exists()

    merged = th.lbt.read_compressed_json(final_path)
    assert str((0, 0)) in merged
    # We expect two transitions total, one per batch.
    assert len(merged[str((0, 0))]) == 2
    assert merged[str((0, 0))][0]["transition"] == "bright"
    assert merged[str((0, 0))][1]["transition"] == "dark"


def test_pixel_timing_plot_writes_plot_and_data(workdir, stub_lbt, monkeypatch):
    # Avoid heavy graphics backends / file format issues by forcing a non-interactive backend (already in module),
    # and avoid writing huge images: keep small.
    out_folder = workdir / "plots"
    frames = [1, 2, 3, 4]
    pixel = "(10, 20)"
    transitions = [
        {"transition": "bright", "to_frame": "frame_0002.png"},
        {"transition": "dark", "to_frame": "frame_0004.png"},
    ]

    processed = th.pixel_timing_plot(pixel, transitions, frames, str(out_folder), checkpoint_data={})
    assert processed == pixel

    # Height is first element of eval(pixel) -> 10
    jpg = out_folder / "10" / f"pixel_{pixel}_timing_plot.jpg"
    data = out_folder / "10" / f"pixel_{pixel}_timing_plot_data.json.gz"
    assert jpg.exists()
    assert data.exists()

    binary_state = th.lbt.read_compressed_json(data)
    # Expect list-of-lists from json; validate shape-ish and frame column.
    assert len(binary_state) == len(frames)
    assert [row[0] for row in binary_state] == frames
