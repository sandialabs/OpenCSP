# test_lookback_tools.py
import json
import gzip
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

import opencsp.app.lookback.lookback_tools as lbt


# -----------------------
# Helpers / fixtures
# -----------------------
class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    # Replace module-level logger with a no-op logger to keep tests quiet and robust
    monkeypatch.setattr(lbt, "logger", DummyLogger(), raising=True)


# -----------------------
# Unit tests
# -----------------------
def test_frame_number_from_img_name():
    assert lbt.frame_number_from_img_name("DSC_2832-09170.png") == 9170
    assert lbt.frame_number_from_img_name("/a/b/c/DSC_0001-00005.png") == 5


def test_lat_long_to_decimal():
    assert lbt.lat_long_to_decimal((35, 30, 0)) == pytest.approx(35.5)
    assert lbt.lat_long_to_decimal((0, 1, 0)) == pytest.approx(1 / 60)


def test_define_observation_time_skyfield_datetime():
    # Basic sanity: should return a skyfield Time object
    src = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t = lbt.define_observation_time_skyfield(src, timedelta(seconds=0))
    # Avoid importing skyfield.Time directly; just check key behavior/attributes
    assert hasattr(t, "utc_datetime")
    assert t.utc_datetime().replace(tzinfo=timezone.utc) == src


def test_custom_serializer_numpy_datetime_set_and_rotation():
    r = lbt.Rotation.from_euler("z", 30, degrees=True)
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    obj = {"arr": np.array([1, 2, 3]), "dt": dt, "s": set(["a", "b"]), "rot": r}
    dumped = json.dumps(obj, default=lbt.custom_serializer)
    loaded = json.loads(dumped)  # raw json loads, no object_hook yet
    assert loaded["arr"] == [1, 2, 3]
    assert loaded["dt"].startswith("2024-01-01T12:00:00")
    assert loaded["rot"]["__rotation__"] is True
    assert isinstance(loaded["rot"]["quat"], list)


def test_custom_deserializer_rotation_roundtrip():
    r = lbt.Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    payload = {"rot": {"__rotation__": True, "quat": r.as_quat().tolist()}}
    # object_hook is called for each dict; simulate json.load(..., object_hook=...)
    out = lbt.custom_deserializer(payload)
    assert "rot" in out
    assert isinstance(out["rot"], lbt.Rotation)
    assert np.allclose(out["rot"].as_quat(), r.as_quat())


def test_write_and_read_json_roundtrip(tmp_path):
    data = {
        "a": 1,
        "b": 2.5,
        "c": "hi",
        "arr": np.array([1, 2, 3]),
        "dt": datetime(2023, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
        "rot": lbt.Rotation.from_euler("z", 45, degrees=True),
    }
    fp = tmp_path / "data.json"
    lbt.write_json(data, str(fp))
    back = lbt.read_json(str(fp))

    assert back["a"] == 1
    assert back["b"] == pytest.approx(2.5)
    assert back["c"] == "hi"
    assert isinstance(back["arr"], np.ndarray)
    assert np.array_equal(back["arr"], np.array([1, 2, 3]))
    assert isinstance(back["dt"], datetime)
    assert isinstance(back["rot"], lbt.Rotation)


def test_write_and_read_compressed_json_roundtrip(tmp_path):
    data = {"x": np.array([9, 8, 7]), "dt": datetime(2024, 1, 2, 3, 4, 5)}
    fp = tmp_path / "data.json.gz"
    lbt.write_compressed_json(data, str(fp), print_path=False)
    back = lbt.read_compressed_json(str(fp))

    assert isinstance(back["x"], np.ndarray)
    assert np.array_equal(back["x"], np.array([9, 8, 7]))
    assert isinstance(back["dt"], datetime)


def test_save_and_load_checkpoint(tmp_path, monkeypatch):
    # save_checkpoint/load_checkpoint depend on the module-level logger already patched
    folder = tmp_path
    name = "ckpt.json"
    payload = {"k": "v", "n": 3}

    lbt.save_checkpoint(str(folder), name, payload, print_path=False)
    back = lbt.load_checkpoint(str(folder), name)
    assert back == payload


def test_load_checkpoint_empty_file_returns_none(tmp_path):
    folder = tmp_path
    name = "empty.json"
    (folder / name).write_text("", encoding="utf-8")
    assert lbt.load_checkpoint(str(folder), name) is None


def test_save_and_read_batch_npz(tmp_path):
    batch = [
        {"image_name": "im1.png", "binary_array": [[0, 1], [1, 0]]},
        {"image_name": "im2.png", "binary_array": [[1, 1], [0, 0]]},
    ]
    fp = tmp_path / "batch.npz"

    lbt.save_batch_to_npz(batch, str(fp))
    back = lbt.read_batch_from_npz(str(fp))

    assert {d["image_name"] for d in back} == {"im1.png", "im2.png"}
    d1 = next(d for d in back if d["image_name"] == "im1.png")
    assert d1["binary_array"] == [[0, 1], [1, 0]]


def test_extract_detailed_video_metadata_parsing(monkeypatch):
    # Mock subprocess.run to return exiftool-like output
    class R:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""

    sample = "\n".join(
        [
            "File Name                        : vid.mp4",
            "File Size                        : 123 MB",
            "File Format                      : MP4",
            "Duration                         : 0:01:10",
            "Video Frame Rate                 : 29.97",
            "Image Width                      : 1920",
            "Image Height                     : 1080",
            "Aspect Ratio                     : 16:9",
            "Video Bitrate                    : 10 Mb/s",
            "Compression                      : AVC",
            "Make                             : Sony",
            "Model                            : ILCE-7M3",
            "Serial Number                    : 123456",
            "Lens Make                        : Sony",
            "Lens Model                       : FE 24-105mm F4 G OSS",
            "Focal Length                     : 35.0 mm",
            "Aperture                         : 4.0",
            "ISO                              : 800",
            "Shutter Speed                    : 1/60",
            "White Balance                    : Auto",
            "Exposure Mode                    : Manual",
            "Metering Mode                    : Multi-segment",
            "Focus Mode                       : AF-C",
            "Image Stabilization              : On",
            "Create Date                      : 2024:01:01 12:00:00",
            "Modify Date                      : 2024:01:01 12:00:01",
            "Media Create Date                : 2024:01:01 12:00:00",
        ]
    )

    def fake_run(*args, **kwargs):
        return R(sample)

    monkeypatch.setattr(lbt.subprocess, "run", fake_run, raising=True)

    md = lbt.extract_detailed_video_metadata("dummy.mp4")
    assert md["file_name"] == "vid.mp4"
    assert md["file_format"] == "MP4"
    assert md["duration"] == pytest.approx(70.0)
    assert md["frame_rate"] == pytest.approx(29.97)
    assert md["resolution"] == "1920x1080"
    assert md["iso"] == 800
    assert md["camera_make"] == "Sony"


def test_accelerate_video_ffmpeg_no_audio_builds_command(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        calls["check"] = check
        return None

    monkeypatch.setattr(lbt.subprocess, "run", fake_run, raising=True)

    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    lbt.accelerate_video_ffmpeg_no_audio(str(inp), str(out), playback_speed=4.0)

    cmd = calls["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    assert str(inp) in cmd
    assert "-an" in cmd
    # filter content check
    assert "setpts=0.25*PTS" in cmd


def test_accelerate_video_ffmpeg_no_audio_rejects_nonpositive_speed(tmp_path):
    with pytest.raises(ValueError):
        lbt.accelerate_video_ffmpeg_no_audio(str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), 0)


def test_write_read_hdf5_roundtrip(tmp_path):
    data = {
        "a": 1,
        "b": 2.0,
        "dt": datetime(2024, 1, 1, 0, 0, 0),
        "arr": np.arange(6).reshape(2, 3),
        "nested": {"x": "hi"},
        "lst": [{"k": 1}, {"k": 2}],
        "st": set([1, 2, 3]),
    }
    fp = tmp_path / "data.h5"

    lbt.write_to_hdf5(data, str(fp))
    back = lbt.read_from_hdf5(str(fp))

    assert back["a"] == 1
    assert back["b"] == 2.0
    assert isinstance(back["dt"], datetime)
    assert np.array_equal(back["arr"], data["arr"])
    assert back["nested"]["x"] == "hi"
    assert isinstance(back["lst"], list)
    assert {d["k"] for d in back["lst"]} == {1, 2}
    # Note: current implementation tries to reconstruct sets in a slightly odd way,
    # so this assertion is intentionally forgiving:
    assert set(back["st"]) == set([1, 2, 3])


def test_save_and_read_hdf5_datasets_compressed_dict_nested(tmp_path):
    data = {
        (1, 2): [{"a": 1, "vec": np.array([1, 2, 3])}, {"a": 2, "vec": np.array([4, 5, 6])}],
        (9, 9): {"scalar": 7, "mat": np.eye(2)},
    }
    fp = tmp_path / "compressed.h5"
    lbt.save_hdf5_datasets_compressed(data=data, datasets=[], file=str(fp))

    out = lbt.read_hdf5_datasets(str(fp))
    # group names are sanitized; just check existence of some top-level groups
    assert "1_2" in out
    assert "9_9" in out
    assert "entry_0" in out["1_2"]
    assert out["1_2"]["entry_0"]["a"] in (1, np.int64(1))


def test_read_hdf5_datasets_specific_dataset(tmp_path):
    fp = tmp_path / "simple.h5"
    lbt.save_hdf5_datasets_compressed(
        data=[np.arange(5), np.arange(6).reshape(2, 3)], datasets=["d1", "d2"], file=str(fp)
    )
    out = lbt.read_hdf5_datasets(str(fp), datasets=["d1"])
    assert list(out.keys()) == ["d1"]
    assert np.array_equal(out["d1"], np.arange(5))
