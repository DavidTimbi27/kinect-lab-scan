"""Pruebas de captura, proyección y registro con escala conocida."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.capture import average_at_pose, capture
from src.model import Frame, Intrinsic, load_config
from src.reconstruct import angle_deg, prepare, project, register_pair
from src.sources.kinect_source import align_color
from src.sources.replay_source import ReplaySource
from src.sources.synthetic_source import SyntheticSource


@pytest.fixture
def config() -> dict:
    """Usa la configuración predeterminada del proyecto."""
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def test_projection_metric(config: dict) -> None:
    """Comprueba rayos, colores y descarte de profundidad inválida."""
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    mm = np.array([[2000, 2000], [0, 4001]], np.uint16)
    frame = Frame(rgb, mm, Intrinsic(2, 2, 2, 2, 0, 0), "", "test", {})
    cloud = project(frame, config["depth"])
    np.testing.assert_allclose(np.asarray(cloud.points), [[0, 0, 2], [1, 0, 2]])
    np.testing.assert_allclose(np.asarray(cloud.colors), [[1, 0, 0], [0, 1, 0]])


def test_capture_replay(tmp_path: Path, config: dict) -> None:
    """Verifica que el manifiesto preserve color, profundidad e intrínsecos."""
    target = tmp_path / "replay"
    expected = SyntheticSource(1).read()
    assert capture(SyntheticSource(1), target, config["depth"]) == 1
    replay = ReplaySource(target)
    actual = replay.read()
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.mm, expected.mm)
    assert actual.intrinsic == expected.intrinsic
    with pytest.raises(StopIteration):
        replay.read()


def test_average_invalid_and_color_occlusion(config: dict) -> None:
    """Comprueba centinelas configurados y prioridad del punto cercano."""
    camera = Intrinsic(2, 1, 2, 2, 0, 0)
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], np.uint8)
    first = Frame(rgb, np.array([[0, 2047]], np.uint16), camera, "", "test", {})
    second = replace(first, mm=np.array([[1000, 2000]], np.uint16))
    depth = {**config["depth"], "invalid_mm": [0, 2047]}
    np.testing.assert_array_equal(average_at_pose([first, second], depth).mm, second.mm)
    xy = np.zeros((1, 2, 2), dtype=int)
    np.testing.assert_array_equal(align_color(rgb, second.mm, xy, second.mm),
                                  [[[255, 0, 0], [0, 0, 0]]])


@pytest.mark.parametrize("method", ["fpfh", "rgbd"])
def test_known_room_transform(config: dict, method: str) -> None:
    """Compara el registro de dos vistas con la pose real del simulador."""
    source = SyntheticSource(33)
    a, b = source.read(), source.read()
    settings = config["registration"]
    views = [prepare(frame, config["depth"], settings["voxel_m"], method == "fpfh")
             for frame in (a, b)]
    result = register_pair(*views, settings, method, np.eye(4))
    expected = (np.linalg.inv(b.calibration["camera_to_world_reference"])
                @ a.calibration["camera_to_world_reference"])
    delta = result.transformation @ np.linalg.inv(expected)
    assert result.fitness > 0.65
    assert result.inlier_rmse < 0.045
    assert angle_deg(delta) < 2.5
    assert np.linalg.norm(delta[:3, 3]) < 0.08

