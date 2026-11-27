"""Captura del SDK v1.8 con color registrado a profundidad."""
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ..model import Frame, Intrinsic, now


def align_color(rgb: np.ndarray, mm: np.ndarray, xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Muestrea el RGB y oculta puntos detrás de otros en el mismo píxel."""
    height, width = rgb.shape[:2]
    x, y = xy[..., 0], xy[..., 1]
    valid = (mm > 0) & (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    pixels = y[valid].astype(np.intp) * width + x[valid]
    depths = z[valid].astype(float)
    nearest = np.full(height * width, np.inf)
    np.minimum.at(nearest, pixels, depths)
    keep = depths <= nearest[pixels] + 40
    aligned = np.zeros((*mm.shape, 3), np.uint8)
    aligned.reshape(-1, 3)[np.flatnonzero(valid)[keep]] = rgb.reshape(-1, 3)[pixels[keep]]
    return aligned


class KinectSource:
    """Lee pares de 640×480 del puente de Windows."""

    def __init__(self, settings: dict, sweep_average: int | None = None) -> None:
        if sys.platform != "win32":
            raise RuntimeError("El Kinect requiere Windows y SDK v1.8. En Linux usa replay.")
        self.intrinsic = Intrinsic(**settings["depth_intrinsic"])
        if (self.intrinsic.width, self.intrinsic.height) != (640, 480):
            raise ValueError("El SDK usa 640×480.")
        self.override = settings.get("color_registration")
        self.sweep = sweep_average is not None
        if sweep_average is not None and sweep_average < 1:
            raise ValueError('El promedio debe ser positivo.')
        options = ['-VerticalSweep', '-FramesPerPose', str(sweep_average)] if self.sweep else []
        self.process = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(Path(__file__).with_name("kinect_bridge.ps1")), *options],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def read(self) -> Frame:
        """Aplica el mapa de fábrica o la calibración explícita."""
        n = 640 * 480
        size = n*10 + (4 if self.sweep else 0)
        packet = self.process.stdout.read(size)
        if len(packet) != size:
            if self.sweep and not packet and self.process.wait(timeout=3) == 0:
                warning = self.process.stderr.read().decode(errors="replace").strip()
                if warning:
                    logging.getLogger(__name__).warning("%s", warning)
                raise StopIteration
            error = self.process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"Sin RGB-D. Revisa alimentación, USB y SDK v1.8. {error}")
        mm = np.frombuffer(packet, "<u2", n).reshape(480, 640)
        rgb = np.ascontiguousarray(np.frombuffer(packet, np.uint8, n*4, n*2)
                                   .reshape(480, 640, 4)[..., 2::-1])
        xy = np.frombuffer(packet, "<i2", n*2, n*6).reshape(480, 640, 2)
        z = mm
        calibration = {"registration": "Kinect SDK 1.8 CoordinateMapper"}
        if self.override is not None:
            v, u = np.indices(mm.shape)
            model = self.intrinsic
            xyz = np.stack(((u-model.cx)/model.fx, (v-model.cy)/model.fy,
                            np.ones_like(u)), axis=-1) * mm[..., None] / 1000
            rotation = np.asarray(self.override["rotation_depth_to_rgb"], dtype=float)
            translation = np.asarray(self.override["translation_depth_to_rgb_m"], dtype=float)
            camera = Intrinsic(**self.override["rgb_intrinsic"])
            if (camera.width, camera.height) != (640, 480):
                raise ValueError("La calibración RGB debe corresponder a 640×480.")
            xyz_rgb = xyz @ rotation.T + translation
            distortion = np.asarray(self.override.get("rgb_distortion", []), dtype=float)
            mapped, _ = cv2.projectPoints(xyz_rgb.reshape(-1, 3), np.zeros(3), np.zeros(3),
                                          camera.matrix().astype(float), distortion)
            xy = np.rint(np.clip(mapped.reshape(*mm.shape, 2), -32768, 32767)).astype(np.int32)
            z = xyz_rgb[..., 2] * 1000
            calibration = {"registration": "pinhole_depth_to_rgb", **self.override}
        if self.sweep:
            calibration['tilt_deg'] = int.from_bytes(packet[n*10:], 'little', signed=True)
        return Frame(align_color(rgb, mm, xy, z), mm, self.intrinsic, now(),
                     "kinect_v1_sdk_1_8", calibration)

    def close(self) -> None:
        """Termina el puente y cierra sus descriptores."""
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()
