"""Lectura del manifiesto de una sesión RGB-D."""
import json
from pathlib import Path

import cv2
import numpy as np

from ..model import Frame, Intrinsic


class ReplaySource:
    """Reproduce capturas en el orden del manifiesto."""
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        with (directory / "metadata.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        self.entries = iter(metadata["frames"])

    def read(self) -> Frame:
        """Carga y comprueba un par de imágenes."""
        item = next(self.entries)
        camera = Intrinsic(**item["intrinsic"])
        rgb = cv2.imdecode(np.fromfile(self.directory / item["rgb"], dtype=np.uint8), cv2.IMREAD_COLOR)
        mm = np.load(self.directory / item["depth"], allow_pickle=False)
        shape = (camera.height, camera.width)
        if rgb is None or rgb.shape != (*shape, 3) or mm.shape != shape or mm.dtype != np.uint16:
            raise ValueError(f"Captura inválida en índice {item['index']}")
        if item["depth_unit"] != "mm" or item["alignment"] != "depth_grid":
            raise ValueError("La sesión debe contener profundidad en mm y RGB alineado a profundidad.")
        return Frame(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), mm, camera, item["timestamp"],
                     item["sensor"], item["calibration"])

    def close(self) -> None:
        """Cada captura se abre y cierra durante la lectura."""

