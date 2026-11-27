"""Contratos RGB-D y carga de configuración."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from datetime import datetime, timezone

import numpy as np
import yaml
from numpy.typing import NDArray


@dataclass(frozen=True)
class Intrinsic:
    """Intrínsecos pinhole en píxeles."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> NDArray:
        """Devuelve la matriz de cámara."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class Frame:
    """Color RGB y profundidad axial en mm sobre la misma cuadrícula."""
    rgb: NDArray[np.uint8]
    mm: NDArray[np.uint16]
    intrinsic: Intrinsic
    timestamp: str
    sensor: str
    calibration: dict


class Source(Protocol):
    """Fuente con fin de flujo señalado por StopIteration."""
    def read(self) -> Frame:
        """Entrega la siguiente captura."""
        ...

    def close(self) -> None:
        """Libera los recursos."""
        ...


def now() -> str:
    """Devuelve una marca UTC ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    """Resuelve las rutas relativas al archivo YAML."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    depth = config["depth"]
    if not 0 < depth["min_mm"] < depth["max_mm"] <= 65535:
        raise ValueError("Configura 0 < min_mm < max_mm <= 65535.")
    config["paths"] = {key: path.parent / value for key, value in config["paths"].items()}
    return config


def clean_depth(mm: NDArray, depth: dict) -> NDArray[np.uint16]:
    """Convierte las muestras fuera de rango y centinelas a cero."""
    valid = (mm >= depth["min_mm"]) & (mm <= depth["max_mm"])
    return np.where(valid & ~np.isin(mm, depth["invalid_mm"]), mm, 0).astype(np.uint16)


def frame_metadata(frame: Frame) -> dict:
    """Serializa la procedencia y calibración de una captura."""
    return dict(timestamp=frame.timestamp, sensor=frame.sensor, intrinsic=asdict(frame.intrinsic),
                calibration=frame.calibration, depth_unit="mm", alignment="depth_grid")

