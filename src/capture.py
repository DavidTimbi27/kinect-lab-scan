"""Captura interactiva y almacenamiento incremental."""
from dataclasses import replace
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .model import Frame, Source, clean_depth, frame_metadata

LOG = logging.getLogger(__name__)


def average_at_pose(frames: list[Frame], depth: dict) -> Frame:
    """Promedia profundidad válida y escoge el color asociado más cercano."""
    first = frames[0]
    if any(frame.intrinsic != first.intrinsic for frame in frames):
        raise ValueError("Cambiaron los intrínsecos durante el promedio.")
    samples = np.stack([clean_depth(frame.mm, depth) for frame in frames])
    counts = np.count_nonzero(samples, axis=0)
    mean = np.divide(samples.sum(axis=0), counts, out=np.zeros(first.mm.shape), where=counts > 0)
    best = np.argmin(np.where(samples > 0, abs(samples - mean), np.inf), axis=0)
    rows, cols = np.indices(first.mm.shape)
    rgb = np.stack([frame.rgb for frame in frames])[best, rows, cols]
    return replace(first, mm=np.rint(mean).astype(np.uint16), rgb=rgb)


def preview(frame: Frame, depth: dict) -> int:
    """Muestra color y profundidad y devuelve la tecla pulsada."""
    mm = clean_depth(frame.mm, depth)
    span = max(1, depth["max_mm"] - depth["min_mm"])
    scaled = np.clip((mm.astype(float)-depth["min_mm"])*255/span, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[mm == 0] = 0
    cv2.imshow("Kinect | S guardar | Q salir", np.hstack((frame.rgb[..., ::-1], color)))
    return cv2.waitKey(1) & 255


def capture(source: Source, directory: Path, depth: dict, average: int = 1,
            interactive: bool = False, limit: int | None = None) -> int:
    """Guarda pares RGB-D y compromete el manifiesto tras cada captura."""
    entries: list[dict] = []
    try:
        if average < 1 or (limit is not None and limit < 1):
            raise ValueError("El promedio y el límite deben ser positivos.")
        directory.mkdir(parents=True, exist_ok=False)
        if interactive:
            LOG.info("Pulsa S o espacio para guardar y Q o Esc para salir.")
        while limit is None or len(entries) < limit:
            try:
                first = source.read()
            except StopIteration:
                break
            if interactive:
                key = preview(first, depth)
                if key in (ord("q"), 27):
                    break
                if key not in (ord("s"), 32):
                    continue
            frames = [first]
            for _ in range(average - 1):
                try:
                    frames.append(source.read())
                except StopIteration:
                    break
            frame = average_at_pose(frames, depth)
            if not np.any(frame.mm):
                LOG.warning("Captura vacía. Apunta a superficies dentro del rango útil.")
                continue
            index = len(entries)
            rgb_name, depth_name = f"rgb_{index:04d}.png", f"depth_{index:04d}.npy"
            success, encoded = cv2.imencode(".png", frame.rgb[..., ::-1])
            if not success:
                raise OSError("No se pudo codificar el PNG RGB.")
            encoded.tofile(directory / rgb_name)
            np.save(directory / depth_name, frame.mm, allow_pickle=False)
            entries.append(dict(index=index, rgb=rgb_name, depth=depth_name,
                                averaged_frames=len(frames), **frame_metadata(frame)))
            pending = directory / "metadata.tmp"
            pending.write_text(json.dumps({"schema": 1, "frames": entries}, indent=2), encoding="utf-8")
            pending.replace(directory / "metadata.json")
            LOG.info("Captura %d guardada en %s", index + 1, directory)
    finally:
        source.close()
        if interactive:
            cv2.destroyAllWindows()
    if not entries:
        raise ValueError("La sesión no contiene capturas guardadas.")
    return len(entries)

