"""Visualizador Open3D y capturas ortográficas reproducibles."""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

VIEWS = {
    "planta": ([0, 0, 1], [0, 1, 0]),
    "frontal": ([0, -1, 0], [0, 0, 1]),
    "lateral": ([1, 0, 0], [0, 0, 1]),
    "isometrica": ([1, -1, 1], [0, 0, 1]),
}


def screenshots(cloud: o3d.geometry.PointCloud, output: Path) -> list[str]:
    """Renderiza la nube con z-buffer en CPU, incluso sin escritorio."""
    xyz = np.asarray(cloud.points)
    if not len(xyz):
        raise ValueError("No hay puntos para generar vistas.")
    colors = np.asarray(cloud.colors) if cloud.has_colors() else np.full_like(xyz, 0.55)
    bgr = np.clip(colors[:, ::-1]*255, 0, 255).astype(np.uint8)
    paths = []
    width, height = 1200, 900
    for name, (direction, up) in VIEWS.items():
        front = np.asarray(direction, dtype=float)
        front /= np.linalg.norm(front)
        right = np.cross(up, front)
        right /= np.linalg.norm(right)
        camera_up = np.cross(front, right)
        uv = xyz @ np.stack((right, -camera_up)).T
        center = (uv.min(axis=0)+uv.max(axis=0))/2
        scale = min((width-100)/max(0.01, np.ptp(uv[:, 0])),
                    (height-100)/max(0.01, np.ptp(uv[:, 1])))
        xy = np.rint((uv-center)*scale + [width/2, height/2]).astype(int)
        frame = np.full((height, width, 3), 247, np.uint8)
        for i in np.argsort(xyz @ front):
            cv2.circle(frame, xy[i], 2, bgr[i].tolist(), -1, cv2.LINE_AA)
        path = output / f"{name}.png"
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise OSError(f"No se pudo generar {path}")
        png.tofile(path)
        paths.append(path.name)
    return paths


def show(cloud: o3d.geometry.PointCloud) -> None:
    """Abre un visor con controles de cámara y tríada de ejes."""
    o3d.visualization.draw_geometries(
        [cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)],
        window_name="Laboratorio RGB-D · metros"
    )

