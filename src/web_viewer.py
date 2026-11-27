"""Exporta la reconstrucción como un visor 3D HTML sin conexión."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .__main__ import session_name
from .model import load_config
from .reconstruct import project
from .sources.replay_source import ReplaySource


def build_cloud(session: Path, report: dict, voxel_m: float) -> o3d.geometry.PointCloud:
    """Fusiona los frames originales a mayor densidad con las poses guardadas."""
    if not 0.008 <= voxel_m <= 0.030:
        raise ValueError('El voxel del visor debe estar entre 8 y 30 mm.')
    replay = ReplaySource(session)
    merged = o3d.geometry.PointCloud()
    tilt = 0.0
    for index, pose in enumerate(report["reconstruction"]["camera_to_global"]):
        frame = replay.read()
        if index == 0:
            tilt = np.deg2rad(frame.calibration.get("tilt_deg", 0))
        view = project(frame, report["parameters"]["depth"]).voxel_down_sample(voxel_m)
        view, _ = view.remove_statistical_outlier(nb_neighbors=14, std_ratio=2.1)
        merged += view.transform(np.asarray(pose))
    geometry = report.get("geometry")
    if geometry:
        merged.transform(np.asarray(geometry["global_to_world"]))
    else:
        c, s = np.cos(tilt), np.sin(tilt)
        merged.rotate(np.array([[1, 0, 0], [0, s, c], [0, -c, s]]), center=(0, 0, 0))
    cloud = merged.voxel_down_sample(voxel_m)
    if len(cloud.points) > 220_000:
        indices = np.linspace(0, len(cloud.points)-1, 220_000, dtype=int)
        cloud = cloud.select_by_index(indices.tolist())
    if not cloud.has_points():
        raise ValueError("La sesión no contiene puntos para visualizar.")
    return cloud


def specification(cloud: o3d.geometry.PointCloud) -> str:
    """Asigna color real y conserva una escala uniforme entre los ejes."""
    xyz = np.asarray(cloud.points, dtype=np.float32)
    rgb = np.rint(np.asarray(cloud.colors)*255).astype(np.uint8)
    rgb[rgb.max(axis=1) < 9] = [79, 93, 108]
    color = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb]
    figure = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=1.8, color=color), hoverinfo="skip",
    ))
    axes = dict(visible=False, showbackground=False, gridcolor="#293948",
                zerolinecolor="#4a6479", color="#aabed0")
    figure.update_layout(
        template=None, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="#0d1723",
        scene=dict(aspectmode="data", bgcolor="#0d1723", dragmode="orbit",
                   xaxis=dict(**axes, title="X (m)"), yaxis=dict(**axes, title="Y (m)"),
                   zaxis=dict(**axes, title="Z (m)")),
    )
    return figure.to_json()


def export_web(session: Path, output: Path, voxel_m: float = 0.015) -> Path:
    """Crea un único HTML con la geometría, estilos y motor de visualización."""
    report = json.loads((output / "reporte.json").read_text(encoding="utf-8"))
    cloud = build_cloud(session, report, voxel_m)
    template = Path(__file__).with_name("visor.html").read_text(encoding="utf-8")
    values = {
        "__PLOTLY_LIBRARY__": get_plotlyjs(),
        "__FIGURE__": specification(cloud),
        "__CAPTURES__": str(report["reconstruction"]["captures"]),
        "__POINTS__": f"{len(cloud.points):,}".replace(",", " "),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    path = output / "visor_3d.html"
    path.write_text(template, encoding="utf-8")
    return path


def main() -> None:
    """Genera el visor de una sesión ya reconstruida."""
    parser = argparse.ArgumentParser(description="Exporta un visor HTML de la nube de puntos.")
    parser.add_argument("session")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"config.yaml")
    parser.add_argument('--voxel-mm', type=float, default=15.0,
                        help='Resolucion del visor en milimetros (8 a 30)')
    args = parser.parse_args()
    config = load_config(args.config)
    session = session_name(args.session)
    try:
        print(export_web(config["paths"]["raw"]/session, config["paths"]["output"]/session, args.voxel_mm/1000))
    except (OSError, KeyError, StopIteration, ValueError) as error:
        parser.exit(1, f"No se pudo generar el visor. Comprueba la sesión y su reconstrucción. {error}\n")


if __name__ == "__main__":
    main()

