"""Segmentación del recinto y exportación de superficies métricas."""
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from .reconstruct import write_cloud

LOG = logging.getLogger(__name__)


def segment(cloud: o3d.geometry.PointCloud, config: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extrae planos grandes y conserva sus puntos de soporte."""
    planes = []
    rest = cloud
    minimum = max(50, int(len(cloud.points)*config["min_plane_fraction"]))
    for _ in range(config["max_planes"]):
        if len(rest.points) < minimum:
            break
        model, inliers = rest.segment_plane(config["plane_threshold_m"], 3, 1400)
        if len(inliers) < minimum:
            break
        plane = np.asarray(model)
        plane /= np.linalg.norm(plane[:3])
        planes.append((plane, np.asarray(rest.points)[inliers]))
        rest = rest.select_by_index(inliers, invert=True)
    return planes


def room_axes(planes: list, up_hint: np.ndarray, forward_hint: np.ndarray) -> tuple:
    """Escoge el plano horizontal inferior y estima ejes desde una pared."""
    up_hint = up_hint / np.linalg.norm(up_hint)
    horizontal = [(plane, points) for plane, points in planes
                  if abs(plane[:3] @ up_hint) > 0.92]
    if not horizontal:
        raise ValueError("No se detectó el piso. Inclúyelo en varias vistas y revisa up_hint_camera.")
    floor, support = min(horizontal, key=lambda item: np.median(item[1] @ up_hint))
    sign = 1 if floor[:3] @ up_hint > 0 else -1
    up, offset = sign*floor[:3], sign*floor[3]
    forward = forward_hint - (forward_hint @ up)*up
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("forward_hint_camera debe tener una componente paralela al piso.")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    vertical = [(plane, points) for plane, points in planes if abs(plane[:3] @ up) < 0.20]
    if vertical:
        wall, _ = max(vertical, key=lambda item: len(item[1]))
        normal = wall[:3] - (wall[:3] @ up)*up
        normal /= np.linalg.norm(normal)
        first = normal if abs(normal @ right) >= abs(normal @ forward) else np.cross(normal, up)
        right = first if first @ right >= 0 else -first
    transform = np.eye(4)
    transform[:3, :3] = np.stack((right, np.cross(up, right), up))
    transform[2, 3] = offset
    return transform, support, bool(vertical)


def extract_geometry(cloud_path: Path, output: Path, config: dict,
                     simulation: bool = False) -> dict:
    """Alinea la nube al piso, estima dimensiones y reconstruye una malla BPA."""
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if len(cloud.points) < 50:
        raise ValueError(f"Nube insuficiente para segmentar planos: {cloud_path}")
    sample = cloud.voxel_down_sample(config["mesh_voxel_m"])
    planes = segment(sample, config)
    transform, floor_points, wall_axes = room_axes(
        planes, np.asarray(config["up_hint_camera"], dtype=float),
        np.asarray(config["forward_hint_camera"], dtype=float),
    )
    cloud.transform(transform)
    write_cloud(cloud, output, "nube_piso")
    xyz = np.asarray(cloud.points)
    low, high = np.quantile(xyz, [0.01, 0.99], axis=0)
    low[2] = 0
    walls: dict[int, list[float]] = {0: [], 1: []}
    ceiling = []
    surfaces: list[dict] = []
    for plane, support in planes:
        aligned = support @ transform[:3, :3].T + transform[:3, 3]
        model = np.linalg.solve(transform.T, plane)
        normal = model[:3]
        axis = int(np.argmax(abs(normal)))
        coordinate = float(np.median(aligned[:, axis]))
        kind = "obstaculo"
        if support is floor_points:
            kind = "piso"
        elif abs(normal[2]) > 0.92 and coordinate >= config["min_room_height_m"]:
            kind = "techo"
            ceiling.append(coordinate)
        elif axis < 2 and abs(normal[axis]) > 0.92 and np.ptp(aligned[:, 2]) >= 1.15:
            kind = "pared"
            walls[axis].append(coordinate)
        surfaces.append(dict(kind=kind, plane=model.tolist(), points=len(support),
                             min_m=aligned.min(axis=0).tolist(), max_m=aligned.max(axis=0).tolist()))
    estimates = {}
    for axis, name in enumerate(("x", "y")):
        if len(walls[axis]) >= 2 and np.ptp(walls[axis]) > 0.8:
            low[axis], high[axis] = min(walls[axis]), max(walls[axis])
            estimates[name] = "paredes opuestas"
        else:
            estimates[name] = "extensión observada (percentiles 1–99)"
    if ceiling:
        high[2] = max(ceiling)
    else:
        LOG.warning("Techo ausente. La altura reporta la extensión observada.")
    estimates["z"] = "plano de techo" if ceiling else "extensión observada (percentil 99)"
    extents = high-low

    mesh_cloud = cloud.voxel_down_sample(config["mesh_voxel_m"])
    if not mesh_cloud.has_normals() or np.any(np.linalg.norm(mesh_cloud.normals, axis=1) < 0.5):
        mesh_cloud.estimate_normals()
        mesh_cloud.orient_normals_consistent_tangent_plane(20)
    voxel = config["mesh_voxel_m"]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        mesh_cloud, o3d.utility.DoubleVector([1.25*voxel, 2*voxel, 3*voxel]))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("BPA no produjo triángulos. Ajusta mesh_voxel_m y revisa cobertura y normales.")
    mesh.compute_vertex_normals()
    for extension in ("obj", "ply"):
        path = output / f"malla.{extension}"
        if not o3d.io.write_triangle_mesh(str(path), mesh, write_triangle_uvs=False):
            raise OSError(f"No se pudo escribir {path}")

    report = dict(dimensions_m=dict(largo=float(max(extents[:2])), ancho=float(min(extents[:2])),
                                    alto=float(extents[2])),
                  dimension_basis=estimates, axes_from_wall=wall_axes,
                  room_bounds_m=dict(min=low.tolist(), max=high.tolist()),
                  cloud_aabb_m=dict(min=xyz.min(axis=0).tolist(), max=xyz.max(axis=0).tolist()),
                  global_to_world=transform.tolist(), planes=surfaces, triangles=len(mesh.triangles))
    if simulation:
        model = dict(units="m", z_up=True, room_bounds_m=report["room_bounds_m"],
                     plane_patches=surfaces)
        (output / "modelo_simulacion.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (output / "geometria.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
