"""Proyección métrica, registro por pares y optimización del grafo."""
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from .model import Frame, clean_depth
from .sources.replay_source import ReplaySource

LOG = logging.getLogger(__name__)
REG = o3d.pipelines.registration
ODO = o3d.pipelines.odometry


@dataclass
class View:
    """Representaciones de una captura para las dos rutas de registro."""
    camera: o3d.camera.PinholeCameraIntrinsic
    rgbd: o3d.geometry.RGBDImage
    cloud: o3d.geometry.PointCloud
    feature: REG.Feature | None


def rgbd_images(frame: Frame, depth: dict, grayscale: bool = False) -> o3d.geometry.RGBDImage:
    """Crea imágenes Open3D con un milímetro por unidad de entrada."""
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(frame.rgb)),
        o3d.geometry.Image(clean_depth(frame.mm, depth)),
        depth_scale=1000, depth_trunc=(depth["max_mm"]+1)/1000,
        convert_rgb_to_intensity=grayscale,
    )


def intrinsic(frame: Frame) -> o3d.camera.PinholeCameraIntrinsic:
    """Adapta intrínsecos pinhole al tipo de Open3D."""
    model = frame.intrinsic
    return o3d.camera.PinholeCameraIntrinsic(model.width, model.height, model.fx, model.fy,
                                           model.cx, model.cy)


def project(frame: Frame, depth: dict) -> o3d.geometry.PointCloud:
    """Proyecta muestras válidas con ejes X derecha, Y abajo y Z delante."""
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_images(frame, depth), intrinsic(frame))


def prepare(frame: Frame, depth: dict, voxel: float, features: bool) -> View:
    """Limpia, submuestrea y calcula normales y descriptores."""
    cloud = project(frame, depth).voxel_down_sample(voxel)
    if len(cloud.points) < 50:
        raise ValueError("Nube vacía o dispersa. Revisa profundidad, calibración y tamaño de vóxel.")
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    cloud, _ = cloud.remove_radius_outlier(nb_points=5, radius=3*voxel)
    if len(cloud.points) < 50:
        raise ValueError("Quedaron menos de 50 puntos tras el filtrado. Revisa el rango y el vóxel.")
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=40))
    cloud.orient_normals_towards_camera_location()
    feature = REG.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=6*voxel, max_nn=100)
    ) if features else None
    return View(intrinsic(frame), rgbd_images(frame, depth, True), cloud, feature)


def angle_deg(transform: np.ndarray) -> float:
    """Mide el ?ngulo de la parte rotacional de una pose."""
    cosine = (np.trace(transform[:3, :3])-1)/2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def register_pair(source: View, target: View, config: dict, method: str,
                  prior: np.ndarray, loop: bool = False) -> REG.RegistrationResult:
    """Estima T de origen a destino y refina con ICP punto a plano."""
    voxel, threshold = config["voxel_m"], config["icp_m"]
    seeds = [prior]
    if method == "fpfh":
        coarse = REG.registration_ransac_based_on_feature_matching(
            source.cloud, target.cloud, source.feature, target.feature, True, 1.8*voxel,
            REG.TransformationEstimationPointToPoint(False), 3,
            [REG.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             REG.CorrespondenceCheckerBasedOnDistance(1.8*voxel)],
            REG.RANSACConvergenceCriteria(50000, 0.999),
        )
        if coarse.fitness > 0:
            seeds.append(coarse.transformation)
    else:
        if not np.allclose(source.camera.intrinsic_matrix, target.camera.intrinsic_matrix):
            raise ValueError("La odometría RGB-D requiere intrínsecos comunes.")
        success, transform, _ = ODO.compute_rgbd_odometry(
            source.rgbd, target.rgbd, source.camera, prior,
            ODO.RGBDOdometryJacobianFromHybridTerm(),
            ODO.OdometryOption(depth_diff_max=threshold),
        )
        if success:
            seeds.append(transform)
        else:
            LOG.warning("Odometría RGB-D sin solución. Se intentará ICP con la pose inicial.")
    results = []
    for seed in seeds:
        if not np.isfinite(seed).all():
            continue
        local = REG.registration_icp(source.cloud, target.cloud, threshold, seed,
                                     REG.TransformationEstimationPointToPlane(),
                                     REG.ICPConvergenceCriteria(max_iteration=55))
        results.append(local)
        for initial in (seed, local.transformation):
            baseline = REG.evaluate_registration(source.cloud, target.cloud, threshold, initial)
            if baseline.fitness < 0.10:
                continue
            results.append(baseline)
            try:
                colored = REG.registration_colored_icp(
                    source.cloud, target.cloud, threshold, initial,
                    REG.TransformationEstimationForColoredICP(0.80),
                    REG.ICPConvergenceCriteria(max_iteration=30),
                )
            except RuntimeError:
                LOG.debug("ICP con color sin correspondencias suficientes.")
            else:
                results.append(colored)
    step_m = config["loop_correction_m"] if loop else config["max_step_m"]
    step_deg = config["loop_correction_deg"] if loop else config["max_step_deg"]
    inverse_prior = np.linalg.inv(prior) if loop else np.eye(4)

    def score(result: REG.RegistrationResult) -> tuple[bool, float]:
        """Prioriza soluciones compatibles con el desplazamiento esperado."""
        delta = result.transformation @ inverse_prior
        within = (np.linalg.norm(delta[:3, 3]) <= step_m and angle_deg(delta) <= step_deg)
        pairs = np.asarray(result.correspondence_set)
        if not len(pairs):
            return False, 0.0
        a = np.asarray(source.cloud.colors)[pairs[:, 0]]
        b = np.asarray(target.cloud.colors)[pairs[:, 1]]
        photometric = float(np.median(np.linalg.norm(a-b, axis=1)))
        cost = result.inlier_rmse/threshold + config["color_weight"]*photometric
        return bool(within), result.fitness * np.exp(-cost)

    return max(results, key=score)


def write_cloud(cloud: o3d.geometry.PointCloud, directory: Path, stem: str) -> None:
    """Escribe la nube coloreada en PLY y PCD."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("ply", "pcd"):
        path = directory / f"{stem}.{extension}"
        if not o3d.io.write_point_cloud(str(path), cloud):
            raise OSError(f"No se pudo escribir {path}")


def reconstruct(session: Path, output: Path, config: dict, method: str) -> dict:
    """Construye el grafo, prueba cierres de lazo y fusiona las vistas."""
    settings = config["registration"]
    source = ReplaySource(session)
    views = []
    while True:
        try:
            frame = source.read()
        except StopIteration:
            break
        try:
            views.append(prepare(frame, config["depth"], settings["voxel_m"], method == "fpfh"))
        except ValueError as error:
            raise ValueError(f"Captura {len(views)}: {error}") from error
    if not views:
        raise ValueError("La sesión no contiene capturas.")
    graph = REG.PoseGraph()
    graph.nodes.append(REG.PoseGraphNode(np.eye(4)))
    pairs = []

    def add_pair(i: int, j: int, prior: np.ndarray, loop: bool) -> np.ndarray | None:
        """Evalúa y registra una arista si supera los umbrales."""
        a, b = views[i], views[j]
        result = register_pair(a, b, settings, method, prior, loop)
        transform = result.transformation
        delta = transform @ np.linalg.inv(prior) if loop else transform
        step_m = settings["loop_correction_m"] if loop else settings["max_step_m"]
        step_deg = settings["loop_correction_deg"] if loop else settings["max_step_deg"]
        consistent = np.linalg.norm(delta[:3, 3]) <= step_m and angle_deg(delta) <= step_deg
        fitness = settings["loop_fitness"] if loop else settings["min_fitness"]
        accepted = bool(result.fitness >= fitness and result.inlier_rmse <= settings["max_rmse_m"]
                        and np.isfinite(transform).all() and consistent)
        pairs.append(dict(source=i, target=j, loop=loop, accepted=accepted,
                          fitness=result.fitness, inlier_rmse_m=result.inlier_rmse,
                          source_to_target=transform.tolist()))
        LOG.info("%d → %d | fitness %.3f | RMSE %.4f m | %s", i, j, result.fitness,
                 result.inlier_rmse, "aceptado" if accepted else "descartado")
        if not accepted:
            return None
        information = REG.get_information_matrix_from_point_clouds(
            a.cloud, b.cloud, settings["icp_m"], transform
        )
        graph.edges.append(REG.PoseGraphEdge(i, j, transform, information, uncertain=loop))
        return transform

    for j in range(1, len(views)):
        prior = np.eye(4)
        if j > 1:
            prior = np.linalg.inv(graph.nodes[j-1].pose) @ graph.nodes[j-2].pose
        transform = add_pair(j-1, j, prior, False)
        if transform is None:
            write_metrics(output, pairs)
            raise ValueError(f"No se pudo unir capturas {j-1} y {j}. Revisa registro_pares.json "
                             "y repite esa transición con más solapamiento o pasos más cortos.")
        graph.nodes.append(REG.PoseGraphNode(graph.nodes[j-1].pose @ np.linalg.inv(transform)))
    for j in range(settings["loop_gap"], len(views)):
        candidates = []
        inverse_target = np.linalg.inv(graph.nodes[j].pose)
        for i in range(j-settings["loop_gap"]+1):
            prior = inverse_target @ graph.nodes[i].pose
            distance, angle = np.linalg.norm(prior[:3, 3]), angle_deg(prior)
            if distance <= settings["loop_distance_m"] and angle <= settings["loop_angle_deg"]:
                candidates.append((distance/settings["loop_distance_m"] + angle/settings["loop_angle_deg"],
                                   i, prior))
        for _, i, prior in sorted(candidates, key=lambda item: item[0])[:4]:
            if add_pair(i, j, prior, True) is not None:
                break
    if len(views) > 1:
        REG.global_optimization(graph, REG.GlobalOptimizationLevenbergMarquardt(),
                                REG.GlobalOptimizationConvergenceCriteria(),
                                REG.GlobalOptimizationOption(
                                    max_correspondence_distance=settings["icp_m"],
                                    edge_prune_threshold=0.25, reference_node=0))
    merged = o3d.geometry.PointCloud()
    for view, node in zip(views, graph.nodes):
        merged += deepcopy(view.cloud).transform(node.pose)
    merged = merged.voxel_down_sample(settings["voxel_m"])
    merged.normalize_normals()
    write_cloud(merged, output, "nube_global")
    write_metrics(output, pairs)
    pending = output / "pose_graph.pending.json"
    o3d.io.write_pose_graph(str(pending), graph)
    if not pending.is_file() or not pending.stat().st_size:
        raise OSError("No se pudo exportar el grafo de poses.")
    pending.replace(output / "pose_graph.json")
    return dict(captures=len(views), method=method, points=len(merged.points), pairs=pairs,
                camera_to_global=[node.pose.tolist() for node in graph.nodes])


def write_metrics(output: Path, pairs: list[dict]) -> None:
    """Conserva el diagnóstico también si falla una arista consecutiva."""
    import json
    output.mkdir(parents=True, exist_ok=True)
    (output / "registro_pares.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

