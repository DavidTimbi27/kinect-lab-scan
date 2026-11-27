"""Une una nueva reconstrucción RGB-D con una sesión previamente alineada."""
import argparse
import json
import logging
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import cv2
import numpy as np
import open3d as o3d

from .__main__ import session_name, write_report
from .model import load_config
from .reconstruct import REG, angle_deg, write_cloud, write_metrics
from .sources.replay_source import ReplaySource
from .web_viewer import export_web

LOG = logging.getLogger(__name__)


def read_sweep(raw: Path, output: Path) -> tuple:
    """Carga las poses y calcula descriptores con coordenadas métricas."""
    report = json.loads((output / "reporte.json").read_text(encoding="utf-8"))
    poses = np.asarray(report["reconstruction"]["camera_to_global"])
    source = ReplaySource(raw)
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.018)
    frames = []
    for pose in poses:
        frame = source.read()
        points, desc = sift.detectAndCompute(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY), None)
        if desc is None:
            frames.append(None)
            continue
        cam = frame.intrinsic
        uv = np.rint([point.pt for point in points]).astype(int)
        uv = uv.clip([0, 0], [cam.width-1, cam.height-1])
        z = frame.mm[uv[:, 1], uv[:, 0]]/1000
        xyz = np.column_stack(((uv[:, 0]-cam.cx)*z/cam.fx,
                               (uv[:, 1]-cam.cy)*z/cam.fy, z))
        frames.append((xyz @ pose[:3, :3].T + pose[:3, 3], desc, z > 0, uv))
    cloud = o3d.io.read_point_cloud(str(output / "nube_global.ply"))
    return report, poses, frames, cloud


def find_matches(source: list, reference: list) -> list:
    """Recoge correspondencias visuales con profundidad válida y deduplica píxeles."""
    matcher = cv2.BFMatcher()
    matches = []
    for i, frame in enumerate(source):
        if frame is None:
            continue
        a, ad, am, au = frame
        for j, target in enumerate(reference):
            if target is None or len(target[1]) < 2:
                continue
            b, bd, bm, bu = target
            seen = set()
            for m, n in matcher.knnMatch(ad, bd, k=2):
                u, v = m.queryIdx, m.trainIdx
                key = (*au[u], *bu[v])
                if m.distance < 0.77*n.distance and am[u] and bm[v] and key not in seen:
                    matches.append((a[u], b[v], i, j))
                    seen.add(key)
    return matches


def estimate(matches: list, source: o3d.geometry.PointCloud,
             reference: o3d.geometry.PointCloud) -> tuple[np.ndarray, dict]:
    """Ajusta una pose robusta a correspondencias RGB y verifica soporte geométrico."""
    if len(matches) < 15:
        raise ValueError("Aparecen muy pocos detalles comunes entre las tomas.")
    a, b = np.asarray([match[:2] for match in matches]).transpose(1, 0, 2)

    def cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def indices(count: int) -> o3d.utility.Vector2iVector:
        ids = np.arange(count)
        return o3d.utility.Vector2iVector(np.column_stack((ids, ids)))

    model = REG.TransformationEstimationPointToPoint(False)
    seed = REG.registration_ransac_based_on_correspondence(
        cloud(a), cloud(b), indices(len(a)), 0.065, model, 3,
        [REG.CorrespondenceCheckerBasedOnEdgeLength(0.85)],
        REG.RANSACConvergenceCriteria(30000, 0.999),
    ).transformation

    def residual(pose: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a @ pose[:3, :3].T + pose[:3, 3] - b, axis=1)

    keep = residual(seed) < 0.065
    if keep.sum() < 15:
        raise ValueError("No hay suficientes correspondencias 3D coherentes.")
    pose = model.compute_transformation(cloud(a[keep]), cloud(b[keep]), indices(int(keep.sum())))
    local = REG.registration_icp(source, reference, 0.085, pose,
                                 REG.TransformationEstimationPointToPlane(),
                                 REG.ICPConvergenceCriteria(max_iteration=40))
    correction = local.transformation @ np.linalg.inv(pose)
    if (np.linalg.norm(correction[:3, 3]) < 0.08 and angle_deg(correction) < 4
            and np.count_nonzero(residual(local.transformation) < 0.065) >= 0.90*keep.sum()):
        pose = local.transformation
    errors = residual(pose)
    keep = errors < 0.065
    support = {(i, j) for flag, (_, _, i, j) in zip(keep, matches) if flag}
    forward = REG.evaluate_registration(source, reference, 0.085, pose)
    reverse = REG.evaluate_registration(reference, source, 0.085, np.linalg.inv(pose))
    stats = dict(fitness=forward.fitness, reverse_fitness=reverse.fitness,
                 inlier_rmse_m=forward.inlier_rmse, visual_inliers=int(keep.sum()),
                 visual_rmse_m=float(np.sqrt(np.mean(errors[keep]**2))),
                 matched_frame_pairs=len(support))
    if (keep.sum() < 15 or len(support) < 2 or forward.fitness < 0.04
            or stats["visual_rmse_m"] > 0.045):
        raise ValueError(f"La unión carece de soporte suficiente. Métricas {stats}")
    return pose, stats


def copy_session(sources: list[Path], target: Path) -> None:
    """Crea una sesión reproducible que conserva la procedencia de sus archivos."""
    target.mkdir(parents=True, exist_ok=False)
    frames = []
    for source in sources:
        manifest = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        for entry in manifest["frames"]:
            item = dict(entry, index=len(frames))
            item.setdefault("source_session", source.name)
            item.setdefault("source_index", entry["index"])
            for key in ("rgb", "depth"):
                original = source / entry[key]
                item[key] = f"{key}_{len(frames):04d}{original.suffix}"
                copyfile(original, target / item[key])
            frames.append(item)
    (target / "metadata.json").write_text(json.dumps({"schema": 1, "frames": frames}, indent=2),
                                          encoding="utf-8")


def merge(reference: str, addition: str, session: str, config: dict) -> Path:
    """Fusiona la sesión entrante y exporta su visor acumulado."""
    start = perf_counter()
    raw, output = (config["paths"][key] for key in ("raw", "output"))
    old, old_poses, old_frames, old_cloud = read_sweep(raw / reference, output / reference)
    new, new_poses, new_frames, new_cloud = read_sweep(raw / addition, output / addition)
    pose, stats = estimate(find_matches(new_frames, old_frames), new_cloud, old_cloud)
    LOG.info("Unión %s", stats)
    destination = output / session
    copy_session([raw / reference, raw / addition], raw / session)
    merged = (old_cloud + new_cloud.transform(pose)).voxel_down_sample(config["registration"]["voxel_m"])
    write_cloud(merged, destination, "nube_global")
    count = len(old_poses)
    metrics = old["reconstruction"]["pairs"] + [
        dict(pair, source=count+pair["source"], target=count+pair["target"])
        for pair in new["reconstruction"]["pairs"]
    ] + [dict(stats, source=count, target=0, accepted=True, loop=False,
              source_to_target=pose.tolist(), fitness_scope="entre las nubes fusionadas")]
    write_metrics(destination, metrics)
    report = dict(session=session, time_s={"fusión": perf_counter()-start},
                  parameters=old["parameters"],
                  geometry=old.get("geometry"),
                  alignment=dict(reference=reference, addition=addition,
                                 previous=old.get("alignment"), **stats),
                  reconstruction=dict(method="Odometría RGB-D por barrido y correspondencias visuales 3D",
                                      captures=count+len(new_poses), points=len(merged.points), pairs=metrics,
                                      camera_to_global=np.concatenate((old_poses, pose @ new_poses)).tolist()))
    write_report(destination / "reporte.json", report)
    return export_web(raw / session, destination)


def main() -> None:
    """Ejecuta la fusión a partir de dos reconstrucciones guardadas."""
    cli = argparse.ArgumentParser(description="Une dos sesiones RGB-D reconstruidas.")
    for name in ("reference", "addition", "session"):
        cli.add_argument(name, type=session_name)
    args = cli.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(merge(args.reference, args.addition, args.session,
                    load_config(Path(__file__).resolve().parents[1] / "config.yaml")))
    except (OSError, ValueError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

