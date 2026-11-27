"""Interfaz de captura, reconstrucción, geometría y visualización."""
import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

from .model import load_config

LOG = logging.getLogger("kinect_lab_scan")


def parser() -> argparse.ArgumentParser:
    """Declara los comandos y sus opciones."""
    cli = argparse.ArgumentParser(description="Gemelo digital RGB-D para Kinect v1")
    cli.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"config.yaml")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("capture", "reconstruct", "geometry", "all", "visualize"):
        command = commands.add_parser(name)
        command.add_argument("session", help="Nombre de sesión, sin directorios")
        if name in ("capture", "all"):
            command.add_argument("--source", choices=("kinect", "synthetic", "replay"),
                                 default="kinect" if name == "capture" else None)
            command.add_argument("--input-session")
            command.add_argument("--count", type=int)
            command.add_argument("--average", type=int)
            command.add_argument("--auto", action="store_true", help="Guarda sin vista previa")
            command.add_argument("--vertical-sweep", action="store_true",
                                 help="Captura tres inclinaciones con el motor del Kinect v1")
        if name in ("reconstruct", "all"):
            command.add_argument("--method", choices=("fpfh", "rgbd"), default="fpfh")
        if name in ("geometry", "all"):
            command.add_argument("--simulation", action="store_true")
        if name in ("geometry", "all", "visualize"):
            command.add_argument("--show", action="store_true")
    return cli


def session_name(value: str) -> str:
    """Verifica que la sesión sea un único nombre de directorio."""
    if not value or value in (".", "..") or any(ch in value for ch in '/\\:*?"<>|'):
        raise ValueError("Indica un nombre de sesión sin directorios ni caracteres reservados.")
    return value


def capture_session(args: argparse.Namespace, config: dict, directory: Path) -> int:
    """Prepara la fuente seleccionada y gestiona el promedio por pose."""
    from .capture import capture
    from .sources.kinect_source import KinectSource
    from .sources.replay_source import ReplaySource
    from .sources.synthetic_source import SyntheticSource

    count = args.count
    sweep = args.vertical_sweep
    if sweep and (args.source != "kinect" or count is not None):
        raise ValueError("--vertical-sweep requiere Kinect y usa tres posiciones fijas.")
    average = args.average if args.average is not None else (
        config["capture"]["average"] if args.source == "kinect" else 1)
    if count is not None and count < 1:
        raise ValueError("--count debe ser positivo.")
    if args.source == "kinect":
        if args.auto and count is None and not sweep:
            raise ValueError("Con Kinect y --auto indica --count. Usa un trípode fijo para promediar.")
        source = KinectSource(config["capture"], sweep_average=average if sweep else None)
    elif args.source == "synthetic":
        source = SyntheticSource(count or 33)
    else:
        if not args.input_session:
            raise ValueError("--source replay requiere --input-session.")
        source = ReplaySource(config["paths"]["raw"] / session_name(args.input_session))
    if args.source != "kinect" and average != 1:
        source.close()
        raise ValueError("Cada frame sintético o de replay corresponde a una pose distinta. Usa --average 1.")
    return capture(source, directory, config["depth"], average,
                   interactive=args.source == "kinect" and not (args.auto or sweep), limit=count)


def write_report(path: Path, report: dict) -> None:
    """Exporta métricas y parámetros en JSON y un resumen en Markdown."""
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    reconstruction = report.get("reconstruction", {})
    lines = [f"# Sesión {report['session']}", "",
             f"- Capturas procesadas: {reconstruction.get('captures', report.get('captures_saved', 'pendiente'))}",
             f"- Registro: {reconstruction.get('method', 'pendiente')}",
             f"- Puntos fusionados: {reconstruction.get('points', 'pendiente')}"]
    lines.append(f"- Tiempo acumulado: {sum(report['time_s'].values()):.2f} s")
    for stage, seconds in report["time_s"].items():
        lines.append(f"- {stage}: {seconds:.2f} s")
    if geometry := report.get("geometry"):
        lines += ["", "## Dimensiones estimadas", ""]
        lines += [f"- {name}: {value:.3f} m" for name, value in geometry["dimensions_m"].items()]
        lines += [f"- Eje {axis}: {basis}" for axis, basis in geometry["dimension_basis"].items()]
    if pairs := reconstruction.get("pairs"):
        lines += ["", "## Registro por pares", "",
                  "| Origen | Destino | Lazo | Aceptado | Fitness | RMSE (m) |",
                  "| --- | --- | --- | --- | ---: | ---: |"]
        for pair in pairs:
            lines.append(f"| {pair['source']} | {pair['target']} | {pair['loop']} | "
                         f"{pair['accepted']} | {pair['fitness']:.3f} | {pair['inlier_rmse_m']:.4f} |")
        lines += ["", "Las matrices origen → destino están en `reporte.json` y `registro_pares.json`."]
    lines += ["", "## Parámetros", "", "```json",
              json.dumps(report["parameters"], ensure_ascii=False, indent=2), "```"]
    path.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Encadena las etapas seleccionadas y actualiza los informes."""
    config = load_config(args.config.resolve())
    session = session_name(args.session)
    raw = config["paths"]["raw"] / session
    output = config["paths"]["output"] / session
    report_path = output / "reporte.json"
    report = dict(session=session, time_s={}, parameters={})
    if report_path.is_file() and args.command in ("reconstruct", "geometry", "visualize"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.command == "capture" or (args.command == "all" and args.source):
        start = perf_counter()
        report["captures_saved"] = capture_session(args, config, raw)
        report["time_s"]["captura"] = perf_counter()-start
        report["parameters"]["depth"] = config["depth"]
        manifest = json.loads((raw / "metadata.json").read_text(encoding="utf-8"))
        frames = manifest["frames"]
        report["parameters"]["capture"] = dict(source=args.source, sensor=frames[0]["sensor"],
                                               intrinsic=frames[0]["intrinsic"],
                                               calibration=frames[0]["calibration"],
                                               averaged_frames=sorted({frame["averaged_frames"]
                                                                       for frame in frames}))
        output.mkdir(parents=True, exist_ok=True)
        write_report(report_path, report)
    if args.command in ("reconstruct", "all"):
        from .reconstruct import reconstruct

        start = perf_counter()
        report["reconstruction"] = reconstruct(raw, output, config, args.method)
        report["time_s"]["reconstrucción"] = perf_counter()-start
        report["parameters"].update({key: config[key] for key in ("depth", "registration")})
        write_report(report_path, report)
    if args.command in ("geometry", "all"):
        from .extract_geometry import extract_geometry

        start = perf_counter()
        geometry = extract_geometry(output / "nube_global.ply", output, config["geometry"], args.simulation)
        report["geometry"] = geometry
        report["time_s"]["geometría"] = perf_counter()-start
        report["parameters"]["geometry"] = config["geometry"]
        write_report(report_path, report)
        dimensions = geometry["dimensions_m"]
        LOG.info("Recinto %.2f × %.2f × %.2f m (largo × ancho × alto)",
                 dimensions["largo"], dimensions["ancho"], dimensions["alto"])
    if args.command in ("geometry", "all", "visualize"):
        import open3d as o3d

        from .visualize import screenshots, show

        start = perf_counter()
        cloud = o3d.io.read_point_cloud(str(output / "nube_piso.ply"))
        report["screenshots"] = screenshots(cloud, output)
        report["time_s"]["vistas"] = perf_counter()-start
        write_report(report_path, report)
        if args.show or args.command == "visualize":
            show(cloud)
    LOG.info("Resultados en %s", output)


def main() -> int:
    """Presenta errores operativos y devuelve un estado útil al shell."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        LOG.info("Operación interrumpida.")
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        LOG.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
