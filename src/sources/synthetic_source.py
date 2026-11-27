"""Simulador RGB-D de un cuarto rectangular con obstáculos."""
import numpy as np

from ..model import Frame, Intrinsic, now


class SyntheticSource:
    """Proyecta cajas con textura analítica desde poses conocidas."""
    intrinsic = Intrinsic(320, 240, 185, 185, 159.5, 119.5)
    # Coordenadas del simulador: X a la derecha, Y hacia abajo, Z al frente.
    room = ((-2.2, -1.25, -2.0), (2.2, 1.35, 2.0))
    boxes = (
        ((-1.7, 0.38, 0.3), (-0.55, 1.35, 1.05)),
        ((0.45, 0.18, 0.8), (1.55, 1.35, 1.45)),
        ((-0.4, 0.58, -1.6), (0.6, 1.35, -0.85)),
        ((1.15, -0.28, -1.45), (1.8, 1.35, -0.95)),
    )

    def __init__(self, count: int = 33) -> None:
        self.count = count
        self.index = 0
        camera = self.intrinsic
        v, u = np.indices((camera.height, camera.width))
        self.rays = np.stack(((u-camera.cx)/camera.fx, (v-camera.cy)/camera.fy,
                              np.ones_like(u)), axis=-1)

    def read(self) -> Frame:
        """Intersecta rayos con el recinto y devuelve profundidad axial."""
        if self.index >= self.count:
            raise StopIteration
        yaw = 2 * np.pi * self.index / max(1, self.count - 1)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        origin = np.array([0.32*s, 0, 0.24*(1-c)])
        ray = self.rays @ rotation.T
        ray = np.where(abs(ray) < 1e-12, 1e-12, ray)
        lo, hi = (np.asarray(bound) for bound in self.room)
        depth = np.maximum((lo-origin)/ray, (hi-origin)/ray).min(axis=-1)
        labels = np.argmin(np.maximum((lo-origin)/ray, (hi-origin)/ray), axis=-1)
        for index, (lo, hi) in enumerate(self.boxes, 3):
            a, b = (lo-origin)/ray, (hi-origin)/ray
            near = np.minimum(a, b).max(axis=-1)
            far = np.maximum(a, b).min(axis=-1)
            hit = (near > 0) & (far >= near) & (near < depth)
            depth[hit], labels[hit] = near[hit], index
        xyz = origin + ray * depth[..., None]
        x, y, z = xyz.transpose(2, 0, 1)
        texture = (20*np.sin(6.7*x+4.3*y+2.9*z) + 18*np.cos(9.1*x-10.3*y+7.2*z)
                   + 21*np.cos(16.7*x+3.1*y-9.3*z) + 15*np.sin(3.9*x-15.5*y-4.3*z)
                   + 25*np.sin(4.8*x*y+5.7*y*z-3.9*x*z))
        palette = np.array([[178, 126, 109], [162, 167, 145], [114, 152, 180],
                            [70, 142, 109], [185, 119, 63], [99, 115, 174], [126, 83, 135]])
        rgb = np.clip(palette[labels] + texture[..., None], 0, 255).astype(np.uint8)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, origin
        self.index += 1
        return Frame(rgb, np.rint(depth*1000).astype(np.uint16), self.intrinsic, now(),
                     "synthetic_room", {"camera_to_world_reference": pose.tolist()})

    def close(self) -> None:
        """El simulador no mantiene recursos externos."""

