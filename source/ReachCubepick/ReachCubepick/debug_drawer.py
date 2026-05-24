import random
from matplotlib.pyplot import draw
import torch
import carb
import numpy as np
from pxr import Gf

from isaacsim.util.debug_draw import _debug_draw



class DebugDrawer:
    """Modern debug drawer using omni.debug_draw, compatible with Isaac Lab 2025+"""

    def __init__(self):
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def clear(self):
        """Clear all debug drawings."""
        self.draw.clear_points()
        self.draw.clear_lines()

    def draw_line(
        self,
        from_point: np.ndarray or torch.Tensor,
        to_point: np.ndarray or torch.Tensor,
        color,
        thickness: float = 1.0,
    ):
        """Draw a line between two points in world frame.

        Args:
            from_point: (3,) array-like [x, y, z]
            to_point: (3,) array-like [x, y, z]
            color: (r, g, b) values in [0, 1]
            thickness: line width
        """
        if isinstance(from_point, torch.Tensor):
            from_point = from_point.cpu().tolist()
        if isinstance(to_point, torch.Tensor):
            to_point = to_point.cpu().tolist()

        # Convert to Gf.Vec3f
        # p0 = Gf.Vec3f(*from_point)
        # p1 = Gf.Vec3f(*to_point)
        # col = carb.Float3(*color)
        p0 = [Gf.Vec3f(*fp) for fp in [from_point]]
        p1 = [Gf.Vec3f(*tp) for tp in [to_point]]
        # Draw line
        # N = 10000
        # point_list_1 = [
        #     (random.uniform(10, 30), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(N)
        # ]
        # point_list_2 = [
        #     (random.uniform(10, 30), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(N)
        # ]
        # colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(N)]
        # sizes = [random.randint(1, 25) for _ in range(N)]
        self.draw.draw_lines(p0, p1, color, thickness)

        # self.draw.draw_lines(from_point, to_point, color, thickness)