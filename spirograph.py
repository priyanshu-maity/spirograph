import math
import itertools
from typing import Self, Literal, Optional
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation


class Arm:
    def __init__(
            self: Self,
            center: tuple[float, float],
            length: float,
            speed: float,
            color: str,
            start_angle: float,
            direction: Literal['CW', 'ACW'] = 'CW'
    ) -> None:
        self.center_x, self.center_y = center
        self.length: float = length
        self.speed: float = speed
        self.color: str = color
        self.direction: Literal['CW', 'ACW'] = direction
        self.direction_factor: int = 1 if direction == 'CW' else -1

        self.start_angle: float = start_angle
        self.angle: float = start_angle

        self.trace_points: np.ndarray = np.zeros((0, 2), dtype=np.float32)


class Animate:
    animation: FuncAnimation
    fig: Figure
    ax: Axes
    arms: list[Arm]
    frames: Optional[int]
    lines: list[Line2D] = []
    traces: list[Line2D] = []
    dots: list[Line2D] = []
    central_dot: Line2D = []
    total_arm_length: float = 0

    DELTA_ANGLE: float = 0.02
    SCALE_FACTOR: int = 10000

    def __init__(self: Self, arms: list[Arm], frames: Optional[int] = None) -> None:
        self.arms = arms
        self.frames = frames

        self.main()

    def main(self: Self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)

        for arm in self.arms:
            self.total_arm_length += arm.length * self.SCALE_FACTOR
            line: Line2D = self.ax.plot([], [], color=arm.color, linewidth=1)[0]
            trace: Line2D = self.ax.plot([], [], color=arm.color, linewidth=1)[0]
            dot: Line2D = self.ax.plot([], [], color=arm.color, markersize=5, marker='o')[0]
            self.lines.append(line)
            self.traces.append(trace)
            self.dots.append(dot)

        self.central_dot: Line2D = self.ax.plot([], [], color='black', markersize=5, marker='o')[0]

        self.ax.set_xlim(-self.total_arm_length - 10, self.total_arm_length + 10)
        self.ax.set_ylim(-self.total_arm_length - 10, self.total_arm_length + 10)

        self.animation = FuncAnimation(
            fig=self.fig,
            func=self.update,
            frames=itertools.count() if self.frames is None else self.frames,
            interval=20,
            blit=True,
            cache_frame_data=False
        )

        plt.show()

    def update(self: Self, frame: int) -> Iterable[Artist]:
        for idx, arm in enumerate(self.arms):
            # Calculate new x, y based on arm angle and length
            x = arm.center_x + arm.length * self.SCALE_FACTOR * math.cos(arm.angle)
            y = arm.center_y + arm.length * self.SCALE_FACTOR * math.sin(arm.angle)

            if idx < len(self.arms) - 1:
                # Update the next arm's center position
                self.arms[idx + 1].center_x = x
                self.arms[idx + 1].center_y = y

            # Update the lines connecting the arms
            self.lines[idx].set_data([arm.center_x, x], [arm.center_y, y])
            self.traces[idx].set_data(arm.trace_points[:, 0], arm.trace_points[:, 1])
            self.dots[idx].set_data([x], [y])
            self.central_dot.set_data([0], [0])

            # Update the arm angle for the next frame
            arm.angle += arm.speed * arm.direction_factor * self.DELTA_ANGLE
            arm.trace_points = np.vstack((arm.trace_points, [x, y]))

        return self.lines + self.traces + self.dots + [self.central_dot]


if __name__ == '__main__':
    arm1 = Arm(center=(0, 0), length=1, speed=10, color='red', start_angle=math.pi, direction='ACW')
    arm2 = Arm(center=(0, 0), length=2, speed=9, color='blue', start_angle=math.pi, direction='CW')
    arm3 = Arm(center=(0, 0), length=3, speed=8, color='green', start_angle=math.pi, direction='ACW')
    arm4 = Arm(center=(0, 0), length=4, speed=7, color='magenta', start_angle=math.pi, direction='CW')
    arm5 = Arm(center=(0, 0), length=5, speed=6, color='purple', start_angle=math.pi, direction='ACW')
    arm6 = Arm(center=(0, 0), length=6, speed=5, color='cyan', start_angle=math.pi, direction='CW')
    arm7 = Arm(center=(0, 0), length=7, speed=4, color='orange', start_angle=math.pi, direction='ACW')
    arm8 = Arm(center=(0, 0), length=8, speed=3, color='yellow', start_angle=math.pi, direction='CW')
    arm9 = Arm(center=(0, 0), length=9, speed=2, color='lime', start_angle=math.pi, direction='ACW')
    arm10 = Arm(center=(0, 0), length=10, speed=1, color='teal', start_angle=math.pi, direction='CW')

    Animate([arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, arm9, arm10])
