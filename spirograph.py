import math
from datetime import datetime
import itertools
from typing import Self, Literal, Optional, Any
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import KeyEvent
from matplotlib.text import Text
import matplotlib.image
from matplotlib.image import AxesImage, imread
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from screeninfo import get_monitors


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
    def __init__(self: Self, name: str, arms: list[Arm], frames: Optional[int] = None, delta_angle: float = 0.02) -> None:
        self.name = name
        self.arms: list[Arm] = arms
        self.frames: Optional[int] = frames

        self.lines: list[Line2D] = []
        self.traces: list[Line2D] = []
        self.dots: list[Line2D] = []

        self.max_arm_length: float = 0.0
        self.delta_angle: float = delta_angle

        self.is_h_pressed: bool = False
        self.h_box: Optional[Text] = None
        self.h_box_dict: dict[str] = {
            "h": "Help",
            "[space]": "Pause/Resume",
            "t": "Start/Stop Tracing",
            "r": "Start/Stop Recording"
        }
        self.h_box_text: str = "\n".join([f"'{key}': {value}" for key, value in self.h_box_dict.items()])

        self.is_recording: bool = False
        self.is_playing: bool = False
        self.is_tracing = False

        self.rec_image: matplotlib.image = imread('rec_button.png')
        self.rec: Optional[AxesImage] = None

        self.setup_window()
        self.main()

    def setup_window(self):
        # Get screen dimensions in inches and pixels
        screen_width_in = get_monitors()[0].width_mm / 25.4
        screen_height_in = get_monitors()[0].height_mm / 25.4
        screen_width_px = get_monitors()[0].width
        screen_height_px = get_monitors()[0].height

        # Set display size (75% of the smaller screen dimension)
        display_size_in = min(screen_width_in, screen_height_in) * 0.75

        # Create figure and axis
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(display_size_in, display_size_in))
        self.ax.axis("off")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.ax.set_aspect('equal', 'box')

        # Remove extra padding around the plot
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Get the Matplotlib figure manager
        manager = plt.get_current_fig_manager()
        manager.window.state('normal')  # Ensure it's not maximized
        manager.canvas.manager.window.title(f"Spirograph - {self.name}")

        # Update window size info
        manager.window.update_idletasks()
        window_width = manager.window.winfo_width()
        window_height = manager.window.winfo_height()

        # Center the window
        pos_x = (screen_width_px - window_width) // 2
        pos_y = (screen_height_px - window_height) // 2
        manager.window.geometry(f"+{pos_x}+{pos_y}")

        # Hide toolbar
        manager.toolbar.pack_forget()

        # Add a central dot
        self.central_dot: Line2D = self.ax.plot([0], [0], color='black', markersize=5, marker='D')[0]

    def main(self: Self):
        for idx, arm in enumerate(self.arms):
            self.max_arm_length += arm.length
            x: float = arm.center_x + arm.length * math.cos(arm.angle)
            y: float = arm.center_y + arm.length * math.sin(arm.angle)

            if idx < len(self.arms) - 1:
                # Update the next arm's center position
                self.arms[idx + 1].center_x = x
                self.arms[idx + 1].center_y = y

            line: Line2D = self.ax.plot([arm.center_x, x], [arm.center_y, y], color=arm.color, linewidth=1)[0]
            trace: Line2D = self.ax.plot([], [], color=arm.color, linewidth=1)[0]
            dot: Line2D = self.ax.plot([x], [y], color=arm.color, markersize=5, marker='o')[0]
            self.lines.append(line)
            self.traces.append(trace)
            self.dots.append(dot)

        self.ax.set_xlim(-self.max_arm_length - 5, self.max_arm_length + 5)
        self.ax.set_ylim(-self.max_arm_length - 5, self.max_arm_length + 5)

        self.animation: FuncAnimation = FuncAnimation(
            fig=self.fig,
            func=self.update,
            frames=itertools.count() if self.frames is None else self.frames,
            interval=10,
            blit=True,
            cache_frame_data=False
        )

        plt.show()

    def update(self: Self, frame: int) -> Iterable[Artist]:
        if self.is_playing:
            total_arm_length: float = 0.0
            for idx, arm in enumerate(self.arms):
                # Calculate new x, y based on arm angle and length
                x = arm.center_x + arm.length * math.cos(arm.angle)
                y = arm.center_y + arm.length * math.sin(arm.angle)

                if idx < len(self.arms) - 1:
                    # Update the next arm's center position
                    self.arms[idx + 1].center_x = x
                    self.arms[idx + 1].center_y = y

                # Update the lines connecting the arms
                self.lines[idx].set_data([arm.center_x, x], [arm.center_y, y])
                self.traces[idx].set_data(arm.trace_points[:, 0], arm.trace_points[:, 1])
                self.dots[idx].set_data([x], [y])

                # Update the arm angle for the next frame
                arm.angle += arm.speed * arm.direction_factor * self.delta_angle

                if self.is_tracing:
                    arm.trace_points = np.vstack((arm.trace_points, [x, y]))

                total_arm_length += arm.length

                self.ax.set_xlim(-self.max_arm_length - 5, self.max_arm_length + 5)
                self.ax.set_ylim(-self.max_arm_length - 5, self.max_arm_length + 5)

            if total_arm_length > self.max_arm_length:
                self.max_arm_length = total_arm_length

        return self.lines + [self.central_dot] + self.traces + self.dots + [self.ax]

    def on_key_press(self: Self, event: KeyEvent) -> Any:
        if event.key == 'h':
            self.is_h_pressed = True
            if self.h_box is None:
                self.h_box = self.ax.text(
                    0.05, 0.95, f"KEY OPTIONS: \n{self.h_box_text}",
                    transform=self.ax.transAxes, ha='left', va='top',
                    fontsize=12, color='white',
                    bbox=dict(
                        facecolor='grey', edgecolor='black',
                        alpha=0.7, boxstyle='round,pad=1', lw=1
                    ),
                    zorder=1000,
                )
        elif event.key == 'r':
            self.is_recording = not self.is_recording
            if self.is_recording:
                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                image_width, image_height = self.rec_image.shape[1], self.rec_image.shape[0]
                image_scale = 0.003
                offset = 0.2
                self.rec = self.ax.imshow(
                    self.rec_image,
                    extent=(
                        xlim[1] - image_width * image_scale - offset, xlim[1] - offset,
                        ylim[1] - image_height * image_scale - offset, ylim[1] - offset
                    )
                )
            else:
                if self.rec is not None:
                    self.rec.remove()
                    self.rec = None
        elif event.key == ' ':
            self.is_playing = not self.is_playing
        elif event.key == 't':
            self.is_tracing = not self.is_tracing
            if not self.is_tracing:
                for arm in self.arms:
                    arm.trace_points = np.zeros((0, 2), dtype=np.float32)
        elif event.key == 's':
            ...

    def on_key_release(self: Self, event: KeyEvent) -> Any:
        if event.key == 'h':
            self.is_h_pressed = False
            if self.h_box is not None:
                self.h_box.remove()
                self.h_box = None


if __name__ == '__main__':
    arm1 = Arm(center=(0, 0), length=1, speed=1, color='black', start_angle=0, direction='CW')
    arm2 = Arm(center=(0, 0), length=1, speed=2, color='black', start_angle=0, direction='CW')
    arm3 = Arm(center=(0, 0), length=1, speed=3, color='black', start_angle=0, direction='ACW')
    # arm4 = Arm(center=(0, 0), length=4, speed=4, color='black', start_angle=-math.pi//2, direction='CW')
    # arm5 = Arm(center=(0, 0), length=5, speed=5, color='black', start_angle=math.pi//3, direction='ACW')

    Animate(
        name="Test",
        arms=[
            arm1,
            arm2,
            arm3,
            # arm4,
            # arm5,
        ]
    )
