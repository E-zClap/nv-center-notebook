import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time


class NVCenter:
    """NV‑center symmetry explorer with an optional single diamond unit cell"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NV Center Symmetry Explorer")
        self.root.geometry("1200x800")

        # Lattice‑visibility toggle
        self.show_lattice = False  # start hidden

        # Core geometry
        self.setup_nv_center()
        self.generate_unit_cell(a=1.2)  # **one** conventional cubic cell

        # Animation parameters
        self.animation_steps = 50
        self.animating = False

        # Build GUI
        self.create_gui()

    # ------------------------------------------------------------------
    # Geometry definitions
    # ------------------------------------------------------------------
    def setup_nv_center(self):
        """Define NV‑center positions (Å, arbitrary units)"""
        self.carbon_positions = np.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, np.sqrt(3) / 2, 0.0],
                [-0.5, -np.sqrt(3) / 2, 0.0],
            ]
        )
        self.nitrogen_position = np.array([0.0, 0.0, 1.0])
        self.vacancy_position = np.zeros(3)

        self.current_carbon = self.carbon_positions.copy()
        self.current_nitrogen = self.nitrogen_position.copy()
        self.current_vacancy = self.vacancy_position.copy()

        self.original_carbon = self.carbon_positions.copy()
        self.original_nitrogen = self.nitrogen_position.copy()
        self.original_vacancy = self.vacancy_position.copy()

    def generate_unit_cell(self, a: float = 1.0):
        """Generate **single** conventional cubic diamond unit cell around the origin."""
        basis = np.array(
            [
                [0, 0, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ]
        )
        # Center the cell at the origin for nicer overlap with NV positions
        self.lattice_positions = a * (basis - 0.5)

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def create_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control = ttk.Frame(main_frame)
        control.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        ttk.Label(control, text="NV Center Symmetries", font=("Arial", 16, "bold")).pack(pady=(0, 20))

        self.create_buttons(control)
        self.plot_scene()

    def create_buttons(self, parent):
        def btn(text, cmd):
            ttk.Button(parent, text=text, width=20, command=cmd).pack(pady=5)

        btn("C3 Rotation (120°)", lambda: self.animate_rotation([0, 0, 1], 2 * np.pi / 3))
        btn("C3² Rotation (240°)", lambda: self.animate_rotation([0, 0, 1], 4 * np.pi / 3))
        btn("σv1 Mirror", self.animate_reflection_v1)
        btn("σv2 Mirror", self.animate_reflection_v2)
        btn("σv3 Mirror", self.animate_reflection_v3)
        btn("Identity", self.animate_identity)
        btn("Toggle Unit Cell", self.toggle_lattice)  # new label
        btn("Reset", self.reset_positions)

        # speed slider
        speed_fr = ttk.Frame(parent)
        speed_fr.pack(pady=(20, 0))
        ttk.Label(speed_fr, text="Animation Speed:").pack()
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(speed_fr, from_=0.1, to=3, variable=self.speed_var, orient=tk.HORIZONTAL).pack(pady=5)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_scene(self):
        self.ax.clear()

        # Optional lattice (light grey)
        if self.show_lattice:
            lp = self.lattice_positions
            self.ax.scatter(lp[:, 0], lp[:, 1], lp[:, 2], c="lightgray", s=60, alpha=0.3, label="Diamond unit cell")

        # NV atoms
        self.ax.scatter(self.current_carbon[:, 0], self.current_carbon[:, 1], self.current_carbon[:, 2], c="gray", s=200, alpha=0.8, label="Carbon")
        self.ax.scatter(*self.current_nitrogen, c="blue", s=250, alpha=0.8, label="Nitrogen")
        self.ax.scatter(*self.current_vacancy, c="red", s=150, alpha=0.6, marker="o", facecolors="none", edgecolors="red", linewidth=3, label="Vacancy")

        # Bonds
        for c in self.current_carbon:
            self.ax.plot([0, c[0]], [0, c[1]], [0, c[2]], "k-", alpha=0.3, linewidth=1)
        self.ax.plot([0, self.current_nitrogen[0]], [0, self.current_nitrogen[1]], [0, self.current_nitrogen[2]], "b-", alpha=0.5, linewidth=2)

        # Aesthetics
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("NV Center Structure (C3v)")
        lim = 1.5
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-0.5, lim)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Symmetry maths + animations
    # ------------------------------------------------------------------
    @staticmethod
    def rotation_matrix(axis, angle):
        axis = np.array(axis) / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def animate_rotation(self, axis, total_angle):
        if self.animating:
            return
        self.animating = True
        steps = int(self.animation_steps / self.speed_var.get())
        for s in range(steps + 1):
            R = self.rotation_matrix(axis, total_angle * s / steps)
            self.current_carbon = self.original_carbon @ R.T
            self.plot_scene()
            self.root.update()
            time.sleep(0.05 / self.speed_var.get())
        self.animating = False

    def animate_reflection(self, normal):
        if self.animating:
            return
        self.animating = True
        steps = int(self.animation_steps / self.speed_var.get())
        n = normal / np.linalg.norm(normal)
        M = np.eye(3) - 2 * np.outer(n, n)
        target = self.original_carbon @ M.T
        for s in range(steps + 1):
            t = s / steps
            self.current_carbon = (1 - t) * self.original_carbon + t * target
            self.plot_scene()
            self.root.update()
            time.sleep(0.05 / self.speed_var.get())
        self.animating = False

    def animate_reflection_v1(self):
        self.animate_reflection(np.array([0, 1, 0]))

    def animate_reflection_v2(self):
        self.animate_reflection(np.array([np.sqrt(3) / 2, 0.5, 0]))

    def animate_reflection_v3(self):
        self.animate_reflection(np.array([np.sqrt(3) / 2, -0.5, 0]))

    def animate_identity(self):
        if self.animating:
            return
        self.animating = True
        steps = int(self.animation_steps / self.speed_var.get())
        for _ in range(steps + 1):
            self.current_carbon = self.original_carbon.copy()
            self.plot_scene()
            self.root.update()
            time.sleep(0.05 / self.speed_var.get())
        self.animating = False

    # ------------------------------------------------------------------
    # Misc controls
    # ------------------------------------------------------------------
    def toggle_lattice(self):
        if self.animating:
            return
        self.show_lattice = not self.show_lattice
        self.plot_scene()

    def reset_positions(self):
        if self.animating:
            return
        self.current_carbon = self.original_carbon.copy()
        self.plot_scene()

    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    NVCenter().run()
