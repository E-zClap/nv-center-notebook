import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib.animation import FuncAnimation


class NVCenter:
    """NV‑center symmetry explorer with optional diamond‑lattice view"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NV Center Symmetry Explorer")
        self.root.geometry("1200x800")

        # Flags / state
        self.show_lattice = False  # Toggle for lattice visibility

        # Initialize atomic coordinates
        self.setup_nv_center()
        self.generate_diamond_lattice(n_cells=1, a=1.2)  # build once; just plotted when toggled

        # Animation parameters
        self.animation_steps = 50
        self.animating = False

        # Build GUI
        self.create_gui()

    # ---------------------------------------------------------------------
    # Geometry setup
    # ---------------------------------------------------------------------
    def setup_nv_center(self):
        """Define NV‑center atomic positions (cartesian, Å arbitrary units)"""
        # Carbon atoms forming equilateral triangle around vacancy
        self.carbon_positions = np.array([
            [1.0, 0.0, 0.0],  # C1
            [-0.5, np.sqrt(3) / 2, 0.0],  # C2
            [-0.5, -np.sqrt(3) / 2, 0.0],  # C3
        ])

        # Nitrogen atom along [111] axis
        self.nitrogen_position = np.array([0.0, 0.0, 1.0])
        # Vacancy at origin
        self.vacancy_position = np.array([0.0, 0.0, 0.0])

        # Working copies (updated during animation)
        self.current_carbon = self.carbon_positions.copy()
        self.current_nitrogen = self.nitrogen_position.copy()
        self.current_vacancy = self.vacancy_position.copy()

        # Originals (for reset)
        self.original_carbon = self.carbon_positions.copy()
        self.original_nitrogen = self.nitrogen_position.copy()
        self.original_vacancy = self.vacancy_position.copy()

    def generate_diamond_lattice(self, n_cells: int = 1, a: float = 1.2):
        """Generate a cubic diamond lattice around the origin.

        Parameters
        ----------
        n_cells : int
            Number of conventional cubic cells in ±x, ±y, ±z (total span = 2*n_cells+1)
        a : float
            Lattice constant used for spacing (same units as NV coordinates)
        """
        # Basis of diamond cubic (fractional coords)
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
        points = []
        rng = range(-n_cells, n_cells + 1)
        for i in rng:
            for j in rng:
                for k in rng:
                    cell_offset = np.array([i, j, k])
                    for b in basis:
                        points.append(a * (cell_offset + b))
        self.lattice_positions = np.array(points)

    # ---------------------------------------------------------------------
    # GUI construction
    # ---------------------------------------------------------------------
    def create_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Matplotlib figure / 3‑D axis
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        title_label = ttk.Label(
            control_frame, text="NV Center Symmetries", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        self.create_symmetry_buttons(control_frame)
        self.plot_nv_center()

    def create_symmetry_buttons(self, parent):
        buttons_info = [
            ("C3 Rotation (120°)", lambda: self.animate_rotation([0, 0, 1], 2 * np.pi / 3)),
            ("C3² Rotation (240°)", lambda: self.animate_rotation([0, 0, 1], 4 * np.pi / 3)),
            ("σv1 Mirror", self.animate_reflection_v1),
            ("σv2 Mirror", self.animate_reflection_v2),
            ("σv3 Mirror", self.animate_reflection_v3),
            ("Identity", self.animate_identity),
            ("Toggle Lattice", self.toggle_lattice),  # NEW BUTTON
            ("Reset", self.reset_positions),
        ]

        for text, command in buttons_info:
            ttk.Button(parent, text=text, command=command, width=20).pack(pady=5)

        # Animation‑speed slider
        speed_frame = ttk.Frame(parent)
        speed_frame.pack(pady=(20, 0))
        ttk.Label(speed_frame, text="Animation Speed:").pack()
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(speed_frame, from_=0.1, to=3.0, variable=self.speed_var, orient=tk.HORIZONTAL).pack(pady=5)

    # ---------------------------------------------------------------------
    # Plotting helpers
    # ---------------------------------------------------------------------
    def plot_nv_center(self):
        self.ax.clear()

        # Optional diamond lattice (light grey)
        if self.show_lattice and len(self.lattice_positions):
            self.ax.scatter(
                self.lattice_positions[:, 0],
                self.lattice_positions[:, 1],
                self.lattice_positions[:, 2],
                c="lightgray",
                s=30,
                alpha=0.25,
                label="Diamond lattice",
            )

        # Carbon atoms (NV)
        self.ax.scatter(
            self.current_carbon[:, 0],
            self.current_carbon[:, 1],
            self.current_carbon[:, 2],
            c="gray",
            s=200,
            alpha=0.8,
            label="Carbon",
        )
        # Nitrogen
        self.ax.scatter(
            *self.current_nitrogen,
            c="blue",
            s=250,
            alpha=0.8,
            label="Nitrogen",
        )
        # Vacancy (hollow red circle)
        self.ax.scatter(
            *self.current_vacancy,
            c="red",
            s=150,
            alpha=0.6,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidth=3,
            label="Vacancy",
        )

        self.draw_bonds()

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("NV Center Structure (C3v Symmetry)")

        # Set equal aspect safely
        max_range = 2.0 if self.show_lattice else 1.5
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-0.5, max_range])

        self.ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        self.canvas.draw()

    def draw_bonds(self):
        # Vacancy‑to‑carbon bonds
        for c_pos in self.current_carbon:
            self.ax.plot(
                [self.current_vacancy[0], c_pos[0]],
                [self.current_vacancy[1], c_pos[1]],
                [self.current_vacancy[2], c_pos[2]],
                "k-",
                alpha=0.3,
                linewidth=1,
            )
        # Vacancy‑to‑nitrogen bond
        self.ax.plot(
            [self.current_vacancy[0], self.current_nitrogen[0]],
            [self.current_vacancy[1], self.current_nitrogen[1]],
            [self.current_vacancy[2], self.current_nitrogen[2]],
            "b-",
            alpha=0.5,
            linewidth=2,
        )

    # ---------------------------------------------------------------------
    # Symmetry operations + animations
    # ---------------------------------------------------------------------
    def rotation_matrix(self, axis, angle):
        axis = np.array(axis) / np.linalg.norm(axis)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    def animate_rotation(self, axis, total_angle):
        if self.animating:
            return
        self.animating = True
        steps = int(self.animation_steps / self.speed_var.get())
        for step in range(steps + 1):
            angle = total_angle * step / steps
            R = self.rotation_matrix(axis, angle)
            self.current_carbon = np.dot(self.original_carbon, R.T)
            self.current_nitrogen = self.original_nitrogen.copy()
            self.current_vacancy = self.original_vacancy.copy()
            self.plot_nv_center()
            self.root.update()
            time.sleep(0.05 / self.speed_var.get())
        self.animating = False

    def animate_reflection(self, normal):
        if self.animating:
            return
        self.animating = True
        steps = int(self.animation_steps / self.speed_var.get())
        normal = normal / np.linalg.norm(normal)
        R = np.eye(3) - 2 * np.outer(normal, normal)
        target_c = np.dot(self.original_carbon, R.T)
        for step in range(steps + 1):
            t = step / steps
            self.current_carbon = self.original_carbon * (1 - t) + target_c * t
            self.current_nitrogen = self.original_nitrogen.copy()
            self.current_vacancy = self.original_vacancy.copy()
            self.plot_nv_center()
            self.root.update()
            time.sleep(0.05 / self.speed_var.get())
        self.animating = False

    def animate_reflection_v1(self):
        self.animate_reflection(np.array([0, 1, 0]))  # xz plane (y=0)

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
            self.current_nitrogen = self.original_nitrogen.copy()
            self.current_vacancy = self.original_vacancy.copy()
            self.plot_nv_center()
            self.root.update()
            time.sleep(0.05 / self.speed_var.get())
        self.animating = False

    # ---------------------------------------------------------------------
    # Misc controls
    # ---------------------------------------------------------------------
    def toggle_lattice(self):
        """Show/hide the diamond lattice without affecting symmetry ops."""
        if self.animating:
            return
        self.show_lattice = not self.show_lattice
        self.plot_nv_center()

    def reset_positions(self):
        if self.animating:
            return
        self.current_carbon = self.original_carbon.copy()
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()
        self.plot_nv_center()

    # ---------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = NVCenter()
    app.run()
