"""
NV Center Symmetry Explorer for Fundamentals

This module provides interactive visualization tools for exploring the symmetry
operations of the NV center. It includes the geometry setup, symmetry operations,
and plotting functionality used in the fundamentals notebook.

Author: NV Center Notebook Project
"""

import numpy as np
import plotly.graph_objects as go
from ipywidgets import Dropdown, interact


def setup_nv_positions():
    """
    Setup NV center atomic positions in the standard coordinate system.
    
    Returns:
        tuple: (carbon_positions, nitrogen_position, vacancy_position)
            - carbon_positions: np.array of shape (3, 3) with C1, C2, C3 coordinates
            - nitrogen_position: np.array of shape (3,) with N coordinates
            - vacancy_position: np.array of shape (3,) with V coordinates
    """
    carbon_positions = np.array([
        [1.0, 0.0, -0.5],      # C1
        [-0.5, np.sqrt(3)/2, -0.5],  # C2
        [-0.5, -np.sqrt(3)/2, -0.5]  # C3
    ])
    nitrogen_position = np.array([0.0, 0.0, 0.0])
    vacancy_position = np.array([0.0, 0.0, 1.0])
    
    return carbon_positions, nitrogen_position, vacancy_position


def create_nv_plot(carbon_pos, nitrogen_pos, vacancy_pos, title):
    """
    Create a 3D Plotly visualization of the NV center structure.
    
    Args:
        carbon_pos (np.array): Carbon atom positions (3x3 array)
        nitrogen_pos (np.array): Nitrogen atom position (3-element array)
        vacancy_pos (np.array): Vacancy position (3-element array)
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D plot
    """
    fig = go.Figure()
    
    # Plot carbon atoms with labels
    fig.add_trace(go.Scatter3d(
        x=carbon_pos[:, 0], y=carbon_pos[:, 1], z=carbon_pos[:, 2],
        mode='markers+text',
        marker=dict(size=10, color='#303030', opacity=0.9),
        name='Carbon',
        text=['C1', 'C2', 'C3'],
        textposition="top center",
        textfont=dict(size=12, color='black')
    ))
    
    # Plot nitrogen atom with label
    fig.add_trace(go.Scatter3d(
        x=[nitrogen_pos[0]], y=[nitrogen_pos[1]], z=[nitrogen_pos[2]],
        mode='markers+text',
        marker=dict(size=12, color='#E63946', opacity=1.0),
        name='Nitrogen',
        text=['N'],
        textposition="top center",
        textfont=dict(size=12, color='black')
    ))
    
    # Plot vacancy with label
    fig.add_trace(go.Scatter3d(
        x=[vacancy_pos[0]], y=[vacancy_pos[1]], z=[vacancy_pos[2]],
        mode='markers+text',
        marker=dict(size=12, color='#4361EE', opacity=0.7),
        name='Vacancy',
        text=['V'],
        textposition="top center",
        textfont=dict(size=12, color='black')
    ))
    
    # Add bonds between nitrogen and carbons
    for carbon in carbon_pos:
        fig.add_trace(go.Scatter3d(
            x=[nitrogen_pos[0], carbon[0]], 
            y=[nitrogen_pos[1], carbon[1]], 
            z=[nitrogen_pos[2], carbon[2]],
            mode='lines',
            line=dict(color='#457B9D', width=5),
            showlegend=False
        ))
    
    # Add N-V bond
    fig.add_trace(go.Scatter3d(
        x=[nitrogen_pos[0], vacancy_pos[0]], 
        y=[nitrogen_pos[1], vacancy_pos[1]], 
        z=[nitrogen_pos[2], vacancy_pos[2]],
        mode='lines',
        line=dict(color='#FF9E00', width=5),
        name='N-V Bond'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], title='X'),
            yaxis=dict(range=[-1.5, 1.5], title='Y'),
            zaxis=dict(range=[-1.5, 1.5], title='Z'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        template='plotly_white'
    )
    
    return fig


class NVSymmetryExplorer:
    """
    Interactive explorer for NV center symmetry operations.
    
    This class provides methods to visualize how the C3v symmetry operations
    (rotations and reflections) transform the carbon atom positions around
    the NV center.
    """
    
    def __init__(self):
        """Initialize the symmetry explorer with the standard NV geometry."""
        self.original_carbon, self.nitrogen_pos, self.vacancy_pos = setup_nv_positions()
    
    def rotation_matrix(self, axis, angle):
        """
        Create rotation matrix using Rodrigues' formula.
        
        Args:
            axis (array-like): Rotation axis (will be normalized)
            angle (float): Rotation angle in radians
            
        Returns:
            np.array: 3x3 rotation matrix
        """
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Skew-symmetric matrix for cross product
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        
        # Rodrigues' formula
        return np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

    def apply_rotation(self, carbon_pos, axis, angle):
        """
        Apply rotation transformation to carbon positions.
        
        Args:
            carbon_pos (np.array): Current carbon positions
            axis (array-like): Rotation axis
            angle (float): Rotation angle in radians
            
        Returns:
            np.array: Transformed carbon positions
        """
        rot_matrix = self.rotation_matrix(axis, angle)
        return np.dot(carbon_pos, rot_matrix.T)

    def apply_reflection(self, carbon_pos, reflection_type):
        """
        Apply reflection transformation to carbon positions.
        
        Args:
            carbon_pos (np.array): Current carbon positions
            reflection_type (str): Type of reflection ('v1', 'v2', or 'v3')
            
        Returns:
            np.array: Transformed carbon positions
        """
        if reflection_type == 'v1':
            # Mirror plane: xz plane (y = 0)
            reflection_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif reflection_type == 'v2':
            # Mirror plane containing C2 atom
            normal = np.array([np.sqrt(3)/2, 0.5, 0])
            reflection_matrix = np.eye(3) - 2 * np.outer(normal, normal)
        elif reflection_type == 'v3':
            # Mirror plane containing C3 atom
            normal = np.array([np.sqrt(3)/2, -0.5, 0])
            reflection_matrix = np.eye(3) - 2 * np.outer(normal, normal)
        else:
            raise ValueError(f"Unknown reflection type: {reflection_type}")
        
        return np.dot(carbon_pos, reflection_matrix.T)
    
    def update_plot(self, operation):
        """
        Update the plot based on the selected symmetry operation.
        
        Args:
            operation (str): Name of the symmetry operation to apply
        """
        carbon_pos = self.original_carbon.copy()
        
        if operation == 'Identity':
            title = "Identity Operation (E)"
        elif operation == 'C3 (120°)':
            carbon_pos = self.apply_rotation(carbon_pos, [0, 0, 1], 2*np.pi/3)
            title = "C3 Rotation (120°)"
        elif operation == 'C3² (240°)':
            carbon_pos = self.apply_rotation(carbon_pos, [0, 0, 1], 4*np.pi/3)
            title = "C3² Rotation (240°)"
        elif operation == 'σv1 Mirror':
            carbon_pos = self.apply_reflection(carbon_pos, 'v1')
            title = "σv1 Mirror Plane (xz plane)"
        elif operation == 'σv2 Mirror':
            carbon_pos = self.apply_reflection(carbon_pos, 'v2')
            title = "σv2 Mirror Plane (containing C2)"
        elif operation == 'σv3 Mirror':
            carbon_pos = self.apply_reflection(carbon_pos, 'v3')
            title = "σv3 Mirror Plane (containing C3)"
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        fig = create_nv_plot(carbon_pos, self.nitrogen_pos, self.vacancy_pos, title)
        fig.show()

    def create_interactive_widget(self):
        """
        Create and return an interactive widget for exploring symmetry operations.
        
        Returns:
            ipywidgets.interact: Interactive widget for symmetry exploration
        """
        return interact(
            self.update_plot, 
            operation=Dropdown(
                options=['Identity', 'C3 (120°)', 'C3² (240°)', 'σv1 Mirror', 'σv2 Mirror', 'σv3 Mirror'],
                value='Identity',
                description='Symmetry Op:'
            )
        )


def create_symmetry_explorer():
    """
    Convenience function to create and display the symmetry explorer.
    
    Returns:
        NVSymmetryExplorer: Configured explorer instance
    """
    explorer = NVSymmetryExplorer()
    explorer.create_interactive_widget()
    
    print("✅ Interactive NV center symmetry explorer is ready!")
    print("\nUse the dropdown menu above to explore different symmetry operations!")
    print("\nThe interactive Plotly visualization allows you to:")
    print("- Rotate by clicking and dragging")
    print("- Zoom with the mouse wheel") 
    print("- Pan by holding Shift while dragging")
    print("- Reset the view by double-clicking")
    
    return explorer
