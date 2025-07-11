"""
NV Center Symmetry Explorer

This module provides classes and functions for exploring the NV (nitrogen-vacancy) center 
symmetries in diamond. The NV center has C3v point group symmetry, consisting of:

- C3 rotations: 120° and 240° rotations about the NV axis
- σv mirror planes: Three vertical mirror planes containing the NV axis
- Identity operation: No change

The NV center consists of:
- A nitrogen atom substituting for a carbon atom
- An adjacent vacancy (missing carbon atom)
- Three carbon atoms in tetrahedral arrangement around the vacancy

Author: Generated from nv_center_symmetry_explorer.ipynb
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt


class NVCenterSymmetryExplorer:
    """
    A class for exploring NV center symmetries through 3D visualization.
    
    This class handles the atomic structure of the NV center and provides
    methods to apply various symmetry operations and visualize the results.
    """
    
    def __init__(self):
        """Initialize the NV center with default atomic positions."""
        self.setup_nv_center()
        self.fig = None
        
    def setup_nv_center(self):
        """Initialize the NV center atomic positions."""
        # Carbon atoms in tetrahedral arrangement around nitrogen
        # Three carbons forming equilateral triangle with nitrogen at center
        self.carbon_positions = np.array([
            [1.0, 0.0, -0.5],      # C1
            [-0.5, np.sqrt(3)/2, -0.5],  # C2
            [-0.5, -np.sqrt(3)/2, -0.5]  # C3
        ])
        
        # Nitrogen atom position at the center of the three carbons
        self.nitrogen_position = np.array([0.0, 0.0, 0.0])
        
        # Vacancy position along z-axis above the nitrogen
        self.vacancy_position = np.array([0.0, 0.0, 1.0])
        
        # Current positions for animation
        self.current_carbon = self.carbon_positions.copy()
        self.current_nitrogen = self.nitrogen_position.copy()
        self.current_vacancy = self.vacancy_position.copy()
        
        # Original positions for reset
        self.original_carbon = self.carbon_positions.copy()
        self.original_nitrogen = self.nitrogen_position.copy()
        self.original_vacancy = self.vacancy_position.copy()
        
    def create_plot(self) -> go.FigureWidget:
        """Create a fresh plotly figure widget."""
        self.fig = go.FigureWidget()
        return self.fig
        
    def plot_nv_center(self, title: str = "NV Center Structure (C3v Symmetry)") -> go.Figure:
        """
        Plot the NV center structure using Plotly.
        
        Args:
            title: Title for the plot
            
        Returns:
            Plotly Figure object
        """
        if self.fig is None:
            self.create_plot()
            
        # Clear existing traces
        self.fig.data = []
        
        # Plot carbon atoms
        self.fig.add_trace(go.Scatter3d(
            x=self.current_carbon[:, 0], 
            y=self.current_carbon[:, 1], 
            z=self.current_carbon[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='#303030',  # Dark gray
                opacity=0.9,
                line=dict(color='#606060', width=1)
            ),
            name='Carbon'
        ))
        
        # Plot nitrogen atom
        self.fig.add_trace(go.Scatter3d(
            x=[self.current_nitrogen[0]],
            y=[self.current_nitrogen[1]],
            z=[self.current_nitrogen[2]],
            mode='markers',
            marker=dict(
                size=12,
                color='#E63946',  # Crimson red
                opacity=1.0,
                line=dict(color='#C1121F', width=1)
            ),
            name='Nitrogen'
        ))
        
        # Plot vacancy
        self.fig.add_trace(go.Scatter3d(
            x=[self.current_vacancy[0]],
            y=[self.current_vacancy[1]],
            z=[self.current_vacancy[2]],
            mode='markers',
            marker=dict(
                size=12,
                color='#4361EE',  # Royal blue
                opacity=0.7,
                line=dict(color='#3A0CA3', width=1)
            ),
            name='Vacancy'
        ))
        
        # Add text labels for atoms
        self.fig.add_trace(go.Scatter3d(
            x=list(self.current_carbon[:, 0]) + [self.current_nitrogen[0], self.current_vacancy[0]],
            y=list(self.current_carbon[:, 1]) + [self.current_nitrogen[1], self.current_vacancy[1]],
            z=list(self.current_carbon[:, 2]) + [self.current_nitrogen[2], self.current_vacancy[2]],
            mode='text',
            text=['C1', 'C2', 'C3', 'N', 'V'],
            textposition="top center",
            textfont=dict(size=12, color='black'),
            name='Labels'
        ))
        
        # Draw bonds
        self.draw_bonds()
        
        # Set layout for a clean, professional look
        self.fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(showticklabels=True, title='X', range=[-1.5, 1.5]),
                yaxis=dict(showticklabels=True, title='Y', range=[-1.5, 1.5]),
                zaxis=dict(showticklabels=True, title='Z', range=[-1.5, 1.5]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0.01, y=0.99),
            template='plotly_white'
        )
        
        # Set camera position for initial view
        self.fig.update_layout(
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=-1.5, z=1.2)
            )
        )
        
        return self.fig
        
    def draw_bonds(self):
        """Draw bonds between atoms using Plotly."""
        # Create lines for bonds from nitrogen to carbon atoms
        for carbon in self.current_carbon:
            x_line = [self.current_nitrogen[0], carbon[0], None]
            y_line = [self.current_nitrogen[1], carbon[1], None]
            z_line = [self.current_nitrogen[2], carbon[2], None]
            
            self.fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='#457B9D', width=5, dash='solid'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Bond from nitrogen to vacancy
        x_line = [self.current_nitrogen[0], self.current_vacancy[0], None]
        y_line = [self.current_nitrogen[1], self.current_vacancy[1], None]
        z_line = [self.current_nitrogen[2], self.current_vacancy[2], None]
        
        self.fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines',
            line=dict(color='#FF9E00', width=5, dash='solid'),
            hoverinfo='none',
            name='N-V Bond'
        ))
        
    def rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Create rotation matrix for given axis and angle using Rodrigues' formula.
        
        Args:
            axis: Rotation axis as numpy array
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        return np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    
    def apply_rotation(self, axis: np.ndarray, angle: float):
        """
        Apply rotation to the structure.
        
        Args:
            axis: Rotation axis as numpy array
            angle: Rotation angle in radians
        """
        rot_matrix = self.rotation_matrix(axis, angle)
        
        # Apply rotation (N and vacancy stay fixed on the rotation axis)
        self.current_carbon = np.dot(self.original_carbon, rot_matrix.T)
        # Nitrogen and vacancy don't move for C3 rotation about z-axis
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()
        
    def apply_reflection(self, reflection_type: str):
        """
        Apply reflection to the structure.
        
        Args:
            reflection_type: Type of reflection ('v1', 'v2', or 'v3')
        """
        if reflection_type == 'v1':
            # Reflection matrix for xz plane (y=0)
            reflection_matrix = np.array([[1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, 1]])
        elif reflection_type == 'v2':
            # Reflection matrix for plane containing C2 and z-axis
            normal = np.array([np.sqrt(3)/2, 0.5, 0])
            reflection_matrix = np.eye(3) - 2 * np.outer(normal, normal)
        elif reflection_type == 'v3':
            # Reflection matrix for plane containing C3 and z-axis
            normal = np.array([np.sqrt(3)/2, -0.5, 0])
            reflection_matrix = np.eye(3) - 2 * np.outer(normal, normal)
        else:
            raise ValueError(f"Unknown reflection type: {reflection_type}")
        
        # Apply reflection
        self.current_carbon = np.dot(self.original_carbon, reflection_matrix.T)
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()
        
    def reset_positions(self):
        """Reset to original positions."""
        self.current_carbon = self.original_carbon.copy()
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()
        
    def apply_symmetry_operation(self, operation: str):
        """
        Apply a symmetry operation by name.
        
        Args:
            operation: Name of the symmetry operation
        """
        self.reset_positions()
        
        if operation == 'Identity':
            pass  # No change
        elif operation == 'C3 (120°)':
            self.apply_rotation([0, 0, 1], 2*np.pi/3)
        elif operation == 'C3² (240°)':
            self.apply_rotation([0, 0, 1], 4*np.pi/3)
        elif operation == 'σv1 Mirror':
            self.apply_reflection('v1')
        elif operation == 'σv2 Mirror':
            self.apply_reflection('v2')
        elif operation == 'σv3 Mirror':
            self.apply_reflection('v3')
        else:
            raise ValueError(f"Unknown symmetry operation: {operation}")
            
    def get_operation_title(self, operation: str) -> str:
        """
        Get the appropriate title for a given symmetry operation.
        
        Args:
            operation: Name of the symmetry operation
            
        Returns:
            Title string for the operation
        """
        titles = {
            'Identity': "Identity Operation (E)",
            'C3 (120°)': "C3 Rotation (120°)",
            'C3² (240°)': "C3² Rotation (240°)",
            'σv1 Mirror': "σv1 Mirror Plane (xz plane)",
            'σv2 Mirror': "σv2 Mirror Plane (containing C2)",
            'σv3 Mirror': "σv3 Mirror Plane (containing C3)"
        }
        return titles.get(operation, operation)
        
    def analyze_symmetry_operations(self) -> Dict[str, Any]:
        """
        Analyze the effect of each symmetry operation on atomic positions.
        
        Returns:
            Dictionary containing analysis results
        """
        operations = {
            'Identity': lambda: None,
            'C3 (120°)': lambda: self.apply_rotation([0, 0, 1], 2*np.pi/3),
            'C3² (240°)': lambda: self.apply_rotation([0, 0, 1], 4*np.pi/3),
            'σv1 Mirror': lambda: self.apply_reflection('v1'),
            'σv2 Mirror': lambda: self.apply_reflection('v2'),
            'σv3 Mirror': lambda: self.apply_reflection('v3')
        }
        
        results = {}
        original_positions = self.original_carbon.copy()
        
        for op_name, operation in operations.items():
            self.reset_positions()
            if operation:
                operation()
            
            # Check which atoms are equivalent after operation
            tolerance = 1e-10
            equivalences = []
            for i in range(3):
                for j in range(3):
                    if np.allclose(original_positions[i], self.current_carbon[j], atol=tolerance):
                        equivalences.append(f"C{i+1} → C{j+1}")
            
            results[op_name] = {
                'original_positions': original_positions.copy(),
                'new_positions': self.current_carbon.copy(),
                'atom_mapping': equivalences
            }
        
        return results
        
    def print_symmetry_analysis(self):
        """Print a detailed analysis of symmetry operations."""
        print("Symmetry Operation Analysis:")
        print("=" * 50)
        
        results = self.analyze_symmetry_operations()
        
        for op_name, data in results.items():
            print(f"\n{op_name}:")
            original = data['original_positions']
            new = data['new_positions']
            
            for i in range(3):
                print(f"  Original C{i+1}: [{original[i][0]:.3f}, {original[i][1]:.3f}, {original[i][2]:.3f}]")
                print(f"  New C{i+1}:      [{new[i][0]:.3f}, {new[i][1]:.3f}, {new[i][2]:.3f}]")
            
            print(f"  Atom mapping: {', '.join(data['atom_mapping'])}")


class CharacterTable:
    """Class for handling the C3v character table."""
    
    @staticmethod
    def get_c3v_character_table() -> pd.DataFrame:
        """
        Get the character table for C3v point group.
        
        Returns:
            Pandas DataFrame containing the character table
        """
        return pd.DataFrame({
            'Irrep': ['A1', 'A2', 'E'],
            'E': [1, 1, 2],
            '2C3': [1, 1, -1],
            '3σv': [1, -1, 0],
            'Linear/Rotational': ['z', 'Rz', '(x,y), (Rx,Ry)'],
            'Quadratic': ['z²', '', '(x²-y², xy), (xz, yz)']
        })
    
    @staticmethod
    def print_character_table():
        """Print the character table for C3v point group."""
        print("Character Table for C3v Point Group:")
        print("=" * 60)
        print(CharacterTable.get_c3v_character_table())


def demo_symmetry_operations():
    """
    Demonstrate the NV center symmetry operations.
    
    This function creates an NV center, applies various symmetry operations,
    and displays the results.
    """
    print("NV Center Symmetry Explorer Demo")
    print("=" * 40)
    
    # Create NV center instance
    nv_center = NVCenterSymmetryExplorer()
    
    # Available symmetry operations
    operations = ['Identity', 'C3 (120°)', 'C3² (240°)', 'σv1 Mirror', 'σv2 Mirror', 'σv3 Mirror']
    
    # Demonstrate each operation
    for operation in operations:
        print(f"\nApplying {operation}...")
        nv_center.apply_symmetry_operation(operation)
        title = nv_center.get_operation_title(operation)
        
        # You can uncomment the following line to show plots
        # fig = nv_center.plot_nv_center(title)
        # fig.show()
    
    # Print symmetry analysis
    print("\n" + "=" * 40)
    nv_center.print_symmetry_analysis()
    
    # Show character table
    print("\n" + "=" * 40)
    CharacterTable.print_character_table()


if __name__ == "__main__":
    """Main execution block for demonstration."""
    demo_symmetry_operations()
