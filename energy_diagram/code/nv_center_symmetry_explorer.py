import numpy as np
import plotly.graph_objects as go
from ipywidgets import interactive, Dropdown, VBox, Layout
from IPython.display import display

class NVCenterNotebook:
    def __init__(self):
        # Initialize NV center coordinates
        self.setup_nv_center()
        
        # Create figure and axis
        self.fig = None
        
    def setup_nv_center(self):
        """Initialize the NV center atomic positions"""
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
        
    def create_plot(self):
        """Create a fresh plotly figure widget"""
        self.fig = go.FigureWidget()
        return self.fig
        
    def plot_nv_center(self, title="NV Center Structure (C3v Symmetry)"):
        """Plot the NV center structure using Plotly"""
        if self.fig is None:
            self.create_plot()
            
        # Clear existing traces
        self.fig.data = []
        
        # Plot carbon atoms
        self.fig.add_trace(go.Scatter3d(
            x=self.current_carbon[:, 0], 
            y=self.current_carbon[:, 1], 
            z=self.current_carbon[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color='#303030',  # Dark gray
                opacity=0.9,
                line=dict(color='#606060', width=1)
            ),
            text=['C1', 'C2', 'C3'],
            textposition="top center",
            textfont=dict(size=12, color='black'),
            name='Carbon'
        ))
        
        # Plot nitrogen atom
        self.fig.add_trace(go.Scatter3d(
            x=[self.current_nitrogen[0]],
            y=[self.current_nitrogen[1]],
            z=[self.current_nitrogen[2]],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#E63946',  # Crimson red
                opacity=1.0,
                line=dict(color='#C1121F', width=1)
            ),
            text=['N'],
            textposition="top center",
            textfont=dict(size=12, color='black'),
            name='Nitrogen'
        ))
        
        # Plot vacancy
        self.fig.add_trace(go.Scatter3d(
            x=[self.current_vacancy[0]],
            y=[self.current_vacancy[1]],
            z=[self.current_vacancy[2]],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#4361EE',  # Royal blue
                opacity=0.7,
                line=dict(color='#3A0CA3', width=1)
            ),
            text=['V'],
            textposition="top center",
            textfont=dict(size=12, color='black'),
            name='Vacancy'
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
        """Draw bonds between atoms using Plotly"""
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
        
    def rotation_matrix(self, axis, angle):
        """Create rotation matrix for given axis and angle using Rodrigues' formula"""
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        return np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    
    def apply_rotation(self, axis, angle):
        """Apply rotation to the structure"""
        rot_matrix = self.rotation_matrix(axis, angle)
        
        # Apply rotation (N and vacancy stay fixed on the rotation axis)
        self.current_carbon = np.dot(self.original_carbon, rot_matrix.T)
        # Nitrogen and vacancy don't move for C3 rotation about z-axis
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()
        
    def apply_reflection(self, reflection_type):
        """Apply reflection to the structure"""
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
        
        # Apply reflection
        self.current_carbon = np.dot(self.original_carbon, reflection_matrix.T)
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()
        
    def reset_positions(self):
        """Reset to original positions"""
        self.current_carbon = self.original_carbon.copy()
        self.current_nitrogen = self.original_nitrogen.copy()
        self.current_vacancy = self.original_vacancy.copy()


def update_plot(operation):
    """Update the plot based on selected symmetry operation"""
    nv_center.reset_positions()
    
    if operation == 'Identity':
        title = "Identity Operation (E)"
    elif operation == 'C3 (120°)':
        nv_center.apply_rotation([0, 0, 1], 2*np.pi/3)
        title = "C3 Rotation (120°)"
    elif operation == 'C3² (240°)':
        nv_center.apply_rotation([0, 0, 1], 4*np.pi/3)
        title = "C3² Rotation (240°)"
    elif operation == 'σv1 Mirror':
        nv_center.apply_reflection('v1')
        title = "σv1 Mirror Plane (xz plane)"
    elif operation == 'σv2 Mirror':
        nv_center.apply_reflection('v2')
        title = "σv2 Mirror Plane (containing C2)"
    elif operation == 'σv3 Mirror':
        nv_center.apply_reflection('v3')
        title = "σv3 Mirror Plane (containing C3)"
    
    # Update the plot in place by directly modifying the figure widget
    with nv_center.fig.batch_update():
        # Clear existing traces
        nv_center.fig.data = []
        
        # Plot carbon atoms
        nv_center.fig.add_trace(go.Scatter3d(
            x=nv_center.current_carbon[:, 0], 
            y=nv_center.current_carbon[:, 1], 
            z=nv_center.current_carbon[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color='#303030',
                opacity=0.9,
                line=dict(color='#606060', width=1)
            ),
            name='Carbon',
            text=['C1', 'C2', 'C3'],
            textposition="top center",
            textfont=dict(size=12, color='black')
        ))
        
        # Plot nitrogen atom
        nv_center.fig.add_trace(go.Scatter3d(
            x=[nv_center.current_nitrogen[0]],
            y=[nv_center.current_nitrogen[1]],
            z=[nv_center.current_nitrogen[2]],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#E63946',
                opacity=1.0,
                line=dict(color='#C1121F', width=1)
            ),
            name='Nitrogen',
            text=['N'],
            textposition="top center",
            textfont=dict(size=12, color='black')
        ))
        
        # Plot vacancy
        nv_center.fig.add_trace(go.Scatter3d(
            x=[nv_center.current_vacancy[0]],
            y=[nv_center.current_vacancy[1]],
            z=[nv_center.current_vacancy[2]],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#4361EE',
                opacity=0.7,
                line=dict(color='#3A0CA3', width=1)
            ),
            name='Vacancy',
            text=['V'],
            textposition="top center",
            textfont=dict(size=12, color='black')
        ))
        
        # Draw bonds from nitrogen to carbon atoms
        for carbon in nv_center.current_carbon:
            nv_center.fig.add_trace(go.Scatter3d(
                x=[nv_center.current_nitrogen[0], carbon[0]],
                y=[nv_center.current_nitrogen[1], carbon[1]],
                z=[nv_center.current_nitrogen[2], carbon[2]],
                mode='lines',
                line=dict(color='#457B9D', width=5),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Bond from nitrogen to vacancy
        nv_center.fig.add_trace(go.Scatter3d(
            x=[nv_center.current_nitrogen[0], nv_center.current_vacancy[0]],
            y=[nv_center.current_nitrogen[1], nv_center.current_vacancy[1]],
            z=[nv_center.current_nitrogen[2], nv_center.current_vacancy[2]],
            mode='lines',
            line=dict(color='#FF9E00', width=5),
            hoverinfo='none',
            name='N-V Bond'
        ))
        
        # Update title
        nv_center.fig.update_layout(title=title)

# Create interactive widget

# Create instance
nv_center = NVCenterNotebook()
print("NV Center class initialized successfully!")

# Create the figure widget - this should be the same as nv_center.fig
nv_center.fig = go.FigureWidget()

symmetry_operations = ['Identity', 'C3 (120°)', 'C3² (240°)', 'σv1 Mirror', 'σv2 Mirror', 'σv3 Mirror']
operation_widget = Dropdown(
    options=symmetry_operations,
    value='Identity',
    description='Symmetry Op:',
    style={'description_width': 'initial'}
)

# Link the dropdown to the update function
interactive_plot = interactive(update_plot, operation=operation_widget)

# Initial plot rendering
nv_center.plot_nv_center("Identity Operation (E)")

# Display the dropdown and the figure widget together
display(VBox([operation_widget, nv_center.fig], layout=Layout(align_items='center')))

print("Use the dropdown menu above to explore different symmetry operations!")
print("\nThe interactive Plotly visualization allows you to:")
print("- Rotate by clicking and dragging")
print("- Zoom with the mouse wheel")
print("- Pan by holding Shift while dragging")
print("- Reset the view by double-clicking")