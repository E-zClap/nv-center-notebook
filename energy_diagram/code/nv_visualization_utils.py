"""
Utility functions for NV center visualization and analysis.
This module contains helper functions for creating diamond lattice structures,
NV centers, and interactive 3D visualizations.
"""

import numpy as np
import plotly.graph_objects as go


def create_diamond_lattice(size=2):
    """Create a diamond lattice of given size."""
    # Diamond lattice constant in Angstroms
    a = 3.57
    
    # Create the FCC lattice points
    positions = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # Add FCC lattice points
                positions.append([i, j, k])
                positions.append([i+0.5, j+0.5, k])
                positions.append([i+0.5, j, k+0.5])
                positions.append([i, j+0.5, k+0.5])
    
    # Add basis atoms
    all_positions = []
    for pos in positions:
        # Original atom
        all_positions.append(np.array(pos) * a)
        # Basis atom - shift by (0.25, 0.25, 0.25)
        all_positions.append((np.array(pos) + np.array([0.25, 0.25, 0.25])) * a)
    
    return np.array(all_positions)


def find_nearest_neighbors(positions, threshold=1.8):
    """Find pairs of atoms that are within threshold distance of each other."""
    bonds = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < threshold:
                bonds.append((i, j))
    return bonds


def create_nv_center(positions):
    """Create an NV center by replacing one atom with nitrogen and removing another."""
    # Choose a carbon atom near the center of the lattice
    center = np.mean(positions, axis=0)
    dists = [np.linalg.norm(pos - center) for pos in positions]
    center_idx = np.argmin(dists)
    
    # Find nearest neighbors to the chosen atom
    dists = [np.linalg.norm(positions[center_idx] - pos) for pos in positions]
    neighbors = np.argsort(dists)[1:5]  # 4 nearest neighbors in diamond
    
    # Choose one neighbor to be the vacancy
    vacancy_idx = neighbors[0]
    
    # Mark the atom types: 0 for carbon, 1 for nitrogen, -1 for vacancy
    atom_types = np.zeros(len(positions), dtype=int)
    atom_types[center_idx] = 1  # Nitrogen
    atom_types[vacancy_idx] = -1  # Vacancy
    
    return atom_types, center_idx, vacancy_idx


def interactive_nv_center(positions, atom_types, bonds):
    """Create an interactive 3D visualization of the NV center using Plotly."""
    
    # Define colors for different atom types
    carbon_color = '#303030'     # Dark gray for carbon
    nitrogen_color = '#E63946'   # Crimson red for nitrogen
    vacancy_color = '#4361EE'    # Royal blue for vacancy
    
    # Separate positions by atom type
    carbon_mask = atom_types == 0
    nitrogen_mask = atom_types == 1
    vacancy_mask = atom_types == -1
    
    carbon_positions = positions[carbon_mask]
    nitrogen_pos = positions[nitrogen_mask][0]
    vacancy_pos = positions[vacancy_mask][0]
    
    # Create figure
    fig = go.Figure()
    
    # Add carbon atoms
    fig.add_trace(go.Scatter3d(
        x=carbon_positions[:, 0],
        y=carbon_positions[:, 1],
        z=carbon_positions[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=carbon_color,
            opacity=0.9,
            line=dict(color='#606060', width=1)
        ),
        name='Carbon'
    ))
    
    # Add nitrogen atom
    fig.add_trace(go.Scatter3d(
        x=[nitrogen_pos[0]],
        y=[nitrogen_pos[1]],
        z=[nitrogen_pos[2]],
        mode='markers',
        marker=dict(
            size=12,
            color=nitrogen_color,
            opacity=1.0,
            line=dict(color='#C1121F', width=1),
            symbol='circle'
        ),
        name='Nitrogen'
    ))
    
    # Add vacancy
    fig.add_trace(go.Scatter3d(
        x=[vacancy_pos[0]],
        y=[vacancy_pos[1]],
        z=[vacancy_pos[2]],
        mode='markers',
        marker=dict(
            size=12,
            color=vacancy_color,
            opacity=0.7,
            line=dict(color='#3A0CA3', width=1),
            symbol='circle'
        ),
        name='Vacancy'
    ))
    
    # Add text labels for N and V
    fig.add_trace(go.Scatter3d(
        x=[nitrogen_pos[0], vacancy_pos[0]],
        y=[nitrogen_pos[1], vacancy_pos[1]],
        z=[nitrogen_pos[2], vacancy_pos[2]],
        mode='text',
        text=['N', 'V'],
        textposition="middle center",
        textfont=dict(size=16, color='black', family='Arial Black'),
        name='Labels'
    ))
    
    # Add bonds
    x_lines, y_lines, z_lines = [], [], []
    colors = []
    
    for i, j in bonds:
        # Skip bonds to/from vacancy
        if atom_types[i] == -1 or atom_types[j] == -1:
            continue
            
        # Add line coordinates with None separators
        x_lines.extend([positions[i, 0], positions[j, 0], None])
        y_lines.extend([positions[i, 1], positions[j, 1], None])
        z_lines.extend([positions[i, 2], positions[j, 2], None])
        
        # Determine color based on atom types
        if atom_types[i] == 1 or atom_types[j] == 1:
            # Bond to nitrogen
            colors.extend(['#FF9E00', '#FF9E00', '#FF9E00'])  # Orange for N-C bonds
        else:
            # Carbon-carbon bond
            colors.extend(['#457B9D', '#457B9D', '#457B9D'])  # Steel blue for C-C bonds
    
    # Add the bonds as lines
    fig.add_trace(go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color=colors, width=5),
        hoverinfo='none',
        name='Bonds'
    ))
    
    # Create unit cell boundary
    min_coords = np.min(positions, axis=0) - 1.0
    max_coords = np.max(positions, axis=0) + 1.0
    
    # Create the edges of the cube
    x_lines, y_lines, z_lines = [], [], []
    
    # Bottom face
    x_lines.extend([min_coords[0], max_coords[0], None, max_coords[0], max_coords[0], None, 
                    max_coords[0], min_coords[0], None, min_coords[0], min_coords[0], None])
    y_lines.extend([min_coords[1], min_coords[1], None, min_coords[1], max_coords[1], None, 
                    max_coords[1], max_coords[1], None, max_coords[1], min_coords[1], None])
    z_lines.extend([min_coords[2], min_coords[2], None, min_coords[2], min_coords[2], None, 
                    min_coords[2], min_coords[2], None, min_coords[2], min_coords[2], None])
    
    # Top face
    x_lines.extend([min_coords[0], max_coords[0], None, max_coords[0], max_coords[0], None, 
                    max_coords[0], min_coords[0], None, min_coords[0], min_coords[0], None])
    y_lines.extend([min_coords[1], min_coords[1], None, min_coords[1], max_coords[1], None, 
                    max_coords[1], max_coords[1], None, max_coords[1], min_coords[1], None])
    z_lines.extend([max_coords[2], max_coords[2], None, max_coords[2], max_coords[2], None, 
                    max_coords[2], max_coords[2], None, max_coords[2], max_coords[2], None])
    
    # Vertical edges
    x_lines.extend([min_coords[0], min_coords[0], None, max_coords[0], max_coords[0], None, 
                    max_coords[0], max_coords[0], None, min_coords[0], min_coords[0], None])
    y_lines.extend([min_coords[1], min_coords[1], None, min_coords[1], min_coords[1], None, 
                    max_coords[1], max_coords[1], None, max_coords[1], max_coords[1], None])
    z_lines.extend([min_coords[2], max_coords[2], None, min_coords[2], max_coords[2], None, 
                    min_coords[2], max_coords[2], None, min_coords[2], max_coords[2], None])
    
    # Add the unit cell as a cube
    fig.add_trace(go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color='#AAAAAA', width=2, dash='dot'),
        hoverinfo='none',
        name='Unit Cell'
    ))
    
    # Set layout for a clean, professional look
    fig.update_layout(
        title='Interactive Nitrogen-Vacancy (NV) Center in Diamond',
        scene=dict(
            xaxis=dict(showticklabels=False, title='', 
                       showgrid=False, zeroline=False, showline=False),
            yaxis=dict(showticklabels=False, title='', 
                       showgrid=False, zeroline=False, showline=False),
            zaxis=dict(showticklabels=False, title='', 
                       showgrid=False, zeroline=False, showline=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99, bordercolor='#AAAAAA', borderwidth=1),
        template='plotly_white'
    )
    
    # Set camera position for initial view - zoomed in closer
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.0, y=-1.0, z=1.0)  # Reduced values for zooming in
        )
    )
    
    return fig


def create_nv_center_demo():
    """Create a complete NV center demonstration with visualization."""
    # Create diamond lattice and NV center
    positions = create_diamond_lattice(size=2)
    print(f"Created diamond lattice with {len(positions)} atoms")

    # Find bonds between atoms
    bonds = find_nearest_neighbors(positions)
    print(f"Found {len(bonds)} bonds between atoms")

    # Create NV center
    atom_types, n_idx, v_idx = create_nv_center(positions)
    print(f"Created NV center: Nitrogen at index {n_idx}, Vacancy at index {v_idx}")

    # Generate interactive 3D visualization
    fig = interactive_nv_center(positions, atom_types, bonds)
    
    return fig

def create_nv_center_full():
    """Create a complete NV center demonstration returning all data."""
    # Create diamond lattice and NV center
    positions = create_diamond_lattice(size=2)
    print(f"Created diamond lattice with {len(positions)} atoms")

    # Find bonds between atoms
    bonds = find_nearest_neighbors(positions)
    print(f"Found {len(bonds)} bonds between atoms")

    # Create NV center
    atom_types, n_idx, v_idx = create_nv_center(positions)
    print(f"Created NV center: Nitrogen at index {n_idx}, Vacancy at index {v_idx}")

    # Generate interactive 3D visualization
    fig = interactive_nv_center(positions, atom_types, bonds)
    
    return fig, positions, atom_types, bonds
