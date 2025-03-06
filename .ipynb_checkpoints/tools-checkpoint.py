import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ase import Atoms
from math import ceil
import torch


def project_atoms_2d(atoms: Atoms, resolution: float = 0.2, direction: str = 'x', sigma: float = 0.1, plot=True) -> None:
    """
    Project 3D atomic positions onto a 2D plane and create density distribution.
    """
    positions = atoms.get_positions(wrap=True)

    # Define projection planes
    proj_dict = {
        'x': (1, 2, 0),  # yz projection
        'y': (0, 2, 1),  # xz projection
        'z': (0, 1, 2)   # xy projection
    }
    
    if direction not in proj_dict:
        raise ValueError("Direction must be 'x', 'y', or 'z'")
        
    idx1, idx2, proj_idx = proj_dict[direction]
    
    # Extract coordinates for chosen projection
    x = positions[:, idx1]
    y = positions[:, idx2]
    
    print(type(x))

    # Create a grid
    xgrid = np.linspace(0, max(x), ceil(max(x)/resolution))
    ygrid = np.linspace(0, max(y), ceil(max(y)/resolution))
    X, Y = np.meshgrid(xgrid, ygrid)
    
    # Calculate density using gaussian KDE
    grid_points = np.vstack([x, y])
    kernel = gaussian_kde(grid_points, bw_method=sigma)
    
    # Evaluate density on the grid
    density = kernel(np.vstack([X.flatten(), Y.flatten()])).reshape(X.shape)
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot of atomic positions
        ax1.scatter(x, y, alpha=0.5)
        ax1.set_aspect('equal')
        ax1.set_xlabel(f'{["x", "y", "z"][idx1]} (Å)')
        ax1.set_ylabel(f'{["x", "y", "z"][idx2]} (Å)')
        ax1.set_title('Atomic positions')
        
        # Density plot
        im = ax2.pcolormesh(X, Y, density, shading='auto', cmap='viridis')
        ax2.set_aspect('equal')
        ax2.set_xlabel(f'{["x", "y", "z"][idx1]} (Å)')
        ax2.set_ylabel(f'{["x", "y", "z"][idx2]} (Å)')
        ax2.set_title('Density distribution')
        plt.colorbar(im, ax=ax2, label='Density')
        
        plt.tight_layout()
        plt.show()


def project_tensor_2d(direction, tensor):
    tensor = tensor.squeeze(0)  # Now shape is (36, 36, 36)
    
    # Project along specified direction
    if direction == 'z':
        projection = torch.mean(tensor, dim=2)  # Average along z-axis
    elif direction == 'y':
        projection = torch.mean(tensor, dim=1)  # Average along y-axis
    elif direction == 'x':
        projection = torch.mean(tensor, dim=0)  # Average along x-axis
    else:
        raise ValueError("Direction must be 'x', 'y', or 'z'")
        
    return projection


def plot_2d_tensor():
    return


def perturb_atoms(atoms: Atoms, sigma):
    positions = atoms.get_positions()
    noise = np.random.normal(loc=0, scale=sigma, size=(len(atoms), 3))
    new_positions = positions + noise
    atoms.set_positions(newpositions=new_positions)
    atoms.wrap()
    return atoms