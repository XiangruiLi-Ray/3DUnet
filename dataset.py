from ase.io import read
from ase import Atom, Atoms
from ase.visualize import view
from scipy.ndimage import gaussian_filter
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import distance
from math import ceil, floor
import matplotlib.pyplot as plt
    

class AseDataset():
    def __init__(self, atoms: Atoms, resolution: float = 0.4, sigma: float = 0.1): #unit A
        self.atoms = atoms
        self.positions = self.atoms.get_positions(wrap=True)
        self.resolution = resolution
        self.sigma = sigma

    def create_grid(self):
        """
        Encode 3D image to 3D kernel density estimation with shape n*n*n*1
        """
        xcell, ycell, zcell = self.atoms.get_cell()
        x = np.linspace(0, max(xcell), ceil(max(xcell)/self.resolution))
        y = np.linspace(0, max(ycell), ceil(max(xcell)/self.resolution))
        z = np.linspace(0, max(zcell), ceil(max(xcell)/self.resolution))
        X, Y, Z = np.meshgrid(x, y, z)
    
        # Reshape grid points for KDE
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        datapoints = self.positions.T

        # Perform KDE
        kde = gaussian_kde(datapoints, bw_method=self.sigma)

        density = kde(grid_points)

        density_reshape = np.array(density).reshape(X.shape[0], Y.shape[0], Z.shape[0])

        # Convert to tensor and reshape to (C, D, H, W) format
        data = torch.from_numpy(density_reshape).float()
        data = data.unsqueeze(0)  # Add channel dimension at the beginning
        
        # Now normalize with the correct shape
        normalize = transforms.Normalize(mean=[0], std=[0.00001])
        ndata = normalize(data)

        return ndata
    
    def padding(self, tensor):
        """
        Padding tensor to make sure it can be divided by 4 (for our 3DUnet structure)
        """
        # Get current dimensions
        shape = tensor.shape
        
        # Calculate new dimensions (next multiple of 4)
        # Skip first dimension (channel)
        new_dims = [shape[0]]  # Keep channel dimension as is
        for dim in shape[1:]:
            # Calculate how much to add to make divisible by 4
            remainder = dim % 4
            padding_needed = 0 if remainder == 0 else 4 - remainder
            new_dims.append(dim + padding_needed)
        
        # Calculate padding
        # Format: (left, right, top, bottom, front, back)
        padding = []
        for i in range(len(shape) - 1, 0, -1):  # Reverse order, skip first dim
            padding.append(0)  # Left/top/front padding
            padding.append(new_dims[i] - shape[i])  # Right/bottom/back padding
        
        # Apply padding
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode = 'circular')
        
        return padded_tensor

    def augmentation(self, image_tensor, num_augmentations=20):
        """
        Augment a single 3D image tensor by applying translations and 90-degree rotations.
        """
        # Create a list to store augmented images
        augmented_images = [image_tensor]
        
        # Get dimensions
        c, d, h, w = image_tensor.shape
        
        for i in range(num_augmentations):
            # Create a copy of the original
            aug_img = image_tensor.clone()
            
            # Randomly decide augmentation type
            aug_type = np.random.choice(['translation', 'rotation', 'both'])
            
            # Apply translation
            if aug_type in ['translation', 'both']:
                # Random translation (-5 to 5 pixels in each dimension)
                shift_d = np.random.randint(-5, 6)
                shift_h = np.random.randint(-5, 6)
                shift_w = np.random.randint(-5, 6)
                
                # Apply periodic boundary shift using roll (circular shift)
                if shift_d != 0:
                    aug_img = torch.roll(aug_img, shifts=shift_d, dims=1)
                if shift_h != 0:
                    aug_img = torch.roll(aug_img, shifts=shift_h, dims=2)
                if shift_w != 0:
                    aug_img = torch.roll(aug_img, shifts=shift_w, dims=3)
    
            
            # Apply rotation (90-degree increments around random axis)
            if aug_type in ['rotation', 'both']:
                # Choose a random axis and number of 90-degree rotations
                axis = np.random.randint(0, 3)  # 0=depth, 1=height, 2=width
                k = np.random.randint(1, 4)  # 1, 2, or 3 times 90 degrees
                
                # Rotate using torch.rot90
                if axis == 0:  # Rotate around depth axis
                    aug_img = torch.rot90(aug_img, k, dims=[2, 3])
                elif axis == 1:  # Rotate around height axis
                    aug_img = torch.rot90(aug_img, k, dims=[1, 3])
                else:  # Rotate around width axis
                    aug_img = torch.rot90(aug_img, k, dims=[1, 2])
            
            augmented_images.append(aug_img)
        
        # Concatenate all augmented images
        augmented_dataset = torch.stack(augmented_images, dim=0)
        
        return augmented_dataset

    def save_npy(self, output) -> None:
        data  = self.create_grid()
        data = self.padding(data)
        data = self.augmentation(data, 20)
        np.save(f'{output}', data)