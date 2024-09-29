import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats.qmc import PoissonDisk
from PoissonDiskSampling import PoissonDiskSampling

class CLEAN:
    def __init__(self):
        self.pos_antennas = None
        self.uv_coverage = None
        self.uv_grid = None

    def set_antenna_array(self, geometry, n_antennas, b_min=None, b_max=None, random_seed=0):
        """
        geometry (str): 'random'
        n_antennas (int): number of antennas
        """
        if n_antennas < 2:
            raise ValueError('At least 2 antennas are required.')
        # Generate antenna configuration
        if geometry == 'random':
            if b_min is None:
                raise ValueError('b_min must be provided for "random" geometry.')
            if b_max is None:
                b_max = 0.5
            pds = PoissonDiskSampling(b_min, b_max, random_seed)
            self.pos_antennas = np.array(pds.random(n_antennas))
        else:
            raise NotImplementedError
        # Construct uv coverage
        self.uv_coverage = []
        for i in range(n_antennas):
            for j in range(n_antennas):
                if i == j:
                    continue
                u = self.pos_antennas[i, 0] - self.pos_antennas[j, 0]
                v = self.pos_antennas[i, 1] - self.pos_antennas[j, 1]
                self.uv_coverage.append((u, v))
        self.uv_coverage = np.array(self.uv_coverage)
        # Plot antenna configuration and uv coverage
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax in axs:
            ax.set_aspect('equal', 'box')

        # Plot antenna configuration
        axs[0].scatter(self.pos_antennas[:, 0], self.pos_antennas[:, 1], c='red', s=5)
        axs[0].set_title('Antenna Configuration')
        axs[0].set_xlabel('X Position')
        axs[0].set_ylabel('Y Position')
        axs[0].set_aspect('equal', 'box')

        # Plot UV coverage
        axs[1].scatter(self.uv_coverage[:, 0], self.uv_coverage[:, 1], c='blue', s=5)
        axs[1].set_title('UV Coverage')
        axs[1].set_xlabel('U (spatial frequency)')
        axs[1].set_ylabel('V (spatial frequency)')
        axs[1].set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()

    def weight_uv_coverage(self, imsize, weighting, robust):
        uv_grid = np.zeros((imsize, imsize))
        uv_step = 1 / imsize
        if weighting == 'natural':
            for u, v in self.uv_coverage:
                u_index = int(u / uv_step) + imsize // 2
                v_index = int(v / uv_step) + imsize // 2
                if 0 <= u_index < imsize and 0 <= v_index < imsize:
                    uv_grid[u_index, v_index] += 1
                else:
                    print(f'Warning: (u, v) = ({u}, {v}) is outside the UV grid')
        elif weighting == 'uniform':
            for u, v in self.uv_coverage:
                u_index = int(u / uv_step) + imsize // 2
                v_index = int(v / uv_step) + imsize // 2
                if 0 <= u_index < imsize and 0 <= v_index < imsize:
                    uv_grid[u_index, v_index] = 1
                else:
                    print(f'Warning: (u, v) = ({u}, {v}) is outside the UV grid')
        elif weighting == 'briggs':
            # w = np.zeros((imsize, imsize))
            # for u, v in self.uv_coverage:
            #     u_index = int(u / uv_step) + imsize // 2
            #     v_index = int(v / uv_step) + imsize // 2
            #     if 0 <= u_index < imsize and 0 <= v_index < imsize:
            #         w[u_index, v_index] += 1
            #     else:
            #         print(f'Warning: (u, v) = ({u}, {v}) is outside the UV grid')
            print('Briggs weighting is not implemented yet.')
        else:
            raise ValueError(f'Invalid weighting scheme "{weighting}".')
        return uv_grid

    def create_psf(self, imsize, weighting='briggs', robust=0.5):
        vis_psf = np.full((imsize, imsize), 1)
        self.uv_grid = self.weight_uv_coverage(imsize, weighting, robust)
        # Create the PSF
        psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.uv_grid))))
        # Create a figure with two subplots (side-by-side)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Plot the UV coverage on the first subplot
        im = ax.imshow(psf)
        fig.colorbar(im, ax=ax)
        ax.set_title('Point Spread Function (PSF)')
        ax.set_aspect('equal')
        
        # Adjust the layout
        plt.tight_layout()
        plt.show()

    def create_dirty_image(self, imsize, weighting, robust):
        pass

    # Implement the CLEAN algorithm
    def clean(self):
        pass

