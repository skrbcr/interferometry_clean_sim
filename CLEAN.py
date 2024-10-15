import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats.qmc import PoissonDisk
from scipy.ndimage import gaussian_filter, rotate
from PoissonDiskSampling import PoissonDiskSampling
from GaussianFitting import fit_psf_gaussian

class CLEAN:
    def __init__(self):
        self.pos_antennas = None
        self.uv_coverage = None


    def set_antenna_array(self, geometry, n_antennas, b_min=None, b_max=None, random_seed=1):
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
            self.pos_antennas = np.array(pds.random(n_antennas, max_iter=10000))
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

        return self.pos_antennas.copy(), self.uv_coverage.copy()


    def weight_uv_coverage(self, imsize, weighting, robust):
        uv_grid = np.zeros((imsize, imsize))
        if weighting == 'natural':
            for u, v in self.uv_coverage:
                u_index = int(u * imsize) + imsize // 2
                v_index = int(v * imsize) + imsize // 2
                if 0 <= u_index < imsize and 0 <= v_index < imsize:
                    uv_grid[u_index, v_index] += 1
                else:
                    print(f'Warning: (u, v) = ({u}, {v}) is outside the UV grid')
        elif weighting == 'uniform':
            for u, v in self.uv_coverage:
                u_index = int(u * imsize) + imsize // 2
                v_index = int(v * imsize) + imsize // 2
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


    def create_psf(self, imsize, weighting, robust):
        # vis_psf = np.full((imsize, imsize), 1)
        uv_grid = self.weight_uv_coverage(imsize, weighting, robust)
        # Create the PSF
        # psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))))
        psf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_grid))).real
        # Normalize the PSF
        psf /= np.max(psf)
        print('PSF created.')
        return psf


    def create_visibility(self, imagefile):
        # Load image as grayscale
        image = cv.imread(imagefile, cv.IMREAD_GRAYSCALE)

        # If the image is not square, crop it to a square
        if image.shape[0] != image.shape[1]:
            print('Warning: This program only supports square images for now. Cropping the image to a square.')
            size = min(image.shape[0], image.shape[1])
            image = image[:size, :size]
        imsize = image.shape[0]

        # Normalize the image
        image = image.astype(float) / 255# * (imsize ** 2)
        # # 180度回転
        # image = np.rot90(image, 2)
        # plt.imshow(image, cmap='hot')
        # plt.show()

        # Create the visibility data
        vis_full = np.fft.fftshift(np.fft.fft2(image))

        # # Sample the visibility data
        # vis_sampled = np.zeros((imsize, imsize), dtype=np.complex128)
        # for u, v in self.uv_coverage:
        #     u_index = int(u * imsize) + imsize // 2
        #     v_index = int(v * imsize) + imsize // 2
        #     if 0 <= u_index < imsize and 0 <= v_index < imsize:
        #         vis_sampled[u_index, v_index] = vis_full[u_index, v_index]
        #     else:
        #         print(f'Warning: (u, v) = ({u}, {v}) is outside the UV grid')

        return vis_full, imsize


    def get_synthesized_beamed_image(self, image, psf):
        sigma_x, sigma_y, theta = fit_psf_gaussian(psf)
        max_value = np.max(image)
        max_index = np.unravel_index(np.argmax(image), image.shape)
        beamed_image = rotate(image, np.degrees(theta), reshape=False)
        beamed_image = gaussian_filter(beamed_image, sigma=[sigma_x, sigma_y])
        beamed_image = rotate(beamed_image, -np.degrees(theta), reshape=False)
        if beamed_image[max_index] != 0:
            beamed_image *= max_value / beamed_image[max_index]
        return beamed_image


    def clean(self, vis, imsize, weighting, robust=0.5, n_iter=0, threshold=0.1, mask=None, gamma=0.2):
        # Create the PSF
        psf = self.create_psf(imsize, weighting, robust)

        # Initialize the model and residual
        model = np.zeros((imsize, imsize), dtype=float)
        uv_grid = self.weight_uv_coverage(imsize, weighting, robust)
        residual = np.fft.ifft2(np.fft.ifftshift(vis * uv_grid)).real

        # mask
        if mask is not None:
            mask = cv.imread(mask, cv.IMREAD_GRAYSCALE)
            mask = mask.astype(float) / 255
            # size of mask must be the same as the residual
            if mask.shape[0] != imsize or mask.shape[1] != imsize:
                raise ValueError('The size of the mask must be the same as the image.')
        else:
            # all pixels are 1
            mask = np.ones((imsize, imsize))

        # Iterate
        if n_iter <= 0:
            print('Warning: n_iter is set to less than 1. I will restore the dirty image.')
        for i in range(n_iter):
            # Find the peak in the residual
            peak = np.unravel_index(np.argmax(np.abs(residual * mask)), residual.shape)
            value = residual[peak]
            if np.abs(value) < threshold:
                print(f'Iteration {i + 1}: Peak value {value} is below threshold {threshold}. Stopping the iteration.')
                break
            value *= gamma
            # Add the peak to the model
            model[peak] += value
            # Shift the PSF to the peak
            shifted_psf = np.roll(np.roll(psf, peak[0] - imsize // 2, axis=0), peak[1] - imsize // 2, axis=1)
            # Subtract the peak from the residual
            residual -= shifted_psf * value

        # Calculate synthesized beam
        sigma_x, sigma_y, theta = fit_psf_gaussian(psf)

        # normalize_factor = total_flux / np.sum(model)
        # model *= normalize_factor
        # residual *= normalize_factor

        # Create image from model and residual
        image = self.get_synthesized_beamed_image(model, psf) + residual
        # image = model + residual

        return psf, model, residual, image

