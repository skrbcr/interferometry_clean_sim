import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, rotate
from CLEAN._PoissonDiskSampling import _PoissonDiskSampling
from CLEAN._GaussianFitting import _fit_psf_gaussian


class CLEAN:
    def __init__(self):
        self.pos_antennas = None
        self.uv_coverage = None

    def set_antenna_array(self, geometry, n_antennas, b_min=None, b_max=None, theta=None, dt=None, Nt=1, random_seed=1):
        """
        Configure the antenna array.

        Args:
            geometry (str): geometry of the antenna array. Currently only 'random' is supported.
            n_antennas (int): number of antennas.
            b_min (float): minimum baseline length.
            b_max (float): maximum baseline length. This should be less than 0.5.
            theta (float): declination angle of the target in radians.
            dt (float): time interval between measurements in radians (2\\pi means 24 hours).
            Nt (int): number of time steps.
            random_seed (int): random seed for random number generation.

        Returns:
            tuple: A tuple containing:
                - pos_antennas (np.ndarray): Positions of antennas.
                - uv_coverage (np.ndarray): UV coverage.
        """
        if n_antennas < 2:
            raise ValueError('At least 2 antennas are required.')
        # Generate antenna configuration
        if geometry == 'random':
            if b_min is None:
                raise ValueError('b_min must be provided for "random" geometry.')
            if b_max is None:
                b_max = 0.5
            pds = _PoissonDiskSampling(b_min, b_max, random_seed)
            self.pos_antennas = np.array(pds.random(n_antennas, max_iter=10000))
        elif geometry == 'east-west':
            if b_min is None and b_max is None:
                raise ValueError('b_min or b_max must be provided for "east-west" geometry.')
            if b_min is None:
                b_min = b_max
            if n_antennas > 2:
                print('Warning: N > 2 is not supported for "east-west" geometry. N is set to 2.')
            self.pos_antennas = np.array([[-b_min / 2, 0], [b_min / 2, 0]])
        else:
            raise NotImplementedError
        # Construct uv coverage
        uv_coverage = []
        for i in range(n_antennas):
            for j in range(n_antennas):
                if i == j:
                    continue
                u = self.pos_antennas[i, 0] - self.pos_antennas[j, 0]
                v = self.pos_antennas[i, 1] - self.pos_antennas[j, 1]
                uv_coverage.append((u, v))

        # Time evolution of the antenna configuration
        uv_coverage_add = []
        if theta is None:
            theta = np.pi / 2
        if dt is None:
            dt = np.pi / 12
        # for u, v in uv_coverage:
        #     a = np.sqrt(u ** 2 + v ** 2 / np.sin(theta) ** 2)
        #     b = a * np.sin(theta)
        #     t0 = np.arctan2(v, u)
        #     for i in range(1, Nt):
        #         u_new = a * np.cos(i * dt + t0)
        #         v_new = b * np.sin(i * dt + t0)
        #         uv_coverage_add.append((u_new, v_new))

        self.uv_coverage = np.array(uv_coverage + uv_coverage_add)

        return self.pos_antennas.copy(), self.uv_coverage.copy()

    def weight_uv_coverage(self, imsize, weighting, robust):
        """
        Create a UV grid from the UV coverage.

        Args:
            imsize (int): size of the image. The UV grid will have the same size.
            weighting (str): weighting scheme. Currently supports 'natural' and 'uniform'.
            robust (float): robust parameter for the Briggs weighting. This is currently meaningless.

        Returns:
            np.ndarray: UV grid.
        """
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
        """
        Create the point spread function (PSF).

        Args:
            imsize (int): size of the image. The PSF will have the same size.
            weighting (str): weighting scheme. Currently supports 'natural' and 'uniform'.
            robust (float): robust parameter for the Briggs weighting. This is currently meaningless.

        Returns:
            np.ndarray: PSF.
        """
        # vis_psf = np.full((imsize, imsize), 1)
        uv_grid = self.weight_uv_coverage(imsize, weighting, robust)
        # Create the PSF
        psf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_grid))).real
        # Normalize the PSF
        psf /= np.max(psf)
        print('PSF created.')
        return psf

    def create_visibility(self, imagefile):
        """
        Create the visibility data from the image.

        Args:
            imagefile (str): path to the image file.

        Returns:
            tuple: A tuple containing:
                - vis_full (np.ndarray): full visibility data.
                - imsize (int): size of the image.
        """
        # Load image as grayscale
        image = cv.imread(imagefile, cv.IMREAD_GRAYSCALE)

        # If the image is not square, crop it to a square
        if image.shape[0] != image.shape[1]:
            print('Warning: This program only supports square images for now. Cropping the image to a square.')
            size = min(image.shape[0], image.shape[1])
            image = image[:size, :size]
        imsize = image.shape[0]

        # Normalize the image
        image = image.astype(float) / 255

        # Create the visibility data
        vis_full = np.fft.fftshift(np.fft.fft2(image))

        return vis_full, imsize

    def get_synthesized_beamed_image(self, image, psf):
        """
        Create the synthesized beamed image from the image and the PSF.

        Args:
            image (np.ndarray): image.
            psf (np.ndarray): PSF.

        Returns:
            np.ndarray: synthesized beamed image.
        """
        sigma_x, sigma_y, theta = _fit_psf_gaussian(psf)
        max_value = np.max(image)
        max_index = np.unravel_index(np.argmax(image), image.shape)
        beamed_image = rotate(image, np.degrees(theta), reshape=False)
        beamed_image = gaussian_filter(beamed_image, sigma=[sigma_x, sigma_y])
        beamed_image = rotate(beamed_image, -np.degrees(theta), reshape=False)
        if beamed_image[max_index] != 0:
            beamed_image *= max_value / beamed_image[max_index]
        return beamed_image

    def clean(self, vis, imsize, weighting, robust=0.5, n_iter=0, threshold=0.1, mask=None, gamma=0.2):
        """
        Clean the image.

        Args:
            vis (np.ndarray): visibility data.
            imsize (int): size of the image.
            weighting (str): weighting scheme. Currently supports 'natural' and 'uniform'.
            robust (float): robust parameter for the Briggs weighting. This is currently meaningless.
            n_iter (int): limit number of iterations.
            threshold (float): threshold for stopping the iteration.
            mask (str): path to the mask file.
            gamma (float): loop gain.

        Returns:
            tuple: A tuple containing:
                - psf (np.ndarray): PSF.
                - model (np.ndarray): model image.
                - residual (np.ndarray): residual image.
                - image (np.ndarray): cleaned image.
        """
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
        else:
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
                if i == n_iter - 1:
                    print(f'Maximum number of iterations {n_iter} reached.')

        # Calculate synthesized beam
        sigma_x, sigma_y, theta = _fit_psf_gaussian(psf)

        # Create image from model and residual
        image = self.get_synthesized_beamed_image(model, psf) + residual

        return psf, model, residual, image
