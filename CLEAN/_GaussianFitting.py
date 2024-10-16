import numpy as np
from scipy.optimize import curve_fit


def _gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, theta, amplitude):
    """
    2D Gaussian function for fitting

    Args:
        x (np.ndarray): x-coordinates
        y (np.ndarray): y-coordinates
        x0 (float): x-coordinate of the center
        y0 (float): y-coordinate of the center
        sigma_x (float): standard deviation in x
        sigma_y (float): standard deviation in y
        theta (float): rotation angle in radians
        amplitude (float): amplitude

    Returns:
        np.ndarray: 2D Gaussian function
    """
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y**2)
    return amplitude * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))


def _fit_psf_gaussian(psf):
    """
    Fit a 2D Gaussian to the PSF to estimate the synthesized beam.

    Parameters:
        psf: 2D numpy array representing the PSF.

    Returns:
        popt: Optimal values for the Gaussian fit (x0, y0, sigma_x, sigma_y, theta, amplitude).
    """
    # Create a grid of x, y coordinates
    imsize = psf.shape[0]
    x = np.arange(0, imsize)
    y = np.arange(0, imsize)
    x, y = np.meshgrid(x, y)

    # Initial guess for the parameters (x0, y0, sigma_x, sigma_y, theta, amplitude)
    x0, y0 = np.unravel_index(np.argmax(psf), psf.shape)
    initial_guess = (1, 1, 0)  # Assume initial sigma_x = sigma_y = 1 and no rotation

    # Flatten the arrays for curve fitting
    x_data = np.ravel(x)
    y_data = np.ravel(y)
    psf_data = np.ravel(psf)

    # Fit the Gaussian model
    popt, _ = curve_fit(lambda xy, sigma_x, sigma_y, theta:
                        _gaussian_2d(xy[0], xy[1], x0, y0, sigma_x, sigma_y, theta, 1),
                        (x_data, y_data), psf_data, p0=initial_guess)

    return popt
