import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import cv2 as cv
sys.path.append('.')
from CLEAN import CLEAN

if __name__ == '__main__':
    clean = CLEAN()
    imagefile = './example/image/structure.png'
    # maskfile = './image/structure_mask.png'
    maskfile = None

    # Set antenna array
    antenna_pos, uv_coverage = clean.set_antenna_array('random', 40, b_min=0.01, random_seed=0, Nt=4, theta=np.pi / 4)

    # Plot antenna positions and uv coverage
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # for ax in axs:
    #     ax.set_aspect('equal', 'box')
    # axs[0].scatter(antenna_pos[:, 0], antenna_pos[:, 1])
    # axs[0].set_title('Antenna Positions')
    # axs[1].scatter(uv_coverage[:, 0], uv_coverage[:, 1])
    # axs[1].set_title('UV Coverage')
    # plt.show()

    # Create visibility
    vis, imsize = clean.create_visibility(imagefile)

    # Clean the image
    psf, model, residual, image = clean.clean(vis, imsize, 'uniform', n_iter=0, threshold=1e-16, mask=maskfile, gamma=0.2)

    # Load the original image (true image)
    true_image = cv.imread(imagefile, cv.IMREAD_GRAYSCALE)
    true_image = true_image.astype(float) / 255.0  # Normalize the image

    # Plot true image, psf, model, and residual in 2x2 grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # True image
    im1 = axs[0, 0].imshow(true_image, cmap='hot')
    axs[0, 0].set_title('True Image')
    plt.colorbar(im1, ax=axs[0, 0])

    # PSF
    im2 = axs[0, 1].imshow(psf, cmap='hot')
    axs[0, 1].set_title('PSF')
    plt.colorbar(im2, ax=axs[0, 1])

    # Model
    im3 = axs[0, 2].imshow(model, cmap='hot')
    axs[0, 2].set_title('Model')
    plt.colorbar(im3, ax=axs[0, 2])

    # Residual
    im4 = axs[1, 0].imshow(residual, cmap='hot')
    axs[1, 0].set_title('Residual')
    plt.colorbar(im4, ax=axs[1, 0])

    # Cleaned image
    im5 = axs[1, 1].imshow(image, cmap='hot')#, vmin=-0.01, vmax=0.15)
    axs[1, 1].set_title('Cleaned Image')
    plt.colorbar(im5, ax=axs[1, 1])

    # Plot ideal image (image obtained by convolving the true image with the synthesized beam)
    ideal_image = clean.get_synthesized_beamed_image(true_image, psf)
    im6 = axs[1, 2].imshow(ideal_image, cmap='hot')
    axs[1, 2].set_title('Ideal Image')
    plt.colorbar(im6, ax=axs[1, 2])

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Calculate SNR
    signal = np.max(image)
    noise = np.std(residual)
    print(f'Signal: {signal}')
    print(f'Noise: {noise}')
    print(f'SNR: {signal / noise}')
