import sys
import numpy as np
sys.path.append('.')

from CLEAN import CLEAN


def test_clean_basic():
    imagefile = './example/image/point.png'

    clean = CLEAN()
    clean.set_antenna_array('random', 20, b_min=0.01)
    vis, imsize = clean.create_visibility(imagefile)
    threshold = 1e-6

    # Dirty image
    psf, _, residual, _ = clean.clean(vis, imsize, 'uniform')
    # Check if residual of point source is the same as psf
    assert psf is not None
    assert np.all(np.isclose(residual, psf * np.max(residual), atol=1e-6))

    # Clean
    psf, model, residual, clean_image = clean.clean(vis, imsize, 'uniform', threshold=threshold, n_iter=10000)

    # Check abs(vis) == 1 for all uv points
    assert np.all(np.isclose(np.abs(vis), 1, atol=1e-6))

    # Check if model have peak at (50, 50) and the rest is zero
    assert model is not None
    assert model[50, 50] != 0
    temp_model = model.copy()
    temp_model[50, 50] = 0
    assert np.all(temp_model == 0) or np.all(np.isclose(temp_model, 0, atol=1e-6))

    # Check abs(residual) < threshold
    assert residual is not None
    assert np.all(np.abs(residual) < threshold)

    assert clean_image is not None
