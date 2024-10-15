import sys
sys.path.append('.')
from CLEAN import CLEAN

def test_clean_basic():
    imagefile = './example/image/point.png'

    clean = CLEAN()
    clean.set_antenna_array('random', 20, b_min=0.01)
    vis, imsize = clean.create_visibility(imagefile)
    psf, model, residual, clean_image = clean.clean(vis, imsize, 'natural')

    assert psf is not None
    assert model is not None
    assert residual is not None
    assert clean_image is not None

