import sys
sys.path.append('..')
from CLEAN import CLEAN

if __name__ == '__main__':
    clean = CLEAN()
    clean.set_antenna_array('random', 40, b_min=0.01, random_seed=0)
    clean.create_psf(100, 'natural', 0)

