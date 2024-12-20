# Interferometry CLEAN Simulation

[![Python package](https://github.com/skrbcr/interferometry_clean_sim/actions/workflows/python-package.yml/badge.svg)](https://github.com/skrbcr/interferometry_clean_sim/actions/workflows/python-package.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skrbcr/interferometry_clean_sim/blob/main/example/1_basic.ipynb)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square&logo=github)](https://skrbcr.github.io/interferometry_clean_sim/)

Simulation of CLEAN algorithm for interferometry

## About

This project simulates the `CLEAN` algorithm used in radio interferometry.
In radio astronomy observation, the raw image retreived from the antennas contains so much noise.
This algorithm removes noise from it and obtains true image.

**Demonstration:**

Noise are removed and only the real object got cleared.

https://github.com/user-attachments/assets/0c978ea0-a2aa-40d7-92ec-e3422dbbad88

## Quick Play

You can try this simulation on Google Colab:

- [1_basic](https://colab.research.google.com/github/skrbcr/interferometry_clean_sim/blob/main/example/1_basic.ipynb)
- [2_minimum](https://colab.research.google.com/github/skrbcr/interferometry_clean_sim/blob/main/example/2_minimum.ipynb)
- [3_video](https://colab.research.google.com/github/skrbcr/interferometry_clean_sim/blob/main/example/3_video.ipynb)

These files are the same as those in the `example` folder of this repository.

## Usage

### Requirements

```
numpy
matplotlib
opencv-python
scipy
```

You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Directory Structure

```
.
├── CLEAN                     # CLEAN simulation module
├── example                   # Usage and examples
│   ├── image                 # Image for `example` directory
├── LICENSE
├── README.md
├── requirements.txt
```

Some folders or files are omitted in the above illustration.

### Usage and examples

Please refer to the `example` folder for usage instructions.

Also, please refer to the [docs](https://skrbcr.github.io/interferometry_clean_sim/).

## To Do

Although I am a bit fatigued with this project, I still have some ideas for improvements:

- [ ] Add support for various antenna configurations.
- [ ] Enable importing real antenna configuration data.
- [ ] Conduct more experiments and demonstrations.
- [ ] In practical CLEAN processes, the size of the synthesized beam is typically about 5 pixels, but this cannot be achieved in the current simulation.
- [ ] Implement Briggs weighting.
- [ ] Improve the UV gridding implementation, as I am not fully confident in its accuracy.
- [ ] Explore additional ways to enhance and refine the simulation.

I welcome any contributions or questions.
Feel free to contact me anytime.

## References

- [Högbom, J. (1974), ``Aperture synthesis with a non-regular distribution of interferometer baselines'', Astrophys. J. Suppl. Ser., **15**, 417-426.](https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H/abstract)
- [The `CLEAN' algorithm --- NRAO](https://www.cv.nrao.edu/~abridle/deconvol/node7.html)
