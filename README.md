# Interferometry CLEAN Simulation

Simulation of CLEAN algorithm for interferometry

## About

This project simulates the CLEAN algorithm used in radio interferometry.

**Demonstration:**

https://github.com/user-attachments/assets/c2ec7665-78c7-4990-b531-f064e3f16574

## Quick Play

You can try this simulation on [Google Colab](https://drive.google.com/drive/folders/1PP8717rmmz6VSvvcDNJT5GyhRdU4WQIE?usp=drive_link).
The files in this Google Drive folder are the same as those in the `example` folder of this repository.

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
```

Some trivial files are omitted in the above illustration.

### Usage and examples

Please refer to the `example` folder for usage instructions.

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

