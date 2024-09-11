# scalar-compton-wavelength

This repository accompanies the scientific paper titled _Galactic Compton Wavelengths in $f(R)$ Screening Theories_. The codebase is designed to calculate the $f(R)$ field for a given density profile and generate the figures presented in the paper.

_This work follows on and uses many similar methods to those in our previous paper ([DOI 10.1088/1475-7516/2024/04/004](https://iopscience.iop.org/article/10.1088/1475-7516/2024/04/004)). The code associated with our previous work is also publicly available at [Bradley-March/GalacticScreeningConditions](https://github.com/Bradley-March/GalacticScreeningConditions)._


## Key Components

#### `Solvers/`

This folder contains the numerical solver to calculate the $f(R)$ field for a given density profile. The finite differencing methods used in this section are explained in Section 3.2 of our previous work.

- `Solvers/fR2D.py`: Implements the `fR2DSolver` class, which solves the $f(R)$ equations of motion.
- `Solvers/utils.py`: Contains utility functions used by the solver.

#### `Packages/`

This folder contains various utility and analysis functions. 

- `Packages/galaxy_relations.py`: Determines the input galactic density. (See section 3.1 of the previous work for details on the galactic pipeline.)
- `Packages/fR_functions.py`: Provides utility functions and performs analysis of the $f(R)$ field.
- `Packages/compt_functions.py`: Calculates the $f(R)$ Compton wavelength and related quantities used in the paper.
- `Packages/utils.py`: Additional utility functions.

#### `constants.py`

This script contains the physical and cosmological parameters used throughout the codebase.

#### `plot_figures.py`

This script generates the figures presented in the paper. It uses the solutions calculated by the solver and performs the necessary plotting.

#### `run_solutions.py`

This script runs all the field solutions required to plot the figures in `plot_figures.py`.

## Usage

1. **Run Solutions**: To calculate the $f(R)$ field solutions, execute the `run_solutions.py` script. _Note: Running these solutions will take considerable time. Pre-saved solutions can be made available upon request (email: [bradley.march@nottingham.ac.uk](mailto:bradley.march@nottingham.ac.uk)).
._
    
2. **Plot Figures**: After producing the solutions, generate the figures by executing the `plot_figures.py` script.

## Dependencies

Ensure you have the following Python packages installed:
- `numpy`
- `matplotlib`
- `h5py`

## Authors

The code contained within the folder ```Solvers``` was created by [**Aneesh P. Naik**](https://github.com/aneeshnaik).

All other code was created by [**Bradley March**](https://github.com/Bradley-March).

## License

Copyright (2024) Bradley March.

`scalar-compton-wavelength` is free software made available under the MIT license. For details see LICENSE.


