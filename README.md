# Multiple Regression for Deep Inelastic Scattering Experiments

This project applies multiple regression using a least squares fit to polynomials of degree 3 and degree 4. The goal is to model the `Rsum` (summed polarization) from electron-proton deep inelastic scattering experiments, analyzing the behavior of the polarization and cross-section in the context of K meson scattering. The results are compared using the reduced chi-squared to determine the goodness of fit between polynomials of different degrees.

## Key Concepts

- **Rsum**: The sum of polarizations in the experiment, dependent on the polarization transferred from the electron beam to the recoil hyperon, the polarization of virtual photons exchanged, and the total cross-section (scattering strength).
- **Cosine Theta (cos(θ))**: The angle between the incident beam and the reaction plane.
- **Extrapolation**: At θ = 0, polarization and Rsum values need to be extrapolated due to detector limitations.
- **Structure Functions**: Used to study quark-gluon dynamics, quark-antiquark momentum distributions, and gluon dynamics. They are crucial for testing Quantum Chromodynamics (QCD) predictions.

## Objective

The goal is to fit `Rsum` as a function of `cos(θ)` using a polynomial regression model, and assess the fit using chi-squared minimization. The models (degree 3 and degree 4) are compared using reduced chi-squared values to determine the best model.

## Methodology

1. **Data**: The data consists of observables from deep inelastic scattering experiments, provided as text files with columns representing variables such as `Q2`, `W`, `cos(θ)`, and polarizations.
2. **Fitting**: The regression uses symbolic algebra for the polynomial fit. The chi-squared function is minimized to optimize the fit parameters.
3. **Error Propagation**: Uncertainties in the measurements (e.g., `dR`) are propagated throughout the fitting procedure.
4. **Covariance Matrix**: The covariance matrix is computed by taking the inverse of the curvature matrix derived from the fitting procedure. This matrix provides uncertainty estimates for the parameters.

## Steps

1. **Load Data**: The program loads multiple datasets of deep inelastic scattering observables from text files.
2. **Fit Polynomial Model**: For each dataset, the program fits a polynomial (degree 3 or degree 4) to the `Rsum` vs `cos(θ)` data.
3. **Minimize Chi-Squared**: The chi-squared value is minimized to determine the optimal fit parameters.
4. **Error Bands**: The error bands for the fit are calculated using the covariance matrix and parameter uncertainties.
5. **Plotting**: Plots of the fitted function with error bands are generated for each dataset, and the reduced chi-squared values are computed and displayed.

## Data Files

- **`.txt` Files**: Each text file contains data separated by bins of invariant energy `W`. The columns include `Q2`, `W`, `cos(θ)`, `eps`, `o`, `do`, `pz1`, `dpz1`, `pz`, `dpz`, `c`, `R`, and `dR`.
- **`all_sigma_odoc1.csv`**: Contains the energies and other related parameters (`o1`, `do1`, `c1`, `wp`) for each dataset.

## Outputs

- **Fitted Functions**: A plot of the fitted function $f(x)$ (polynomial regression) with error bands for each dataset.
- **Chi-Squared Results**: The reduced chi-squared value for each polynomial fit (degree 3 and 4).
- **Parameter Estimates**: The optimized parameters for the polynomial models.

## Project Structure  
```
├── code/  
│   ├── polarization_analysis.py  # Main Python script performing regression analysis  
├── plots/ # Rsum plots for all datasets - degree 3 and degree 4 fits
├── data/ # All data files (.txt, .csv) needed to run the script
└── README.md
```


## Usage

1. Ensure you have the required data files (`*.txt` and `all_sigma_odoc1.csv`) in the `data` directory.
2. Run the script located in `src/regression_analysis.py`. It will:
   - Read the data from the text files and the CSV file.
   - Fit the polynomial models (degree 3 and 4) to the data.
   - Generate plots of the fitted function and error bands.
   - Print the reduced chi-squared value for each fit.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `sympy`

## Example Plot

The plots display the fitted polynomial (degree 3 or 4) with shaded error bands and data points with uncertainties.

## Notes

- The data at `cos(θ) = 1` needs to be extrapolated due to detector limitations.
- This analysis is useful for investigating quark-gluon dynamics in high-energy physics.
